"""
Dream-7B Single-GPU SFT Trainer
Ported from official DreamLM/Dream repo (fsdp_sft_trainer.py)
- q_sample(): gen_utils.py 그대로
- loss 계산: _compute_loss_and_backward() 에서 FSDP 제거
- 데이터: SFTDataset의 loss_mask 로직 (response-only)
- attention: 4D bidirectional mask
"""

import os
import json
import math
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt


# ============================================================
# 1. q_sample — 공식 gen_utils.py 그대로
# ============================================================
def q_sample(
    input_ids,
    maskable_mask,
    mask_token_id,
    min=0.0,
    max=1.0,
    eos_token_id=None,
    t=None,
    t_mask=None,
):
    x_0 = input_ids

    if t_mask is None:
        if t is None:
            t = torch.rand((x_0.shape[0],), dtype=torch.float, device=input_ids.device)
            t = min + (max - min) * t
        u = torch.rand_like(x_0, dtype=torch.float)
        t_mask = (u < t[:, None]) & maskable_mask

    x_t = x_0.masked_fill(t_mask, mask_token_id)

    if eos_token_id is not None:
        last_non_eos_token_idx = ((input_ids != eos_token_id) | (~maskable_mask)).sum(
            dim=-1
        ) - 1
        seq_len = x_0.shape[1]

        for i in range(x_0.shape[0]):
            if last_non_eos_token_idx[i] < seq_len - 1:
                t_mask_at_eos = t_mask[i, last_non_eos_token_idx[i] + 1]
                if t_mask_at_eos:
                    x_t[i, last_non_eos_token_idx[i] + 1 :] = mask_token_id
                    t_mask[i, last_non_eos_token_idx[i] + 1 :] = True
                else:
                    x_t[i, last_non_eos_token_idx[i] + 1 :] = eos_token_id
                    t_mask[i, last_non_eos_token_idx[i] + 1 :] = False

    return x_t, t, t_mask


# ============================================================
# 2. Context-Adaptive Reweighting (CART) — 공식 코드 그대로
# ============================================================
def context_adaptive_reweight(seq_len, distribution="symmetric-geometric", **kwargs):
    position_ids_l = np.arange(seq_len).reshape(-1, 1)
    position_ids_r = np.arange(seq_len).reshape(1, -1)
    distance = position_ids_l - position_ids_r
    distance = torch.from_numpy(distance)

    def geometric_distribution(k, cart_p=0.8, **kwargs):
        if not 0 < cart_p <= 1:
            raise ValueError("p must be between 0 and 1")
        res = (math.log(cart_p) + (k.abs() - 1) * math.log(1 - cart_p)).exp() * 0.5
        res.masked_fill_(k == 0, 0)
        return res

    if distribution == "symmetric-geometric":
        matrix = geometric_distribution(distance, **kwargs)
    else:
        raise ValueError(f"Unknown distribution {distribution}")

    return matrix


# ============================================================
# 3. Dataset — 공식 SFTDataset의 loss_mask 로직 포팅
# ============================================================
class DreamSFTDataset(Dataset):
    """
    공식 코드의 SFTDataset과 동일한 로직:
    - chat_template으로 prompt/response 토큰화
    - loss_mask: prompt=0, response=1
    - max_length까지 padding
    """

    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

        with open(jsonl_path) as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        # 공식 코드: apply_chat_template으로 prompt 생성
        prompt_messages = [{"role": "user", "content": ex["instruction"]}]
        prompt_str = self.tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=False
        )
        response_str = ex["output"] + self.tokenizer.eos_token

        # 공식 코드: prompt와 response 별도 토큰화
        prompt_out = self.tokenizer(
            prompt_str, return_tensors="pt", add_special_tokens=False
        )
        prompt_ids = prompt_out["input_ids"][0]
        prompt_attn = prompt_out["attention_mask"][0]

        response_out = self.tokenizer(
            response_str, return_tensors="pt", add_special_tokens=False
        )
        response_ids = response_out["input_ids"][0]
        response_attn = response_out["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]

        # 결합
        input_ids = torch.cat([prompt_ids, response_ids])
        attention_mask = torch.cat([prompt_attn, response_attn])

        # padding or truncation to max_length
        seq_len = input_ids.shape[0]
        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), self.pad_token_id, dtype=input_ids.dtype)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(pad_len, dtype=attention_mask.dtype)  # 공식 코드: pad도 attn=1
            ])
        elif seq_len > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        # loss_mask: prompt=0, response=1 (공식 코드 동일)
        loss_mask = attention_mask.clone()
        loss_mask[:min(prompt_length, loss_mask.size(0))] = 0

        # position_ids (공식 코드: compute_position_id_with_mask)
        position_ids = torch.cumsum(attention_mask, dim=0) - 1
        position_ids = position_ids.clamp(min=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask.bool(),
            "position_ids": position_ids,
            "loss_mask": loss_mask.bool(),
        }


# ============================================================
# 4. Loss 계산 — 공식 _compute_loss_and_backward() 포팅
# ============================================================
def compute_dream_loss(
    model,
    batch,
    mask_token_id,
    vocab_size,
    pad_eos_token_id=None,
    time_reweighting="original",
    token_reweighting=False,
    alpha=0.25,
    gamma=2.0,
    cart_p=0.1,
    treat_eos_as_one=False,
):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    position_ids = batch["position_ids"]
    loss_mask = batch["loss_mask"]

    labels = input_ids.contiguous()
    batch_size = input_ids.shape[0]

    # 공식 코드: q_sample으로 response 영역만 마스킹
    masked_input_ids, t, loss_mask_nonflatten = q_sample(
        input_ids,
        maskable_mask=loss_mask,
        mask_token_id=mask_token_id,
        eos_token_id=pad_eos_token_id if treat_eos_as_one else None,
    )
    loss_mask_flat = loss_mask_nonflatten.reshape(-1)

    # 공식 코드: 4D bidirectional attention mask
    attn_mask_4d = torch.logical_and(
        attention_mask.unsqueeze(1).unsqueeze(-2),
        attention_mask.unsqueeze(1).unsqueeze(-1),
    )

    # Forward
    output = model(
        input_ids=masked_input_ids,
        attention_mask=attn_mask_4d,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output.logits

    # 공식 코드: Dream 특유의 logit shift (자기 위치 예측)
    # cat([logits[:,0:1], logits[:,:-1]]) — AR의 next-token이 아님
    shift_logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1).contiguous()
    shift_labels = labels.contiguous()

    # Flatten
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1).to(shift_logits.device)

    # CE loss (reduction=none)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits, shift_labels)

    # 마스크된 위치만
    loss_mask_flat = loss_mask_flat.to(loss.device)
    loss = loss.masked_fill(~loss_mask_flat, 0)

    # 공식 코드: token reweighting (focal loss)
    if token_reweighting:
        loss = alpha * (1 - torch.exp(-loss)) ** gamma * loss

    # 공식 코드: time reweighting
    if time_reweighting == "original":
        weight = 1 / t[:, None].float().expand(labels.size())
    elif time_reweighting == "linear":
        weight = 1 - t[:, None].float().expand(labels.size())
    elif time_reweighting == "cart":
        seq_len = input_ids.shape[-1]
        weight_matrix = context_adaptive_reweight(seq_len, cart_p=cart_p)
        _weight_matrix = weight_matrix[:seq_len, :seq_len].to(loss_mask_flat.device)
        non_mask = ~loss_mask_nonflatten.to(loss.device)
        weight = (
            non_mask.type_as(_weight_matrix)
            .matmul(_weight_matrix)
            .masked_fill(non_mask, 0)
        )
    else:
        weight = t.new_ones((batch_size, 1)).float().expand(labels.size())

    loss = loss * weight.reshape(-1)

    # 공식 코드: masked token 수로 나누기
    valid_tokens = torch.sum(loss_mask_flat)
    loss = torch.sum(loss) / valid_tokens

    return loss, t.mean().item(), valid_tokens.item()


# ============================================================
# 5. Training Loop
# ============================================================
def train(args):
    print(f"{'='*60}")
    print(f"Dream-7B Official SFT (Single-GPU Port)")
    print(f"{'='*60}")

    # Disk safety
    usage = int(os.popen("df -h . | awk 'NR==2 {print $5}' | tr -d '%'").read().strip())
    if usage >= 95:
        print("DISK 95% — halted")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Model & Tokenizer ----
    print(f"\n[1/5] Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True
    )
    mask_token_id = tokenizer.mask_token_id
    assert mask_token_id is not None, "mask_token_id is None"
    print(f"  mask_token_id: {mask_token_id}")
    print(f"  pad_token_id:  {tokenizer.pad_token_id}")
    print(f"  eos_token_id:  {tokenizer.eos_token_id}")

    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    vram_base = torch.cuda.memory_allocated() / 1e9
    print(f"  Model class: {type(model).__name__}")
    print(f"  VRAM (base): {vram_base:.2f} GB")

    # ---- LoRA ----
    if args.lora_rank > 0:
        print(f"\n[2/5] Applying LoRA (r={args.lora_rank}, alpha={args.lora_alpha})")
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("\n[2/5] Full fine-tuning (no LoRA)")
        for p in model.parameters():
            p.requires_grad = True

    vram_lora = torch.cuda.memory_allocated() / 1e9
    vocab_size = model.config.vocab_size

    # ---- Data ----
    print(f"\n[3/5] Loading data: {args.train_file}")
    train_ds = DreamSFTDataset(args.train_file, tokenizer, max_length=args.max_length)
    val_ds = DreamSFTDataset(args.val_file, tokenizer, max_length=args.max_length)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.micro_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.micro_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
    )

    # ---- Optimizer & Scheduler (공식 config 동일) ----
    print(f"\n[4/5] Optimizer: AdamW lr={args.lr}, betas=(0.9, 0.95), wd=0.01")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    total_steps = args.epochs * len(train_loader) // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"  Total steps: {total_steps}, Warmup: {warmup_steps}")

    # ---- Training ----
    print(f"\n[5/5] Training for {args.epochs} epochs")
    print(f"  time_reweighting: {args.time_reweighting}")
    print(f"  token_reweighting: {args.token_reweighting}")
    print(f"  gradient_accumulation: {args.gradient_accumulation_steps}")
    print(f"  effective_batch: {args.micro_batch_size * args.gradient_accumulation_steps}")
    print(f"{'='*60}\n")

    train_log = []
    val_log = []
    best_val_loss = float("inf")
    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            # Move to GPU
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, t_mean, n_valid = compute_dream_loss(
                    model=model,
                    batch=batch,
                    mask_token_id=mask_token_id,
                    vocab_size=vocab_size,
                    pad_eos_token_id=tokenizer.pad_token_id,
                    time_reweighting=args.time_reweighting,
                    token_reweighting=args.token_reweighting,
                    treat_eos_as_one=args.treat_eos_as_one,
                )
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            epoch_losses.append(loss.item() * args.gradient_accumulation_steps)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg_loss = np.mean(epoch_losses[-args.gradient_accumulation_steps:])
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"  Epoch {epoch+1} | Step {global_step} | "
                        f"loss={avg_loss:.4f} | t_mean={t_mean:.3f} | "
                        f"valid_tok={n_valid:.0f} | lr={lr_now:.2e}"
                    )
                    train_log.append({
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "loss": avg_loss,
                        "lr": lr_now,
                    })

        # ---- Validation ----
        model.eval()
        v_losses = []
        with torch.no_grad():
            for vbatch in val_loader:
                vbatch = {k: v.cuda(non_blocking=True) for k, v in vbatch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    vloss, _, _ = compute_dream_loss(
                        model=model,
                        batch=vbatch,
                        mask_token_id=mask_token_id,
                        vocab_size=vocab_size,
                        pad_eos_token_id=tokenizer.pad_token_id,
                        time_reweighting=args.time_reweighting,
                        token_reweighting=args.token_reweighting,
                        treat_eos_as_one=args.treat_eos_as_one,
                    )
                v_losses.append(vloss.item())

        val_loss = np.mean(v_losses)
        train_loss = np.mean(epoch_losses)
        elapsed = (time.time() - start_time) / 60

        print(
            f"\n  === Epoch {epoch+1}/{args.epochs} done | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"elapsed={elapsed:.1f}min ==="
        )
        val_log.append({
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, "adapter_best")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  ** Best model saved → {save_path}")

    # ---- Final save ----
    total_time = (time.time() - start_time) / 60
    save_path = os.path.join(args.output_dir, "adapter_final")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # ---- Loss curve ----
    fig, ax = plt.subplots(figsize=(8, 4))
    if train_log:
        ax.plot(
            [d["global_step"] for d in train_log],
            [d["loss"] for d in train_log],
            label="Train Loss", alpha=0.7,
        )
    if val_log:
        ax.plot(
            [d["global_step"] for d in val_log],
            [d["val_loss"] for d in val_log],
            label="Val Loss", linestyle="--", marker="o",
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Dream-7B Official SFT | {args.time_reweighting} | lr={args.lr}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(
        os.path.join(args.output_dir, "loss_curve.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # ---- Meta ----
    meta = {
        "model": args.model_id,
        "method": "official_dream_sft_port",
        "data_train": args.train_file,
        "data_val": args.val_file,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.micro_batch_size * args.gradient_accumulation_steps,
        "time_reweighting": args.time_reweighting,
        "token_reweighting": args.token_reweighting,
        "treat_eos_as_one": args.treat_eos_as_one,
        "clip_grad": args.clip_grad,
        "vram_base_gb": round(vram_base, 2),
        "vram_lora_gb": round(vram_lora, 2),
        "total_steps": global_step,
        "train_time_min": round(total_time, 1),
        "final_train_loss": round(train_loss, 4) if epoch_losses else None,
        "final_val_loss": round(val_loss, 4) if v_losses else None,
        "best_val_loss": round(best_val_loss, 4),
        "train_log": train_log,
        "val_log": val_log,
    }
    json.dump(meta, open(os.path.join(args.output_dir, "training_meta.json"), "w"), indent=2)
    print(f"\nDone | {total_time:.1f}min | best_val={best_val_loss:.4f}")
    print(f"Outputs → {args.output_dir}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dream-7B Official SFT (Single-GPU)")

    # Model
    parser.add_argument("--model_id", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Data
    parser.add_argument("--train_file", default="data/train_1k.jsonl")
    parser.add_argument("--val_file", default="data/val_1k.jsonl")
    parser.add_argument("--max_length", type=int, default=512)

    # Training (공식 config 기본값)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)  # 공식: 1e-5
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    # Diffusion loss (공식 config 기본값)
    parser.add_argument("--time_reweighting", default="original",
                        choices=["original", "linear", "cart", "none"])
    parser.add_argument("--token_reweighting", action="store_true")  # 공식: false
    parser.add_argument("--treat_eos_as_one", action="store_true")

    # Output
    parser.add_argument("--output_dir", default="outputs/dream_official")
    parser.add_argument("--log_every", type=int, default=5)

    args = parser.parse_args()
    train(args)
