"""
Fast-dLLM v2 7B SFT — en→fi (OPUS-100 10K).

Uses the official objective: the model's own forward() applies block diffusion
masking + complementary mask + token shift internally. We just provide
input_ids and labels (labels=-100 for prompt + PAD tokens, real ids for
assistant response tokens), padded to a multiple of bd_size with mask_id.

Repo: https://github.com/NVlabs/Fast-dLLM   (v2/train_scripts/finetune.py)
Model: Efficient-Large-Model/Fast_dLLM_v2_7B
"""
import os, json, argparse, math
import torch
# Compat: transformers 5.x dropped "default" rope_type key
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
def _default_rope(config, device=None, seq_len=None, **kwargs):
    base = config.rope_theta
    dim = int(config.head_dim if hasattr(config, "head_dim") and config.head_dim else config.hidden_size // config.num_attention_heads)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, 1.0
ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope)
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments, default_data_collator)

MODEL_ID = "Efficient-Large-Model/Fast_dLLM_v2_7B"
MASK_TOK = "|<MASK>|"


def build_example(tok, ex, mask_id, bd_size, max_len):
    """Tokenize one (instruction, output) pair using the model's chat template.
    Returns (input_ids, labels) padded to multiple of bd_size."""
    user_msgs = [{"role": "user", "content": ex["instruction"]}]
    full_msgs = user_msgs + [{"role": "assistant", "content": ex["output"]}]
    def _tok(msgs, add_gen):
        r = tok.apply_chat_template(msgs, add_generation_prompt=add_gen, tokenize=True)
        if isinstance(r, list) and r and isinstance(r[0], int):
            return r
        # transformers 5.x BatchEncoding
        ids = r["input_ids"] if hasattr(r, "__getitem__") else r.input_ids
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            return ids[0]
        return list(ids)
    prompt_ids = _tok(user_msgs, True)
    full_ids = _tok(full_msgs, False)
    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]
    p = min(len(prompt_ids), len(full_ids))
    ids = list(full_ids)
    labels = [-100] * p + list(full_ids[p:])

    pad_len = (bd_size - len(ids) % bd_size) % bd_size
    if pad_len:
        ids.extend([mask_id] * pad_len)
        labels.extend([-100] * pad_len)
    attn = [1] * len(ids)
    return {"input_ids": ids, "labels": labels, "attention_mask": attn}


class JsonlDataset(Dataset):
    def __init__(self, path, tok, mask_id, bd_size, max_len):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok, self.mask_id, self.bd_size, self.max_len = tok, mask_id, bd_size, max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        return build_example(self.tok, self.rows[i], self.mask_id, self.bd_size, self.max_len)


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        n = L - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * n)
        out["labels"].append(b["labels"] + [-100] * n)
        out["attention_mask"].append(b["attention_mask"] + [0] * n)
    return {k: torch.tensor(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", default="data/train_10k.jsonl")
    p.add_argument("--output_dir", default="outputs/fastdllm_v2_10k")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--save_steps", type=int, default=4000)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--lora_rank", type=int, default=0, help="0 = full FT")
    p.add_argument("--lora_alpha", type=int, default=0)
    p.add_argument("--lora_target", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                   help="comma-sep modules, or 'all-linear'. Exclude lm_head: known bf16 NaN trigger.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer/model from {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    mask_id = tok.encode(MASK_TOK, add_special_tokens=False)[0]
    pad_id = tok.pad_token_id
    print(f"mask_id={mask_id}, pad_id={pad_id}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    bd_size = model.config.bd_size
    print(f"bd_size={bd_size}")
    # gradient_checkpointing disabled: transformers 5.x + bf16 + Fast-dLLM custom forward
    # produces NaN loss via Trainer path despite direct forward being finite.
    # model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if args.lora_rank > 0:
        from peft import LoraConfig, get_peft_model
        alpha = args.lora_alpha or args.lora_rank * 2
        tm = "all-linear" if args.lora_target == "all-linear" else args.lora_target.split(",")
        lcfg = LoraConfig(r=args.lora_rank, lora_alpha=alpha,
                          lora_dropout=0.05, bias="none",
                          target_modules=tm)
        model.enable_input_require_grads()
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()

    ds = JsonlDataset(args.train_file, tok, mask_id, bd_size, args.max_len)
    print(f"train examples: {len(ds)}")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        gradient_checkpointing=False,
        optim="adamw_torch",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=args.seed,
        max_grad_norm=1.0,
    )

    class NanGuardTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            out = model(**inputs)
            loss = out.loss if hasattr(out, "loss") else out["loss"]
            if not torch.isfinite(loss):
                # surface the event but keep training alive: zero loss with grad-graph attached
                print(f"[NanGuard] non-finite loss={loss.item()} → skipping update")
                loss = sum(p.sum() * 0.0 for p in model.parameters() if p.requires_grad)
            return (loss, out) if return_outputs else loss

    trainer = NanGuardTrainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=lambda b: collate(b, pad_id),
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tok.save_pretrained(os.path.join(args.output_dir, "final"))
    print("done.")


if __name__ == "__main__":
    main()
