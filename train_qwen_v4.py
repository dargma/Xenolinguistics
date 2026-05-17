"""
Qwen2.5-7B v4 — Same conditions as Dream v4 for fair comparison
all-linear LoRA r=16, same data, same eval
"""
import torch, time, json, os, matplotlib.pyplot as plt, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

def train(args):
    print(f"{'='*60}")
    print(f"Qwen2.5-7B v4 | target={args.target_modules} | r={args.lora_rank} | lr={args.lr}")
    print(f"{'='*60}")

    os.makedirs(args.output_dir, exist_ok=True)

    def load_ds(path):
        data = []
        for line in open(path):
            d = json.loads(line)
            data.append({"text": (
                f"<|im_start|>user\n{d['instruction']}<|im_end|>\n"
                f"<|im_start|>assistant\n{d['output']}<|im_end|>"
            )})
        return Dataset.from_list(data)

    train_ds = load_ds(args.train_file)
    val_ds = load_ds(args.val_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id,
        torch_dtype=torch.bfloat16, device_map="auto")

    if args.target_modules == "all-linear":
        target_modules = "all-linear"
    else:
        target_modules = args.target_modules.split(",")

    if args.lora_rank > 0:
        lora_cfg = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=0.05, bias="none",
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        print("Full FT mode (lora_rank=0)")

    per_dev_bs = 4 if args.lora_rank > 0 else 1
    grad_accum = 2 if args.lora_rank > 0 else 8
    sft_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_dev_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps", eval_steps=200,
        save_steps=2000,
        save_total_limit=2,
        report_to="none",
        max_length=256,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(model=model, args=sft_args,
        train_dataset=train_ds, eval_dataset=val_ds, processing_class=tokenizer)

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    save_subdir = "adapter" if args.lora_rank > 0 else "final"
    model.save_pretrained(f"{args.output_dir}/{save_subdir}")
    tokenizer.save_pretrained(f"{args.output_dir}/{save_subdir}")

    history = trainer.state.log_history
    train_loss = [(h["step"], h["loss"]) for h in history if "loss" in h and "eval_loss" not in h]
    eval_loss = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]

    meta = {
        "model": args.model_id,
        "target_modules": args.target_modules,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "data_train": args.train_file,
        "epochs": args.epochs, "lr": args.lr,
        "train_time_min": round(elapsed / 60, 1),
        "final_train_loss": train_loss[-1][1] if train_loss else None,
        "final_eval_loss": eval_loss[-1][1] if eval_loss else None,
    }
    json.dump(meta, open(f"{args.output_dir}/training_meta.json", "w"), indent=2)
    print(f"\nDone | {elapsed/60:.1f}min | train={train_loss[-1][1]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--target_modules", default="all-linear")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--train_file", default="data/train_1k.jsonl")
    parser.add_argument("--val_file", default="data/val_1k.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_dir", default="outputs/qwen_v4_1k")
    args = parser.parse_args()
    train(args)
