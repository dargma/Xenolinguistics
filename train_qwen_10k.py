"""
Qwen2.5-7B LoRA SFT — 10k data
이전 1k와 동일 설정, 데이터만 10k로 확대
"""
import torch, time, json, os, matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

os.makedirs("outputs/qwen_10k", exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_TAG = "10k"
EPOCHS = 3
LR = 2e-4
MAX_SEQ = 256

def load_ds(path):
    data = []
    for line in open(path):
        d = json.loads(line)
        data.append({"text": (
            f"<|im_start|>user\n{d['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{d['output']}<|im_end|>"
        )})
    return Dataset.from_list(data)

train_ds = load_ds(f"data/train_{DATA_TAG}.jsonl")
val_ds = load_ds(f"data/val_{DATA_TAG}.jsonl")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
    torch_dtype=torch.bfloat16, device_map="auto")

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

args = SFTConfig(
    output_dir="outputs/qwen_10k",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps", eval_steps=200,
    save_steps=500,
    report_to="none",
)

trainer = SFTTrainer(model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds, processing_class=tokenizer)

start = time.time()
trainer.train()
elapsed = time.time() - start

# Save
model.save_pretrained("outputs/qwen_10k/adapter")
tokenizer.save_pretrained("outputs/qwen_10k/adapter")

# Loss curve
history = trainer.state.log_history
train_loss = [(h["step"], h["loss"]) for h in history if "loss" in h and "eval_loss" not in h]
eval_loss = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]

fig, ax = plt.subplots(figsize=(8, 4))
if train_loss: ax.plot(*zip(*train_loss), label="Train Loss")
if eval_loss: ax.plot(*zip(*eval_loss), label="Eval Loss", linestyle="--")
ax.set_xlabel("Step"); ax.set_ylabel("Loss")
ax.set_title(f"Qwen2.5-7B LoRA SFT — {DATA_TAG} / lr={LR} / r=16")
ax.legend(); ax.grid(True, alpha=0.3)
fig.savefig(f"outputs/qwen_10k/loss_curve_{DATA_TAG}.png", dpi=150, bbox_inches="tight")

# Save trainer state for comparison plots
json.dump(trainer.state.log_history, open("outputs/qwen_10k/log_history.json", "w"), indent=2)

meta = {
    "model": MODEL_ID, "data_tag": DATA_TAG, "epochs": EPOCHS, "lr": LR,
    "lora_r": 16, "effective_batch_size": 8,
    "train_time_min": round(elapsed / 60, 1),
    "final_train_loss": train_loss[-1][1] if train_loss else None,
    "final_eval_loss": eval_loss[-1][1] if eval_loss else None,
}
json.dump(meta, open("outputs/qwen_10k/training_meta.json", "w"), indent=2)
print(f"Qwen 10k SFT done | {elapsed/60:.1f}min | train={train_loss[-1][1]:.4f}")
