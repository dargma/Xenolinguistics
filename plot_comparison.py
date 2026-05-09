"""
Qwen vs Dream 1k loss curve 비교 + 10k 결과 추가 (있을 경우)
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# ---- 1k 데이터 로드 ----
# Qwen: trainer_state.json
qwen_state = json.load(open("outputs/qwen/checkpoint-300/trainer_state.json"))
qwen_train = [(h["step"], h["loss"]) for h in qwen_state["log_history"] if "loss" in h and "eval_loss" not in h]
qwen_eval = [(h["step"], h["eval_loss"]) for h in qwen_state["log_history"] if "eval_loss" in h]

# Dream v2: training_meta.json
dream_meta = json.load(open("outputs/dream_official/training_meta.json"))
dream_train = [(d["global_step"], d["loss"]) for d in dream_meta["train_log"]]
dream_eval = [(d["global_step"], d["val_loss"]) for d in dream_meta["val_log"]]

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

Y_MAX = 5.0

# Left: Qwen
ax = axes[0]
ax.plot(*zip(*qwen_train), label="Train Loss", alpha=0.7, color="#2196F3")
ax.plot(*zip(*qwen_eval), label="Val Loss", linestyle="--", marker="o", color="#F44336", markersize=5)
ax.set_xlabel("Step")
ax.set_ylabel("Loss (CE)")
ax.set_title("Qwen2.5-7B + LoRA (1k)\nlr=2e-4 | AR Causal LM")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, Y_MAX)

# Right: Dream
ax = axes[1]
ax.plot(*zip(*dream_train), label="Train Loss", alpha=0.7, color="#4CAF50")
ax.plot(*zip(*dream_eval), label="Val Loss", linestyle="--", marker="o", color="#FF9800", markersize=5)
ax.set_xlabel("Step")
ax.set_title("Dream-7B + LoRA v2 (1k)\nlr=1e-5 | Official Diffusion SFT")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, Y_MAX)

fig.suptitle("Qwen (AR) vs Dream (Diffusion) — 1k Data, Same LoRA r=16", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig("reports/fig-qwen-vs-dream-1k-loss.png", dpi=150, bbox_inches="tight")
print("Saved: reports/fig-qwen-vs-dream-1k-loss.png")

# ---- Summary table ----
print("\n=== 1k 학습 비교 ===")
print(f"{'':20} {'Qwen (AR)':>15} {'Dream v2 (Diff)':>18}")
print(f"{'Final train loss':20} {qwen_train[-1][1]:>15.4f} {dream_train[-1][1]:>18.4f}")
print(f"{'Final val loss':20} {qwen_eval[-1][1]:>15.4f} {dream_eval[-1][1]:>18.4f}")
print(f"\nNote: Loss scales differ — Qwen=CE(next-token), Dream=CE(masked)/t")
