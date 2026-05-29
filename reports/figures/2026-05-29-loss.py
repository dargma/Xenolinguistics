"""4-panel loss curves for the 2026-05-29 multi-axis AR-vs-Diffusion report.
Panels = architecture × dataset (AR/Diffusion × Klingon/Khalani).
Each Klingon panel overlays forward(en→tlh) & reverse(tlh→en); AR panels add eval loss.
Diffusion has no val set → train loss only. Khalani has no reverse run & no val set.

Parses the raw training logs (loss/eval_loss dicts with 'epoch' field):
  outputs/klingon_qwen_fullft.log          (FWD AR)
  outputs/klingon_qwen_tlh2en.log          (REV AR)
  outputs/klingon_fastdllm_fullft.log      (FWD Diff)
  /content/local_fast/outputs/klingon_fastdllm_tlh2en_train.log (REV Diff)
  outputs/khalani_qwen_fullft.log          (Khalani AR)
  outputs/khalani_fastdllm_fullft.log      (Khalani Diff)
Output: reports/figures/2026-05-29-loss.png
"""
import re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOSS = re.compile(r"\{'loss': ([0-9.]+).*?'epoch': ([0-9.]+)\}")
EVAL = re.compile(r"\{'eval_loss': ([0-9.]+).*?'epoch': ([0-9.]+)\}")

def parse(path, pat):
    if not os.path.exists(path):
        return [], []
    txt = open(path).read()
    xs, ys = [], []
    for m in pat.finditer(txt):
        ys.append(float(m.group(1))); xs.append(float(m.group(2)))
    return xs, ys

O = "outputs"
LF = "/content/local_fast/outputs"
runs = {
    "fwd_ar":   f"{O}/klingon_qwen_fullft.log",
    "rev_ar":   f"{O}/klingon_qwen_tlh2en.log",
    "fwd_df":   f"{O}/klingon_fastdllm_fullft.log",
    "rev_df":   f"{LF}/klingon_fastdllm_tlh2en_train.log",
    "kha_ar":   f"{O}/khalani_qwen_fullft.log",
    "kha_df":   f"{O}/khalani_fastdllm_fullft.log",
}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

def parse_text(txt, pat):
    xs, ys = [], []
    for m in pat.finditer(txt):
        ys.append(float(m.group(1))); xs.append(float(m.group(2)))
    return xs, ys

# khalani 역방향: AR+Diffusion이 한 로그에 순차 기록 → "TRAIN Diffusion" 기준 분리
_kr = open(f"{LF}/khalani_fastdllm_kha2en_fullft".rsplit("/",2)[0] + "/khalani_rev.log").read() \
      if False else open("/content/local_fast/khalani_rev.log").read()
_split = _kr.find("TRAIN Diffusion")
KHA_REV_AR_TXT = _kr[:_split]
KHA_REV_DF_TXT = _kr[_split:]

def plot_text(ax, txt, pat, label, color, style="-"):
    xs, ys = parse_text(txt, pat)
    if xs:
        ax.plot(xs, ys, style, label=f"{label} ({ys[-1]:.2f})", color=color, markersize=3, linewidth=1.4)

def plot_curve(ax, path, pat, label, color, style="-"):
    xs, ys = parse(path, pat)
    if xs:
        ax.plot(xs, ys, style, label=f"{label} ({ys[-1]:.2f})", color=color,
                markersize=3, linewidth=1.4)

# (0,0) AR · Klingon — fwd/rev × train/eval
ax = axes[0, 0]
plot_curve(ax, runs["fwd_ar"], LOSS, "fwd train", "C0")
plot_curve(ax, runs["fwd_ar"], EVAL, "fwd eval", "C0", "s--")
plot_curve(ax, runs["rev_ar"], LOSS, "rev train", "C3")
plot_curve(ax, runs["rev_ar"], EVAL, "rev eval", "C3", "s--")
ax.set_title("AR (Qwen2.5-7B) · Klingon"); ax.set_xlabel("epoch"); ax.set_ylabel("CE loss")
ax.legend(fontsize=8); ax.grid(alpha=.3)

# (0,1) Diffusion · Klingon — fwd/rev train (no eval)
ax = axes[0, 1]
plot_curve(ax, runs["fwd_df"], LOSS, "fwd train", "C0")
plot_curve(ax, runs["rev_df"], LOSS, "rev train", "C3")
ax.set_title("Diffusion (Fast-dLLM v2 7B) · Klingon"); ax.set_xlabel("epoch")
ax.set_ylabel("block-diffusion loss")
ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.annotate("no val set in diffusion trainer (train only)", xy=(0.97, 0.95),
            xycoords="axes fraction", ha="right", va="top", fontsize=8,
            color="gray", style="italic")

# (1,0) AR · Khalani — fwd/rev train (no val), epochs 20
ax = axes[1, 0]
plot_curve(ax, runs["kha_ar"], LOSS, "fwd train", "C0")
plot_text(ax, KHA_REV_AR_TXT, LOSS, "rev train", "C3")
ax.set_title("AR (Qwen2.5-7B) · Khalani (55 pairs, 20 ep)"); ax.set_xlabel("epoch")
ax.set_ylabel("CE loss"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.annotate("no val set; train→0 (memorizes 44)", xy=(0.97, 0.95),
            xycoords="axes fraction", ha="right", va="top", fontsize=8,
            color="gray", style="italic")

# (1,1) Diffusion · Khalani — fwd/rev train
ax = axes[1, 1]
plot_curve(ax, runs["kha_df"], LOSS, "fwd train", "C0")
plot_text(ax, KHA_REV_DF_TXT, LOSS, "rev train", "C3")
ax.set_title("Diffusion (Fast-dLLM v2 7B) · Khalani (55 pairs, 20 ep)")
ax.set_xlabel("epoch"); ax.set_ylabel("block-diffusion loss")
ax.legend(fontsize=8); ax.grid(alpha=.3)

fig.suptitle("Loss curves — AR vs Diffusion × Klingon vs Khalani (2026-05-29)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "reports/figures/2026-05-29-loss.png"
fig.savefig(out, dpi=130)
print("saved", out)
# quick sanity: print final losses
for k, p in runs.items():
    _, ys = parse(p, LOSS)
    print(f"  {k}: {len(ys)} pts, final train loss = {ys[-1] if ys else 'NA'}")
