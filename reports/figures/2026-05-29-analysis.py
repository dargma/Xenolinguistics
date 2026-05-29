"""Analysis figure for §4: (A) output/reference length ratio, (B) per-sentence chrF wins.
Reads outputs/analysis_klingon.json. Output: reports/figures/2026-05-29-analysis.png
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("outputs/analysis_klingon.json"))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

# (A) length ratio
labels = ["en→tlh\nAR", "en→tlh\nDiff", "tlh→en\nAR", "tlh→en\nDiff"]
ratios = [d["en2tlh_ar"]["len_ratio"], d["en2tlh_diffusion"]["len_ratio"],
          d["tlh2en_ar"]["len_ratio"], d["tlh2en_diffusion"]["len_ratio"]]
colors = ["#4C72B0", "#DD8452", "#4C72B0", "#DD8452"]
bars = a1.bar(labels, ratios, color=colors)
a1.axhline(1.0, ls="--", c="gray", lw=1); a1.text(3.45, 1.02, "ref length", color="gray", fontsize=8, ha="right")
for b, r in zip(bars, ratios):
    a1.text(b.get_x()+b.get_width()/2, r+0.03, f"{r:.2f}", ha="center", fontsize=9)
a1.set_title("(A) Output length / reference length"); a1.set_ylabel("ratio"); a1.set_ylim(0, 1.8); a1.grid(axis="y", alpha=.3)

# (B) per-sentence chrF wins (stacked)
dirs = ["en→tlh", "tlh→en"]
ar  = [d["paired_en2tlh"]["AR_win"], d["paired_tlh2en"]["AR_win"]]
tie = [d["paired_en2tlh"]["tie"],    d["paired_tlh2en"]["tie"]]
dff = [d["paired_en2tlh"]["Diff_win"], d["paired_tlh2en"]["Diff_win"]]
a2.barh(dirs, ar, color="#4C72B0", label="AR win")
a2.barh(dirs, tie, left=ar, color="#CCCCCC", label="tie")
a2.barh(dirs, dff, left=[a+t for a, t in zip(ar, tie)], color="#DD8452", label="Diffusion win")
for i, (a, t, f) in enumerate(zip(ar, tie, dff)):
    a2.text(a/2, i, str(a), va="center", ha="center", fontsize=9, color="white")
    a2.text(a+t+f/2, i, str(f), va="center", ha="center", fontsize=9, color="white")
a2.set_title("(B) Per-sentence chrF: who wins (n=300)"); a2.set_xlabel("sentences"); a2.legend(fontsize=8, loc="lower right")

fig.suptitle("Klingon analysis — Diffusion stops late (longer output); wins when output is English", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("reports/figures/2026-05-29-analysis.png", dpi=130)
print("saved reports/figures/2026-05-29-analysis.png")
