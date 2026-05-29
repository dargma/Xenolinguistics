"""Loss curves for the Klingon AR-vs-Diffusion full-FT runs (2026-05-28).
Points captured from training logs during the run (per-step trainer_state.json was
pruned by save_total_limit=1 + disk cleanup). AR=Qwen2.5-7B, DIFF=Fast-dLLM v2 7B.
Output: reports/figures/2026-05-28-klingon-loss.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Qwen (AR) — cross-entropy loss
ar_tr = [(10,6.047),(30,2.938),(50,2.114),(100,1.927),(150,1.739),(200,1.678),
         (300,1.466),(400,1.456),(1570,1.071),(1590,1.044)]
ar_ev = [(200,1.677),(400,1.444),(600,1.285),(800,1.193),(1000,1.146),(1200,1.133),(1400,1.129)]
# Fast-dLLM (Diffusion) — block-diffusion loss (different scale)
df_tr = [(10,12.67),(60,7.88),(160,7.16),(260,6.25),(410,5.69),(560,5.76),(610,4.7),
         (760,4.84),(910,4.32),(1110,4.1),(1160,3.79),(1310,3.86),(1510,3.71),(1590,3.81)]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
a1.plot(*zip(*ar_tr), "o-", label="train loss", color="C0")
a1.plot(*zip(*ar_ev), "s--", label="eval loss", color="C3")
a1.set_title("AR — Qwen2.5-7B full-FT (lr 2e-5, 1 ep)")
a1.set_xlabel("step"); a1.set_ylabel("CE loss"); a1.legend(); a1.grid(alpha=.3)

a2.plot(*zip(*df_tr), "o-", label="train loss", color="C1")
a2.set_title("Diffusion — Fast-dLLM v2 7B full-FT (lr 2e-5, 1 ep)")
a2.set_xlabel("step"); a2.set_ylabel("block-diffusion loss"); a2.legend(); a2.grid(alpha=.3)
a2.annotate("eval/val loss not logged\n(no val set in diffusion trainer)",
            xy=(0.96, 0.82), xycoords="axes fraction", ha="right", va="top",
            fontsize=8, color="gray", style="italic",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=.7))

fig.suptitle("Klingon en→tlh full fine-tune — loss curves (2026-05-28)")
fig.tight_layout()
fig.savefig("reports/figures/2026-05-28-klingon-loss.png", dpi=130)
print("saved reports/figures/2026-05-28-klingon-loss.png")
