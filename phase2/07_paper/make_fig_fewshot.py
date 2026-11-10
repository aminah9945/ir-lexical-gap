"""Re-draw 04_improvements/figures/e4_fewshot.png for the paper (same data and logic as summarize4.py; only the
legend text differs). Data: 04_improvements/results/e4_fewshot_dataset_selection.csv, e4_splithalf.csv,
03_experiments/results/w1_router_macro.csv. Only k values available for all 8 datasets are plotted."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R4 = os.path.join(HERE, "..", "04_improvements", "results")
R3 = os.path.join(HERE, "..", "03_experiments", "results")
fs = pd.read_csv(os.path.join(R4, "e4_fewshot_dataset_selection.csv"))
sh = pd.read_csv(os.path.join(R4, "e4_splithalf.csv"))
rm = pd.read_csv(os.path.join(R3, "w1_router_macro.csv"))
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.5, 2.9), dpi=300)
for m, col, ls, lab in [("bgebase", "#1f6fb4", "-.", "BGE-b"), ("bge", "#6250d6", "-", "BGE-s"), ("minilm", "#eb6834", "--", "MiniLM")]:
    g = fs[fs.model == m]
    nds = g.dataset.nunique()
    g = g.groupby("k").filter(lambda z: len(z) == nds)
    mac = g.groupby("k").gain_mean.mean()
    ax.plot(mac.index, mac.values, ls, marker="o", ms=3, color=col, label=f"pick fixed system from k labels, D={lab}")
    ax.axhline(sh[sh.model == m].oracle_ds_gain_insample.mean(), color=col, lw=0.7, ls=":")
for cfg, col in [("A_bgebase_std_k8", "#1f6fb4"), ("A_bge_std_k8", "#6250d6"), ("A_minilm_std_k8", "#eb6834")]:
    v = rm[(rm.config == cfg) & (rm.variant == "selected")].router_minus_H.values[0]
    ax.scatter([3], [v], marker="x", color=col, zorder=3)
ax.scatter([], [], marker="x", color="#6b6a66", label="LODO query router (no target labels)")
ax.axhline(0, color="#1baf7a", lw=1, label="always-hybrid")
ax.plot([], [], ":", color="#6b6a66", lw=0.7, label="dataset oracle (all labels)")
ax.set_xscale("log"); ax.set_xticks([3, 5, 10, 20, 30]); ax.set_xticklabels(["router", "5", "10", "20", "30"])
ax.minorticks_off()
ax.set_xlabel("labelled target queries k"); ax.set_ylabel(r"macro $\Delta$ nDCG@10 vs always-H")
ax.legend(frameon=False, fontsize=5.6, ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.28))
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figs", "fig_fewshot.png"), bbox_inches="tight")
