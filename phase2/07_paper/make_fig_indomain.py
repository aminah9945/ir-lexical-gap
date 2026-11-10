"""Re-draw 04_improvements/figures/e4_indomain_bars.png at single-column size (same data and logic as summarize4.py).
Data: 04_improvements/results/e4_indomain_perdataset.csv (A_bge_std_k8), 03_experiments/results/w1_router_perdataset.csv."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
p = pd.read_csv(os.path.join(HERE, "..", "04_improvements", "results", "e4_indomain_perdataset.csv"))
p = p[p.config == "A_bge_std_k8"].set_index("dataset")
lodo = pd.read_csv(os.path.join(HERE, "..", "03_experiments", "results", "w1_router_perdataset.csv"))
lodo = lodo[(lodo.config == "A_bge_std_k8") & (lodo.variant == "selected")].set_index("dataset").router
PRETTY = {"nfcorpus": "NFCorpus", "scifact": "SciFact", "fiqa": "FiQA", "scidocs": "SciDocs", "arguana": "ArguAna",
          "trec-covid": "TREC-COVID", "quora": "Quora", "webis-touche2020": "Touché"}
dss = list(p.index); x = np.arange(len(dss)); base = p.always_H
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)
for i, (v, lab, col) in enumerate([(p.indomain_best_fixed - base, "in-domain best fixed (CV)", "#6b6a66"),
                                   (p.indomain_router - base, "in-domain router (CV)", "#6250d6"),
                                   (lodo.reindex(dss) - base, "LODO router", "#a99cf0"),
                                   (p.oracle_query - base, "per-query oracle", "#1f1f1e")]):
    ax.bar(x + (i - 1.5) * 0.2, v, 0.2, color=col, label=lab)
ax.axhline(0, color="#1baf7a", lw=1)
ax.set_xticks(x); ax.set_xticklabels([PRETTY[d] for d in dss], rotation=35, ha="right", fontsize=6.5)
ax.set_ylabel(r"$\Delta$ nDCG@10 vs always-H(BGE-s)", fontsize=7)
ax.legend(ncol=2, frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figs", "fig_indomain.png"), bbox_inches="tight")
