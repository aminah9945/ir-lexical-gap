"""Re-draw Fig. 1 of 03_experiments (fig1_winshare_bge.png) for the paper, with a non-overlapping legend.
Data: phase2/03_experiments/results/fig1_winshare.csv (model == 'bge'); no numbers are recomputed."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
t = pd.read_csv(os.path.join(HERE, "..", "03_experiments", "results", "fig1_winshare.csv"))
t = t[t.model == "bge"].reset_index(drop=True)
PRETTY = {"nfcorpus": "NFCorpus", "scifact": "SciFact", "fiqa": "FiQA", "scidocs": "SciDocs", "arguana": "ArguAna",
          "trec-covid": "TREC-COVID", "quora": "Quora", "webis-touche2020": "Touché"}
cols = [("BM25", "#2a78d6", "BM25 best"), ("dense", "#eb6834", "BGE-s best"),
        ("hybrid", "#1baf7a", "hybrid best"), ("tie", "#d9d8d3", "tie (<=0.01)")]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)
x = np.arange(len(t)); bottom = np.zeros(len(t))
for c, colr, lab in cols:
    ax.bar(x, t[c], 0.72, bottom=bottom, color=colr, label=lab); bottom += t[c].values
for xi, v in zip(x, t.oracle_minus_H):
    ax.text(xi, 1.02, f"+{v:.3f}", ha="center", fontsize=5.6, color="#1f1f1e")
ax.set_xticks(x); ax.set_xticklabels([PRETTY[d] for d in t.dataset], rotation=35, ha="right", fontsize=6.5)
ax.set_ylabel("share of queries"); ax.set_ylim(0, 1.1)
ax.legend(ncol=4, frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.06), handlelength=1.0,
          columnspacing=0.8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figs", "fig_winshare_bge.png"))
