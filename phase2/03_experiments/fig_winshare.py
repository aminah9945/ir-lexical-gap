"""Fig. 1 candidate: per-query winner shares among {BM25, D, H(D)} per dataset (strict winner by > 0.01 nDCG@10 over
both others; otherwise 'tie'), with the per-query-oracle headroom over always-H annotated.
Output figures/fig1_winshare_<model>.png and results/fig1_winshare.csv"""
import os
import sys

import numpy as np
import pandas as pd

import common2 as c2


def main(model):
    rows = []
    for d in c2.ALL_DATASETS:
        w = c2.perquery(d)
        h = f"hybrid_cc_{model}"
        if not {"bm25", model, h} <= set(w.columns):
            continue
        Y = w[["bm25", model, h]].dropna().values
        srt = np.sort(Y, 1)
        win = np.where(srt[:, -1] - srt[:, -2] > c2.TIE, Y.argmax(1), 3)
        sh = np.bincount(win, minlength=4) / len(win)
        rows.append(dict(model=model, dataset=d, BM25=sh[0], dense=sh[1], hybrid=sh[2], tie=sh[3],
                         oracle_minus_H=Y.max(1).mean() - Y[:, 2].mean()))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    plt = c2.mpl_style()
    fig, ax = plt.subplots(figsize=(7, 2.8))
    x = np.arange(len(t))
    bottom = np.zeros(len(t))
    for col, c, lab in [("BM25", c2.C_BM25, "BM25 best"), ("dense", c2.C_DENSE, f"{c2.MODEL_PRETTY[model]} best"),
                        ("hybrid", c2.C_HYB, "hybrid best"), ("tie", c2.C_GRID, "tie (≤0.01)")]:
        ax.bar(x, t[col], 0.7, bottom=bottom, color=c, label=lab)
        bottom += t[col].values
    for xi, v in zip(x, t.oracle_minus_H):
        ax.text(xi, 1.02, f"+{v:.3f}", ha="center", fontsize=7, color=c2.C_INK)
    ax.set_xticks(x)
    ax.set_xticklabels([c2.PRETTY[d] for d in t.dataset], fontsize=7.5)
    ax.set_ylabel("share of queries")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.22))
    ax.set_title("numbers: per-query oracle − always-hybrid (nDCG@10)", fontsize=7.5, loc="right", y=1.12)
    fig.savefig(os.path.join(c2.FIG2, f"fig1_winshare_{model}.png"))
    return t


if __name__ == "__main__":
    ts = [main(m) for m in (sys.argv[1:] or ["minilm", "bge", "bgebase"])]
    pd.concat([t for t in ts if len(t)]).to_csv(os.path.join(c2.RES2, "fig1_winshare.csv"), index=False)
