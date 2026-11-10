"""W1 standalone QPP evaluation (family F3): Kendall tau and Pearson r between each query-time predictor and the
per-query nDCG@10 of BM25 and of each dense model, per dataset. Holm correction over all Kendall tests in F3.

Predictors: pre-retrieval (Tier A, evaluated against every system), BM25-list predictors (B_*) vs BM25 nDCG,
dense-list predictors (D_*, Tier B) vs that dense model's nDCG.
Outputs results/w1_qpp_correlations.csv, results/w1_qpp_summary.csv, figures/w1_qpp_heatmap.png
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

import common2 as c2

QDIR = os.path.join(c2.RES2, "qpp")
PRE = ["q_len", "q_terms", "q_oov_frac", "q_jargon", "idf_mean", "idf_max", "idf_min", "idf_std", "avictf", "scs",
       "scq_sum", "scq_mean", "scq_max", "var_sum", "var_max"]
POST = ["wig5", "nqc100", "smv100", "clarity100", "s1", "s1_s10", "s1_s10_rel", "mu10_mu100"]


def corr_rows(ds, system, list_label, x, y):
    out = []
    for c in x.columns:
        v = pd.concat([x[c], y], axis=1).dropna()
        if len(v) < 10 or v.iloc[:, 0].nunique() < 2:
            continue
        t = stats.kendalltau(v.iloc[:, 0], v.iloc[:, 1])
        r = stats.pearsonr(v.iloc[:, 0], v.iloc[:, 1])
        out.append(dict(dataset=ds, system=system, list=list_label, predictor=c, n=len(v), kendall_tau=t.statistic,
                        p_tau=t.pvalue, pearson_r=r.statistic, p_r=r.pvalue))
    return out


def main():
    rows = []
    for ds in c2.ALL_DATASETS:
        fa = os.path.join(QDIR, f"tierA_{ds}.csv")
        if not os.path.exists(fa):
            continue
        A = pd.read_csv(fa, dtype={"qid": str}).set_index("qid")
        w = c2.perquery(ds)
        systems = ["bm25"] + [m for m in c2.DENSE_MODELS if m in w]
        for s in systems:
            y = w[s].reindex(A.index)
            rows += corr_rows(ds, s, "pre", A[PRE], y)
        rows += corr_rows(ds, "bm25", "B-list", A[["B_" + p for p in POST]].rename(columns=lambda c: c[2:]),
                          w["bm25"].reindex(A.index))
        for m in c2.DENSE_MODELS:
            fb = os.path.join(QDIR, f"tierB_{m}_{ds}.csv")
            if os.path.exists(fb) and m in w:
                B = pd.read_csv(fb, dtype={"qid": str}).set_index("qid")
                rows += corr_rows(ds, m, "D-list", B[["D_" + p for p in POST]].rename(columns=lambda c: c[2:]),
                                  w[m].reindex(B.index))
    df = pd.DataFrame(rows)
    df["p_tau_holm"] = c2.holm(df.p_tau.values)
    df.to_csv(os.path.join(c2.RES2, "w1_qpp_correlations.csv"), index=False)
    summ = df.groupby(["list", "system", "predictor"]).agg(
        k=("dataset", "size"), mean_tau=("kendall_tau", "mean"), min_tau=("kendall_tau", "min"),
        max_tau=("kendall_tau", "max"), mean_r=("pearson_r", "mean"),
        n_sig_holm=("p_tau_holm", lambda p: int((p < 0.05).sum()))).reset_index()
    summ.to_csv(os.path.join(c2.RES2, "w1_qpp_summary.csv"), index=False)
    print(summ[summ.list != "pre"].round(3).to_string())
    heatmap(df)


def heatmap(df):
    plt = c2.mpl_style()
    from matplotlib.colors import LinearSegmentedColormap
    sub = df[df.list.isin(["B-list", "D-list"])].copy()
    sub["col"] = sub.list.str[0] + ":" + sub.system.map(lambda s: c2.MODEL_PRETTY.get(s, "BM25"))
    piv = sub.pivot_table(index="predictor", columns=["col", "dataset"], values="kendall_tau")
    piv = piv.reindex(POST)
    cmap = LinearSegmentedColormap.from_list("div", ["#b8431a", "#f7f6f2", c2.C_BM25])
    fig, ax = plt.subplots(figsize=(min(16, 0.32 * piv.shape[1] + 2), 2.8))
    im = ax.imshow(piv.values, cmap=cmap, vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"{a} {c2.SHORT[b]}" for a, b in piv.columns], rotation=90, fontsize=6.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Kendall τ with nDCG@10", shrink=0.8)
    ax.set_title("Post-retrieval QPP on BM25 lists (B:) and dense lists (D:)")
    fig.savefig(os.path.join(c2.FIG2, "w1_qpp_heatmap.png"))


if __name__ == "__main__":
    main()
