"""W7: per-dataset effect sizes, bootstrap CIs, randomization tests, Holm (family F1 = one comparison type
across datasets) and DerSimonian-Laird random-effects pooling, for the Phase-1 system comparisons.

Delta = nDCG@10(a) - nDCG@10(b) per query. Outputs:
  results/w7_effect_sizes.csv   long table (per dataset rows + pooled rows)
  figures/w7_forest.png          forest plot of the main comparisons
Usage: python effect_sizes.py [metric]   (default ndcg@10; recall@100 as secondary)
"""
import os
import sys

import numpy as np
import pandas as pd

import common2 as c2

METRIC = sys.argv[1] if len(sys.argv) > 1 else "ndcg@10"
SMALL = {"trec-covid", "webis-touche2020"}

COMPARISONS = [  # (name, a, b)
    ("BM25 - TF-IDF", "bm25", "tfidf"),
    ("stemming (BM25 - BM25 no-stem)", "bm25", "bm25[stop]"),
    ("RM3 - BM25", "bm25_rm3", "bm25"),
    ("WordNet - BM25", "bm25_wordnet", "bm25"),
]
for m in c2.DENSE_MODELS:
    P = c2.MODEL_PRETTY[m]
    COMPARISONS += [(f"{P} - BM25", m, "bm25"), (f"H({P}) - BM25", f"hybrid_cc_{m}", "bm25"),
                    (f"H({P}) - {P}", f"hybrid_cc_{m}", m)]
COMPARISONS += [("BM25-LODO - BM25", "bm25lodo", "bm25")]


def pooled_rows(sub, name, label, dss):
    s = sub[sub.dataset.isin(dss)]
    if len(s) < 2:
        return []
    r = c2.dl_meta(s.delta.values, s.se_boot.values)
    loo = [c2.dl_meta(s[s.dataset != d].delta.values, s[s.dataset != d].se_boot.values)["est"] for d in s.dataset] \
        if len(s) > 2 else [np.nan]
    return [dict(dataset=label, comparison=name, n=int(s.n.sum()), delta=r["est"], ci_lo=r["lo"], ci_hi=r["hi"],
                 se_boot=r["se"], p=r["p"], tau2=r["tau2"], I2=r["I2"], k=r["k"],
                 sign_pos=int((s.delta > 0).sum()), sign_neg=int((s.delta < 0).sum()),
                 sig_pos_holm=int(((s.delta > 0) & (s.p_holm < .05)).sum()),
                 sig_neg_holm=int(((s.delta < 0) & (s.p_holm < .05)).sum()),
                 loo_min=float(np.min(loo)), loo_max=float(np.max(loo)),
                 macro_delta=float(s.delta.mean()))]


def main():
    rows = []
    cache = {}
    for name, a, b in COMPARISONS:
        for d in c2.ALL_DATASETS:
            if d not in cache:
                cache[d] = c2.perquery(d, METRIC)
            w = cache[d]
            if a not in w or b not in w:
                continue
            x = w[[a, b]].dropna()
            r = c2.paired_summary(x[a].values, x[b].values)
            rows.append(dict(dataset=d, comparison=name, a=a, b=b, **r))
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for name, g in df.groupby("comparison"):
        df.loc[g.index, "p_holm"] = c2.holm(g.p.values)
    pooled = []
    for name, g in df.groupby("comparison", sort=False):
        dss = list(g.dataset)
        pooled += pooled_rows(g, name, f"POOLED_DL(k={len(dss)})", dss)
        pooled += pooled_rows(g, name, "POOLED_DL_excl_small", [d for d in dss if d not in SMALL])
    out = pd.concat([df, pd.DataFrame(pooled)], ignore_index=True)
    order = {n: i for i, (n, _, _) in enumerate(COMPARISONS)}
    out["_o"] = out.comparison.map(order)
    out = out.sort_values(["_o"], kind="stable").drop(columns="_o")
    suffix = "" if METRIC == "ndcg@10" else "_" + METRIC.replace("@", "")
    out.to_csv(os.path.join(c2.RES2, f"w7_effect_sizes{suffix}.csv"), index=False)
    pr = out[out.dataset.str.startswith("POOLED_DL(")]
    print(pr[["comparison", "k", "delta", "ci_lo", "ci_hi", "p", "I2", "sign_pos", "sign_neg", "sig_pos_holm",
              "sig_neg_holm", "loo_min", "loo_max"]].round(4).to_string())
    if METRIC == "ndcg@10":
        forest(out)


def forest(out):
    plt = c2.mpl_style()
    comps = [n for n, _, _ in COMPARISONS if n in set(out.comparison)]
    keep = ["BM25 - TF-IDF", "RM3 - BM25", "MiniLM - BM25", "H(MiniLM) - BM25", "BGE-s - BM25", "H(BGE-s) - BM25",
            "H(BGE-s) - BGE-s", "BGE-b - BM25", "H(BGE-b) - BM25"]
    comps = [c for c in keep if c in comps]
    fig, axes = plt.subplots(1, len(comps), figsize=(1.9 * len(comps), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    dss = [d for d in c2.ALL_DATASETS if d in set(out.dataset)]
    labels = [c2.PRETTY[d] for d in dss] + ["Pooled (DL)"]
    for ax, cname in zip(axes, comps):
        g = out[out.comparison == cname].set_index("dataset")
        for i, d in enumerate(dss):
            if d not in g.index:
                continue
            r = g.loc[d]
            col = c2.C_DENSE if ("MiniLM" in cname or "BGE" in cname) and not cname.startswith("H(") else \
                (c2.C_HYB if cname.startswith("H(") else c2.C_BM25)
            sig = r.p_holm < 0.05
            ax.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=1.4)
            ax.plot(r.delta, i, "o", color=col, ms=4.5, mfc=col if sig else "white")
        pk = [k for k in g.index if str(k).startswith("POOLED_DL(")]
        if pk:
            r = g.loc[pk[0]]
            ax.plot([r.ci_lo, r.ci_hi], [len(dss), len(dss)], color=c2.C_INK, lw=1.8)
            ax.plot(r.delta, len(dss), "D", color=c2.C_INK, ms=5)
        ax.axvline(0, color=c2.C_MUTED, lw=0.8)
        ax.set_title(cname.replace(" (BM25 - BM25 no-stem)", ""), fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis() if ax is axes[0] else None
        ax.set_xlabel("Δ nDCG@10")
    fig.suptitle("Paired Δ nDCG@10 with 95% bootstrap CI (filled = Holm p<0.05); diamond = DL random-effects pool",
                 fontsize=8.5, y=1.02)
    fig.savefig(os.path.join(c2.FIG2, "w7_forest.png"))


if __name__ == "__main__":
    main()
