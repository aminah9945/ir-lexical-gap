"""Hypothesis tests H1-H3, H5, H6 plus E2/E5 summaries, tables and figures.

Each hypothesis has a decision rule fixed in code (see VERDICT RULES comments) so the
verdict is mechanical, not post-hoc. Writes results/*.csv, results/verdicts.json, figures/*.png
"""
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from common import DATASETS, CORE, PRETTY, RES, FIG, SEED

warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)
ALPHA = 0.05
METRICS = ["ndcg@10", "map@100", "recall@100", "mrr@10"]

# palette (validated reference palette, light mode)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"


# ----------------------------------------------------------------------------- loading
def load():
    pq = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "perquery_*.csv"))])
    pq["qid"] = pq.qid.astype(str)
    feats = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "features_*.csv"))])
    feats["qid"] = feats.qid.astype(str)
    dom = pd.read_csv(os.path.join(RES, "domain_stats.csv"))
    return pq, feats, dom


def wide(pq, metric="ndcg@10"):
    return pq.pivot_table(index=["dataset", "qid"], columns="system", values=metric).reset_index()


# ----------------------------------------------------------------------------- stats helpers
def perm_test(a, b, n=10000):
    """Two-sided paired randomization (sign-flip) test on mean difference."""
    d = np.asarray(a) - np.asarray(b)
    obs = abs(d.mean())
    cnt, done = 0, 0
    while done < n:
        m = min(1000, n - done)
        s = rng.choice([-1, 1], size=(m, len(d)))
        cnt += (np.abs((s * d).mean(1)) >= obs - 1e-12).sum()
        done += m
    return d.mean(), (cnt + 1) / (n + 1)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1, run)
    return adj


def boot_ci_mean(x, n=2000):
    x = np.asarray(x)
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(bs, [2.5, 97.5])


def partial_spearman(df, x, y, controls, nboot=1000):
    """Spearman partial correlation: rank all, regress out controls, correlate residuals."""
    d = df[[x, y] + controls].dropna()
    R = d.rank()

    def resid(v, cols):
        if not cols:
            return v - v.mean()
        X = np.column_stack([np.ones(len(R))] + [R[c].values for c in cols])
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        return v - X @ beta

    rx, ry = resid(R[x].values, controls), resid(R[y].values, controls)
    r, p = stats.pearsonr(rx, ry)
    idx = rng.integers(0, len(rx), (nboot, len(rx)))
    bs = [np.corrcoef(rx[i], ry[i])[0, 1] for i in idx]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return r, p, lo, hi, len(d)


def mixed(df, formula):
    m = smf.mixedlm(formula, df, groups=df["dataset"]).fit(reml=True)
    return m


# ----------------------------------------------------------------------------- tables
def main_table(pq):
    g = pq.groupby(["dataset", "system"])[METRICS].mean().reset_index()
    ci = pq.groupby(["dataset", "system"])["ndcg@10"].apply(lambda x: boot_ci_mean(x)).reset_index()
    g["ndcg_lo"] = [c[0] for c in ci["ndcg@10"]]
    g["ndcg_hi"] = [c[1] for c in ci["ndcg@10"]]
    g.to_csv(os.path.join(RES, "table_main.csv"), index=False)
    return g


def h1(W):
    """H1: BM25 > TF-IDF > Boolean.
    VERDICT RULE: SUPPORTED if both orderings are significant (Holm) on all datasets;
    PARTIALLY SUPPORTED if each holds significantly on a majority of datasets;
    NOT SUPPORTED otherwise. Reversals (significant opposite direction) are listed."""
    rows = []
    pairs = [("bm25", "tfidf"), ("tfidf", "bool_or"), ("bool_or", "bool_and"), ("bm25", "bool_or")]
    for d in DATASETS:
        w = W[W.dataset == d]
        for a, b in pairs:
            diff, p = perm_test(w[a], w[b])
            rows.append(dict(dataset=d, a=a, b=b, mean_a=w[a].mean(), mean_b=w[b].mean(), diff=diff, p=p))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p)
    df["sig_claimed"] = (df.p_holm < ALPHA) & (df["diff"] > 0)
    df["sig_reversal"] = (df.p_holm < ALPHA) & (df["diff"] < 0)
    df.to_csv(os.path.join(RES, "h1_tests.csv"), index=False)
    n = len(DATASETS)
    sub = {}
    for a, b in pairs[:2]:
        s = df[(df.a == a) & (df.b == b)]
        sub[f"{a}>{b}"] = dict(holds=int(s.sig_claimed.sum()), reversals=s[s.sig_reversal].dataset.tolist())
    k1, k2 = sub["bm25>tfidf"]["holds"], sub["tfidf>bool_or"]["holds"]
    if k1 == n and k2 == n:
        v = "SUPPORTED"
    elif k1 > n / 2 and k2 > n / 2:
        v = "PARTIALLY SUPPORTED"
    else:
        v = "NOT SUPPORTED"
    return dict(verdict=v, detail=sub, n_datasets=n)


def h2(dom, tab):
    """H2: lexical effectiveness falls as domain vocabulary specificity rises.
    Specificity measured (not assumed): corpus jargon rate (primary), query jargon rate, mean query IDF.
    VERDICT RULE (n=8 domains, low power): SUPPORTED if Spearman rho(corpus_jargon, BM25 nDCG@10) < 0
    with p < .05; NOT SUPPORTED if rho >= 0 or p >= .05 (we report which).
    Also checks the proposal's assumed ordering NFCorpus < SciFact < FiQA."""
    bm = tab[tab.system == "bm25"].set_index("dataset")["ndcg@10"]
    d = dom.set_index("dataset").copy()
    d["bm25_ndcg"] = bm
    out = {}
    for spec in ["corpus_jargon", "q_jargon", "q_mean_idf"]:
        r, p = stats.spearmanr(d[spec], d["bm25_ndcg"])
        out[spec] = dict(rho=r, p=p)
    dense = [s for s in ["minilm", "bge"] if s in set(tab.system)]
    if "minilm" in dense:
        mn = tab[tab.system == "minilm"].set_index("dataset")["ndcg@10"]
        d["gap_minilm"] = mn - bm
        r, p = stats.spearmanr(d["corpus_jargon"], d["gap_minilm"].reindex(d.index))
        out["corpus_jargon_vs_dense_gap"] = dict(rho=r, p=p)
    d.to_csv(os.path.join(RES, "h2_domain_table.csv"))
    prim = out["corpus_jargon"]
    v = "SUPPORTED" if prim["rho"] < 0 and prim["p"] < ALPHA else "NOT SUPPORTED"
    order = d.loc[CORE, "corpus_jargon"].sort_values().index.tolist()
    out["measured_core_order_low_to_high_jargon"] = order
    out["assumed_order_matches"] = order == ["nfcorpus", "scifact", "fiqa"]
    out["verdict"] = v
    return out, d


def h3(W, feats):
    """H3: per-query overlap (cov_mean) predicts BM25 effectiveness.
    VERDICT RULE: SUPPORTED if (a) mixed-model coefficient of overlap > 0 with p < .05 and
    (b) partial Spearman (controls: #query terms, #relevant docs) is positive & significant in >= 6/8
    datasets. PARTIALLY if only one of (a)/(b). Circularity check: same analysis for dense nDCG."""
    df = W.merge(feats, on=["dataset", "qid"])
    df["log_nrel"] = np.log(df.n_rel)
    rows = []
    for d in DATASETS:
        s = df[df.dataset == d]
        for sysname in ["bm25", "tfidf", "minilm"]:
            if sysname not in s or s[sysname].isna().all():
                continue
            for x in ["cov_mean", "idfcov_mean", "cov_max", "jac_mean"]:
                r0, p0 = stats.spearmanr(s[x], s[sysname])
                r, p, lo, hi, n = partial_spearman(s, x, sysname, ["q_terms", "log_nrel"],
                                                   nboot=500 if x == "cov_mean" else 100)
                rows.append(dict(dataset=d, system=sysname, feature=x, rho=r0, p_rho=p0,
                                 partial_rho=r, p_partial=p, ci_lo=lo, ci_hi=hi, n=n))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, "h3_correlations.csv"), index=False)
    z = df.copy()
    for c in ["cov_mean", "log_nrel", "q_terms"]:
        z[c + "_z"] = (z[c] - z[c].mean()) / z[c].std()
    mods = {}
    for sysname in ["bm25", "minilm"]:
        if sysname in z:
            zz = z.dropna(subset=[sysname])
            m = mixed(zz, f"Q('{sysname}') ~ cov_mean_z + log_nrel_z + q_terms_z")
            mods[sysname] = dict(coef=m.params["cov_mean_z"], p=m.pvalues["cov_mean_z"],
                                 ci=m.conf_int().loc["cov_mean_z"].tolist())
    prim = t[(t.system == "bm25") & (t.feature == "cov_mean")]
    k = int(((prim.partial_rho > 0) & (prim.p_partial < ALPHA)).sum())
    a = mods["bm25"]["coef"] > 0 and mods["bm25"]["p"] < ALPHA
    b = k >= 6
    v = "SUPPORTED" if a and b else ("PARTIALLY SUPPORTED" if a or b else "NOT SUPPORTED")
    return dict(verdict=v, mixed=mods, n_datasets_sig_positive=k,
                per_dataset=prim[["dataset", "partial_rho", "ci_lo", "ci_hi", "p_partial"]].to_dict("records")), df


def h5(df, dense="minilm"):
    """H5: dense beats BM25 mainly on low-overlap queries (gap = dense - BM25 decreases with overlap).
    VERDICT RULE: SUPPORTED if mixed-model coefficient of overlap on gap < 0 (p<.05) and partial
    Spearman negative & significant in >= 6/8 datasets; PARTIALLY if one; else NOT SUPPORTED."""
    if dense not in df:
        return None
    df = df.copy()
    df["gap"] = df[dense] - df["bm25"]
    rows = []
    for d in DATASETS:
        s = df[df.dataset == d].dropna(subset=["gap"])
        if len(s) == 0:
            continue
        r, p, lo, hi, n = partial_spearman(s, "cov_mean", "gap", ["q_terms", "log_nrel"], nboot=500)
        bins = pd.cut(s.cov_mean, [-.01, .25, .5, .75, 1.0])
        wins = s.groupby(bins).apply(lambda x: pd.Series(dict(n=len(x), dense_win=(x.gap > 0).mean(),
                                                               bm25_win=(x.gap < 0).mean(), mean_gap=x.gap.mean())))
        rows.append(dict(dataset=d, partial_rho=r, ci_lo=lo, ci_hi=hi, p=p, n=n,
                         **{f"gap_bin{i}": v for i, v in enumerate(wins.mean_gap.values)}))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, f"h5_gap_{dense}.csv"), index=False)
    z = df.dropna(subset=["gap"]).copy()
    for c in ["cov_mean", "log_nrel", "q_terms"]:
        z[c + "_z"] = (z[c] - z[c].mean()) / z[c].std()
    m = mixed(z, "gap ~ cov_mean_z + log_nrel_z + q_terms_z")
    coef, p = m.params["cov_mean_z"], m.pvalues["cov_mean_z"]
    k = int(((t.partial_rho < 0) & (t.p < ALPHA)).sum())
    a, b = coef < 0 and p < ALPHA, k >= 6
    v = "SUPPORTED" if a and b else ("PARTIALLY SUPPORTED" if a or b else "NOT SUPPORTED")
    return dict(verdict=v, dense=dense, mixed=dict(coef=coef, p=p, ci=m.conf_int().loc["cov_mean_z"].tolist()),
                n_datasets_sig_negative=k, per_dataset=t[["dataset", "partial_rho", "ci_lo", "ci_hi", "p"]].to_dict("records"))


def h6(df, dense="minilm"):
    """H6: a single overlap threshold tells you when BM25 alone is defensible.
    Rule 'use dense if cov_mean < t' learned on 7 datasets, tested on the held-out one (LODO),
    target = dense beats BM25 (queries with |gap| < 0.01 excluded as ties).
    VERDICT RULE: SUPPORTED if mean LODO balanced accuracy >= 0.60 and >= 6/8 folds >= 0.55;
    PARTIALLY if mean >= 0.55; else NOT SUPPORTED. Threshold-free AUC also reported."""
    if dense not in df:
        return None
    z = df.dropna(subset=[dense]).copy()
    z["gap"] = z[dense] - z["bm25"]
    z = z[z.gap.abs() >= 0.01]
    z["y"] = (z.gap > 0).astype(int)
    grid = np.round(np.arange(0.05, 1.0001, 0.05), 2)

    def bacc(s, t):
        pred = (s.cov_mean < t).astype(int)
        tpr = ((pred == 1) & (s.y == 1)).sum() / max(1, (s.y == 1).sum())
        tnr = ((pred == 0) & (s.y == 0)).sum() / max(1, (s.y == 0).sum())
        return (tpr + tnr) / 2

    rows = []
    for d in DATASETS:
        tr, te = z[z.dataset != d], z[z.dataset == d]
        if len(te) == 0 or te.y.nunique() < 2:
            continue
        # balance training datasets equally (Quora would otherwise dominate)
        scores = [np.mean([bacc(tr[tr.dataset == dd], t) for dd in tr.dataset.unique()]) for t in grid]
        t = grid[int(np.argmax(scores))]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(te.y, -te.cov_mean)
        rows.append(dict(held_out=d, threshold=t, train_bacc=max(scores), test_bacc=bacc(te, t),
                         auc_low_overlap_predicts_dense=auc, n=len(te), dense_win_rate=te.y.mean()))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, f"h6_threshold_{dense}.csv"), index=False)
    mb = t.test_bacc.mean()
    k = int((t.test_bacc >= 0.55).sum())
    v = "SUPPORTED" if mb >= 0.60 and k >= 6 else ("PARTIALLY SUPPORTED" if mb >= 0.55 else "NOT SUPPORTED")
    return dict(verdict=v, mean_lodo_bacc=mb, folds_ge_055=k, thresholds=t.threshold.tolist(),
                threshold_median=float(t.threshold.median()), mean_auc=t.auc_low_overlap_predicts_dense.mean(),
                per_fold=t.to_dict("records"))


def ablations(W, df):
    """E2 (stemming/stopping) and E5 (RM3, WordNet) deltas vs BM25 with Holm-corrected tests;
    plus whether expansion gains concentrate on low-overlap queries."""
    comps = [("bm25", "bm25[stop]", "stemming"), ("bm25", "bm25[stem]", "stopword removal"),
             ("bm25_rm3", "bm25", "RM3 expansion"), ("bm25_wordnet", "bm25", "WordNet expansion")]
    for dn in ["minilm", "bge"]:
        if dn in W:
            comps += [(dn, "bm25", f"dense ({dn})"), (f"hybrid_rrf_{dn}", "bm25", f"hybrid RRF ({dn})"),
                      (f"hybrid_cc_{dn}", "bm25", f"hybrid convex ({dn})")]
    rows = []
    for d in DATASETS:
        w = W[W.dataset == d]
        f = df[df.dataset == d]
        for a, b, lab in comps:
            if a not in w or w[a].isna().all():
                continue
            diff, p = perm_test(w[a], w[b])
            gain = (f[a] - f[b]) if a in f else None
            r = stats.spearmanr(f.cov_mean, gain)[0] if gain is not None else np.nan
            rows.append(dict(dataset=d, comparison=lab, a=a, b=b, diff=diff, p=p, rho_gain_vs_overlap=r))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    t.to_csv(os.path.join(RES, "ablation_tests.csv"), index=False)
    return t


# ----------------------------------------------------------------------------- figures
def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def figures(tab, dtab, df, abl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                         "axes.labelcolor": INK, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    seq = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
    div = LinearSegmentedColormap.from_list("div", ["#c0392b", "#e88a84", "#f0efec", "#86b6ef", "#1c5cab"])

    # Fig 1: heatmap nDCG@10
    order = ["bool_and", "bool_or", "tfidf", "bm25", "bm25_rm3", "bm25_wordnet", "minilm", "bge",
             "hybrid_rrf_minilm", "hybrid_cc_minilm"]
    order = [s for s in order if s in set(tab.system)]
    M = tab.pivot(index="dataset", columns="system", values="ndcg@10").loc[DATASETS, order]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    im = ax.imshow(M.values, cmap=seq, vmin=0, vmax=0.9, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.45 else INK)
    ax.set_xticks(range(len(order)), [s.replace("hybrid_", "hyb_").replace("_minilm", "") for s in order],
                  rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(DATASETS)), [PRETTY[d] for d in DATASETS], fontsize=8)
    ax.set_title("nDCG@10 by dataset and system", loc="left", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_heatmap.png"), dpi=200)
    plt.close(fig)

    # Fig 2: domain-level specificity vs BM25
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3))
    for ax, ycol, ttl in ((axes[0], "bm25_ndcg", "BM25 nDCG@10"),
                          (axes[1], "gap_minilm", "MiniLM minus BM25 (nDCG@10)")):
        if ycol not in dtab:
            continue
        ax.scatter(dtab.corpus_jargon, dtab[ycol], s=40, color=BLUE, edgecolor="#fcfcfb", lw=1.5, zorder=3)
        for d, r in dtab.iterrows():
            ax.annotate(PRETTY[d], (r.corpus_jargon, r[ycol]), xytext=(4, 3), textcoords="offset points",
                        fontsize=7, color=INK)
        if ycol == "gap_minilm":
            ax.axhline(0, color=MUTED, lw=0.8)
        rho, p = stats.spearmanr(dtab.corpus_jargon, dtab[ycol])
        ax.set_title(f"{ttl}\nSpearman rho={rho:.2f}, p={p:.2f} (n=8)", loc="left", fontsize=9)
        ax.set_xlabel("corpus jargon rate (share of tokens with Zipf < 3)")
        style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_domain.png"), dpi=200)
    plt.close(fig)

    # Fig 3: small multiples, nDCG vs overlap bins
    bins = [-.01, .25, .5, .75, 1.0]
    labs = ["0-.25", ".25-.5", ".5-.75", ".75-1"]
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.8), sharey=True)
    series = [("bm25", BLUE, "BM25"), ("minilm", ORANGE, "MiniLM"), ("hybrid_rrf_minilm", AQUA, "Hybrid RRF")]
    for ax, d in zip(axes.ravel(), DATASETS):
        s = df[df.dataset == d].copy()
        s["bin"] = pd.cut(s.cov_mean, bins, labels=labs)
        cnt = s.bin.value_counts().reindex(labs)
        for col, c, lab in series:
            if col in s and not s[col].isna().all():
                m = s.groupby("bin")[col].mean().reindex(labs)
                ok = cnt.values >= 5
                ax.plot(np.arange(4)[ok], m.values[ok], color=c, lw=2, marker="o", ms=4, label=lab)
        ax.set_title(f"{PRETTY[d]}", loc="left", fontsize=9)
        ax.set_xticks(range(4), [f"{l}\n(n={n})" for l, n in zip(labs, cnt.fillna(0).astype(int))], fontsize=6.5)
        style(ax)
    axes[0, 0].set_ylabel("mean nDCG@10")
    axes[1, 0].set_ylabel("mean nDCG@10")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Effectiveness by query-term coverage of relevant docs (bins with n<5 hidden)", x=0.01,
                 ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_overlap_bins.png"), dpi=200)
    plt.close(fig)

    # Fig 4: forest plot of partial correlations (H3 BM25, H5 gap)
    h3t = pd.read_csv(os.path.join(RES, "h3_correlations.csv"))
    h3t = h3t[(h3t.system == "bm25") & (h3t.feature == "cov_mean")].set_index("dataset").loc[DATASETS]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    y = np.arange(len(DATASETS))
    ax.errorbar(h3t.partial_rho, y + 0.12, xerr=[h3t.partial_rho - h3t.ci_lo, h3t.ci_hi - h3t.partial_rho],
                fmt="o", color=BLUE, ms=5, lw=1.5, label="overlap vs BM25 nDCG@10 (H3)")
    p5 = os.path.join(RES, "h5_gap_minilm.csv")
    if os.path.exists(p5):
        h5t = pd.read_csv(p5).set_index("dataset").reindex(DATASETS)
        ax.errorbar(h5t.partial_rho, y - 0.12, xerr=[h5t.partial_rho - h5t.ci_lo, h5t.ci_hi - h5t.partial_rho],
                    fmt="s", color=ORANGE, ms=5, lw=1.5, label="overlap vs (MiniLM - BM25) gap (H5)")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, [PRETTY[d] for d in DATASETS])
    ax.invert_yaxis()
    ax.set_xlabel("partial Spearman rho (controls: #query terms, #relevant), 95% bootstrap CI")
    ax.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1)
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_forest.png"), dpi=200)
    plt.close(fig)

    # Fig 5: ablation deltas (diverging heatmap)
    A = abl.pivot(index="dataset", columns="comparison", values="diff").reindex(DATASETS)
    P = abl.pivot(index="dataset", columns="comparison", values="p_holm").reindex(DATASETS)
    cols = [c for c in ["stemming", "stopword removal", "RM3 expansion", "WordNet expansion",
                        "dense (minilm)", "dense (bge)", "hybrid RRF (minilm)", "hybrid convex (minilm)"] if c in A]
    A, P = A[cols], P[cols]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    lim = np.nanmax(np.abs(A.values))
    im = ax.imshow(A.values, cmap=div, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A.values[i, j]
            if not np.isnan(v):
                star = "*" if P.values[i, j] < ALPHA else ""
                ax.text(j, i, f"{v:+.3f}{star}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(v) > 0.6 * lim else INK)
    ax.set_xticks(range(len(cols)), cols, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(DATASETS)), [PRETTY[d] for d in DATASETS], fontsize=8)
    ax.set_title("Change in nDCG@10 vs the reference (* = Holm-adjusted p < .05)", loc="left", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_ablation.png"), dpi=200)
    plt.close(fig)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


if __name__ == "__main__":
    pq, feats, dom = load()
    tab = main_table(pq)
    W = wide(pq)
    verdicts = {}
    verdicts["H1"] = h1(W)
    verdicts["H2"], dtab = h2(dom, tab)
    verdicts["H3"], df = h3(W, feats)
    verdicts["H5"] = h5(df, "minilm")
    if "bge" in df and df.bge.notna().any():
        verdicts["H5_bge_robustness"] = h5(df[df.bge.notna()], "bge")
    verdicts["H6"] = h6(df, "minilm")
    abl = ablations(W, df)
    figures(tab, dtab, df, abl)
    # Boolean set-based summary & BM25 grid summary
    bs = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "boolean_sets_*.csv"))])
    bsum = bs.groupby("dataset").agg(empty_frac=("set_size", lambda x: (x == 0).mean()),
                                     median_set=("set_size", "median"), P=("precision", "mean"),
                                     R=("recall", "mean"), F1=("f1", "mean")).reindex(DATASETS)
    bsum.to_csv(os.path.join(RES, "boolean_set_summary.csv"))
    gr = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "bm25_grid_*.csv"))])
    gsum = gr.groupby("dataset").apply(lambda g: pd.Series(dict(
        default=g[(g.k1 == 0.9) & (g.b == 0.4)].ndcg10.iloc[0], best=g.ndcg10.max(),
        best_k1=g.loc[g.ndcg10.idxmax()].k1, best_b=g.loc[g.ndcg10.idxmax()].b))).reindex(DATASETS)
    gsum.to_csv(os.path.join(RES, "bm25_grid_summary.csv"))
    with open(os.path.join(RES, "verdicts.json"), "w") as f:
        json.dump(jsonable(verdicts), f, indent=1)
    for k, v in verdicts.items():
        if v:
            print(k, v["verdict"])
