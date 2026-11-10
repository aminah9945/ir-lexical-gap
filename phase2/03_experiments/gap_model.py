"""W3: model the dense-minus-BM25 gap against query/relevant-document term coverage (explanatory analysis:
cov_mean uses qrels, which is legitimate here because nothing is deployed; it is NOT a router input).

gap_q = nDCG@10_D - nDCG@10_B, D in {MiniLM, BGE-s, (BGE-b once the Colab runs exist)}.
Predictor z_cov = cov_mean z-scored within dataset. Controls: log(q_terms), z(q_mean_idf) within dataset, log(n_rel).
  Model A: MixedLM (REML), random intercept + random slope on z_cov by dataset (fallback: random intercept).
  Model B (primary): per-dataset OLS with HC3 SEs, DerSimonian-Laird meta-analysis of the z_cov slopes.
  Circularity: Model B with nDCG_B and nDCG_D as outcomes; slope_B - slope_D, pooled with the DL weights of the
               gap model, dataset-stratified query bootstrap CI (2,000 resamples).
  Robustness: 3-class outcome (D wins / tie |gap|<=0.01 / B wins), multinomial logit with dataset fixed effects.
  W2 (if results/qpp/tierB_<model>_<ds>.csv exist): Model B with qrel-free proxies (pcov_D10, pcov_gap) in place of
     cov_mean, and Spearman rho(proxy, cov_mean) per dataset with bootstrap CIs.
Outputs: results/w3_*.csv, figures/w3_gap_bins.png
"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

import common2 as c2

warnings.filterwarnings("ignore")
NB = 2000


def zs(x):
    x = np.asarray(x, float)
    s = x.std(ddof=0)
    return (x - x.mean()) / s if s > 0 else x * 0


def build(model):
    frames = []
    for d in c2.ALL_DATASETS:
        w = c2.perquery(d)
        if model not in w or "bm25" not in w:
            continue
        f = pd.read_csv(os.path.join(c2.P1_RES, f"features_{d}.csv"), dtype={"qid": str})
        x = f.merge(w[["bm25", model]].reset_index(), on="qid")
        x = x[(x.q_terms > 0) & (x.n_rel > 0)].copy()
        x["dataset"] = d
        x["gap"] = x[model] - x["bm25"]
        x["yB"], x["yD"] = x["bm25"], x[model]
        x["z_cov"] = zs(x.cov_mean)
        x["z_idf"] = zs(x.q_mean_idf)
        x["log_terms"] = np.log(x.q_terms)
        x["log_nrel"] = np.log(x.n_rel)
        frames.append(x)
    return pd.concat(frames, ignore_index=True) if frames else None


XCOLS = ["z_cov", "log_terms", "z_idf", "log_nrel"]


def ols_hc3(y, X):
    Xc = sm.add_constant(X, has_constant="add")
    r = sm.OLS(y, Xc).fit(cov_type="HC3")
    return r


def model_b(df, outcome, pred="z_cov", xcols=None):
    xcols = xcols or [pred] + [c for c in XCOLS if c != "z_cov"]
    rows = []
    for d, g in df.groupby("dataset", sort=False):
        g = g.dropna(subset=[outcome] + xcols)
        X = g[xcols].copy()
        X = X.loc[:, X.std() > 0]
        r = ols_hc3(g[outcome].values, X)
        rows.append(dict(dataset=d, n=len(g), slope=r.params.get(pred, np.nan), se=r.bse.get(pred, np.nan),
                         p=r.pvalues.get(pred, np.nan), r2=r.rsquared))
    t = pd.DataFrame(rows)
    meta = c2.dl_meta(t.slope.values, t.se.values)
    return t, meta


def slopes_fast(df, outcome, idx_by_ds, rng=None):
    """Per-dataset OLS z_cov slope with plain lstsq (for the bootstrap)."""
    out = []
    for d, (Xd, yd) in idx_by_ds.items():
        n = len(yd[outcome])
        ii = rng.integers(0, n, n) if rng is not None else np.arange(n)
        X = Xd[ii]
        y = yd[outcome][ii]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.append(beta[1])
    return np.array(out)


def circularity(df, gap_meta_weights):
    tb, mb = model_b(df, "yB")
    td, md = model_b(df, "yD")
    prep = {}
    for d, g in df.groupby("dataset", sort=False):
        X = np.column_stack([np.ones(len(g)), g[XCOLS].values])
        prep[d] = (X, {"yB": g.yB.values, "yD": g.yD.values})
    w = gap_meta_weights / gap_meta_weights.sum()
    point = float(np.sum(w * (tb.slope.values - td.slope.values)))
    rng = np.random.default_rng(c2.SEED)
    bs = np.empty(NB)
    for i in range(NB):
        # identical query resample for B and D (re-seeded identically)
        seed = rng.integers(1 << 31)
        sB = slopes_fast(df, "yB", prep, np.random.default_rng(seed))
        sD = slopes_fast(df, "yD", prep, np.random.default_rng(seed))
        bs[i] = np.sum(w * (sB - sD))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    per = tb[["dataset", "n", "slope", "se"]].rename(columns={"slope": "slope_B", "se": "se_B"}).merge(
        td[["dataset", "slope", "se"]].rename(columns={"slope": "slope_D", "se": "se_D"}), on="dataset")
    per["diff_B_minus_D"] = per.slope_B - per.slope_D
    summ = dict(pooled_slope_B=mb["est"], pooled_slope_B_lo=mb["lo"], pooled_slope_B_hi=mb["hi"],
                pooled_slope_D=md["est"], pooled_slope_D_lo=md["lo"], pooled_slope_D_hi=md["hi"],
                diff_weighted=point, diff_boot_lo=lo, diff_boot_hi=hi, ratio_D_over_B=md["est"] / mb["est"])
    return per, summ


def mixed(df):
    try:
        m = smf.mixedlm("gap ~ z_cov + log_terms + z_idf + log_nrel", df, groups=df["dataset"],
                        re_formula="~z_cov").fit(reml=True, method="lbfgs")
        kind = "random intercept + slope"
        if not m.converged:
            raise RuntimeError("no convergence")
    except Exception as e:  # noqa
        m = smf.mixedlm("gap ~ z_cov + log_terms + z_idf + log_nrel", df, groups=df["dataset"]).fit(reml=True)
        kind = f"random intercept only (fallback: {e})"
    ci = m.conf_int().loc["z_cov"]
    return dict(kind=kind, coef=m.params["z_cov"], lo=ci[0], hi=ci[1], p=m.pvalues["z_cov"],
                re_var_slope=float(m.cov_re.iloc[-1, -1]) if m.cov_re.shape[0] > 1 else np.nan,
                converged=bool(m.converged))


def mnlogit(df):
    y = np.where(df.gap > c2.TIE, 2, np.where(df.gap < -c2.TIE, 0, 1))  # 0 = B wins, 1 = tie, 2 = D wins
    X = pd.concat([df[XCOLS], pd.get_dummies(df.dataset, drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    r = sm.MNLogit(y, X).fit(disp=False, maxiter=200)
    out = []
    for j, lab in [(0, "tie vs B wins"), (1, "D wins vs B wins")]:
        out.append(dict(contrast=lab, coef_z_cov=r.params.iloc[:, j]["z_cov"], se=r.bse.iloc[:, j]["z_cov"],
                        p=r.pvalues.iloc[:, j]["z_cov"], odds_ratio=float(np.exp(r.params.iloc[:, j]["z_cov"]))))
    return out, dict(B_wins=float(np.mean(y == 0)), tie=float(np.mean(y == 1)), D_wins=float(np.mean(y == 2)))


def proxies(df, model):
    """W2: qrel-free proxies vs cov_mean, and Model B with proxies (needs tier-B feature files)."""
    frames = []
    for d in df.dataset.unique():
        p = os.path.join(c2.RES2, "qpp", f"tierB_{model}_{d}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p, dtype={"qid": str})[["dataset", "qid", "pcov_B10", "pcov_D10", "pcov_gap",
                                                                "pcov_any_B10", "pcov_any_D10"]])
    if not frames:
        return None, None
    x = df.merge(pd.concat(frames), on=["dataset", "qid"])
    rho_rows = []
    rng = np.random.default_rng(c2.SEED)
    for d, g in x.groupby("dataset", sort=False):
        for pr in ["pcov_B10", "pcov_D10", "pcov_gap", "pcov_any_B10", "pcov_any_D10"]:
            gg = g[[pr, "cov_mean"]].dropna()
            rho = stats.spearmanr(gg[pr], gg.cov_mean).statistic
            bs = []
            for _ in range(1000):
                ii = rng.integers(0, len(gg), len(gg))
                bs.append(stats.spearmanr(gg[pr].values[ii], gg.cov_mean.values[ii]).statistic)
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            rho_rows.append(dict(model=model, dataset=d, proxy=pr, rho=rho, lo=lo, hi=hi, n=len(gg)))
    fits = []
    for pr in ["pcov_D10", "pcov_gap"]:
        x["z_" + pr] = x.groupby("dataset")[pr].transform(zs)
        t, meta = model_b(x, "gap", pred="z_" + pr, xcols=["z_" + pr, "log_terms", "z_idf"])
        t["proxy"] = pr
        t["model"] = model
        fits.append((t, dict(model=model, proxy=pr, **{k: meta[k] for k in ("est", "lo", "hi", "p", "tau2", "I2", "k")})))
    return pd.DataFrame(rho_rows), fits


def gap_bins_figure(dfs):
    plt = c2.mpl_style()
    dss = [d for d in c2.ALL_DATASETS if any(d in set(df.dataset) for df in dfs.values())]
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.2), sharex=True, sharey=True)
    cols = {"minilm": c2.C_DENSE, "bge": "#b8431a", "bgebase": "#7a2a0e"}
    for ax, d in zip(axes.ravel(), dss):
        for m, df in dfs.items():
            g = df[df.dataset == d]
            if g.empty:
                continue
            bins = pd.cut(g.cov_mean, [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0])
            s = g.groupby(bins, observed=True).gap.agg(["mean", "sem", "size"])
            xs = [iv.mid for iv in s.index]
            ax.errorbar(xs, s["mean"], yerr=1.96 * s["sem"].fillna(0), color=cols[m], marker="o", ms=3, lw=1.2,
                        capsize=2, label=c2.MODEL_PRETTY[m])
        ax.axhline(0, color=c2.C_MUTED, lw=0.8)
        ax.set_title(c2.PRETTY[d])
    for ax in axes[1]:
        ax.set_xlabel("coverage of query terms in relevant docs")
    for ax in axes[:, 0]:
        ax.set_ylabel("nDCG@10 dense − BM25")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(c2.FIG2, "w3_gap_bins.png"))


def main():
    rows_b, rows_meta, rows_mix, rows_circ, rows_circ_per, rows_mn, rho_all, prox_fit, prox_meta = \
        [], [], [], [], [], [], [], [], []
    dfs = {}
    for m in c2.DENSE_MODELS:
        df = build(m)
        if df is None or df.dataset.nunique() < 3:
            continue
        dfs[m] = df
        t, meta = model_b(df, "gap")
        t["model"] = m
        rows_b.append(t)
        rows_meta.append(dict(model=m, outcome="gap", **meta, sign_neg=int((t.slope < 0).sum()),
                              sig_neg=int(((t.slope < 0) & (t.p < .05)).sum()), n_queries=len(df)))
        # LODO pooled estimates and sensitivity without the two small sets
        loo = [c2.dl_meta(t[t.dataset != d].slope, t[t.dataset != d].se)["est"] for d in t.dataset]
        rows_meta[-1].update(loo_min=min(loo), loo_max=max(loo))
        s = t[~t.dataset.isin(["trec-covid", "webis-touche2020"])]
        ms = c2.dl_meta(s.slope, s.se)
        rows_meta[-1].update(excl_small_est=ms["est"], excl_small_lo=ms["lo"], excl_small_hi=ms["hi"])
        rows_mix.append(dict(model=m, **mixed(df)))
        w = 1 / (t.se.values ** 2 + meta["tau2"])
        per, summ = circularity(df, w)
        per["model"] = m
        rows_circ_per.append(per)
        rows_circ.append(dict(model=m, **summ))
        mn, shares = mnlogit(df)
        for r in mn:
            rows_mn.append(dict(model=m, **r, **shares))
        rho, fits = proxies(df, m)
        if rho is not None:
            rho_all.append(rho)
            for t2, mt in fits:
                prox_fit.append(t2)
                prox_meta.append(mt)
    W = lambda x, f: x.to_csv(os.path.join(c2.RES2, f), index=False)  # noqa
    W(pd.concat(rows_b), "w3_modelB_perdataset.csv")
    W(pd.DataFrame(rows_meta), "w3_modelB_meta.csv")
    W(pd.DataFrame(rows_mix), "w3_mixedlm.csv")
    W(pd.concat(rows_circ_per), "w3_circularity_perdataset.csv")
    W(pd.DataFrame(rows_circ), "w3_circularity.csv")
    W(pd.DataFrame(rows_mn), "w3_mnlogit.csv")
    if rho_all:
        W(pd.concat(rho_all), "w2_proxy_rho.csv")
        W(pd.concat(prox_fit), "w2_proxy_modelB_perdataset.csv")
        W(pd.DataFrame(prox_meta), "w2_proxy_modelB_meta.csv")
    print(pd.DataFrame(rows_meta).round(4).to_string())
    print(pd.DataFrame(rows_mix).round(4).to_string())
    print(pd.DataFrame(rows_circ).round(4).to_string())
    print(pd.DataFrame(rows_mn).round(4).to_string())
    if prox_meta:
        print(pd.DataFrame(prox_meta).round(4).to_string())
        print(pd.concat(rho_all).groupby(["model", "proxy"]).rho.describe().round(3).to_string())
    gap_bins_figure(dfs)


if __name__ == "__main__":
    main()
