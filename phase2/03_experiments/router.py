"""W1: query-time router among S = {B, D, H(D)} with leave-one-dataset-out (LODO) evaluation.

Pre-declared (before any router result was seen; see EXPERIMENT_LOG.md):
  * primary hypothesis: macro-average nDCG@10 (router) - macro (always-H) > 0, Tier A and Tier B reported separately;
  * headline router = cost-sensitive regression, model family chosen by an inner LODO on the training datasets
    between Ridge(alpha=1) and HGB (fixed hyper-parameters), standard scaling fitted on the training datasets;
  * secondary: each model family on its own, multinomial logistic classification, within-dataset percentile
    features (transductive), BM25-LODO (W6) as B.
Protocol: for each held-out dataset h, everything (scaler, imputer, models, operating point) is fitted on the other
datasets only; each training dataset has total sample weight 1. No qrel-derived feature is used (asserted).

Usage: python router.py            (runs every configuration whose inputs exist)
Outputs: results/w1_router_perdataset.csv, results/w1_router_macro.csv, results/w1_router_tests.csv,
         results/w1_efficiency_curve.csv, results/w1_efficiency_operating.csv, results/w1_router_perquery/*.csv.gz,
         figures/w1_router_bars_<config>.png, figures/w1_efficiency_<config>.png
"""
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")  # HGB OpenMP oversubscription made fits ~35x slower; configs run in parallel

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common2 as c2

warnings.filterwarnings("ignore")
QDIR = os.path.join(c2.RES2, "qpp")
PQOUT = os.path.join(c2.RES2, "w1_router_perquery")
os.makedirs(PQOUT, exist_ok=True)
MARGIN = 0.005          # pre-declared cost tie-break margin
EFF_LOSS = 0.005        # pre-declared efficiency operating-point tolerance (macro, training datasets)
SYS = ["B", "D", "H"]   # cost order: B (no encoder) < D (encoder only) < H (both)
ID_COLS = {"dataset", "qid"}


def hgb():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50,
                                         l2_regularization=1.0, random_state=c2.SEED)


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


def hgb_pipe():
    # HGB handles NaN natively; scaling is monotone and irrelevant for trees but kept for symmetry
    return make_pipeline(StandardScaler(), hgb())


MAKERS = {"ridge": ridge, "hgb": hgb_pipe}


# ----------------------------------------------------------------------------- data
def assemble(model, tier, bvar="bm25"):
    hsys = f"hybrid_cc_{model}" if bvar == "bm25" else f"hybrid_cc_{model}_{bvar}"
    frames = []
    for d in c2.ALL_DATASETS:
        fa = os.path.join(QDIR, f"tierA_{d}.csv")
        fb = os.path.join(QDIR, f"tierB_{model}_{d}.csv")
        if not os.path.exists(fa) or (tier == "B" and not os.path.exists(fb)):
            continue
        w = c2.perquery(d)
        if not {bvar, model, hsys} <= set(w.columns):
            continue
        lab = w[[bvar, model, hsys]].rename(columns={bvar: "B", model: "D", hsys: "H"}).dropna().reset_index()
        X = pd.read_csv(fa, dtype={"qid": str})
        if tier == "B":
            X = X.merge(pd.read_csv(fb, dtype={"qid": str}).drop(columns="dataset"), on="qid")
        x = lab.merge(X, on="qid", how="inner")
        assert len(x) == len(lab), (d, len(x), len(lab))
        x["dataset"] = d
        frames.append(x)
    if len(frames) < 3:
        return None, None
    df = pd.concat(frames, ignore_index=True)
    feats = [c for c in df.columns if c not in ID_COLS | set(SYS)]
    c2.assert_no_qrel_features(feats)
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan).astype(float)
    return df, feats


def pct_transform(df, feats):
    out = df.copy()
    out[feats] = df.groupby("dataset")[feats].rank(pct=True)
    return out


def weights(df):
    return 1.0 / df.groupby("dataset")["qid"].transform("size").values


# ----------------------------------------------------------------------------- routing
def choose(pred_dB, pred_dD, margin=MARGIN):
    v = np.column_stack([pred_dB, pred_dD, np.zeros(len(pred_dB))])      # values relative to H
    best = v.max(1, keepdims=True)
    ok = v >= best - margin
    return np.argmax(ok, axis=1)                                          # cheapest system within the margin


def fit_predict_reg(kind, tr, te, feats):
    w = weights(tr)
    preds = []
    for tgt in ("B", "D"):
        y = tr[tgt].values - tr["H"].values
        m = MAKERS[kind]()
        last = m.steps[-1][0]
        m.fit(tr[feats].values, y, **{f"{last}__sample_weight": w})
        preds.append(m.predict(te[feats].values))
    return preds


def fit_predict_clf(tr, te, feats):
    Y = tr[SYS].values
    srt = np.sort(Y, 1)
    margin = srt[:, -1] - srt[:, -2]
    lab = Y.argmax(1)
    keep = margin > 0
    w = weights(tr) * margin
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(C=1.0, max_iter=2000))
    m.fit(tr[feats].values[keep], lab[keep], logisticregression__sample_weight=w[keep])
    proba = np.zeros((len(te), 3))
    proba[:, m.classes_] = m.predict_proba(te[feats].values)
    return proba.argmax(1)


def routed_value(df, choice):
    return df[SYS].values[np.arange(len(df)), choice]


def macro_of(df, vals):
    return pd.Series(vals, index=df.index).groupby(df.dataset).mean().mean()


def inner_select(tr, feats):
    """Inner LODO over the training datasets: choose ridge vs hgb by macro routed nDCG; also return OOF dB preds."""
    dss = tr.dataset.unique()
    res, oof = {}, {}
    for kind in MAKERS:
        val = np.empty(len(tr))
        pb = np.empty(len(tr))
        for d in dss:
            itr, ite = tr[tr.dataset != d], tr[tr.dataset == d]
            p_b, p_d = fit_predict_reg(kind, itr, ite, feats)
            ii = np.where(tr.dataset.values == d)[0]
            val[ii] = routed_value(ite, choose(p_b, p_d))
            pb[ii] = p_b
        res[kind] = macro_of(tr, val)
        oof[kind] = pb
    best = max(res, key=res.get)
    return best, res, oof[best]


def eff_threshold(tr, oof_dB):
    """Largest BM25 share (route to B iff predicted dB > tau) with macro loss vs always-H <= EFF_LOSS on training."""
    taus = np.unique(np.quantile(oof_dB, np.linspace(0, 1, 201)))
    base = macro_of(tr, tr.H.values)
    best_tau, best_share = np.inf, 0.0
    for t in taus:
        toB = oof_dB > t
        val = np.where(toB, tr.B.values, tr.H.values)
        loss = base - macro_of(tr, val)
        share = pd.Series(toB, index=tr.index).groupby(tr.dataset).mean().mean()
        if loss <= EFF_LOSS and share > best_share:
            best_tau, best_share = t, share
    return best_tau


# ----------------------------------------------------------------------------- one configuration
def run_config(name, df, feats, transform="std"):
    t0 = time.time()
    if transform == "pct":
        df = pct_transform(df, feats)
    dss = list(dict.fromkeys(df.dataset))
    # "selected_argmax" (post-hoc, added after seeing the first results): same predictions as "selected" but plain
    # argmax without the pre-declared 0.005 cost tie-break.
    out = {k: np.empty(len(df), int) for k in ["ridge", "hgb", "selected", "logreg", "selected_argmax"]}
    pred_dB_sel = np.empty(len(df))
    chosen_kind, eff_rows = {}, []
    for h in dss:
        tr, te = df[df.dataset != h], df[df.dataset == h]
        ii = np.where(df.dataset.values == h)[0]
        preds = {}
        for kind in MAKERS:
            preds[kind] = fit_predict_reg(kind, tr, te, feats)
            out[kind][ii] = choose(*preds[kind])
        kind, inner_scores, oof_dB = inner_select(tr, feats)
        chosen_kind[h] = (kind, inner_scores)
        out["selected"][ii] = out[kind][ii]
        out["selected_argmax"][ii] = choose(*preds[kind], margin=0.0)
        pred_dB_sel[ii] = preds[kind][0]
        out["logreg"][ii] = fit_predict_clf(tr, te, feats)
        tau = eff_threshold(tr, oof_dB)
        toB = preds[kind][0] > tau
        from scipy.stats import kendalltau
        eff_rows.append(dict(config=name, dataset=h, model_kind=kind, tau=tau, share_B=float(toB.mean()),
                             ktau_pred_vs_true_dB=float(kendalltau(preds[kind][0], te.B.values - te.H.values).statistic),
                             ktau_pred_vs_true_dD=float(kendalltau(preds[kind][1], te.D.values - te.H.values).statistic),
                             ndcg_eff=float(np.where(toB, te.B, te.H).mean()), ndcg_H=float(te.H.mean()),
                             ndcg_B=float(te.B.mean())))
    print(f"  {name}: fitted in {time.time()-t0:.0f}s; inner choice {[(h, k) for h, (k, _) in chosen_kind.items()]}",
          flush=True)
    return df, dss, out, pred_dB_sel, chosen_kind, pd.DataFrame(eff_rows)


def evaluate(name, df, dss, out, chosen_kind):
    rng = np.random.default_rng(c2.SEED)
    rows, pq_frames = [], []
    Y = df[SYS].values
    oracle_q = Y.max(1)
    srt = np.sort(Y, 1)
    unique_best = srt[:, -1] > srt[:, -2]
    olab = Y.argmax(1)
    for h in dss:
        m = df.dataset.values == h
        tr = df[~m]
        means_tr = tr.groupby("dataset")[SYS].mean().mean()
        bft = SYS.index(means_tr.idxmax())
        yh = Y[m]
        base = dict(config=name, dataset=h, n=int(m.sum()), always_B=yh[:, 0].mean(), always_D=yh[:, 1].mean(),
                    always_H=yh[:, 2].mean(), best_fixed_train=yh[:, bft].mean(), best_fixed_train_sys=SYS[bft],
                    oracle_dataset=yh.mean(0).max(), oracle_dataset_sys=SYS[int(yh.mean(0).argmax())],
                    oracle_query=oracle_q[m].mean())
        pq = pd.DataFrame({"qid": df.qid.values[m], "B": yh[:, 0], "D": yh[:, 1], "H": yh[:, 2],
                           "oracle": oracle_q[m]})
        for variant, ch in out.items():
            c = ch[m]
            val = yh[np.arange(len(c)), c]
            mix = np.bincount(c, minlength=3) / len(c)
            draws = np.array([yh[np.arange(len(c)), rng.choice(3, len(c), p=mix)].mean() for _ in range(1000)])
            hr_den = base["oracle_query"] - base["always_H"]
            acc = np.mean(c[unique_best[m]] == olab[m][unique_best[m]])
            rows.append(dict(base, variant=variant, router=val.mean(), random_same_mix=draws.mean(),
                             pct_B=mix[0], pct_D=mix[1], pct_H=mix[2],
                             HR=(val.mean() - base["always_H"]) / hr_den if hr_den >= 0.01 else np.nan,
                             acc_vs_oracle=acc, n_unique_best=int(unique_best[m].sum()),
                             inner_choice=chosen_kind[h][0] if variant == "selected" else variant))
            pq[f"router_{variant}"] = val
            pq[f"choice_{variant}"] = np.array(SYS)[c]
        pq["best_fixed_train"] = yh[:, bft]
        pq["dataset"] = h
        pq_frames.append(pq)
    per = pd.DataFrame(rows)
    pqa = pd.concat(pq_frames, ignore_index=True)
    pqa.to_csv(os.path.join(PQOUT, f"{name}.csv.gz"), index=False)
    return per, pqa


def tests(name, per, pqa):
    rows = []
    for variant in per.variant.unique():
        for base in ["H", "best_fixed_train", "B", "D"]:
            sub = []
            for h, g in pqa.groupby("dataset", sort=False):
                d = g[f"router_{variant}"].values - g[base].values
                r = c2.paired_summary(g[f"router_{variant}"].values, g[base].values)
                sub.append(dict(config=name, variant=variant, baseline=base, dataset=h, **r))
            s = pd.DataFrame(sub)
            s["p_holm"] = c2.holm(s.p.values)
            meta = c2.dl_meta(s.delta.values, s.se_boot.values)
            macro, p2, p1 = c2.perm_test_macro([g[f"router_{variant}"].values - g[base].values
                                                for _, g in pqa.groupby("dataset", sort=False)])
            rows += s.to_dict("records")
            rows.append(dict(config=name, variant=variant, baseline=base, dataset="MACRO", delta=macro, p=p2,
                             p_one_sided=p1, n=int(s.n.sum())))
            rows.append(dict(config=name, variant=variant, baseline=base, dataset="POOLED_DL", delta=meta["est"],
                             ci_lo=meta["lo"], ci_hi=meta["hi"], p=meta["p"], n=int(s.n.sum()),
                             win=float((s.delta > 0).sum()), loss=float((s.delta < 0).sum())))
    return pd.DataFrame(rows)


def macro_table(per):
    cols = ["always_B", "always_D", "always_H", "best_fixed_train", "oracle_dataset", "oracle_query", "router",
            "random_same_mix", "pct_B", "pct_D", "pct_H", "acc_vs_oracle"]
    m = per.groupby(["config", "variant"])[cols].mean().reset_index()
    m["k"] = per.groupby(["config", "variant"]).size().values
    m["HR_macro"] = (m.router - m.always_H) / (m.oracle_query - m.always_H)
    m["router_minus_H"] = m.router - m.always_H
    # micro (query-weighted) secondary
    micro = per.assign(w=per.n).groupby(["config", "variant"]).apply(
        lambda g: pd.Series({"router_micro": np.average(g.router, weights=g.w),
                             "always_H_micro": np.average(g.always_H, weights=g.w)})).reset_index()
    return m.merge(micro, on=["config", "variant"])


def efficiency_curve(name, df, pred_dB):
    rows = []
    qs = np.unique(np.quantile(pred_dB, np.linspace(0, 1, 101)))
    for t in list(qs) + [np.inf]:
        toB = pred_dB > t
        val = np.where(toB, df.B.values, df.H.values)
        g = pd.DataFrame({"d": df.dataset.values, "v": val, "b": toB})
        rows.append(dict(config=name, tau=t, share_B_macro=g.groupby("d").b.mean().mean(),
                         ndcg_macro=g.groupby("d").v.mean().mean(), kind="router"))
    # oracle ordering (true dB) and random assignment, for reference
    dB = df.B.values - df.H.values
    for frac in np.linspace(0, 1, 51):
        vo, vr, sh = [], [], []
        for _, g in df.groupby("dataset"):
            k = int(round(frac * len(g)))
            order = np.argsort(-(g.B.values - g.H.values))
            sel = np.zeros(len(g), bool)
            sel[order[:k]] = True
            vo.append(np.where(sel, g.B, g.H).mean())
            vr.append(frac * g.B.mean() + (1 - frac) * g.H.mean())
        rows.append(dict(config=name, tau=np.nan, share_B_macro=frac, ndcg_macro=np.mean(vo), kind="oracle"))
        rows.append(dict(config=name, tau=np.nan, share_B_macro=frac, ndcg_macro=np.mean(vr), kind="random"))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def fig_bars(name, per):
    plt = c2.mpl_style()
    p = per[per.variant == "selected"].set_index("dataset")
    dss = list(p.index)
    x = np.arange(len(dss) + 1)
    cols = [("always_B", "always-BM25", c2.C_BM25), ("always_D", "always-dense", c2.C_DENSE),
            ("always_H", "always-hybrid", c2.C_HYB), ("router", "router", c2.C_ROUTER)]
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    wd = 0.2
    for i, (c, lab, col) in enumerate(cols):
        vals = list(p[c]) + [p[c].mean()]
        ax.bar(x + (i - 1.5) * wd, vals, wd, color=col, label=lab)
    oq = list(p.oracle_query) + [p.oracle_query.mean()]
    for xi, v in zip(x, oq):
        ax.plot([xi - 2 * wd, xi + 2 * wd], [v, v], color=c2.C_INK, lw=1.2)
    ax.plot([], [], color=c2.C_INK, lw=1.2, label="per-query oracle")
    ax.set_xticks(x)
    ax.set_xticklabels([c2.PRETTY[d] for d in dss] + ["Macro avg"], rotation=0, fontsize=7.5)
    ax.set_ylabel("nDCG@10")
    ax.legend(ncol=5, frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.15))
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(c2.FIG2, f"w1_router_bars_{name}.png"))
    plt.close(fig)


def fig_eff(name, curve, ops, per):
    plt = c2.mpl_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for kind, col, lab, ls in [("oracle", c2.C_MUTED, "oracle ordering", "--"), ("random", c2.C_GRID, "random", ":"),
                               ("router", c2.C_ROUTER, "Tier-A router (LODO)", "-")]:
        g = curve[curve.kind == kind].sort_values("share_B_macro")
        ax.plot(g.share_B_macro, g.ndcg_macro, ls, color=col if kind != "random" else c2.C_MUTED, lw=1.4, label=lab)
    o = ops
    ax.plot(o.share_B.mean(), o.ndcg_eff.mean(), "o", color=c2.C_ROUTER, ms=6, mfc="white",
            label=f"operating point (≤{EFF_LOSS} loss on train)")
    ax.axhline(o.ndcg_H.mean(), color=c2.C_HYB, lw=0.9, label="always-hybrid")
    ax.axhline(o.ndcg_B.mean(), color=c2.C_BM25, lw=0.9, label="always-BM25")
    ax.set_xlabel("share of queries served by BM25 only (macro)")
    ax.set_ylabel("macro nDCG@10")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.savefig(os.path.join(c2.FIG2, f"w1_efficiency_{name}.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def configs():
    cfg = []
    for m in c2.DENSE_MODELS:
        cfg.append((f"A_{m}_std", m, "A", "bm25", "std"))
        cfg.append((f"A_{m}_pct", m, "A", "bm25", "pct"))
        cfg.append((f"B_{m}_std", m, "B", "bm25", "std"))
        cfg.append((f"B_{m}_pct", m, "B", "bm25", "pct"))
        cfg.append((f"A_{m}_bm25lodo_std", m, "A", "bm25lodo", "std"))
    return cfg


PARTS = os.path.join(c2.RES2, "w1_parts")
os.makedirs(PARTS, exist_ok=True)
OUTS = ["perdataset", "tests", "efficiency_curve", "efficiency_operating", "inner_selection"]


def main(only=None):
    for name, m, tier, bvar, tf in configs():
        if only and name not in only:
            continue
        df, feats = assemble(m, tier, bvar)
        if df is None:
            print(f"skip {name}: inputs missing")
            continue
        name = f"{name}_k{df.dataset.nunique()}"
        print(f"== {name}: {df.dataset.nunique()} datasets, {len(df)} queries, {len(feats)} features", flush=True)
        df2, dss, out, pred_dB, chosen, eff = run_config(name, df, feats, tf)
        per, pqa = evaluate(name, df2, dss, out, chosen)
        cur = efficiency_curve(name, df2, pred_dB)
        inner = pd.DataFrame([dict(config=name, dataset=h, chosen=k, **{f"inner_{kk}": v for kk, v in s.items()})
                              for h, (k, s) in chosen.items()])
        parts = dict(perdataset=per, tests=tests(name, per, pqa), efficiency_curve=cur, efficiency_operating=eff,
                     inner_selection=inner)
        pd.to_pickle(parts, os.path.join(PARTS, f"{name}.pkl"))
        fig_bars(name, per)
        if tier == "A":
            fig_eff(name, cur, eff, per)
        mt = macro_table(per)
        print(mt[["variant", "always_B", "always_D", "always_H", "best_fixed_train", "oracle_dataset", "oracle_query",
                  "router", "HR_macro", "pct_B", "pct_D", "pct_H"]].round(4).to_string(), flush=True)


def collect():
    """Merge every per-config part into the final CSVs (stale parts from smaller dataset sets are kept; the config
    name carries the number of datasets k)."""
    import glob
    parts = [pd.read_pickle(f) for f in sorted(glob.glob(os.path.join(PARTS, "*.pkl")))]
    if not parts:
        return
    fname = dict(perdataset="w1_router_perdataset.csv", tests="w1_router_tests.csv",
                 efficiency_curve="w1_efficiency_curve.csv", efficiency_operating="w1_efficiency_operating.csv",
                 inner_selection="w1_inner_selection.csv")
    for k in OUTS:
        pd.concat([p[k] for p in parts]).to_csv(os.path.join(c2.RES2, fname[k]), index=False)
    per = pd.concat([p["perdataset"] for p in parts])
    macro_table(per).to_csv(os.path.join(c2.RES2, "w1_router_macro.csv"), index=False)


if __name__ == "__main__":
    if sys.argv[1:] == ["collect"]:
        collect()
    elif sys.argv[1:] == ["list"]:
        print(" ".join(c[0] for c in configs()))
    else:
        main(sys.argv[1:] or None)
        collect()
