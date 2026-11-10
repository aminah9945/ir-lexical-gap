"""E1 (pre-declared, 04_interpretation.md): per-query continuous fusion weight alpha_q predicted from query-time
features, leave-one-dataset-out.

For each grid alpha g != 0.5 a Ridge(alpha=1) regresses nDCG(g) - nDCG(0.5) on Tier A + Tier B features
(standard scaling and median imputation fitted on training datasets; each training dataset has total weight 1).
Choice = argmax over predictions (0 for g = 0.5); every g whose prediction lies within 0.005 of the maximum is a
candidate, and the candidate closest to 0.5 wins (conservative tie-break).
Secondary (reported, not in F4): HGB variant (Agent 3's fixed hyper-parameters), LODO-global alpha (best training
macro), dataset-oracle alpha and per-query oracle alpha (upper bounds, use qrels of the held-out set).

Usage: python e1_alpha_router.py [model]   (default minilm; bge/bgebase once Colab runs exist)
Outputs: results/e1_perdataset_<model>.csv, results/e1_perquery_<model>.csv.gz, results/e1_tests_<model>.csv
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "03_experiments"))
import common2 as c2  # noqa: E402

RES4 = os.path.join(HERE, "results")
QDIR = os.path.join(c2.RES2, "qpp")
ALPHAS = [round(a, 1) for a in np.linspace(0, 1, 11)]
MARGIN = 0.005


def assemble(model):
    frames = []
    for ds in c2.ALL_DATASETS:
        fg = os.path.join(RES4, f"alpha_grid_{model}_{ds}.csv.gz")
        fa, fb = os.path.join(QDIR, f"tierA_{ds}.csv"), os.path.join(QDIR, f"tierB_{model}_{ds}.csv")
        if not all(os.path.exists(f) for f in (fg, fa, fb)):
            continue
        g = pd.read_csv(fg, dtype={"qid": str}).pivot(index="qid", columns="alpha", values="ndcg@10")
        g.columns = [f"y{a:.1f}" for a in g.columns]
        X = pd.read_csv(fa, dtype={"qid": str}).merge(
            pd.read_csv(fb, dtype={"qid": str}).drop(columns="dataset"), on="qid")
        x = g.reset_index().merge(X, on="qid", how="inner")
        assert len(x) == len(g), (ds, len(x), len(g))
        x["dataset"] = ds
        frames.append(x)
    df = pd.concat(frames, ignore_index=True)
    ycols = [f"y{a:.1f}" for a in ALPHAS]
    feats = [c for c in df.columns if c not in {"dataset", "qid", *ycols}]
    c2.assert_no_qrel_features(feats)
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan).astype(float)
    return df, feats, ycols


def w_of(tr):
    return 1.0 / tr.groupby("dataset")["qid"].transform("size").values


def make(kind):
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    return make_pipeline(StandardScaler(), HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50, l2_regularization=1.0,
        random_state=c2.SEED))


def choose(P):
    """P: (n, 11) predicted gains over alpha=0.5 -> chosen grid index (closest to 0.5 within MARGIN of max)."""
    best = P.max(1, keepdims=True)
    ok = P >= best - MARGIN
    dist = np.abs(np.array(ALPHAS) - 0.5)[None, :]
    return np.argmin(np.where(ok, dist, np.inf), axis=1)


def main(model="minilm"):
    df, feats, ycols = assemble(model)
    Y = df[ycols].values
    i5 = ALPHAS.index(0.5)
    dss = list(dict.fromkeys(df.dataset))
    print(f"E1 {model}: {len(dss)} datasets {dss}, {len(df)} queries, {len(feats)} features", flush=True)
    ch = {k: np.zeros(len(df), int) for k in ("ridge", "hgb", "global")}
    rows = []
    for h in dss:
        te_m = df.dataset.values == h
        tr, te = df[~te_m], df[te_m]
        w = w_of(tr)
        for kind in ("ridge", "hgb"):
            P = np.zeros((te_m.sum(), len(ALPHAS)))
            for j, yc in enumerate(ycols):
                if j == i5:
                    continue
                m = make(kind)
                m.fit(tr[feats].values, tr[yc].values - tr[f"y{0.5:.1f}"].values,
                      **{f"{m.steps[-1][0]}__sample_weight": w})
                P[:, j] = m.predict(te[feats].values)
            ch[kind][te_m] = choose(P)
        trmac = tr.groupby("dataset")[ycols].mean().mean().values
        ch["global"][te_m] = int(np.argmax(trmac))
    pq = pd.DataFrame({"dataset": df.dataset, "qid": df.qid, "H": Y[:, i5],
                       "oracle_q": Y.max(1), "alpha_oracle_q": np.array(ALPHAS)[Y.argmax(1)]})
    for k, c in ch.items():
        pq[f"e1_{k}"] = Y[np.arange(len(df)), c]
        pq[f"alpha_{k}"] = np.array(ALPHAS)[c]
    dso = {}
    for h in dss:
        m = df.dataset.values == h
        j = int(np.argmax(Y[m].mean(0)))
        dso[h] = (ALPHAS[j], Y[m][:, j])
    pq["oracle_ds"] = np.concatenate([dso[h][1] for h in dss])
    pq.to_csv(os.path.join(RES4, f"e1_perquery_{model}.csv.gz"), index=False)
    for h, g in pq.groupby("dataset", sort=False):
        rows.append(dict(dataset=h, n=len(g), H=g.H.mean(), e1_ridge=g.e1_ridge.mean(), e1_hgb=g.e1_hgb.mean(),
                         e1_global=g.e1_global.mean(), alpha_global=g.alpha_global.iloc[0],
                         oracle_ds=g.oracle_ds.mean(), alpha_oracle_ds=dso[h][0], oracle_q=g.oracle_q.mean(),
                         mean_alpha_ridge=g.alpha_ridge.mean(), share_alpha_ridge_ne_05=(g.alpha_ridge != 0.5).mean(),
                         mean_alpha_hgb=g.alpha_hgb.mean()))
    per = pd.DataFrame(rows)
    per.loc[len(per)] = dict(dataset="MACRO", n=per.n.sum(), **per.drop(columns=["dataset", "n"]).mean())
    per.to_csv(os.path.join(RES4, f"e1_perdataset_{model}.csv"), index=False)
    print(per.round(4).to_string(), flush=True)
    tests = []
    for v in ("e1_ridge", "e1_hgb", "e1_global"):
        sub = []
        for h, g in pq.groupby("dataset", sort=False):
            sub.append(dict(variant=v, dataset=h, **c2.paired_summary(g[v].values, g.H.values)))
        s = pd.DataFrame(sub)
        s["p_holm"] = c2.holm(s.p.values)
        meta = c2.dl_meta(s.delta.values, s.se_boot.values)
        mac, p2, p1 = c2.perm_test_macro([g[v].values - g.H.values for _, g in pq.groupby("dataset", sort=False)])
        tests += s.to_dict("records")
        tests.append(dict(variant=v, dataset="MACRO", delta=mac, p=p2, p_one_sided=p1))
        tests.append(dict(variant=v, dataset="POOLED_DL", delta=meta["est"], ci_lo=meta["lo"], ci_hi=meta["hi"],
                          p=meta["p"]))
    t = pd.DataFrame(tests)
    t.to_csv(os.path.join(RES4, f"e1_tests_{model}.csv"), index=False)
    print(t[t.dataset.isin(["MACRO", "POOLED_DL"])].round(4).to_string(), flush=True)


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["minilm"]))
