"""E4 (pre-declared, 04_interpretation.md): why does query-time routing fail?

(a) In-domain ceiling: the same features, model families and decision rule as Agent 3's pre-declared "selected"
    router (03_experiments/router.py), but trained IN-DOMAIN with 5-fold CV inside each dataset (KFold, shuffle,
    seed 42); Ridge vs HGB chosen by an inner 4-fold CV on the training folds. Compared with always-H (P4, primary:
    config A_bge_std) and with the in-domain best-fixed system chosen on the training folds (added before running,
    after E1: separates a learnable dataset prior from query-level signal).
(b) Realizable dataset-level headroom: best fixed system chosen on a random 50% of a dataset's queries, evaluated on
    the other 50% (1,000 splits); expected gain over always-H.
(c) Concentration of the query-oracle headroom: share of sum(max(B,D,H) - H) contributed by queries whose oracle
    gain is <= 0.05, (0.05, 0.2], > 0.2.
(d) Exploratory label-consistency proxy: agreement of sign(B - H) under nDCG@10 vs recall@100 and MRR@10 (only
    queries where both differences exceed 0.01; post-hoc fix) and headroom share from queries with <= 2 relevant docs.
(e) Exploratory/post-hoc: few-shot dataset-level system selection with k labelled target queries (k=5..100).

Usage: python e4_why_routing_fails.py
Outputs: results/e4_indomain_perdataset.csv, results/e4_indomain_tests.csv, results/e4_splithalf.csv,
         results/e4_headroom_concentration.csv, results/e4_label_consistency.csv,
         results/e4_fewshot_dataset_selection.csv
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "03_experiments"))
import common2 as c2  # noqa: E402
import router as R  # noqa: E402  (Agent 3's router: assemble, fit_predict_reg, choose, MAKERS)
from pooling_bias import read_qrels  # noqa: E402

RES4 = os.path.join(HERE, "results")
CONFIGS = [("A_bge_std", "bge", "A"), ("A_minilm_std", "minilm", "A"), ("B_minilm_std", "minilm", "B")]
CONFIGS += [(f"A_bgebase_std", "bgebase", "A"), ("B_bge_std", "bge", "B"), ("B_bgebase_std", "bgebase", "B")]


def cv_route(g, feats, seed=c2.SEED):
    """In-domain 5-fold CV router for one dataset frame g -> (chosen idx, best-fixed-on-train idx, kinds)."""
    n = len(g)
    choice, bft, kinds = np.zeros(n, int), np.zeros(n, int), []
    for tr_i, te_i in KFold(5, shuffle=True, random_state=seed).split(np.arange(n)):
        tr, te = g.iloc[tr_i], g.iloc[te_i]
        bft[te_i] = int(np.argmax(tr[R.SYS].mean().values))
        # inner 4-fold: ridge vs hgb by mean routed nDCG
        score = {}
        for kind in R.MAKERS:
            v = np.zeros(len(tr))
            for itr, ite in KFold(4, shuffle=True, random_state=seed).split(np.arange(len(tr))):
                a, b = tr.iloc[itr], tr.iloc[ite]
                pb, pd_ = R.fit_predict_reg(kind, a, b, feats)
                v[ite] = R.routed_value(b, R.choose(pb, pd_))
            score[kind] = v.mean()
        kind = max(score, key=score.get)
        kinds.append(kind)
        pb, pd_ = R.fit_predict_reg(kind, tr, te, feats)
        choice[te_i] = R.choose(pb, pd_)
    return choice, bft, kinds


def indomain():
    per, pqs = [], []
    for name, m, tier in CONFIGS:
        df, feats = R.assemble(m, tier)
        if df is None:
            continue
        name = f"{name}_k{df.dataset.nunique()}"
        for ds, g in df.groupby("dataset", sort=False):
            g = g.reset_index(drop=True)
            ch, bft, kinds = cv_route(g, feats)
            Y = g[R.SYS].values
            val = Y[np.arange(len(g)), ch]
            fixed = Y[np.arange(len(g)), bft]
            mix = np.bincount(ch, minlength=3) / len(ch)
            per.append(dict(config=name, dataset=ds, n=len(g), always_B=Y[:, 0].mean(), always_D=Y[:, 1].mean(),
                            always_H=Y[:, 2].mean(), indomain_best_fixed=fixed.mean(), indomain_router=val.mean(),
                            oracle_dataset=Y.mean(0).max(), oracle_query=Y.max(1).mean(),
                            pct_B=mix[0], pct_D=mix[1], pct_H=mix[2], kinds="/".join(kinds)))
            pqs.append(pd.DataFrame(dict(config=name, dataset=ds, qid=g.qid, H=Y[:, 2], router=val, fixed=fixed)))
            print(per[-1], flush=True)
    per = pd.DataFrame(per)
    pq = pd.concat(pqs, ignore_index=True)
    per.to_csv(os.path.join(RES4, "e4_indomain_perdataset.csv"), index=False)
    pq.to_csv(os.path.join(RES4, "e4_indomain_perquery.csv.gz"), index=False)
    rows = []
    for cfg, gc in pq.groupby("config", sort=False):
        for a, b in (("router", "H"), ("router", "fixed"), ("fixed", "H")):
            sub = []
            for ds, g in gc.groupby("dataset", sort=False):
                sub.append(dict(config=cfg, comparison=f"{a}-{b}", dataset=ds,
                                **c2.paired_summary(g[a].values, g[b].values)))
            s = pd.DataFrame(sub)
            s["p_holm"] = c2.holm(s.p.values)
            mac, p2, p1 = c2.perm_test_macro([g[a].values - g[b].values for _, g in gc.groupby("dataset", sort=False)])
            meta = c2.dl_meta(s.delta.values, s.se_boot.values)
            rows += s.to_dict("records")
            rows.append(dict(config=cfg, comparison=f"{a}-{b}", dataset="MACRO", delta=mac, p=p2, p_one_sided=p1,
                             sig_pos_holm=int(((s.p_holm < .05) & (s.delta > 0)).sum()),
                             sig_neg_holm=int(((s.p_holm < .05) & (s.delta < 0)).sum())))
            rows.append(dict(config=cfg, comparison=f"{a}-{b}", dataset="POOLED_DL", delta=meta["est"],
                             ci_lo=meta["lo"], ci_hi=meta["hi"], p=meta["p"]))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES4, "e4_indomain_tests.csv"), index=False)
    print(per.groupby("config")[["always_H", "indomain_best_fixed", "indomain_router", "oracle_dataset",
                                 "oracle_query"]].mean().round(4).to_string())
    print(t[t.dataset == "MACRO"].round(4).to_string())


def labels(m):
    out = {}
    for ds in c2.ALL_DATASETS:
        cols = {"bm25", m, f"hybrid_cc_{m}"}
        ws = {k: c2.perquery(ds, k) for k in ("ndcg@10", "recall@100", "mrr@10")}
        if not cols <= set(ws["ndcg@10"].columns):
            continue
        out[ds] = {k: w[["bm25", m, f"hybrid_cc_{m}"]].dropna().rename(
            columns={"bm25": "B", m: "D", f"hybrid_cc_{m}": "H"}) for k, w in ws.items()}
    return out


def descriptive():
    rng = np.random.default_rng(c2.SEED)
    sh, conc, cons, few = [], [], [], []
    for m in c2.DENSE_MODELS:
        L = labels(m)
        for ds, w in L.items():
            Y = w["ndcg@10"].values
            n = len(Y)
            gains = []
            for _ in range(1000):
                p = rng.permutation(n)
                a, b = p[: n // 2], p[n // 2:]
                j = int(np.argmax(Y[a].mean(0)))
                gains.append(Y[b, j].mean() - Y[b, 2].mean())
            sh.append(dict(model=m, dataset=ds, n=n, oracle_ds_gain_insample=Y.mean(0).max() - Y[:, 2].mean(),
                           splithalf_gain_mean=np.mean(gains), splithalf_gain_lo=np.percentile(gains, 2.5),
                           splithalf_gain_hi=np.percentile(gains, 97.5),
                           ))
            og = Y.max(1) - Y[:, 2]
            tot = og.sum()
            conc.append(dict(model=m, dataset=ds, n=n, headroom=og.mean(),
                             share_queries_gain_gt0=np.mean(og > 0),
                             share_from_le_005=og[og <= 0.05].sum() / tot if tot else np.nan,
                             share_from_005_02=og[(og > 0.05) & (og <= 0.2)].sum() / tot if tot else np.nan,
                             share_from_gt_02=og[og > 0.2].sum() / tot if tot else np.nan,
                             share_queries_gt_02=np.mean(og > 0.2)))
            # (d) post-hoc fix: agreement only where BOTH metrics show |dB| > 0.01 (sign(0) counted as disagreement before)
            dB = {k: (v["B"] - v["H"]).values for k, v in w.items()}
            nz = np.abs(dB["ndcg@10"]) > 0.01
            r = dict(model=m, dataset=ds, n=n, n_nontie=int(nz.sum()), share_ties_ndcg=1 - nz.mean())
            for k in ("recall@100", "mrr@10"):
                both = nz & (np.abs(dB[k]) > 0.01)
                r[f"n_both_{k}"] = int(both.sum())
                r[f"sign_agree_{k}"] = float(np.mean(np.sign(dB["ndcg@10"][both]) == np.sign(dB[k][both]))) if both.any() else np.nan
            # n_rel (explanatory only): headroom share from queries with <= 2 relevant docs
            qr = read_qrels(ds)
            nrel = np.array([sum(1 for x in qr[q].values() if x > 0) for q in w["ndcg@10"].index])
            r["median_nrel"] = float(np.median(nrel))
            r["share_q_nrel_le2"] = float(np.mean(nrel <= 2))
            r["headroom_share_nrel_le2"] = float(og[nrel <= 2].sum() / tot) if tot else np.nan
            cons.append(r)
            # (e) exploratory, post-hoc: few-shot dataset-level selection with k labelled target queries
            for k in (5, 10, 20, 30, 50, 100):
                if k >= n:
                    continue
                gk = []
                for _ in range(1000):
                    p = rng.permutation(n)
                    a, b = p[:k], p[k:]
                    j = int(np.argmax(Y[a].mean(0)))
                    gk.append(Y[b, j].mean() - Y[b, 2].mean())
                few.append(dict(model=m, dataset=ds, k=k, gain_mean=np.mean(gk), gain_lo=np.percentile(gk, 2.5),
                                gain_hi=np.percentile(gk, 97.5), p_loss=np.mean(np.array(gk) < 0)))
    for name, rows in (("e4_splithalf", sh), ("e4_headroom_concentration", conc), ("e4_label_consistency", cons), ("e4_fewshot_dataset_selection", few)):
        d = pd.DataFrame(rows)
        d.to_csv(os.path.join(RES4, f"{name}.csv"), index=False)
        print(name)
        print(d.round(3).to_string())


if __name__ == "__main__":
    what = sys.argv[1:] or ["indomain", "descriptive"]
    if "descriptive" in what:
        descriptive()
    if "indomain" in what:
        indomain()
