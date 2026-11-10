"""E2 and E3 evaluation from cached cross-encoder logits (see ce_score.py; pre-declared in 04_interpretation.md).

E3: re-rank the top-100 of BM25 (all datasets) and of H_0.5(D) (datasets with D runs) by CE logit; nDCG@10 on the
    scored queries (CPU: sample S_d of <=150 queries; Colab: all queries). Primary P3 = CE(H(D)) - H(D) macro.
E2: per dataset, alpha_hat = argmax over the alpha grid of the mean pseudo-nDCG@10 on the tuning sample S_d
    (gains = sigmoid(CE logit), ideal = the 10 largest gains in the scored pool; docs without a CE score get gain 0
    and are counted). Evaluated with real qrels on the queries NOT in S_d (primary P2) and on all queries.
CE caches are read from cache/ce_<ds>.pkl and from colab_runs/ce/ce_<ds>.pkl (Colab, full scale); the one with more
scored queries wins. The tuning sample S_d is always the seed-42 sample of 150 queries, also for Colab caches.

Usage: python ce_eval.py
Outputs: results/e3_ce_perdataset.csv, results/e3_ce_perquery.csv.gz, results/e3_ce_tests.csv,
         results/e2_alpha_perdataset.csv, results/e2_tests.csv
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "03_experiments"))
import common2 as c2  # noqa: E402
from pooling_bias import read_qrels  # noqa: E402
from ce_score import sample_qids, TOPK  # noqa: E402

RES4 = os.path.join(HERE, "results")
CACHE4 = os.path.join(HERE, "cache")
ALPHAS = [round(a, 1) for a in np.linspace(0, 1, 11)]


def load_ce(ds):
    best = None
    for p in (os.path.join(CACHE4, f"ce_{ds}.pkl"), os.path.join(c2.COLAB, "ce", f"ce_{ds}.pkl")):
        if os.path.exists(p):
            s = pickle.load(open(p, "rb"))
            qs = {q for q, _ in s["scores"]}
            if best is None or len(qs) > len(best[1]):
                best = (s["scores"], qs, p)
    return best


def rerank(lst, sc, q):
    top = [d for d, _ in lst[:TOPK]]
    missing = [d for d in top if (q, d) not in sc]
    assert not missing, f"unscored docs for {q}: {len(missing)}"
    return sorted(top, key=lambda d: -sc[(q, d)])  # stable: ties keep first-stage order


def sig(x):
    return 1 / (1 + np.exp(-x))


def pseudo_ndcg(top10, q, sc, ideal):
    g = np.array([sig(sc[(q, d)]) if (q, d) in sc else 0.0 for d in top10])
    disc = 1 / np.log2(np.arange(2, 2 + len(g)))
    idcg = (ideal[:10] * (1 / np.log2(np.arange(2, 2 + len(ideal[:10]))))).sum()
    return float((g * disc).sum() / idcg) if idcg > 0 else 0.0, int(sum((q, d) not in sc for d in top10))


def e3():
    rows, pqs = [], []
    for ds in c2.ALL_DATASETS:
        got = load_ce(ds)
        if got is None:
            continue
        sc, scored, src = got
        qrels = read_qrels(ds)
        pq = c2.perquery(ds)
        systems = [("bm25", "bm25")] + [(f"hybrid_cc_{m}", m) for m in c2.DENSE_MODELS if c2.has_run(ds, m)]
        for sysname, m in systems:
            run = c2.load_run(ds, "bm25") if sysname == "bm25" else None
            if run is None:
                p = os.path.join(c2.P1_RUNS, f"{ds}__{sysname}.pkl")
                run = c2.load_run(ds, sysname) if os.path.exists(p) else c2.convex(
                    {q: c2.load_run(ds, "bm25").get(q, []) for q in scored}, c2.load_run(ds, m))
            qs = sorted(q for q in scored if q in qrels)
            try:
                rr = {q: rerank(run.get(q, []), sc, q) for q in qs}
            except AssertionError as e:
                print(f"skip {ds}/{sysname}: {e}")
                continue
            ce_nd = {q: c2.common.evaluate_query(rr[q], qrels[q])["ndcg@10"] for q in qs}
            d = pd.DataFrame({"dataset": ds, "first_stage": sysname, "qid": qs,
                              "ce": [ce_nd[q] for q in qs], "base": pq.loc[qs, sysname].values})
            for other in ("bm25", "bge", "hybrid_cc_bge", "minilm", "hybrid_cc_minilm", "bgebase", "hybrid_cc_bgebase"):
                if other in pq.columns:
                    d[f"ref_{other}"] = pq.loc[qs, other].values
            pqs.append(d)
            rows.append(dict(dataset=ds, first_stage=sysname, n=len(qs), source=os.path.relpath(src, HERE),
                             base=d.base.mean(), ce=d.ce.mean(), delta=d.ce.mean() - d.base.mean(),
                             **{c: d[c].mean() for c in d.columns if c.startswith("ref_")}))
    if not rows:
        print("E3: no CE caches yet")
        return
    per = pd.DataFrame(rows)
    pqa = pd.concat(pqs, ignore_index=True)
    per.to_csv(os.path.join(RES4, "e3_ce_perdataset.csv"), index=False)
    pqa.to_csv(os.path.join(RES4, "e3_ce_perquery.csv.gz"), index=False)
    print(per.round(4).to_string())
    tests = []
    comps = [(fs, "ce", "base") for fs in pqa.first_stage.unique()]
    comps += [("bm25", "ce", "ref_hybrid_cc_bge"), ("bm25", "ce", "ref_bge")]
    for fs in pqa.first_stage.unique():
        if fs.startswith("hybrid_cc_"):
            comps.append((fs, "ce", "ref_hybrid_cc_bge"))
    for fs, a, b in comps:
        g0 = pqa[pqa.first_stage == fs]
        if b not in g0.columns:
            continue
        g0 = g0.dropna(subset=[b])
        if g0.empty:
            continue
        sub = [dict(first_stage=fs, comparison=f"CE({fs})-{b}", dataset=ds,
                    **c2.paired_summary(g[a].values, g[b].values)) for ds, g in g0.groupby("dataset", sort=False)]
        s = pd.DataFrame(sub)
        s["p_holm"] = c2.holm(s.p.values)
        mac, p2, p1 = c2.perm_test_macro([g[a].values - g[b].values for _, g in g0.groupby("dataset", sort=False)])
        meta = c2.dl_meta(s.delta.values, s.se_boot.values)
        tests += s.to_dict("records")
        tests.append(dict(first_stage=fs, comparison=f"CE({fs})-{b}", dataset="MACRO", n=len(s), delta=mac, p=p2,
                          p_one_sided=p1, sig_pos_holm=int(((s.p_holm < .05) & (s.delta > 0)).sum()),
                          sig_neg_holm=int(((s.p_holm < .05) & (s.delta < 0)).sum())))
        tests.append(dict(first_stage=fs, comparison=f"CE({fs})-{b}", dataset="POOLED_DL", n=len(s), delta=meta["est"],
                          ci_lo=meta["lo"], ci_hi=meta["hi"], p=meta["p"], I2=meta["I2"]))
    t = pd.DataFrame(tests)
    t.to_csv(os.path.join(RES4, "e3_ce_tests.csv"), index=False)
    print(t[t.dataset.isin(["MACRO", "POOLED_DL"])].round(4).to_string())


def e2():
    rows, diffs = [], {}
    for m in c2.DENSE_MODELS:
        for ds in c2.ALL_DATASETS:
            fg = os.path.join(RES4, f"alpha_grid_{m}_{ds}.csv.gz")
            ft = os.path.join(CACHE4, f"fused_top10_{m}_{ds}.pkl")
            got = load_ce(ds)
            if not (os.path.exists(fg) and os.path.exists(ft) and got):
                continue
            sc, scored, _ = got
            qrels = read_qrels(ds)
            S = [q for q in sample_qids(qrels) if q in scored]
            t10 = pickle.load(open(ft, "rb"))
            pool = {}
            for (q, d), v in sc.items():
                pool.setdefault(q, []).append(sig(v))
            ideal = {q: np.sort(np.array(v))[::-1] for q, v in pool.items()}
            pn, miss = {}, 0
            for a in ALPHAS:
                vals = []
                for q in S:
                    v, mi = pseudo_ndcg(t10[a][q], q, sc, ideal[q])
                    vals.append(v)
                    miss += mi
                pn[a] = float(np.mean(vals))
            best = max(pn.values())
            cands = [a for a in ALPHAS if pn[a] >= best - 1e-12]
            ahat = min(cands, key=lambda a: abs(a - 0.5))
            g = pd.read_csv(fg, dtype={"qid": str}).pivot(index="qid", columns="alpha", values="ndcg@10")
            g.columns = [round(c, 1) for c in g.columns]
            rest = [q for q in g.index if q not in set(S)]
            orc = max(ALPHAS, key=lambda a: g[a].mean())
            rows.append(dict(model=m, dataset=ds, n_tune=len(S), n_rest=len(rest), alpha_hat=ahat,
                             alpha_oracle_all=orc, missing_ce_in_fused_top10=miss,
                             H05_rest=g.loc[rest, 0.5].mean(), Hhat_rest=g.loc[rest, ahat].mean(),
                             H05_all=g[0.5].mean(), Hhat_all=g[ahat].mean(), Horacle_all=g[orc].mean(),
                             corr_pseudo_true_over_alpha=float(np.corrcoef([pn[a] for a in ALPHAS],
                                                                           [g.loc[S, a].mean() for a in ALPHAS])[0, 1]),
                             **{f"pseudo_{a}": pn[a] for a in ALPHAS}))
            diffs[(m, ds)] = (g.loc[rest, ahat].values - g.loc[rest, 0.5].values,
                              g[ahat].values - g[0.5].values)
            print(rows[-1]["model"], ds, "alpha_hat", ahat, "oracle", orc, flush=True)
    if not rows:
        print("E2: nothing to evaluate yet")
        return
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(RES4, "e2_alpha_perdataset.csv"), index=False)
    print(per[["model", "dataset", "alpha_hat", "alpha_oracle_all", "H05_rest", "Hhat_rest", "H05_all", "Hhat_all",
               "Horacle_all", "corr_pseudo_true_over_alpha", "missing_ce_in_fused_top10"]].round(4).to_string())
    tests = []
    for m in per.model.unique():
        for j, scope in enumerate(("rest", "all")):
            keys = [k for k in diffs if k[0] == m]
            sub = [dict(model=m, scope=scope, dataset=k[1], **c2.paired_summary(diffs[k][j], np.zeros_like(diffs[k][j])))
                   for k in keys]
            s = pd.DataFrame(sub)
            s["p_holm"] = c2.holm(s.p.values)
            mac, p2, p1 = c2.perm_test_macro([diffs[k][j] for k in keys])
            tests += s.to_dict("records")
            tests.append(dict(model=m, scope=scope, dataset="MACRO", delta=mac, p=p2, p_one_sided=p1, n=len(keys)))
    t = pd.DataFrame(tests)
    t.to_csv(os.path.join(RES4, "e2_tests.csv"), index=False)
    print(t[t.dataset == "MACRO"].round(4).to_string())


if __name__ == "__main__":
    what = sys.argv[1:] or ["e3", "e2"]
    if "e3" in what:
        e3()
    if "e2" in what:
        e2()
