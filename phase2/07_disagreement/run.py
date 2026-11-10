"""E7, rules in Drive PREREG_E7_disagreement_labelling_2026-11-05.md (createdTime 2026-11-05T09:15:51Z)."""
import os, json, numpy as np, pandas as pd
P2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R4 = os.path.join(P2, "04_improvements", "results"); Q = os.path.join(P2, "03_experiments", "results", "qpp")
DS = ["nfcorpus", "scifact", "fiqa", "scidocs", "arguana", "trec-covid", "quora", "webis-touche2020"]
A = [round(x / 10, 1) for x in range(11)]; prio = np.array([-abs(a - 0.5) for a in A]) * 1e-9; i05 = A.index(0.5)
out = {}
for m in ["bgebase", "bge", "minilm"]:
    for k in [20, 10]:
        dis, rnd, rnd_draws = {}, {}, []
        G = np.zeros((1000, len(DS)))
        for j, ds in enumerate(DS):
            g = pd.read_csv(os.path.join(R4, f"alpha_grid_{m}_{ds}.csv.gz"), dtype={"qid": str}).pivot(index="qid", columns="alpha", values="ndcg@10")
            g.columns = [round(c, 1) for c in g.columns]; M = g[A]
            f = pd.read_csv(os.path.join(Q, f"tierB_{m}_{ds}.csv"), dtype={"qid": str}).set_index("qid").rbo100
            f = f.reindex(M.index)
            # DEVIATION (logged): 15 NFCorpus queries have no rbo100; they are never chosen for labelling but are evaluated.
            order = sorted(f.dropna().index, key=lambda q: (f[q], q))
            lab = order[:k]; rest = M.index.difference(lab)
            a = int(np.argmax(M.loc[lab].values.mean(0) + prio))
            dis[ds] = dict(gain=float(M.loc[rest].values[:, a].mean() - M.loc[rest].values[:, i05].mean()), alpha=A[a])
            X = M.values; n = len(X); rng = np.random.default_rng(42)
            for t in range(1000):
                idx = rng.choice(n, k, replace=False); mask = np.ones(n, bool); mask[idx] = False
                aa = int(np.argmax(X[idx].mean(0) + prio)); G[t, j] = X[mask, aa].mean() - X[mask, i05].mean()
        dm = np.mean([v["gain"] for v in dis.values()]); rm = G.mean(1)
        noT = [j for j, d in enumerate(DS) if d != "webis-touche2020"]
        dmT = np.mean([dis[DS[j]]["gain"] for j in noT]); rmT = G[:, noT].mean(1)
        r = dict(DIS=float(dm), RAND=float(rm.mean()), diff=float(dm - rm.mean()), DIS_noT=float(dmT), RAND_noT=float(rmT.mean()),
                 diff_noT=float(dmT - rmT.mean()), share_rand_ge_DIS=float((rm >= dm).mean()),
                 per_ds={d: dict(dis=round(dis[d]["gain"], 4), rand=round(float(G[:, j].mean()), 4), alpha_dis=dis[d]["alpha"]) for j, d in enumerate(DS)})
        r["success"] = bool(r["diff"] >= 0.003 and r["diff_noT"] >= 0.003) if (m == "bgebase" and k == 20) else None
        out[f"{m}/k{k}"] = r
        print(f"{m:8s} k={k:2d} DIS {dm:+.4f} RAND {rm.mean():+.4f} diff {dm - rm.mean():+.4f} | noT diff {dmT - rmT.mean():+.4f} | rand>=DIS {r['share_rand_ge_DIS']:.3f} success={r['success']}")
        print("     ", {d: (v['dis'], v['rand'], v['alpha_dis']) for d, v in r["per_ds"].items()})
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w"), indent=1)
