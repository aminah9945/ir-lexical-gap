import os, json, numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_improvements", "results")
DS = ["nfcorpus", "scifact", "fiqa", "scidocs", "arguana", "trec-covid", "quora", "webis-touche2020"]
A = [round(x / 10, 1) for x in range(11)]
prio = np.array([-abs(a - 0.5) for a in A]) * 1e-9  # tie-break toward 0.5
out = {}
for m in ["minilm", "bge", "bgebase"]:
    mats = {}
    for ds in DS:
        g = pd.read_csv(os.path.join(R, f"alpha_grid_{m}_{ds}.csv.gz"), dtype={"qid": str}).pivot(index="qid", columns="alpha", values="ndcg@10")
        g.columns = [round(c, 1) for c in g.columns]
        mats[ds] = g[A].values
    for k in [10, 20, 30]:
        rng = np.random.default_rng(42)
        G = np.zeros((1000, len(DS))); AH = np.zeros((1000, len(DS)))
        for j, ds in enumerate(DS):
            M = mats[ds]; n = len(M); i05 = A.index(0.5)
            for t in range(1000):
                idx = rng.choice(n, k, replace=False); mask = np.ones(n, bool); mask[idx] = False
                a = int(np.argmax(M[idx].mean(0) + prio))
                G[t, j] = M[mask, a].mean() - M[mask, i05].mean(); AH[t, j] = A[a]
        mac = G.mean(1)
        sub = lambda ex: G[:, [j for j, d in enumerate(DS) if d not in ex]].mean(1)
        out[f"{m}/k{k}"] = dict(
            macro=float(mac.mean()), lo=float(np.percentile(mac, 2.5)), hi=float(np.percentile(mac, 97.5)),
            share_neg=float((mac < 0).mean()),
            excl_touche=float(sub(["webis-touche2020"]).mean()),
            excl_touche_covid=float(sub(["webis-touche2020", "trec-covid"]).mean()),
            per_ds={d: round(float(G[:, j].mean()), 4) for j, d in enumerate(DS)},
            median_alpha={d: float(np.median(AH[:, j])) for j, d in enumerate(DS)})
    print(m, "done", flush=True)
fs = pd.read_csv(os.path.join(R, "e4_fewshot_dataset_selection.csv"))
out["system_choice_ref"] = {f"{m}/k{k}": float(fs[(fs.model == m) & (fs.k == k)].gain_mean.mean()) for m in ["minilm", "bge", "bgebase"] for k in [10, 20, 30]}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w"), indent=1)
for key, v in out.items():
    if key == "system_choice_ref": continue
    print(f"{key:12s} macro {v['macro']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] neg {v['share_neg']:.2f} -T {v['excl_touche']:+.4f} -T-C {v['excl_touche_covid']:+.4f}  sys-choice {out['system_choice_ref'][key]:+.4f}")
    print("     ", v["per_ds"]); print("      alpha", v["median_alpha"])
