"""P2 (E2) on held-out queries, rebuilt without qrels: sample_qids() sorts the qrels keys (queries with >=1 positive
judgement), which is exactly the evaluated query set in alpha_grid_*.csv.gz. Checked against e2_tests.csv per-dataset deltas."""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); P2 = os.path.abspath(os.path.join(HERE, "..", "..")); ROOT = os.path.dirname(P2)
sys.path[:0] = [os.path.join(P2, "03_experiments"), os.path.join(ROOT, "src")]
import common2 as c2  # noqa
sys.path.insert(0, HERE)
from extra_numbers_v2_lib import macro_boot  # noqa
A4 = os.path.join(P2, "04_improvements", "results")
e2p = pd.read_csv(os.path.join(A4, "e2_alpha_perdataset.csv")); e2t = pd.read_csv(os.path.join(A4, "e2_tests.csv"))
out = {}
for m in ["minilm", "bge", "bgebase"]:
    diffs, five, chk = [], [], {}
    for _, r in e2p[e2p.model == m].iterrows():
        if r.n_rest == 0: continue
        g = pd.read_csv(os.path.join(A4, f"alpha_grid_{m}_{r.dataset}.csv.gz"), dtype={"qid": str}).pivot(index="qid", columns="alpha", values="ndcg@10")
        g.columns = [round(c, 1) for c in g.columns]
        qs = sorted(g.index); rng = np.random.default_rng(c2.SEED); S = set(rng.choice(qs, 150, replace=False).tolist())
        rest = [q for q in g.index if q not in S]
        d = g.loc[rest, round(r.alpha_hat, 1)].values - g.loc[rest, 0.5].values
        ref = e2t[(e2t.model == m) & (e2t.scope == "rest") & (e2t.dataset == r.dataset)].delta.iloc[0]
        chk[r.dataset] = dict(n=len(rest), n_ref=int(r.n_rest), delta=float(d.mean()), ref=float(ref))
        diffs.append(d)
        if r.dataset in ["nfcorpus", "scifact", "fiqa", "scidocs", "arguana"]: five.append(d)
    out[m] = dict(six=macro_boot(diffs), five=macro_boot(five), check=chk)
json.dump(out, open(os.path.join(HERE, "p2_rest_v2.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
