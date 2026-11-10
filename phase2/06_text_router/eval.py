"""Evaluate E5 per PREREGISTRATION.md. Writes results.json."""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "..", "07_paper", "analysis_extra")]
from extra_numbers_v2_lib import macro_boot  # same bootstrap + sign-flip as the paper
from scipy.stats import kendalltau
DS = ["nfcorpus", "scifact", "fiqa", "scidocs", "arguana", "trec-covid", "quora", "webis-touche2020"]
SMALL = ["webis-touche2020", "trec-covid"]
def route(pB, pD):
    v = np.stack([np.zeros_like(pB), pB, pD], 1)  # H, B, D gains vs H
    mx = v.max(1, keepdims=True); ok = v >= mx - 0.005
    # cheapest first: B < D < H
    return np.where(ok[:, 1], "B", np.where(ok[:, 2], "D", "H"))
out = {}
for m in ["bgebase", "bge", "minilm"]:
    fs = [os.path.join(HERE, "preds", f"{m}__{d}.csv") for d in DS]
    if not all(os.path.exists(f) for f in fs): continue
    d = pd.concat([pd.read_csv(f, dtype={"qid": str}) for f in fs])
    c = route(d.pB.values, d.pD.values); d["choice"] = c
    d["R"] = np.select([c == "B", c == "D"], [d.B, d.D], d.H)
    by = lambda ex=(): [g.R.values - g.H.values for _, g in d[~d.dataset.isin(ex)].groupby("dataset", sort=False)]
    r = dict(all=macro_boot(by()), excl_touche=macro_boot(by(["webis-touche2020"])), excl_touche_covid=macro_boot(by(SMALL)))
    r["macro_router"] = float(d.groupby("dataset").R.mean().mean()); r["macro_H"] = float(d.groupby("dataset").H.mean().mean())
    r["mix"] = {k: float((c == k).mean()) for k in "BDH"}
    r["per_ds"] = {ds: dict(delta=float((g.R - g.H).mean()), mix={k: float((g.choice == k).mean()) for k in "BDH"},
                            tau_B=float(kendalltau(g.pB, g.dB).statistic), tau_D=float(kendalltau(g.pD, g.dD).statistic)) for ds, g in d.groupby("dataset")}
    r["tau_B_mean"] = float(np.mean([v["tau_B"] for v in r["per_ds"].values()])); r["tau_D_mean"] = float(np.mean([v["tau_D"] for v in r["per_ds"].values()]))
    a = r["all"]; r["success"] = bool(a["delta"] >= 0.005 and a["p_one"] < 0.05)
    out[m] = r
    print(m, f"router {r['macro_router']:.4f} H {r['macro_H']:.4f} d {a['delta']:+.4f} [{a['lo']:+.4f},{a['hi']:+.4f}] p1 {a['p_one']:.4f} success={r['success']}",
          f"-T {r['excl_touche']['delta']:+.4f} -T-C {r['excl_touche_covid']['delta']:+.4f} tauB {r['tau_B_mean']:.3f} tauD {r['tau_D_mean']:.3f} mix", {k: round(v, 3) for k, v in r['mix'].items()})
    for ds, v in r["per_ds"].items(): print("   ", ds, round(v["delta"], 4), {k: round(x, 2) for k, x in v["mix"].items()}, round(v["tau_B"], 3))
json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
