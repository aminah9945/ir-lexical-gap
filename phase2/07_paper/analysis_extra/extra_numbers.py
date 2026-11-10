"""Extra numbers requested by the 08 critique (MUST-2,3,4,5,6,7,9,10; SHOULD-3,8,11; COULD-7).
All computed from existing per-query result files; nothing is tuned. Everything here is EXPLORATORY / post hoc
(added after the review) except the micro average, which was pre-declared as secondary (EXPERIMENT_LOG sec. 0).
Bootstrap: stratified percentile bootstrap of the macro difference (resample queries within each dataset),
10,000 resamples; sign-flip test: common2.perm_test_macro (10,000). Seed 42.
Output: analysis_extra/extra_numbers.json and printed tables."""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.dirname(P2)
sys.path.insert(0, os.path.join(P2, "03_experiments"))
sys.path.insert(0, os.path.join(P2, "04_improvements"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import common2 as c2  # noqa: E402

A3 = os.path.join(P2, "03_experiments", "results")
A4 = os.path.join(P2, "04_improvements", "results")
P1 = os.path.join(ROOT, "results")
SMALL = ["webis-touche2020", "trec-covid"]
B = 10000
out = {}


def macro_boot(diffs, n=B, seed=42):
    rng = np.random.default_rng(seed)
    ds = [np.asarray(v, float) for v in diffs]
    obs = float(np.mean([v.mean() for v in ds]))
    bs = np.zeros(n)
    for v in ds:
        for s0 in range(0, n, 1000):
            m = min(1000, n - s0)
            bs[s0:s0 + m] += v[rng.integers(0, len(v), (m, len(v)))].mean(1)
    bs /= len(ds)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    _, p2, p1 = c2.perm_test_macro(ds)
    return dict(delta=obs, lo=float(lo), hi=float(hi), p_one=float(p1), p_two=float(p2), k=len(ds))


def by_ds(df, a, b, excl=()):
    d = df[~df.dataset.isin(excl)]
    return [g[a].values - g[b].values for _, g in d.groupby("dataset", sort=False)]


# ---- MUST-2/4/7: router vs always-H CIs, sensitivity, micro
rt = {}
for cfg in ["A_bge_std_k8", "A_minilm_std_k8", "B_minilm_std_k5", "A_minilm_bm25lodo_std_k5", "A_bge_pct_k8",
            "A_minilm_pct_k8", "B_minilm_pct_k5"]:
    d = pd.read_csv(os.path.join(A3, "w1_router_perquery", f"{cfg}.csv.gz"))
    for var in ["selected", "ridge", "hgb", "selected_argmax"]:
        col = f"router_{var}"
        r = {"all": macro_boot(by_ds(d, col, "H"))}
        if d.dataset.isin(SMALL).any():
            r["excl_touche"] = macro_boot(by_ds(d, col, "H", ["webis-touche2020"]))
            r["excl_touche_covid"] = macro_boot(by_ds(d, col, "H", SMALL))
        r["micro_router"] = float(d[col].mean())
        r["micro_H"] = float(d["H"].mean())
        rt[f"{cfg}/{var}"] = r
out["router"] = rt
rm = pd.read_csv(os.path.join(A3, "w1_router_macro.csv"))
out["logreg_pctD"] = rm[rm.variant == "logreg"].set_index("config").pct_D.round(3).to_dict()

# ---- P1 (E1 ridge) CI, P3 CE(H) CI, CE(BM25) CI
e1 = pd.read_csv(os.path.join(A4, "e1_perquery_minilm.csv.gz"))
out["P1_e1_ridge"] = macro_boot(by_ds(e1, "e1_ridge", "H"))
ce = pd.read_csv(os.path.join(A4, "e3_ce_perquery.csv.gz"))
out["P3_ce_hybrid"] = macro_boot(by_ds(ce[ce.first_stage == "hybrid_cc_minilm"], "ce", "base"))
cb = ce[ce.first_stage == "bm25"]
out["ce_bm25_all"] = macro_boot(by_ds(cb, "ce", "base"))
out["ce_bm25_excl_touche_covid"] = macro_boot(by_ds(cb, "ce", "base", SMALL))

# ---- P2 (E2) CI, reconstructing held-out queries exactly as ce_eval.e2()
from ce_score import sample_qids  # noqa: E402
from pooling_bias import read_qrels  # noqa: E402
e2p = pd.read_csv(os.path.join(A4, "e2_alpha_perdataset.csv"))
diffs, chk = [], {}
for _, r in e2p.iterrows():
    g = pd.read_csv(os.path.join(A4, f"alpha_grid_minilm_{r.dataset}.csv.gz"), dtype={"qid": str}).pivot(
        index="qid", columns="alpha", values="ndcg@10")
    g.columns = [round(c, 1) for c in g.columns]
    S = set(sample_qids(read_qrels(r.dataset)))
    rest = [q for q in g.index if q not in S]
    diffs.append(g.loc[rest, r.alpha_hat].values - g.loc[rest, 0.5].values)
    chk[r.dataset] = (len(rest), int(r.n_rest))
out["P2_e2_rest"] = macro_boot(diffs)
out["P2_nrest_check"] = chk

# ---- P4 in-domain: router - H, router - fixed (CI), sensitivity
ind = pd.read_csv(os.path.join(A4, "e4_indomain_perquery.csv.gz"))
ib = ind[ind.config == "A_bge_std_k8"]
out["P4_router_H"] = macro_boot(by_ds(ib, "router", "H"))
out["P4_router_H_excl_touche_covid"] = macro_boot(by_ds(ib, "router", "H", SMALL))
out["P4_router_fixed"] = macro_boot(by_ds(ib, "router", "fixed"))

# ---- MUST-3 / SHOULD-3: few-shot and split-half sensitivity
fs = pd.read_csv(os.path.join(A4, "e4_fewshot_dataset_selection.csv"))
fsr = {}
for m in ["bge", "minilm"]:
    for k in [5, 10, 20, 30]:
        g = fs[(fs.model == m) & (fs.k == k)]
        fsr[f"{m}/k{k}"] = dict(all8=float(g.gain_mean.mean()),
                                excl_touche=float(g[g.dataset != "webis-touche2020"].gain_mean.mean()),
                                excl_touche_covid=float(g[~g.dataset.isin(SMALL)].gain_mean.mean()),
                                excl_touche_arguana=float(g[~g.dataset.isin(["webis-touche2020", "arguana"])].gain_mean.mean()))
out["fewshot"] = fsr
sh = pd.read_csv(os.path.join(A4, "e4_splithalf.csv"))
out["splithalf_excl_touche"] = {m: float(sh[(sh.model == m) & (sh.dataset != "webis-touche2020")].splithalf_gain_mean.mean())
                                for m in ["bge", "minilm"]}

# ---- MUST-6: per-query oracle noise floor between near-identical lexical systems
nf = {}
for other in ["bm25[stop]", "bm25_wordnet"]:
    gains, mdiff = [], []
    for ds in c2.ALL_DATASETS:
        w = pd.read_csv(os.path.join(P1, f"perquery_lexical_{ds}.csv"), dtype={"qid": str}).pivot(
            index="qid", columns="system", values="ndcg@10")[["bm25", other]].dropna()
        gains.append((w.max(1) - w["bm25"]).mean())
        mdiff.append(w["bm25"].mean() - w[other].mean())
    nf[other] = dict(oracle_gain_macro=float(np.mean(gains)), macro_diff=float(np.mean(mdiff)),
                     per_ds=dict(zip(c2.ALL_DATASETS, np.round(gains, 4).tolist())))
# three-way headroom over BGE-s hybrid, for reference
rmb = rm[(rm.config == "A_bge_std_k8") & (rm.variant == "selected")].iloc[0]
nf["threeway_headroom_bge"] = float(rmb.oracle_query - rmb.always_H)
out["noise_floor"] = nf

# ---- MUST-10: QPP same-5-dataset comparison
q = pd.read_csv(os.path.join(A3, "w1_qpp_correlations.csv"))
out["qpp_cols"] = q.columns.tolist()
json.dump(out, open(os.path.join(HERE, "extra_numbers.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:20000])

# ---- added: remaining "without Touche" sensitivities for the sensitivity table
out["P4_router_H_excl_touche"] = macro_boot(by_ds(ib, "router", "H", ["webis-touche2020"]))
out["ce_bm25_excl_touche"] = macro_boot(by_ds(cb, "ce", "base", ["webis-touche2020"]))
out["splithalf_excl_touche_covid"] = {m: float(sh[(sh.model == m) & (~sh.dataset.isin(SMALL))].splithalf_gain_mean.mean())
                                      for m in ["bge", "minilm"]}
out["splithalf_all"] = {m: float(sh[sh.model == m].splithalf_gain_mean.mean()) for m in ["bge", "minilm"]}
json.dump(out, open(os.path.join(HERE, "extra_numbers.json"), "w"), indent=1, default=float)
for k in ["P4_router_H_excl_touche", "ce_bm25_excl_touche"]:
    x = out[k]
    print(k, f"{x['delta']:+.4f} [{x['lo']:+.4f},{x['hi']:+.4f}] p1={x['p_one']:.4f} k={x['k']}")
print("splithalf", out["splithalf_all"], out["splithalf_excl_touche"], out["splithalf_excl_touche_covid"])
