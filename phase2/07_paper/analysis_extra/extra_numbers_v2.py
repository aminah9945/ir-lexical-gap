"""extra_numbers_v2: recompute the paper's bootstrap CIs and sensitivities from the CURRENT result files
(after the 2026-10-22 full Colab ingest: MiniLM, BGE-s, BGE-b on all 8 datasets; Tier B for all three).
Same procedures as extra_numbers.py (stratified percentile bootstrap, 10^4, seed 42; common2.perm_test_macro).
P2 is not recomputed here (needs qrels); its 5-collection value is read from e2_tests.csv per-dataset rows.
Output: analysis_extra/extra_numbers_v2.json. Everything here is post hoc / exploratory except as labelled in the paper."""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.dirname(P2)
sys.path.insert(0, os.path.join(P2, "03_experiments")); sys.path.insert(0, os.path.join(ROOT, "src"))
import common2 as c2  # noqa

A3 = os.path.join(P2, "03_experiments", "results"); A4 = os.path.join(P2, "04_improvements", "results")
SMALL = ["webis-touche2020", "trec-covid"]; FIVE = ["nfcorpus", "scifact", "fiqa", "scidocs", "arguana"]
B = 10000; out = {}

def macro_boot(diffs, n=B, seed=42):
    rng = np.random.default_rng(seed)
    ds = [np.asarray(v, float) for v in diffs]
    obs = float(np.mean([v.mean() for v in ds])); bs = np.zeros(n)
    for v in ds:
        for s0 in range(0, n, 1000):
            m = min(1000, n - s0); bs[s0:s0 + m] += v[rng.integers(0, len(v), (m, len(v)))].mean(1)
    bs /= len(ds); lo, hi = np.percentile(bs, [2.5, 97.5]); _, p2, p1 = c2.perm_test_macro(ds)
    return dict(delta=obs, lo=float(lo), hi=float(hi), p_one=float(p1), p_two=float(p2), k=len(ds))

def by_ds(df, a, b, excl=(), only=None):
    d = df[~df.dataset.isin(excl)]
    if only is not None: d = d[d.dataset.isin(only)]
    return [g[a].values - g[b].values for _, g in d.groupby("dataset", sort=False)]

def sens(d, a, b):
    r = {"all": macro_boot(by_ds(d, a, b))}
    if d.dataset.isin(SMALL).any():
        r["excl_touche"] = macro_boot(by_ds(d, a, b, ["webis-touche2020"]))
        r["excl_touche_covid"] = macro_boot(by_ds(d, a, b, SMALL))
    return r

rt = {}
JOBS = [(c, "selected") for c in ["A_bge_std_k8", "A_minilm_std_k8", "A_bgebase_std_k8", "B_bge_std_k8", "B_minilm_std_k8",
                                  "B_bgebase_std_k8", "A_minilm_bm25lodo_std_k5", "A_bgebase_bm25lodo_std_k8"]]
JOBS += [("A_bgebase_bm25lodo_std_k8", "ridge"), ("A_bge_std_k8", "ridge"), ("A_bgebase_std_k8", "ridge")]
for cfg, var in JOBS:
    d = pd.read_csv(os.path.join(A3, "w1_router_perquery", f"{cfg}.csv.gz")); col = f"router_{var}"
    r = sens(d, col, "H"); r["micro_router"] = float(d[col].mean()); r["micro_H"] = float(d["H"].mean())
    rt[f"{cfg}/{var}"] = r; print("router", cfg, var, flush=True)
out["router"] = rt

for m in ["minilm", "bge", "bgebase"]:
    e1 = pd.read_csv(os.path.join(A4, f"e1_perquery_{m}.csv.gz"))
    out[f"P1_e1_ridge_{m}"] = sens(e1, "e1_ridge", "H")
    out[f"P1_e1_ridge_{m}_five"] = macro_boot(by_ds(e1, "e1_ridge", "H", only=FIVE))
    out[f"P1_e1_hgb_{m}"] = macro_boot(by_ds(e1, "e1_hgb", "H"))
    out[f"P1_e1_global_{m}"] = macro_boot(by_ds(e1, "e1_global", "H"))
    out[f"P1_e1_n_ds_{m}"] = int(e1.dataset.nunique())

e2 = pd.read_csv(os.path.join(A4, "e2_tests.csv"))
p2 = {}
for m in ["minilm", "bge", "bgebase"]:
    x = e2[(e2.model == m) & (e2.scope == "rest") & (~e2.dataset.isin(["MACRO"])) & (e2.n > 0)]
    p2[m] = dict(macro_five=float(x[x.dataset.isin(FIVE)].delta.mean()), macro_all_with_rest=float(x.delta.mean()),
                 n_ds_with_rest=int(len(x)), per_ds=dict(zip(x.dataset, x.delta.round(4))))
out["P2_from_e2_tests"] = p2

ind = pd.read_csv(os.path.join(A4, "e4_indomain_perquery.csv.gz"))
for cfg in ind.config.unique():
    ib = ind[ind.config == cfg]
    out[f"P4_{cfg}_router_H"] = sens(ib, "router", "H")
    out[f"P4_{cfg}_router_fixed"] = macro_boot(by_ds(ib, "router", "fixed"))

fs = pd.read_csv(os.path.join(A4, "e4_fewshot_dataset_selection.csv")); fsr = {}
for m in fs.model.unique():
    for k in [5, 10, 20, 30]:
        g = fs[(fs.model == m) & (fs.k == k)]
        fsr[f"{m}/k{k}"] = dict(all8=float(g.gain_mean.mean()), excl_touche=float(g[g.dataset != "webis-touche2020"].gain_mean.mean()),
                                excl_touche_covid=float(g[~g.dataset.isin(SMALL)].gain_mean.mean()))
out["fewshot"] = fsr
sh = pd.read_csv(os.path.join(A4, "e4_splithalf.csv"))
out["splithalf"] = {m: dict(all8=float(sh[sh.model == m].splithalf_gain_mean.mean()),
                            insample=float(sh[sh.model == m].oracle_ds_gain_insample.mean()),
                            excl_touche=float(sh[(sh.model == m) & (sh.dataset != "webis-touche2020")].splithalf_gain_mean.mean()),
                            excl_touche_covid=float(sh[(sh.model == m) & (~sh.dataset.isin(SMALL))].splithalf_gain_mean.mean()))
                    for m in sh.model.unique()}
hc = pd.read_csv(os.path.join(A4, "e4_headroom_concentration.csv"))
out["headroom_conc"] = {m: dict(share_q_gt02=[float(g.share_queries_gt_02.min()), float(g.share_queries_gt_02.max())],
                                share_head_gt02=[float(g.share_from_gt_02.min()), float(g.share_from_gt_02.max())])
                        for m, g in hc.groupby("model")}
lc = pd.read_csv(os.path.join(A4, "e4_label_consistency.csv"))
out["headroom_nrel_le2"] = {m: g[g.dataset.isin(["arguana", "scifact", "quora"])].set_index("dataset").headroom_share_nrel_le2.round(4).to_dict()
                            for m, g in lc.groupby("model")}
json.dump(out, open(os.path.join(HERE, "extra_numbers_v2.json"), "w"), indent=1, default=float)
print("done")
