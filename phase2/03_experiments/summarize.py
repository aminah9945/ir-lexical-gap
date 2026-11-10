"""Print compact markdown tables of the headline numbers (input for RESULTS_SUMMARY.md) and write
results/summary_tables.md. Reads only the result CSVs written by the other scripts."""
import os

import numpy as np
import pandas as pd

import common2 as c2

R = c2.RES2
out = []


def P(s=""):
    out.append(s)


def md(df, floatfmt=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.values]
    return "\n".join(lines)


def pfmt(p):
    return "" if pd.isna(p) else ("<1e-4" if p < 1e-4 else f"{p:.4f}")


def sanity():
    t = pd.read_csv(os.path.join(c2.P1_RES, "table_main.csv"))
    rows = []
    for d in c2.ALL_DATASETS:
        w = c2.perquery(d)
        ref = t[(t.dataset == d) & (t.system == "bm25")]["ndcg@10"].values[0]
        rows.append(dict(dataset=d, bm25_perquery_mean=w["bm25"].mean(), table_main=ref,
                         absdiff=abs(w["bm25"].mean() - ref)))
    s = pd.DataFrame(rows)
    P("### Sanity: always-BM25 vs Phase-1 table_main.csv (nDCG@10)")
    P(md(s, 5))
    P(f"max |diff| = {s.absdiff.max():.2e}; macro = {s.bm25_perquery_mean.mean():.4f}")
    P()


def w7():
    f = os.path.join(R, "w7_effect_sizes.csv")
    if not os.path.exists(f):
        return
    t = pd.read_csv(f)
    pr = t[t.dataset.str.startswith("POOLED_DL(")].copy()
    pr["p"] = pr.p.map(pfmt)
    P("### W7 pooled (DerSimonian-Laird) paired differences, nDCG@10")
    P(md(pr[["comparison", "k", "delta", "ci_lo", "ci_hi", "p", "I2", "sign_pos", "sign_neg", "sig_pos_holm",
             "sig_neg_holm", "loo_min", "loo_max"]]))
    ex = t[t.dataset == "POOLED_DL_excl_small"][["comparison", "k", "delta", "ci_lo", "ci_hi"]]
    P()
    P("Sensitivity without TREC-COVID and Touche:")
    P(md(ex))
    P()


def w3():
    f = os.path.join(R, "w3_modelB_meta.csv")
    if not os.path.exists(f):
        return
    P("### W3 gap model (gap = dense - BM25 on z(cov_mean))")
    m = pd.read_csv(f)
    m["p"] = m.p.map(pfmt)
    P(md(m[["model", "k", "est", "lo", "hi", "p", "tau2", "I2", "sign_neg", "sig_neg", "loo_min", "loo_max",
            "excl_small_est", "excl_small_lo", "excl_small_hi"]], 4))
    P()
    x = pd.read_csv(os.path.join(R, "w3_mixedlm.csv"))
    x["p"] = x.p.map(pfmt)
    P(md(x[["model", "kind", "coef", "lo", "hi", "p", "re_var_slope"]], 4))
    P()
    c = pd.read_csv(os.path.join(R, "w3_circularity.csv"))
    P(md(c, 4))
    P()
    f2 = os.path.join(R, "w2_proxy_modelB_meta.csv")
    if os.path.exists(f2):
        P("W2 proxies in place of cov_mean (Model B, DL pooled slope):")
        q = pd.read_csv(f2)
        q["p"] = q.p.map(pfmt)
        P(md(q, 4))
        r = pd.read_csv(os.path.join(R, "w2_proxy_rho.csv"))
        P()
        P("Spearman rho(proxy, cov_mean), per dataset:")
        P(md(r.pivot_table(index=["model", "proxy"], columns="dataset", values="rho").reset_index(), 2))
    P()


def w1():
    f = os.path.join(R, "w1_router_macro.csv")
    if not os.path.exists(f):
        return
    m = pd.read_csv(f)
    t = pd.read_csv(os.path.join(R, "w1_router_tests.csv"))
    mac = t[(t.dataset == "MACRO")].set_index(["config", "variant", "baseline"])
    rows = []
    for _, r in m.iterrows():
        k = (r.config, r.variant)
        rows.append(dict(config=r.config, variant=r.variant, B=r.always_B, D=r.always_D, H=r.always_H,
                         best_fixed_tr=r.best_fixed_train, oracle_ds=r.oracle_dataset, oracle_q=r.oracle_query,
                         router=r.router, minus_H=r.router_minus_H, HR=r.HR_macro,
                         p1_vs_H=pfmt(mac.loc[k + ("H",)].p_one_sided) if k + ("H",) in mac.index else "",
                         p2_vs_H=pfmt(mac.loc[k + ("H",)].p) if k + ("H",) in mac.index else "",
                         pctB=r.pct_B, pctD=r.pct_D, pctH=r.pct_H))
    P("### W1 router, macro nDCG@10 over held-out datasets (LODO)")
    P(md(pd.DataFrame(rows), 4))
    P()
    hol = t[(t.dataset != "MACRO") & (t.dataset != "POOLED_DL") & (t.variant == "selected") & (t.baseline == "H")]
    g = hol.groupby("config").apply(lambda x: pd.Series(dict(
        sig_better=int(((x.delta > 0) & (x.p_holm < .05)).sum()), sig_worse=int(((x.delta < 0) & (x.p_holm < .05)).sum()),
        k=len(x)))).reset_index()
    P("Selected router vs always-H, per dataset (Holm, family F2): # significantly better / worse")
    P(md(g))
    P()
    per = pd.read_csv(os.path.join(R, "w1_router_perdataset.csv"))
    for cfg in per.config.unique():
        p = per[(per.config == cfg) & (per.variant == "selected")]
        P(f"Per dataset, {cfg} (selected):")
        P(md(p[["dataset", "n", "always_B", "always_D", "always_H", "best_fixed_train_sys", "oracle_dataset_sys",
                "oracle_query", "router", "HR", "pct_B", "pct_D", "pct_H", "inner_choice"]], 3))
        P()
    eo = os.path.join(R, "w1_efficiency_operating.csv")
    if os.path.exists(eo):
        e = pd.read_csv(eo)
        g = e.groupby("config")[["share_B", "ndcg_eff", "ndcg_H", "ndcg_B"]].mean().reset_index()
        g["loss_vs_H"] = g.ndcg_H - g.ndcg_eff
        P("Efficiency operating point (B vs H, tau chosen on training folds; macro):")
        P(md(g, 4))
        P()


def w6():
    f = os.path.join(R, "w6_lodo_selection.csv")
    if not os.path.exists(f):
        return
    P("### W6 BM25 LODO-selected (k1, b)")
    P(md(pd.read_csv(f), 4))
    P()


def w9():
    f = os.path.join(R, "w9_judged.csv")
    if not os.path.exists(f):
        return
    P("### W9 judged@k")
    P(md(pd.read_csv(f), 3))
    P()


def qpp():
    f = os.path.join(R, "w1_qpp_summary.csv")
    if not os.path.exists(f):
        return
    s = pd.read_csv(f)
    P("### QPP (F3): mean Kendall tau across datasets, post-retrieval predictors")
    P(md(s[s.list != "pre"].pivot_table(index="predictor", columns="system", values="mean_tau").reset_index(), 3))
    P()
    P("Pre-retrieval predictors vs BM25 nDCG (mean tau, #datasets Holm-significant):")
    P(md(s[(s.list == "pre") & (s.system == "bm25")][["predictor", "k", "mean_tau", "min_tau", "max_tau",
                                                      "n_sig_holm"]], 3))
    P()


if __name__ == "__main__":
    sanity()
    w7()
    w3()
    w6()
    qpp()
    w1()
    w9()
    txt = "\n".join(out)
    open(os.path.join(R, "summary_tables.md"), "w").write(txt)
    print(txt)
