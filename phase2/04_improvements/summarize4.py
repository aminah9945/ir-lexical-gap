"""Collect E1-E4 results: the primary family F4 (Holm over P1-P4) and figures.

Usage: python summarize4.py
Outputs: results/f4_primary.csv, results/summary4.md, figures/e1_alpha_bars.png, figures/e4_fewshot.png,
         figures/e3_ce_bars.png, figures/e4_indomain_bars.png
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "03_experiments"))
import common2 as c2  # noqa: E402

RES4, FIG4 = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
os.makedirs(FIG4, exist_ok=True)


def r(p):
    return os.path.join(RES4, p)


def md(df, fl=4):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(f"{v:.{fl}f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return "\n".join(out)


def primary():
    rows = []
    if os.path.exists(r("e1_tests_minilm.csv")):
        t = pd.read_csv(r("e1_tests_minilm.csv"))
        x = t[(t.variant == "e1_ridge") & (t.dataset == "MACRO")].iloc[0]
        rows.append(dict(test="P1 E1 per-query alpha (Ridge, Tier A+B, MiniLM, LODO) - H0.5", k=5, delta=x.delta,
                         p_one_sided=x.p_one_sided))
    if os.path.exists(r("e2_tests.csv")):
        t = pd.read_csv(r("e2_tests.csv"))
        x = t[(t.model == "minilm") & (t.scope == "rest") & (t.dataset == "MACRO")]
        if len(x):
            x = x.iloc[0]
            rows.append(dict(test="P2 E2 CE-pseudo-label alpha (MiniLM, held-out queries) - H0.5", k=int(x.n),
                             delta=x.delta, p_one_sided=x.p_one_sided))
    if os.path.exists(r("e3_ce_tests.csv")):
        t = pd.read_csv(r("e3_ce_tests.csv"))
        x = t[(t.comparison == "CE(hybrid_cc_minilm)-base") & (t.dataset == "MACRO")]
        if len(x):
            x = x.iloc[0]
            rows.append(dict(test="P3 E3 CE re-rank of H(MiniLM) top-100 - H(MiniLM) (sample)", k=int(x.n),
                             delta=x.delta, p_one_sided=x.p_one_sided))
    if os.path.exists(r("e4_indomain_tests.csv")):
        t = pd.read_csv(r("e4_indomain_tests.csv"))
        x = t[(t.config == "A_bge_std_k8") & (t.comparison == "router-H") & (t.dataset == "MACRO")].iloc[0]
        rows.append(dict(test="P4 E4 in-domain router (A_bge_std, 5-fold CV) - always-H", k=8, delta=x.delta,
                         p_one_sided=x.p_one_sided))
    f = pd.DataFrame(rows)
    f["p_holm"] = c2.holm(f.p_one_sided.values)
    f["improvement"] = (f.p_holm < 0.05) & (f.delta >= 0.005)
    f.to_csv(r("f4_primary.csv"), index=False)
    return f


def figs():
    plt = c2.mpl_style()
    # E1 bars
    if os.path.exists(r("e1_perdataset_minilm.csv")):
        p = pd.read_csv(r("e1_perdataset_minilm.csv"))
        x = np.arange(len(p))
        fig, ax = plt.subplots(figsize=(7.0, 2.8))
        for i, (c, lab, col) in enumerate([("H", "hybrid α=0.5", c2.C_HYB),
                                           ("e1_global", "LODO-global α", c2.C_MUTED),
                                           ("e1_ridge", "per-query α (Ridge)", c2.C_ROUTER),
                                           ("e1_hgb", "per-query α (HGB)", "#a99cf0")]):
            ax.bar(x + (i - 1.5) * 0.2, p[c], 0.2, color=col, label=lab)
        for xi, v in zip(x, p.oracle_q):
            ax.plot([xi - 0.4, xi + 0.4], [v, v], color=c2.C_INK, lw=1.1)
        ax.plot([], [], color=c2.C_INK, lw=1.1, label="per-query α oracle")
        ax.set_xticks(x)
        ax.set_xticklabels([c2.PRETTY.get(d, "Macro avg") for d in p.dataset], fontsize=7.5)
        ax.set_ylabel("nDCG@10")
        ax.legend(ncol=5, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.16))
        ax.grid(axis="x", visible=False)
        fig.savefig(os.path.join(FIG4, "e1_alpha_bars.png"))
        plt.close(fig)
    # few-shot curve (macro over datasets per model)
    if os.path.exists(r("e4_fewshot_dataset_selection.csv")):
        fs = pd.read_csv(r("e4_fewshot_dataset_selection.csv"))
        sh = pd.read_csv(r("e4_splithalf.csv"))
        fig, ax = plt.subplots(figsize=(4.2, 2.9))
        for m, col, ls in [("bge", c2.C_ROUTER, "-"), ("minilm", c2.C_DENSE, "--"), ("bgebase", c2.C_INK, ":")]:
            g = fs[fs.model == m]
            if g.empty:
                continue
            nds = g.dataset.nunique()
            g = g.groupby("k").filter(lambda z: len(z) == nds) if m else g
            mac = g.groupby("k").gain_mean.mean()
            # datasets with n < k drop out; keep only k present for all datasets
            ax.plot(mac.index, mac.values, ls, marker="o", ms=3, color=col, label=f"k labelled queries, D={c2.MODEL_PRETTY[m]}")
            ax.axhline(sh[sh.model == m].oracle_ds_gain_insample.mean(), color=col, lw=0.7, ls=":")
        rm = pd.read_csv(os.path.join(c2.RES2, "w1_router_macro.csv"))
        for cfg, col in [("A_bge_std_k8", c2.C_ROUTER), ("A_minilm_std_k8", c2.C_DENSE)]:
            v = rm[(rm.config == cfg) & (rm.variant == "selected")].router_minus_H.values[0]
            ax.scatter([3], [v], marker="x", color=col, zorder=3)
        ax.scatter([], [], marker="x", color=c2.C_MUTED, label="LODO query router (Agent 3)")
        ax.axhline(0, color=c2.C_HYB, lw=1, label="always-hybrid")
        ax.plot([], [], ":", color=c2.C_MUTED, lw=0.7, label="dataset oracle (all labels)")
        ax.set_xscale("log")
        ax.set_xticks([3, 5, 10, 20, 30])
        ax.set_xticklabels(["0\n(router)", "5", "10", "20", "30"])
        ax.minorticks_off()
        ax.set_xlabel("labelled target queries k used to pick one fixed system")
        ax.set_ylabel("macro Δ nDCG@10 vs always-H")
        ax.legend(frameon=False, fontsize=6.5, loc="lower right")
        fig.savefig(os.path.join(FIG4, "e4_fewshot.png"))
        plt.close(fig)
    # E3 bars
    if os.path.exists(r("e3_ce_perdataset.csv")):
        p = pd.read_csv(r("e3_ce_perdataset.csv"))
        b = p[p.first_stage == "bm25"].set_index("dataset")
        h = p[p.first_stage == "hybrid_cc_minilm"].set_index("dataset")
        dss = list(b.index)
        x = np.arange(len(dss))
        fig, ax = plt.subplots(figsize=(7.5, 2.8))
        ax.bar(x - 0.3, b.base, 0.15, color=c2.C_BM25, label="BM25")
        ax.bar(x - 0.15, b.ce, 0.15, color=c2.C_BM25, alpha=0.45, hatch="//", label="CE(BM25 top-100)")
        if "ref_hybrid_cc_bge" in b:
            ax.bar(x, b.ref_hybrid_cc_bge, 0.15, color=c2.C_HYB, label="H(BGE-s)")
        hv = [h.base.get(d, np.nan) for d in dss]
        hc = [h.ce.get(d, np.nan) for d in dss]
        ax.bar(x + 0.15, hv, 0.15, color=c2.C_DENSE, label="H(MiniLM)")
        ax.bar(x + 0.3, hc, 0.15, color=c2.C_ROUTER, label="CE(H(MiniLM) top-100)")
        ax.set_xticks(x)
        ax.set_xticklabels([c2.PRETTY[d] for d in dss], fontsize=7.5)
        ax.set_ylabel("nDCG@10 (CE sample queries)")
        ax.legend(ncol=5, frameon=False, fontsize=6.8, loc="upper left", bbox_to_anchor=(0, 1.16))
        ax.grid(axis="x", visible=False)
        fig.savefig(os.path.join(FIG4, "e3_ce_bars.png"))
        plt.close(fig)
    # E4 in-domain bars
    if os.path.exists(r("e4_indomain_perdataset.csv")):
        p = pd.read_csv(r("e4_indomain_perdataset.csv"))
        p = p[p.config == "A_bge_std_k8"].set_index("dataset")
        lodo = pd.read_csv(os.path.join(c2.RES2, "w1_router_perdataset.csv"))
        lodo = lodo[(lodo.config == "A_bge_std_k8") & (lodo.variant == "selected")].set_index("dataset").router
        dss = list(p.index)
        x = np.arange(len(dss))
        fig, ax = plt.subplots(figsize=(7.5, 2.8))
        base = p.always_H
        for i, (v, lab, col) in enumerate([(p.indomain_best_fixed - base, "in-domain best fixed (CV)", c2.C_MUTED),
                                           (p.indomain_router - base, "in-domain router (CV)", c2.C_ROUTER),
                                           (lodo.reindex(dss) - base, "LODO router", "#a99cf0"),
                                           (p.oracle_query - base, "per-query oracle", c2.C_INK)]):
            ax.bar(x + (i - 1.5) * 0.2, v, 0.2, color=col, label=lab)
        ax.axhline(0, color=c2.C_HYB, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([c2.PRETTY[d] for d in dss], fontsize=7.5)
        ax.set_ylabel("Δ nDCG@10 vs always-H(BGE-s)")
        ax.legend(ncol=4, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.16))
        ax.grid(axis="x", visible=False)
        fig.savefig(os.path.join(FIG4, "e4_indomain_bars.png"))
        plt.close(fig)


def main():
    f = primary()
    figs()
    parts = ["# Agent 4 machine summary (regenerate with `python summarize4.py`)", "", "## F4 primary family (Holm)",
             md(f)]
    for name in ["e1_perdataset_minilm", "e2_alpha_perdataset", "e3_ce_perdataset", "e4_indomain_perdataset",
                 "e4_splithalf", "e4_headroom_concentration", "e4_label_consistency"]:
        if os.path.exists(r(f"{name}.csv")):
            d = pd.read_csv(r(f"{name}.csv"))
            d = d[[c for c in d.columns if not c.startswith("pseudo_") and c not in ("kinds", "source")]]
            parts += ["", f"## {name}", md(d, 4)]
    for name, filt in [("e1_tests_minilm", None), ("e3_ce_tests", None), ("e4_indomain_tests", None), ("e2_tests", None)]:
        if os.path.exists(r(f"{name}.csv")):
            d = pd.read_csv(r(f"{name}.csv"))
            d = d[d.dataset.isin(["MACRO", "POOLED_DL"])].dropna(axis=1, how="all")
            parts += ["", f"## {name} (macro / pooled)", md(d, 4)]
    if os.path.exists(r("e4_fewshot_dataset_selection.csv")):
        fs = pd.read_csv(r("e4_fewshot_dataset_selection.csv"))
        mac = fs.groupby(["model", "k"]).agg(gain_macro=("gain_mean", "mean"), n_ds=("dataset", "nunique")).reset_index()
        parts += ["", "## few-shot dataset-level selection (exploratory), macro gain vs always-H", md(mac, 4)]
    open(r("summary4.md"), "w").write("\n".join(parts) + "\n")
    print("\n".join(parts[:4]))


if __name__ == "__main__":
    main()
