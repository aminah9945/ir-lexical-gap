"""Build report/report.pdf from results/*.csv, results/*.json, figures/*.png and report/narrative.py."""
import json
import os
import sys

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASETS, PRETTY, RES, FIG, ROOT, DOMAIN  # noqa

sys.path.insert(0, os.path.join(ROOT, "report"))
import narrative as N  # noqa

INK = colors.HexColor("#1f1f1e")
MUTED = colors.HexColor("#6b6a66")
RULE = colors.HexColor("#d6d5d0")
HEAD = colors.HexColor("#eef3fb")
VCOL = {"SUPPORTED": colors.HexColor("#0ca30c"), "PARTIALLY SUPPORTED": colors.HexColor("#c98500"),
        "NOT SUPPORTED": colors.HexColor("#d03b3b")}

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, textColor=INK, spaceAfter=6, spaceBefore=10)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, textColor=INK, spaceAfter=4, spaceBefore=8)
BODY = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.2, leading=12.6, textColor=INK, alignment=TA_LEFT,
                      spaceAfter=5)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=7.6, leading=9.6, textColor=MUTED)
CELL = ParagraphStyle("c", parent=BODY, fontSize=7.4, leading=9, spaceAfter=0)
TITLE = ParagraphStyle("t", parent=H1, fontSize=19, leading=23, spaceAfter=4)


def P(t, st=BODY):
    return Paragraph(t, st)


def table(data, widths=None, header=True, font=7.4):
    rows = [[c if hasattr(c, "wrap") else Paragraph(str(c), ParagraphStyle("x", parent=CELL, fontSize=font))
             for c in r] for r in data]
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
          ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), HEAD))
    t.setStyle(TableStyle(st))
    return t


def fig(name, width=17, caption=None):
    path = os.path.join(FIG, name)
    from PIL import Image as PI
    w, h = PI.open(path).size
    out = [Image(path, width=width * cm, height=width * cm * h / w)]
    if caption:
        out.append(P(caption, SMALL))
    return KeepTogether(out)


def verdict_para(v):
    c = VCOL.get(v, INK)
    return Paragraph(f'<font color="{c.hexval().replace("0x", "#")}"><b>{v}</b></font>', CELL)


def f3(x):
    return f"{x:.3f}"


def build():
    V = json.load(open(os.path.join(RES, "verdicts.json")))
    V["H4"] = json.load(open(os.path.join(RES, "h4_verdict.json")))
    tab = pd.read_csv(os.path.join(RES, "table_main.csv"))
    dom = pd.read_csv(os.path.join(RES, "domain_stats.csv")).set_index("dataset").reindex(DATASETS)
    story = []
    story += [P(N.TITLE, TITLE), P(N.SUBTITLE, SMALL), Spacer(1, 6)]
    story += [P("Summary of verdicts", H1), P(N.EXEC_INTRO)]
    rows = [["#", "Claim (from proposal)", "Test", "Verdict", "Key evidence"]]
    for k, claim, test in N.CLAIMS:
        rows.append([k, claim, test, verdict_para(V[k]["verdict"]), N.EVIDENCE[k](V)])
    story.append(table(rows, widths=[0.8 * cm, 4.2 * cm, 3.6 * cm, 2.4 * cm, 6.4 * cm]))
    story += [Spacer(1, 6), P("What you may not want to hear", H2)] + [P("&bull; " + t) for t in N.BAD_NEWS(V)]

    story += [P("1. Setup", H1)] + [P(t) for t in N.SETUP]
    rows = [["Dataset", "Domain", "Docs", "Test q.", "Avg doc len", "Avg q len", "Rel/q", "Corpus jargon",
             "Query jargon", "Mean q IDF", "Mean coverage"]]
    for d, r in dom.iterrows():
        rows.append([PRETTY[d], DOMAIN[d], f"{int(r.n_docs):,}", int(r.n_queries), f"{r.avg_doc_len:.0f}",
                     f"{r.avg_q_len:.1f}", f"{r.avg_rel:.1f}", f"{r.corpus_jargon:.3f}", f"{r.q_jargon:.3f}",
                     f"{r.q_mean_idf:.2f}", f"{r.cov_mean:.2f}"])
    story += [table(rows, font=7), P(N.STATS_NOTE, SMALL)]
    story += [P("Validation against published references", H2), P(N.VALIDATION)]
    ref = N.PYSERINI_BM25
    bm = tab[tab.system == "bm25"].set_index("dataset")["ndcg@10"]
    rows = [["Dataset"] + [PRETTY[d] for d in DATASETS],
            ["Ours (BM25, k1=0.9 b=0.4)"] + [f3(bm.get(d, float("nan"))) for d in DATASETS],
            ["Pyserini BM25 flat (Kamalloo et al. 2024)"] + [f3(ref[d]) for d in DATASETS]]
    story.append(table(rows, font=7))
    story.append(PageBreak())

    story += [P("2. Results: effectiveness (E1, E2, E5, E6)", H1), fig("fig1_heatmap.png", 17, N.CAP_FIG1)]
    for m in ["ndcg@10", "recall@100", "map@100", "mrr@10"]:
        order = [s for s in N.SYSTEM_ORDER if s in set(tab.system)]
        M = tab.pivot(index="system", columns="dataset", values=m).reindex(order)[DATASETS]
        rows = [[m] + [PRETTY[d] for d in DATASETS]]
        for s, r in M.iterrows():
            best = r.max()
            rows.append([N.SYSNAME.get(s, s)] + [("<b>%s</b>" % f3(v)) if v == best else
                                                  ("-" if pd.isna(v) else f3(v)) for v in r.values])
        story += [Spacer(1, 4), table(rows, font=6.8)]
    story += [P(N.CAP_TABLES, SMALL)]
    story += [P("Boolean retrieval as a set (its native output)", H2), P(N.BOOLEAN_TEXT)]
    bs = pd.read_csv(os.path.join(RES, "boolean_set_summary.csv")).set_index("dataset")
    rows = [["Dataset", "Empty AND result", "Median set size", "Set precision", "Set recall", "Set F1"]]
    for d, r in bs.iterrows():
        rows.append([PRETTY[d], f"{r.empty_frac:.0%}", f"{r.median_set:.0f}", f3(r.P), f3(r.R), f3(r.F1)])
    story.append(table(rows, font=7))
    story += [fig("fig5_ablation.png", 17, N.CAP_FIG5)] + [P(t) for t in N.ABLATION_TEXT(V)]
    gs = pd.read_csv(os.path.join(RES, "bm25_grid_summary.csv")).set_index("dataset")
    rows = [["Dataset", "BM25 default (0.9/0.4)", "Oracle best", "best k1", "best b"]]
    for d, r in gs.iterrows():
        rows.append([PRETTY[d], f3(r["default"]), f3(r.best), r.best_k1, r.best_b])
    story += [P("BM25 parameter sensitivity (oracle, tuned on test - upper bound only)", H2), table(rows, font=7),
              P(N.GRID_TEXT, SMALL)]
    story.append(PageBreak())

    story += [P("3. Hypothesis tests", H1)]
    for k, title, texts, figs in N.SECTIONS(V):
        story.append(P(title, H2))
        story += [P(t) for t in texts]
        for fn, cap in figs:
            story.append(fig(fn, 16.5, cap))
    story.append(PageBreak())
    story += [P("4. Failure-mode catalogue (paraphrased examples)", H1)] + [P(t) for t in N.CATALOGUE_INTRO]
    rows = [["Mode", "Domain", "Paraphrased query", "What the relevant document says instead"]]
    rows += N.CATALOGUE
    story.append(table(rows, widths=[2.0 * cm, 2.1 * cm, 6.2 * cm, 7.1 * cm]))
    rr = pd.read_csv(os.path.join(RES, "failure_rescue_rates.csv")).set_index("primary")
    cols = [c for c in rr.columns if c.startswith("rescued_") and c != "rescued_bm25"]
    rows = [["Primary mode"] + [N.SYSNAME.get(c.replace("rescued_", ""), c) for c in cols]]
    for m, r in rr.iterrows():
        rows.append([m] + ["-" if pd.isna(r[c]) else f"{r[c]:.0%}" for c in cols])
    story += [P("Share of BM25 failures rescued (nDCG@10 >= 0.1) by each alternative", H2), table(rows, font=7),
              P(N.RESCUE_TEXT, SMALL)]
    story += [P("5. Decision guideline", H1)] + [P(t) for t in N.GUIDELINE(V)]
    story += [P("6. Threats to validity", H1)] + [P("&bull; " + t) for t in N.THREATS]
    story += [P("7. Reproducibility", H1)] + [P(t) for t in N.REPRO]

    doc = SimpleDocTemplate(os.path.join(ROOT, "report", "report.pdf"), pagesize=A4, leftMargin=1.8 * cm,
                            rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title=N.TITLE, author="Project report")

    def footer(c, d):
        c.setFont("Helvetica", 7)
        c.setFillColor(MUTED)
        c.drawString(1.8 * cm, 1.0 * cm, "Lexical-gap analysis across BEIR domains")
        c.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, str(d.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote report/report.pdf")


if __name__ == "__main__":
    build()
