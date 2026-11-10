"""Colab setup for the IR_Lexical_Gap project (Agent 5). Embedded verbatim in 00_SETUP_drive.ipynb;
the other notebooks copy it from MyDrive/IR_Lexical_Gap/02_code/phase2/setup_drive.py.

What it does (idempotent, safe to re-run):
  1. creates the Drive project layout  MyDrive/IR_Lexical_Gap/project/{src,report,colab,results,figures,cache,phase2/...}
     (the datasets stay in MyDrive/IR_Lexical_Gap/data; link_proj() builds /content/proj with symlinks into Drive)
  2. downloads the 8 BEIR datasets into MyDrive/IR_Lexical_Gap/data/<ds>/ (skipped if corpus.jsonl exists)
  3. downloads every open-access paper PDF of phase2/literature/reading_list.md into
     MyDrive/IR_Lexical_Gap/04_literature/pdfs/<topic>/FirstAuthorYear_ShortTitle.pdf
     (skips existing files, checks the %PDF magic bytes, never aborts on a failure, writes download_report.txt)

Usage (Colab):  python setup_drive.py [all|data|layout|papers|link]      (default: all)
Env:  IRGAP_DRIVE_ROOT  (default /content/drive/MyDrive/IR_Lexical_Gap)  -- set it to a scratch dir to test locally.
      IRGAP_PAPERS      comma-separated bib keys to download only a subset (testing).
"""
import os
import re
import sys
import time
import zipfile

import requests

DRIVE = os.environ.get("IRGAP_DRIVE_ROOT", "/content/drive/MyDrive/IR_Lexical_Gap")
DATA = os.path.join(DRIVE, "data")
PROJ = os.path.join(DRIVE, "project")
LIT = os.path.join(DRIVE, "04_literature", "pdfs")
BEIR = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
DATASETS = ["nfcorpus", "scifact", "fiqa", "scidocs", "arguana", "trec-covid", "quora", "webis-touche2020"]
UA = {"User-Agent": "Mozilla/5.0 (IR_Lexical_Gap student project; Colab setup script)"}

# Every entry of reading_list.md that has an open PDF link (33 papers). Spot-checked 2026-10-22 from the project
# machine (16 URLs): all arXiv / ACL Anthology / NeurIPS / TREC links returned application/pdf; the two Springer
# links returned an HTML paywall page (no university network); the author-hosted Robertson & Zaragoza link was not
# reachable from that machine. All are tried anyway; failures are reported, never fatal.
# (topic folder, bib key, file name, [urls in order of preference])
PAPERS = [
    ("01_classical_ir_bm25", "robertson2009prf", "Robertson2009_ProbabilisticRelevanceFramework_BM25",
     ["https://staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf"]),
    ("01_classical_ir_bm25", "abduljaleel2004umass", "AbdulJaleel2004_UMass_TREC2004_RM3",
     ["https://trec.nist.gov/pubs/trec13/papers/umass.novelty.hard.pdf"]),
    ("02_beir_dense_vs_lexical", "thakur2021beir", "Thakur2021_BEIR",
     ["https://arxiv.org/pdf/2104.08663",
      "https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf"]),
    ("02_beir_dense_vs_lexical", "kamalloo2024resources", "Kamalloo2024_ResourcesForBrewingBEIR",
     ["https://arxiv.org/pdf/2306.07471"]),
    ("02_beir_dense_vs_lexical", "thakur2024touche", "Thakur2024_Touche2020_NeuralRetrievalEvaluation",
     ["https://arxiv.org/pdf/2407.07790"]),
    ("02_beir_dense_vs_lexical", "formal2022match", "Formal2022_MatchYourWords_LexicalMatchingNeuralIR",
     ["https://arxiv.org/pdf/2112.05662"]),
    ("02_beir_dense_vs_lexical", "sciavolino2021simple", "Sciavolino2021_EntityCentricQuestionsDenseRetrievers",
     ["https://aclanthology.org/2021.emnlp-main.496.pdf"]),
    ("02_beir_dense_vs_lexical", "ren2023thorough", "Ren2023_ZeroShotDenseRetrievalExamination",
     ["https://aclanthology.org/2023.findings-emnlp.1057.pdf"]),
    ("02_beir_dense_vs_lexical", "lin2021pyserini", "Lin2021_Pyserini", ["https://arxiv.org/pdf/2102.10073"]),
    ("02_beir_dense_vs_lexical", "muennighoff2023mteb", "Muennighoff2023_MTEB",
     ["https://aclanthology.org/2023.eacl-main.148.pdf"]),
    ("02_beir_dense_vs_lexical", "formal2021splade", "Formal2021_SPLADE", ["https://arxiv.org/pdf/2107.05720"]),
    ("02_beir_dense_vs_lexical", "ma2021replication", "Ma2021_DPR_ReplicationStudy",
     ["https://arxiv.org/pdf/2104.05740"]),
    ("02_beir_dense_vs_lexical", "lin2021fewbrief", "Lin2021_FewBriefNotes_DeepImpact_COIL",
     ["https://arxiv.org/pdf/2106.14807"]),
    ("04_query_performance_prediction", "meng2023qpp", "Meng2023_QPP_AdhocToConversational",
     ["https://arxiv.org/pdf/2305.10923"]),
    ("04_query_performance_prediction", "faggioli2023qpp", "Faggioli2023_QPP_NeuralIR_AreWeThereYet",
     ["https://arxiv.org/pdf/2302.09947"]),
    ("05_selective_retrieval_routing", "arabzadeh2021predicting", "Arabzadeh2021_SparseVsDenseStrategySelection",
     ["https://arxiv.org/pdf/2109.10739"]),
    ("05_selective_retrieval_routing", "khramtsova2023selecting", "Khramtsova2023_SelectingDenseRetrieverZeroShot",
     ["https://arxiv.org/pdf/2309.09403"]),
    ("05_selective_retrieval_routing", "mallen2023when", "Mallen2023_WhenNotToTrustLanguageModels",
     ["https://aclanthology.org/2023.acl-long.546.pdf"]),
    ("05_selective_retrieval_routing", "jeong2024adaptive", "Jeong2024_AdaptiveRAG",
     ["https://aclanthology.org/2024.naacl-long.389.pdf"]),
    ("06_hybrid_fusion", "bruch2024fusion", "Bruch2024_FusionFunctionsHybridRetrieval",
     ["https://arxiv.org/pdf/2210.11934"]),
    ("06_hybrid_fusion", "chen2022out", "Chen2022_OutOfDomainSemantics_ZeroShotHybrid",
     ["https://arxiv.org/pdf/2201.10582"]),
    ("06_hybrid_fusion", "kuzi2020leveraging", "Kuzi2020_SemanticLexicalMatching_Hybrid",
     ["https://arxiv.org/pdf/2010.01195"]),
    ("07_domain_specific_2025", "mori2025assessing", "Mori2025_BM25vsDense_CJEU_Legal",
     ["https://arxiv.org/pdf/2506.12895"]),
    ("07_domain_specific_2025", "cunha2025medlink", "Cunha2025_MedLink_Demo",
     ["https://link.springer.com/content/pdf/10.1007/978-3-031-88720-8_13.pdf"]),
    ("07_domain_specific_2025", "ateia2025bioragent", "Ateia2025_BioRAGent_Demo",
     ["https://link.springer.com/content/pdf/10.1007/978-3-031-88720-8_1.pdf"]),
    ("07_domain_specific_2025", "doostmohammadi2025studying", "Doostmohammadi2025_RALMTraining_QueryOverlap",
     ["https://aclanthology.org/2025.emnlp-main.1363.pdf"]),
    ("07_domain_specific_2025", "akarsu2026bm25", "Akarsu2026_BM25toCorrectiveRAG_Finance",
     ["https://arxiv.org/pdf/2604.01733"]),
    ("07_domain_specific_2025", "pandey2026domain", "Pandey2026_DomainAdaptiveDenseRetrieval",
     ["https://arxiv.org/pdf/2602.00899"]),
    ("07_domain_specific_2025", "sager2025deep", "Sager2025_DeepRetrieval_CheckThat2025",
     ["https://arxiv.org/pdf/2505.23250"]),
    ("08_dense_retrievers", "reimers2019sbert", "Reimers2019_SentenceBERT", ["https://aclanthology.org/D19-1410.pdf"]),
    ("08_dense_retrievers", "wang2020minilm", "Wang2020_MiniLM",
     ["https://proceedings.neurips.cc/paper/2020/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"]),
    ("08_dense_retrievers", "xiao2024cpack", "Xiao2024_CPack_BGE", ["https://arxiv.org/pdf/2309.07597"]),
    ("08_dense_retrievers", "wang2022e5", "Wang2022_E5_TextEmbeddings", ["https://arxiv.org/pdf/2212.03533"]),
]


def extra_from_reading_list(path):
    """PDF links in reading_list.md that are not in PAPERS (in case the list was extended later)."""
    known = {p[1] for p in PAPERS}
    out, sec = [], "99_other"
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        m = re.match(r"## (\d+)\. (.*)", line)
        if m:
            sec = f"{int(m.group(1)):02d}_" + re.sub(r"[^a-z0-9]+", "_", m.group(2).lower()).strip("_")[:40]
            continue
        if not line.startswith("| ") or line.startswith("| Pri") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        key = cells[1]
        urls = [u for u in re.findall(r"https?://[^\s;)]+", cells[-1]) if "doi.org" not in u and "huggingface" not in u]
        if key in known or not urls or not re.match(r"^[a-z_]+\d{4}[a-z]*$", key):
            continue
        m = re.match(r"([a-z_]+)(\d{4})(.*)", key)
        name = f"{m.group(1).capitalize()}{m.group(2)}_{m.group(3).capitalize() or 'paper'}"
        out.append((sec, key, name, urls))
    return out


# Output folders that live in Drive. Code folders (src/, phase2/03_experiments/ ...) stay REAL local folders:
# Python resolves symlinks for sys.path[0], so a symlinked src/ would make common.ROOT point into Drive/project,
# where data/ does not exist (Drive cannot hold symlinks).
PROJ_LINKS = ["results", "figures", "cache", "report",
              "phase2/03_experiments/results", "phase2/03_experiments/figures",
              "phase2/03_experiments/cache", "phase2/03_experiments/logs",
              "phase2/04_improvements/results", "phase2/04_improvements/figures",
              "phase2/04_improvements/cache", "phase2/04_improvements/logs"]
LOCAL_DIRS = ["src", "colab", "phase2/03_experiments", "phase2/04_improvements", "phase2/colab"]


def layout():
    """Create the Drive project folders (real folders: Google Drive cannot hold symlinks)."""
    for d in PROJ_LINKS + ["cache/runs", "src", "phase2/colab"]:
        os.makedirs(os.path.join(PROJ, d), exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(os.path.join(DRIVE, "colab_runs"), exist_ok=True)
    print("project layout ready under", PROJ)


def link_proj(local="/content/proj"):
    """Build /content/proj (local disk) with the local project layout: code folders are real local folders,
    output folders are symlinks INTO Drive (PROJ/<d>), data -> DRIVE/data, and
    phase2/03_experiments/colab_runs -> DRIVE/colab_runs. Everything the pipelines write lands in Drive."""
    layout()
    if os.path.isdir(local) and not os.path.islink(local):
        stale = [d for d in PROJ_LINKS + ["data"] if os.path.exists(os.path.join(local, d))
                 and not os.path.islink(os.path.join(local, d))]
        if stale:
            bak = f"{local}_local_{int(time.time())}"
            os.rename(local, bak)
            print(f"moved an existing non-Drive {local} (real {stale[0]}/) to {bak}")
    for d in LOCAL_DIRS:
        os.makedirs(os.path.join(local, d), exist_ok=True)
    links = {d: os.path.join(PROJ, d) for d in PROJ_LINKS}
    links["data"] = DATA
    links["phase2/03_experiments/colab_runs"] = os.path.join(DRIVE, "colab_runs")
    for d, target in links.items():
        link = os.path.join(local, d)
        if not os.path.lexists(link):
            os.symlink(target, link)
    print(f"{local}: code folders local; {', '.join(sorted(links))} -> Google Drive")


def get(url, tries=4, **kw):
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=120, **kw)
            if r.status_code in (429, 503) and i < tries - 1:
                time.sleep(5 * 2 ** i)
                continue
            return r
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(5 * 2 ** i)
    return r


def data():
    os.makedirs(DATA, exist_ok=True)
    for ds in DATASETS:
        corpus = os.path.join(DATA, ds, "corpus.jsonl")
        if os.path.exists(corpus) and os.path.exists(os.path.join(DATA, ds, "qrels", "test.tsv")):
            print(f"{ds:18s} present, skipped")
            continue
        t = time.time()
        r = get(BEIR.format(ds), stream=True)
        r.raise_for_status()
        tmp = os.path.join("/tmp" if os.path.isdir("/tmp") else DATA, f"{ds}.zip")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(DATA)  # archives contain <ds>/corpus.jsonl, queries.jsonl, qrels/*.tsv
        os.remove(tmp)
        n = sum(1 for _ in open(corpus, encoding="utf-8"))
        print(f"{ds:18s} {n:>8d} docs  ({time.time() - t:.0f}s)")


def papers():
    only = set(filter(None, os.environ.get("IRGAP_PAPERS", "").split(",")))
    lst = PAPERS + extra_from_reading_list(os.path.join(DRIVE, "04_literature", "reading_list.md"))
    if only:
        lst = [p for p in lst if p[1] in only]
    rows = []
    for topic, key, name, urls in lst:
        d = os.path.join(LIT, topic)
        os.makedirs(d, exist_ok=True)
        out = os.path.join(d, name + ".pdf")
        if os.path.exists(out) and os.path.getsize(out) > 10000:
            rows.append((key, "present", out))
            continue
        status = "FAILED"
        for u in urls:
            try:
                r = get(u, allow_redirects=True)
                if r.status_code == 200 and r.content[:5] == b"%PDF-":
                    open(out, "wb").write(r.content)
                    status = f"ok {len(r.content) // 1024} KB"
                    break
                status = f"FAILED (HTTP {r.status_code}, {r.headers.get('content-type', '?')[:30]}) {u}"
            except Exception as e:  # noqa: BLE001 -- report and continue
                status = f"FAILED ({type(e).__name__}) {u}"
            if "arxiv.org" in u:
                time.sleep(3)  # arXiv asks for polite crawling
        rows.append((key, status, out))
        print(f"{key:28s} {status}")
    ok = sum(r[1].startswith(("ok", "present")) for r in rows)
    rep = [f"PDF download report ({time.strftime('%Y-%m-%d %H:%M')}): {ok}/{len(rows)} available", ""]
    rep += [f"{k:28s} {s}  -> {os.path.relpath(p, LIT)}" for k, s, p in rows]
    rep += ["", "FAILED items: open the DOI/URL in reading_list.md from the university network (library access)",
            "and save the PDF into the same folder with the file name shown above."]
    os.makedirs(LIT, exist_ok=True)
    open(os.path.join(LIT, "download_report.txt"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep[:1]))
    return rows


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "layout"):
        layout()
    if what == "link":
        link_proj()
    if what in ("all", "data"):
        data()
    if what in ("all", "papers"):
        papers()
