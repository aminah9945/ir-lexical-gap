"""Builds the Drive-based Colab notebooks (Agent 5):
  phase2/colab/00_SETUP_drive.ipynb    mount Drive, project layout, 8 BEIR datasets, literature PDFs, pip packages
  phase2/colab/phase2_analysis.ipynb   Agent 3 (03_experiments/run_all.sh) + Agent 4 (04_improvements/run_all4.sh), CPU
  phase2/colab/phase1_full_pipeline_drive.ipynb  runs colab/lexical_gap_full_pipeline.ipynb (unchanged, read from
                                       Drive) cell by cell with results/, cache/, figures/, report/ linked into Drive
phase2_analysis copies the code from Drive 02_code/ and checks md5s recorded at build time.
Re-run after any change to the code:  python phase2/colab/build_colab_notebooks.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
P = "/content/proj"
DRIVE = "/content/drive/MyDrive/IR_Lexical_Gap"
META = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"}}
META_CPU = {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}


def md(t, tag=None):
    c = {"cell_type": "markdown", "metadata": {}, "source": t}
    if tag:
        c["metadata"]["tags"] = [tag]
    return c


def code(t, tag=None):
    c = {"cell_type": "code", "metadata": {}, "source": t, "outputs": [], "execution_count": None}
    if tag:
        c["metadata"]["tags"] = [tag]
    return c


def nb(cells, meta):
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


def save(obj, rel):
    p = os.path.join(ROOT, rel)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {rel} ({os.path.getsize(p) // 1024} KB, {len(obj['cells'])} cells)")


MOUNT = "from google.colab import drive\ndrive.mount('/content/drive')"
# every notebook takes setup_drive.py from Drive (02_code/phase2 = phase2/colab/setup_drive.py, uploaded by Agent 5)
GET_SETUP = (f"import shutil\nshutil.copy('{DRIVE}/02_code/phase2/setup_drive.py', '/content/setup_drive.py')  "
             "# Drive copy of phase2/colab/setup_drive.py")
LINK = f"!python /content/setup_drive.py link\n%cd {P}"

# ----------------------------------------------------------------------------- 00_SETUP_drive
setup = nb([
    md(f"""# 00 SETUP: Google Drive, datasets, literature

Run this notebook **once** (re-running is safe: whatever already exists is skipped). No GPU needed (*Runtime > Change runtime type > CPU* is fine).

It does four things:

1. It mounts Google Drive. Allow access when asked.
2. It creates the project layout in `MyDrive/IR_Lexical_Gap/project/`. The folders `src/`, `results/`, `cache/`, `figures/`, `report/` and `phase2/03_experiments|04_improvements/...` mirror the local project.
3. It downloads the 8 BEIR test collections into `MyDrive/IR_Lexical_Gap/data/<dataset>/`: nfcorpus, scifact, fiqa, scidocs, arguana, trec-covid, quora and webis-touche2020. That is about 1.3 GB unzipped and takes about 5-10 min.
4. It downloads every open-access PDF from `04_literature/reading_list.md` into `MyDrive/IR_Lexical_Gap/04_literature/pdfs/<topic>/FirstAuthorYear_ShortTitle.pdf` and writes `pdfs/download_report.txt`.
   - Papers marked "DOI only" in the reading list are paywalled (ACM, Springer, Elsevier). Fetch those from the university network.
   - The two Springer demo papers and the author-hosted Robertson & Zaragoza PDF may fail. The report lists every failure; nothing aborts.

**Then run, in this order:**
1. `phase1_full_pipeline_drive` (T4 GPU). It runs `lexical_gap_full_pipeline` with its outputs in Drive.
2. `phase2_dense_runs` (T4)
3. `phase2_ce_rerank` (T4)
4. `phase2_analysis` (CPU)

See `01_RUN_ORDER.md` in the Drive root."""),
    code(MOUNT),
    code("!pip -q install sentence-transformers PyStemmer wordfreq pytrec-eval-terrier reportlab statsmodels requests\n"
         "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True); print('packages ok')"),
    md("The setup code is `02_code/phase2/setup_drive.py` on Drive. It is copied to the runtime and run from there; open it in Drive to read it."),
    code(GET_SETUP),
    code("!python /content/setup_drive.py layout"),
    md("### BEIR datasets -> `MyDrive/IR_Lexical_Gap/data/`"),
    code("!python /content/setup_drive.py data"),
    md("### Literature PDFs -> `MyDrive/IR_Lexical_Gap/04_literature/pdfs/<topic>/`"),
    code("!python /content/setup_drive.py papers"),
    code(f"""import os, shutil
# keep a copy of the setup script in the Drive project mirror
shutil.copy('/content/setup_drive.py', '{DRIVE}/project/phase2/colab/setup_drive.py')
for root, dirs, files in sorted(os.walk('{DRIVE}/04_literature/pdfs')):
    pdfs = [f for f in files if f.endswith('.pdf')]
    if pdfs:
        print(root.split('pdfs/')[-1] + '/', len(pdfs), 'PDFs')
for d in sorted(os.listdir('{DRIVE}/data')):
    p = '{DRIVE}/data/' + d + '/corpus.jsonl'
    if os.path.exists(p):
        print(f'data/{{d}}: {{os.path.getsize(p) / 1e6:.0f}} MB corpus')
print('\\nSetup finished. Next: open 01_notebooks/phase1_full_pipeline_drive.ipynb (T4 GPU) and run all.')"""),
], META_CPU)
save(setup, "phase2/colab/00_SETUP_drive.ipynb")

# ----------------------------------------------------------------------------- phase2_analysis
A3 = sorted(glob.glob(os.path.join(ROOT, "phase2/03_experiments/*.py"))) + [os.path.join(ROOT, "phase2/03_experiments/run_all.sh")]
A4 = sorted(glob.glob(os.path.join(ROOT, "phase2/04_improvements/*.py"))) + [os.path.join(ROOT, "phase2/04_improvements/run_all4.sh")]

cells = [
    md(f"""# Phase 2 analysis (Agents 3 + 4), run from Google Drive. CPU only

This notebook re-runs the full Phase-2 analysis with the exact code of the project:
- **Agent 3**: `phase2/03_experiments/run_all.sh`. It covers Colab ingest + sanity gates, the W6 BM25 LODO grid, the W1/W2 query-time features, W7 effect sizes, the W3 gap model, W1 QPP + routers and W9 pooling bias, and writes `summary_tables.md`.
- **Agent 4**: `phase2/04_improvements/run_all4.sh`. It covers the alpha grid, E1 per-query alpha, the E4 diagnostics, the E2/E3 cross-encoder evaluation and `summary4.md`.

All outputs (CSVs, figures, logs) are written to `MyDrive/IR_Lexical_Gap/project/phase2/03_experiments/{{results,figures,logs,cache}}` and `.../04_improvements/{{...}}`. The figures for the paper are therefore regenerated in Drive.

**Prerequisites. Run these notebooks first, all with Drive mounted:**
1. `00_SETUP_drive`: the datasets.
2. `phase1_full_pipeline_drive`: it runs `lexical_gap_full_pipeline` and writes the Phase-1 results and rankings into `project/results` and `project/cache/runs`.
3. `phase2_dense_runs`: MiniLM / BGE-s / BGE-b rankings in `MyDrive/IR_Lexical_Gap/colab_runs`.
4. `phase2_ce_rerank`: cross-encoder scores in `colab_runs/ce`.

Steps 3 and 4 are optional. Without them, the scripts skip the models and datasets whose inputs are missing; they discover inputs from the files present. The numbers then cover fewer models than `RESULTS_SUMMARY.md`. The prerequisite check below tells you what is missing.

**Runtime: CPU is enough.** Do not waste GPU quota on this notebook.

**Honest limits:**
- **RAM.** The code was developed and run on a 15 GB RAM / 4-core machine. Free Colab has about 12.7 GB RAM and 2 cores. The heaviest steps load Quora's and Touche's top-1000 rankings: each Quora run pickle is about 160 MB on disk and more than 1 GB in RAM. We did **not** measure their peak RAM in Colab.
  - If the session crashes with *"Your session crashed after using all available RAM"*, re-run the notebook. Some finished steps are cached: the W6 BM25 grid, the QPP feature files and the alpha grid are skipped when their files exist. The routers re-run, at about 5 min each.
  - If it crashes again at the same step, use a High-RAM runtime (Colab Pro) or run the two shell scripts on a 16 GB machine. Do not drop datasets: the verdicts assume all 8.
- **Time.** On 2 cores, expect roughly **2-4 h**. The W6 BM25 grid on Quora alone took 53 min on 4 cores. Keep the tab open; free Colab disconnects idle sessions.
- **Numbers.** The Colab run uses rankings that you regenerated yourself. MiniLM/hybrid runs then exist for all 8 datasets, not only the 5 local ones. Tables can therefore differ slightly from `RESULTS_SUMMARY.md` and `04_improvements/RESULTS.md`. The narrative `.md` files are not regenerated; only the CSV/PNG/summary tables are."""),
    md("## 0. Mount Drive, link the project layout, install packages"),
    code(MOUNT),
    code("!pip -q install PyStemmer wordfreq statsmodels\n!nproc; free -g"),
    code(GET_SETUP),
    code(LINK),
    md("""## 1. Code: copied from Drive `02_code/` into the local project layout
These are the exact project files (`src/` = Phase-1 modules, read-only; `02_code/phase2/` = Agent 3; `02_code/phase2/improvements/` = Agent 4). Their md5 sums are checked against the versions this notebook was built for. A mismatch means somebody edited the Drive copy: the run still works, but say so in the paper's reproducibility note."""),
]
import hashlib
COPY = []  # (drive path relative to 02_code, local path relative to project)
for rel in sorted(glob.glob(os.path.join(ROOT, "src/*.py"))):
    COPY.append(("src/" + os.path.basename(rel), "src/" + os.path.basename(rel)))
for f in A3:
    COPY.append(("phase2/" + os.path.basename(f), os.path.relpath(f, ROOT)))
for f in A4:
    COPY.append(("phase2/improvements/" + os.path.basename(f), os.path.relpath(f, ROOT)))
MD5 = {loc: hashlib.md5(open(os.path.join(ROOT, loc), "rb").read()).hexdigest() for _, loc in COPY}
cells.append(code(f"""import hashlib, os, shutil
C = '{DRIVE}/02_code'
COPY = {json.dumps(COPY, indent=0)}
MD5 = {json.dumps(MD5, indent=0)}
bad = []
for src, dst in COPY:
    os.makedirs(os.path.dirname('{P}/' + dst), exist_ok=True)
    shutil.copy(f'{{C}}/{{src}}', '{P}/' + dst)
    h = hashlib.md5(open('{P}/' + dst, 'rb').read()).hexdigest()
    if h != MD5[dst]:
        bad.append(dst)
print(len(COPY), 'files copied;', 'all md5 match the built version' if not bad else f'CHANGED on Drive: {{bad}}')"""))
cells += [
    md("## 2. Prerequisite check"),
    code(f"""import glob, os
os.chdir('{P}')
DS = ['nfcorpus', 'scifact', 'fiqa', 'scidocs', 'arguana', 'trec-covid', 'quora', 'webis-touche2020']
def have(pattern): return sorted(os.path.basename(p) for p in glob.glob(pattern))
missing = []
for d in DS:
    row = {{'data': os.path.exists(f'data/{{d}}/corpus.jsonl'),
           'perquery_lexical': os.path.exists(f'results/perquery_lexical_{{d}}.csv'),
           'features': os.path.exists(f'results/features_{{d}}.csv'),
           'bm25 run': os.path.exists(f'cache/runs/{{d}}__bm25.pkl'),
           'hybrid_cc_minilm run': os.path.exists(f'cache/runs/{{d}}__hybrid_cc_minilm.pkl'),
           'colab dense npz': len(have(f'phase2/03_experiments/colab_runs/run_*_{{d}}.npz')),
           'colab CE': os.path.exists(f'phase2/03_experiments/colab_runs/ce/ce_{{d}}.pkl')}}
    print(f'{{d:18s}}', '  '.join(f'{{k}}={{v}}' for k, v in row.items()))
    missing += [f'{{d}}: {{k}}' for k in ('data', 'perquery_lexical', 'features', 'bm25 run') if not row[k]]
if missing:
    print('\\nREQUIRED INPUTS MISSING (run 00_SETUP_drive and phase1_full_pipeline_drive first):')
    print('\\n'.join(missing))
else:
    print('\\nall required Phase-1 inputs present')"""),
    md("## 3. Agent 3 pipeline (`run_all.sh`)\nThe log of each step is in `phase2/03_experiments/logs/` on Drive. `COLAB_RUNS` points at `MyDrive/IR_Lexical_Gap/colab_runs`."),
    code(f"""%%bash
export COLAB_RUNS={DRIVE}/colab_runs
cd {P} && bash phase2/03_experiments/run_all.sh 2>&1 | tail -60"""),
    md("## 4. Agent 4 pipeline (`run_all4.sh`)"),
    code(f"""%%bash
export COLAB_RUNS={DRIVE}/colab_runs
cd {P} && bash phase2/04_improvements/run_all4.sh 2>&1 | tail -60"""),
    md("## 5. Results and figures (already saved in Drive)"),
    code(f"""import glob
from IPython.display import Image, Markdown, display
for f in ['phase2/03_experiments/results/summary_tables.md', 'phase2/04_improvements/results/summary4.md']:
    if os.path.exists(f):
        display(Markdown(open(f).read()))
for f in sorted(glob.glob('phase2/03_experiments/figures/*.png')) + sorted(glob.glob('phase2/04_improvements/figures/*.png')):
    print(f); display(Image(f, width=800))"""),
    md("## 6. Mirror the code into the Drive project and flush Drive"),
    code(f"""import shutil
D = '{DRIVE}/project'
for pat in ['src/*.py', 'phase2/03_experiments/*.py', 'phase2/03_experiments/*.sh',
            'phase2/04_improvements/*.py', 'phase2/04_improvements/*.sh']:
    for f in glob.glob(pat):
        os.makedirs(os.path.dirname(f'{{D}}/{{f}}'), exist_ok=True)
        shutil.copy(f, f'{{D}}/{{f}}')
from google.colab import drive
drive.flush_and_unmount()
print('Drive flushed. Outputs are in MyDrive/IR_Lexical_Gap/project/phase2/.')"""),
]
save(nb(cells, META_CPU), "phase2/colab/phase2_analysis.ipynb")

# ----------------------------------------------------------------------------- Phase-1 Drive runner
# The Phase-1 notebook (colab/lexical_gap_full_pipeline.ipynb) is left unchanged. This small runner mounts Drive,
# builds /content/proj with output folders linked into Drive, and executes the Phase-1 notebook's code cells
# (read from 01_notebooks on Drive) one by one in the same kernel, so all Phase-1 outputs land in Drive.
FP_DRIVE = f"{DRIVE}/01_notebooks/lexical_gap_full_pipeline.ipynb"
runner = nb([
    md(f"""# Phase 1 full pipeline, Drive mode

This notebook runs **`lexical_gap_full_pipeline.ipynb` unchanged**, with every output written to Google Drive. Open that notebook to read the code; it embeds every source file.

1. *Runtime > Change runtime type > T4 GPU*. Only the dense step needs the GPU.
2. *Runtime > Run all*. Allow Drive access when asked. It takes about 1-1.5 h. At the end, `report.pdf` and `results.zip` are also downloaded to your computer.

**How it works.**
- It builds `/content/proj` with `results/`, `cache/`, `figures/` and `report/` linked into `MyDrive/IR_Lexical_Gap/project/`, and `data/` linked to `MyDrive/IR_Lexical_Gap/data`. The data is filled by `00_SETUP_drive`; if it is there, the download step is skipped.
- It then executes the code cells of `01_notebooks/lexical_gap_full_pipeline.ipynb` one after another, in this kernel.
- `phase2_analysis` needs these outputs, in particular `project/results/*.csv` and `project/cache/runs/*.pkl` (about 2 GB).

**If it stops** (error or disconnect): fix the cause, set `START` in the loop cell (the second-to-last code cell) to the cell number printed in the error, and run that cell again. Re-run the mount and link cells first if the runtime was reset."""),
    code(MOUNT),
    code(GET_SETUP),
    code(LINK),
    code(f"""import json
from IPython import get_ipython
NB = '{FP_DRIVE}'
START = 0  # code-cell index to (re)start from
cells = [c for c in json.load(open(NB))['cells'] if c['cell_type'] == 'code']
for i in range(START, len(cells)):
    src = cells[i]['source']
    src = ''.join(src) if isinstance(src, list) else src
    print(f'\\n===== Phase-1 code cell {{i}}/{{len(cells) - 1}}: {{src.strip().splitlines()[0][:90]}}', flush=True)
    res = get_ipython().run_cell(src)
    if not res.success:
        raise RuntimeError(f'Phase-1 code cell {{i}} failed. Fix the cause, set START = {{i}} and run this cell again.')
print('\\nPhase 1 finished: outputs in MyDrive/IR_Lexical_Gap/project/{{results,cache,figures,report}}')"""),
    code(f"""import glob, shutil
for f in glob.glob('{P}/src/*.py'):
    shutil.copy(f, '{DRIVE}/project/src/')
shutil.copy('{P}/report/report.pdf', '{DRIVE}/05_reports_and_paper/report_regenerated.pdf')
from google.colab import drive
drive.flush_and_unmount()
print('Drive flushed. Next: phase2_dense_runs (T4), phase2_ce_rerank (T4), phase2_analysis (CPU).')"""),
], META)
save(runner, "phase2/colab/phase1_full_pipeline_drive.ipynb")
