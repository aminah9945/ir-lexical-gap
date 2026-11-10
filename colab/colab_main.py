"""Colab GPU runner: BM25 (needed for hybrids) + MiniLM + bge-small dense + hybrids on all 8 datasets.
Produces results_colab.zip containing per-query CSVs to upload back for analysis."""
import os, sys, zipfile, subprocess, time, json, platform
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.environ.setdefault("DEVICE", "cuda")
os.environ.setdefault("BATCH", "256")
os.environ.setdefault("MAXLEN", "256")
import numpy as np, pandas as pd, torch
from common import DATASETS, DATA, RES, Index, bm25_matrix, query_matrix, search, to_run, evaluate_run, save_run, per_query_rows
import run_dense
from sentence_transformers import SentenceTransformer

assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 GPU"
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
for d in DATASETS:
    if not os.path.exists(os.path.join(DATA, d, "corpus.jsonl")):
        print("downloading", d, flush=True)
        subprocess.run(f"cd {DATA} && wget -q {URL.format(d)} && unzip -q -o {d}.zip && rm {d}.zip", shell=True, check=True)

check = []
for d in DATASETS:  # BM25 run, identical code/params to the main pipeline
    t = time.time()
    idx = Index(d)
    C, vocab, an = idx.variant(True, True)
    qids = list(idx.qrels)
    Q = query_matrix([idx.queries[q] for q in qids], vocab, an)
    W, _ = bm25_matrix(C)
    run = to_run(search(W, Q), qids, idx.doc_ids)
    save_run(run, d, "bm25")
    m = np.mean([v["ndcg@10"] for v in evaluate_run(run, idx.qrels).values()])
    check.append(dict(dataset=d, bm25_ndcg10=m))
    print(f"bm25 {d}: {m:.4f} ({time.time()-t:.0f}s)", flush=True)
    del idx, C, W, Q
pd.DataFrame(check).to_csv(os.path.join(RES, "colab_bm25_check.csv"), index=False)

for mkey in ["minilm", "bge"]:
    model = SentenceTransformer(run_dense.MODELS[mkey][0], device="cuda")
    model.max_seq_length = int(os.environ["MAXLEN"])
    for d in DATASETS:
        t = time.time()
        run_dense.main(mkey, d, model)
        print(f"  {mkey} {d} done in {time.time()-t:.0f}s", flush=True)

with open(os.path.join(RES, "colab_env.json"), "w") as f:
    json.dump(dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, python=platform.python_version(),
                   maxlen=os.environ["MAXLEN"], batch=os.environ["BATCH"]), f, indent=1)
out = "/content/results_colab.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in os.listdir(RES):
        if fn.startswith(("perquery_minilm", "perquery_bge", "colab_")):
            z.write(os.path.join(RES, fn), fn)
print("WROTE", out)
