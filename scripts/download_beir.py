"""Downloads the 8 BEIR test collections into ./data (about 1 GB unzipped). Safe to re-run."""
import os, urllib.request, zipfile
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(os.path.dirname(HERE), "data"); os.makedirs(DATA, exist_ok=True)
for d in ["nfcorpus", "scifact", "fiqa", "scidocs", "arguana", "trec-covid", "quora", "webis-touche2020"]:
    if os.path.exists(os.path.join(DATA, d, "corpus.jsonl")):
        print("have", d); continue
    z = os.path.join(DATA, d + ".zip"); print("downloading", d, flush=True)
    urllib.request.urlretrieve(f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{d}.zip", z)
    zipfile.ZipFile(z).extractall(DATA); os.remove(z)
print("data ready")
