"""Step 0 (CIC-DDoS2019 only): build the sampled working files from the original release.

Streams the CSVs out of CSV-01-12.zip and CSV-03-11.zip and keeps
  - every BENIGN flow, and
  - a seeded Bernoulli sample of attack flows, at a per-file rate targeting ~60,000 attack flows per file.
Raw lines are copied unchanged (header kept). The run is resumable: finished files get a .done marker.

Usage:
    python reproduce/sample_cicddos2019.py <folder containing CSV-01-12.zip and CSV-03-11.zip>
Output: data/raw/CICDDoS2019_sampled/*.csv.gz and sampling_log.json
"""
import zipfile, gzip, random, sys, os, json, time
if len(sys.argv) != 2: print(__doc__); sys.exit(1)
SRC = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "CICDDoS2019_sampled")
os.makedirs(OUT, exist_ok=True)
TARGET = 60000
BYTES_PER_ROW = 465.0
random.seed(2019)
log = {}
for zname in ["CSV-03-11.zip", "CSV-01-12.zip"]:
    zf = zipfile.ZipFile(os.path.join(SRC, zname))
    for info in zf.infolist():
        if not info.filename.endswith(".csv") or "lock" in info.filename: continue
        day, fname = info.filename.split("/")
        dst = os.path.join(OUT, f"{day}_{fname}.gz")
        if os.path.exists(dst + ".done"): continue
        rate = min(1.0, TARGET / (info.file_size / BYTES_PER_ROW))
        t0 = time.time(); n = kept = ben = 0; labels = {}
        with zf.open(info) as f, gzip.open(dst, "wt", compresslevel=3) as g:
            header = f.readline().decode("utf-8", "replace"); g.write(header)
            for raw in f:
                line = raw.decode("utf-8", "replace"); n += 1
                lab = line.rstrip("\r\n").rsplit(",", 1)[-1].strip()
                labels[lab] = labels.get(lab, 0) + 1
                if lab == "BENIGN":
                    g.write(line); kept += 1; ben += 1
                elif random.random() < rate:
                    g.write(line); kept += 1
        log[f"{day}/{fname}"] = dict(rows=n, kept=kept, benign=ben, rate=rate, labels=labels, sec=round(time.time() - t0))
        open(dst + ".done", "w").write(json.dumps(log[f"{day}/{fname}"]))
        print(day, fname, log[f"{day}/{fname}"], flush=True)
json.dump(log, open(os.path.join(OUT, "sampling_log.json"), "w"), indent=1)
print("ALL DONE", flush=True)
