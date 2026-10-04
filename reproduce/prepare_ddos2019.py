"""Step 2: build the CIC-DDoS2019 working set (data/ddos2019/all_sampled.pkl) from the sampled CSVs."""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "CICDDoS2019_sampled"
DST = ROOT / "data" / "ddos2019"; DST.mkdir(parents=True, exist_ok=True)
files = sorted(SRC.glob("*.csv.gz"))
assert len(files) == 18, f"expected 18 sampled files, found {len(files)} in {SRC}"
dfs = []
for f in files:
    d = pd.read_csv(f, low_memory=False); d.columns = d.columns.str.strip()
    d["file"] = f.name[:-7]; dfs.append(d); print("read", f.name, len(d), flush=True)
df = pd.concat(dfs, ignore_index=True)
df["Label"] = df["Label"].str.strip(); df["day"] = df.file.str[:5]
df["ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df.to_pickle(DST / "all_sampled.pkl")
print("rows", len(df)); print(df.groupby(["day", "Label"]).size())
