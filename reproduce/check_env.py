"""Step 1: check that every package is installed and record the machine used for the runs."""
import json, platform, os, sys, datetime
from pathlib import Path
info = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
        "os": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()}
missing = []
for mod in ["numpy", "pandas", "sklearn", "lightgbm", "torch", "shap", "pyarrow", "joblib"]:
    try:
        m = __import__(mod); info[mod] = getattr(m, "__version__", "ok")
    except Exception as e:
        missing.append(mod)
try:
    import torch
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available(): info["gpu"] = torch.cuda.get_device_name(0)
except Exception: pass
try:
    import ctypes
    class M(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong)] + [(f"x{i}", ctypes.c_ulonglong) for i in range(6)]
    m = M(); m.dwLength = ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    info["ram_gb"] = round(m.ullTotalPhys / 2**30, 1)
except Exception: pass
out = Path(__file__).resolve().parents[1] / "results_local"; out.mkdir(exist_ok=True)
json.dump(info, open(out / "run_environment.json", "w"), indent=1)
print(json.dumps(info, indent=1))
if missing:
    print("\nMISSING PACKAGES:", missing, "\nInstall with:  pip install " + " ".join("scikit-learn" if m == "sklearn" else m for m in missing)); sys.exit(1)
print("\nEnvironment OK. Next: python reproduce\\run_all.py step2")
