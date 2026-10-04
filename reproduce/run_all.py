"""Runs the experiments step by step on your own machine.
Usage (from the agentic-ids folder, with the env activated):
    python reproduce\\run_all.py step1     (check environment)
    python reproduce\\run_all.py step2     (prepare CIC-DDoS2019 working set)
    python reproduce\\run_all.py step3     (CIC-IDS2017 flow level: blocked CV, ~6 min)
    python reproduce\\run_all.py step4     (CIC-IDS2017 flow level: 5 seeds, ~15 min)
    python reproduce\\run_all.py step5     (CIC-IDS2017 incident level + leave-one-type-out, ~2 min)
    python reproduce\\run_all.py step6     (CIC-DDoS2019 flow level: cross-day, unseen types, seen, ~18 min)
    python reproduce\\run_all.py step7     (CIC-DDoS2019 incident level + cross-dataset transfer, ~5 min)
    python reproduce\\run_all.py step7b    (cross-dataset transfer only, <1 min)
Times are for an Intel Core i5 12th gen, 16 GB RAM, CPU only.
Each step writes a log to reproduce\\logs\\ and results to results_local\\ or results_ddos2019_local\\.
"""
import subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; LOGS = HERE / "logs"; LOGS.mkdir(exist_ok=True)
STEPS = {
    "step1": [["check_env.py"]],
    "step2": [["prepare_ddos2019.py"]],
    "step3": [["flow_experiments.py", "F2"]],
    "step4": [["flow_experiments.py", "F1"]],
    "step5": [["incident_experiments.py"], ["extra_incident.py"]],
    "step6": [["ddos2019_flow.py", "X1"], ["ddos2019_flow.py", "L1"], ["ddos2019_flow.py", "S1"]],
    "step7": [["ddos2019_incidents.py"], ["ddos2019_summary.py"], ["crossdataset.py"]],
    "step7b": [["crossdataset.py"]],
}
step = sys.argv[1] if len(sys.argv) > 1 else ""
if step not in STEPS: print(__doc__); sys.exit(1)
for cmd in STEPS[step]:
    name = "_".join(c.replace(".py", "") for c in cmd); t0 = time.time()
    print(f"\n=== {step}: running {' '.join(cmd)}  (log: reproduce\\logs\\{name}.log)", flush=True)
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        p = subprocess.Popen([sys.executable, "-u"] + cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            print(line, end=""); log.write(line)
        p.wait()
    print(f"=== finished in {(time.time()-t0)/60:.1f} min, exit code {p.returncode}", flush=True)
    if p.returncode != 0:
        print("STOPPED: this step failed. Send me the log file above."); sys.exit(p.returncode)
nxt = {"step1": "step2", "step2": "step3", "step3": "step4", "step4": "step5", "step5": "step6", "step6": "step7"}.get(step)
print(f"\nDone. Next: python reproduce\\run_all.py {nxt}" if nxt else "\nALL STEPS DONE. Tell Claude so the paper can be rebuilt from your results.")
