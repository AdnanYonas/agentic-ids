import json, pandas as pd, numpy as np
from pathlib import Path
O = Path(__file__).resolve().parents[1] / "results_ddos2019_local"
L1 = json.load(open(O/"flow_L1_unseen.json")); P = pd.read_csv(O/"incident_I1_predictions.csv")
rows = []
for typ in L1["order"]:
    f = L1["folds"][typ]; g = P[P.held_out == typ]
    pos = g[g.label == 1]; neg = g[g.label == 0]
    rows.append(dict(type=typ, test_flows=f["n_test"], test_benign=f["n_test_benign"],
        flow_recall_LR=f["LR"]["recall_by_type"].get(typ), flow_recall_LGBM=f["LightGBM"]["recall_by_type"].get(typ),
        flow_recall_MA=f["MonitoringAgent"]["recall_by_type"].get(typ), benign_fpr_MA=f["MonitoringAgent"]["benign_fpr"],
        fp_alert_flows=int(round(f["MonitoringAgent"]["benign_fpr"] * f["n_test_benign"])),
        cand=len(g), pos=len(pos), inc_recall_all=float((pos.p_escalate_all >= .5).mean()) if len(pos) else np.nan,
        inc_recall_CA=float((pos.p_ca_aug >= .5).mean()) if len(pos) else np.nan,
        false_inc_all=int((neg.p_escalate_all >= .5).sum()), false_inc_CA=int((neg.p_ca_aug >= .5).sum())))
T = pd.DataFrame(rows); T.to_csv(O/"unseen_type_table.csv", index=False); print(T.round(4).to_string())
