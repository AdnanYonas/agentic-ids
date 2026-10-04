import json, numpy as np, pandas as pd
from incident_experiments import *
from common import dump
tau = json.load(open(OUT/"flow_F2_oof_summary.json"))["threshold_median"]
f = build_flows(); real = real_candidates(f, 1000, tau)
# leave-one-type-out: augmentation without type t, test on held-out type t
loto = {t: [] for t in TYPES}
for k in range(K):
    tr_r = real[real.fold != k]
    syn_tr = synth(f[f.fold != k], tau, 40, seed=100 + k)
    syn_te = synth(f[f.fold == k], tau, 20, seed=900 + k)
    for t in TYPES:
        aug = pd.concat([tr_r, syn_tr[syn_tr.type != t]])
        m = model().fit(aug[FEATURES], aug.label)
        te = syn_te[syn_te.type == t]
        loto[t] += list(((m.predict_proba(te[FEATURES])[:, 1] >= 0.5).astype(int) == te.label).astype(int))
loto = {t: float(np.mean(v)) for t, v in loto.items()}
print("LOTO", loto)
# where do false-positive alerts end up?
R = pd.read_csv(OUT/"incident_real_predictions.csv")
f2 = f.assign(win=f.row // 1000)
fpal = f2[(f2.p >= tau) & (f2.y == 0)].groupby(["win","port"]).size().rename("fp_alerts").reset_index()
M = R.merge(fpal, on=["win","port"], how="left").fillna({"fp_alerts":0})
esc = M.p_ca_aug >= 0.5
res = dict(loto=loto, fp_alerts_in_escalated_aug=int(M.loc[esc,"fp_alerts"].sum()),
           fp_alerts_in_rejected_aug=int(M.loc[~esc,"fp_alerts"].sum()),
           fp_alerts_in_escalated_real=int(M.loc[M.p_ca_real>=0.5,"fp_alerts"].sum()),
           tp_alerts_in_rejected_aug=int(M.loc[~esc,"n_alerts"].sum()-M.loc[~esc,"fp_alerts"].sum()),
           fn_flows_total=int(((f.p<tau)&(f.y==1)).sum()),
           neg_by_port=real[real.label==0].port.value_counts().to_dict(),
           neg_median_nflows=float(real[real.label==0].n_flows.median()),
           neg_median_nalerts=float(real[real.label==0].n_alerts.median()),
           pos_median_nflows=float(real[real.label==1].n_flows.median()),
           wrong_aug=M.loc[(M.p_ca_aug>=0.5)!=(M.label==1),["win","port","n_flows","n_alerts","alert_fraction","ddos_share","label","p_ca_aug","p_ca_real"]].to_dict(orient="records"),
           wrong_real=M.loc[(M.p_ca_real>=0.5)!=(M.label==1),["win","port","n_flows","n_alerts","alert_fraction","ddos_share","label","p_ca_aug","p_ca_real"]].to_dict(orient="records"))
dump(res, "incident_extra.json"); print(json.dumps(res, indent=1, default=str)[:3000])
