"""Cross-dataset transfer of the Correlation Agent (detector stays local to each dataset)."""
import json, numpy as np, pandas as pd
import incident_experiments as IE
from incident_experiments import synth, score, FEATURES
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import OUT, dump
RATIO = ["alert_fraction", "prob_mean_alerts", "prob_mean_all", "critical_fraction"]
AGNOSTIC = ["log_n_flows", "log_n_alerts", "alert_fraction", "prob_mean_alerts", "prob_mean_all", "critical_fraction"]
O2 = OUT.parent / "results_ddos2019_local"
# CIC-IDS2017
tau17 = json.load(open(OUT / "flow_F2_oof_summary.json"))["threshold_median"]
f17 = IE.build_flows(); real17 = pd.read_csv(OUT / "incident_real_candidates.csv")
syn17 = synth(f17, tau17, 40, seed=7); syn17_test = synth(f17, tau17, 20, seed=4242)
# CIC-DDoS2019
c19a = pd.read_csv(O2 / "incident_I1_predictions.csv"); c19b = pd.read_csv(O2 / "incident_I2_predictions.csv")
from ddos2019_flow import load
from ddos2019_incidents import flows_table
df, X = load(); L1 = pd.read_parquet(O2 / "oof_L1.parquet"); f19 = flows_table(df, X, L1)
syn19 = synth(f19, 0.5, 40, seed=777); syn19_test = synth(f19, 0.5, 20, seed=4243)

def fit(train, feats):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")).fit(train[feats], train.label)

def ev(m, test, feats):
    p = m.predict_proba(test[feats])[:, 1]; s = score(test.label.values, p)
    return dict(f1=s["f1"], precision=s["precision"], recall=s["recall"], false_incidents=int(((p >= .5) & (test.label == 0)).sum()),
                missed=int(((p < .5) & (test.label == 1)).sum()), n=len(test), pos=int(test.label.sum()))

def esc_all(test):
    s = score(test.label.values, np.ones(len(test)))
    return dict(f1=s["f1"], precision=s["precision"], recall=1.0, false_incidents=int((test.label == 0).sum()), missed=0, n=len(test), pos=int(test.label.sum()))

res = {}
tr17 = pd.concat([real17, syn17]); tr19 = pd.concat([c19a[real17.columns.intersection(c19a.columns)], syn19])
tests = {"CIC-DDoS2019 01-12 (real)": c19a, "CIC-DDoS2019 03-11 (real)": c19b, "CIC-DDoS2019 synthetic": syn19_test,
         "CIC-IDS2017 (real)": real17, "CIC-IDS2017 synthetic": syn17_test}
for fname, feats in [("all10", FEATURES), ("agnostic6", AGNOSTIC), ("ratio4", RATIO)]:
    m17, m19 = fit(tr17, feats), fit(tr19, feats)
    for tname, t in tests.items():
        r = {"escalate_all": esc_all(t)}
        if "DDoS2019" in tname: r["transfer_from_CIC-IDS2017"] = ev(m17, t, feats)
        else: r["transfer_from_CIC-DDoS2019"] = ev(m19, t, feats)
        res.setdefault(fname, {})[tname] = r
        print(fname, tname, {k: (round(v["f1"], 3), v["false_incidents"], v["missed"]) for k, v in r.items()}, flush=True)

# how much attack traffic sits in the candidates the transferred agent (CIC-IDS2017 -> CIC-DDoS2019, agnostic6) misses
m17 = fit(tr17, AGNOSTIC); miss = {}
for day, t in (("01-12", c19a), ("03-11", c19b)):
    p = m17.predict_proba(t[AGNOSTIC])[:, 1]; mm = (p < .5) & (t.label == 1); att = t.label == 1
    tot = int(t.loc[att, "n_alerts"].sum()); inm = int(t.loc[mm, "n_alerts"].sum())
    miss[day] = dict(missed=int(mm.sum()), missed_max_flows=int(t.loc[mm, "n_flows"].max()) if mm.any() else 0,
                     attack_alerts_total=tot, attack_alerts_in_missed=inm, alert_coverage=1 - inm / tot)
    print("miss analysis", day, miss[day], flush=True)
res["agnostic6_miss_analysis"] = miss
dump(res, "../results_ddos2019_local/crossdataset_transfer.json")
