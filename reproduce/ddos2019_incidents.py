"""CIC-DDoS2019 incident-level experiments with real timestamps.

Incident candidate = (destination IP, W-second window) group with >= 1 Monitoring-Agent alert.
I1: leave-one-attack-interval-out on 01-12 (scores and Correlation Agent both blind to the held-out type).
I2: cross-day: Correlation Agent trained on all 01-12 candidates (+synthetic), tested on 03-11.
"""
import json, sys
import numpy as np, pandas as pd
import shap
from incident_experiments import aggregate, synth, model, score, boot_ci, FEATURES, PRETTY, TYPES
from ddos2019_flow import load, O2
from common import dump

TAU = 0.5


def flows_table(df, X, pred, idxcol="idx", pcol="p_MonitoringAgent"):
    s = df.iloc[pred[idxcol].values]
    return pd.DataFrame(dict(dst=s["Destination IP"].values, ts=s.ts.values, port=s["Destination Port"].values,
                             y=pred.y.values, type=pred.type.values, p=pred[pcol].values,
                             fold=pred["fold"].values if "fold" in pred else -1,
                             dur=X.loc[s.index, "Flow Duration"].values, pps=X.loc[s.index, "Flow Packets/s"].values,
                             bps=X.loc[s.index, "Flow Bytes/s"].values, iat=X.loc[s.index, "Flow IAT Mean"].values)).fillna({"pps": 0, "bps": 0, "iat": 0, "dur": 0})


def candidates(f, W):
    t0 = f.ts.min()
    f = f.assign(win=((f.ts - t0).dt.total_seconds() // W).astype(int))
    rows = []
    for (dst, w), g in f.groupby(["dst", "win"], sort=False):
        if (g.p >= TAU).any():
            r = aggregate(g, TAU)
            att = g[g.y == 1]
            r.update(dst=dst, win=int(w), fold=int(g.fold.mode().iloc[0]),
                     itype=att.type.mode().iloc[0] if r["label"] == 1 else "benign", source="real",
                     port=int(g.port.mode().iloc[0]))
            rows.append(r)
    return pd.DataFrame(rows)


def evaluate(train_real, train_syn, test, rule_from=None):
    best = max(range(1, 51), key=lambda c: score(train_real.label.values, (train_real.n_alerts >= c).astype(float).values)["f1"])
    ca_real = model().fit(train_real[FEATURES], train_real.label)
    aug = pd.concat([train_real, train_syn]); ca_aug = model().fit(aug[FEATURES], aug.label)
    t = test.copy()
    t["p_escalate_all"] = 1.0; t["p_rule"] = (t.n_alerts >= best).astype(float); t["kappa"] = best
    t["p_ca_real"] = ca_real.predict_proba(t[FEATURES])[:, 1]; t["p_ca_aug"] = ca_aug.predict_proba(t[FEATURES])[:, 1]
    return t, ca_aug


METHODS = ["p_escalate_all", "p_rule", "p_ca_real", "p_ca_aug"]


def summarise(P, f, with_ci=False):
    out = {"n": len(P), "pos": int(P.label.sum()), "alerts": int(P.n_alerts.sum()), "fp_alert_flows": int(((f.p >= TAU) & (f.y == 0)).sum())}
    for m in METHODS:
        s = score(P.label.values, P[m].values)
        if with_ci: s["f1_ci"] = boot_ci(P.label.values, P[m].values, "f1")
        s["recall_by_type"] = {t: float((g[m] >= 0.5).mean()) for t, g in P[P.label == 1].groupby("itype")}
        s["false_incidents"] = int(((P[m] >= 0.5) & (P.label == 0)).sum())
        s["escalated"] = int((P[m] >= 0.5).sum())
        out[m] = s
    return out


def main():
    df, X = load()
    L1 = pd.read_parquet(O2 / "oof_L1.parquet"); X1p = pd.read_parquet(O2 / "pred_X1.parquet")
    order = json.load(open(O2 / "flow_L1_unseen.json"))["order"]
    f1 = flows_table(df, X, L1)
    f2 = flows_table(df, X, X1p)
    res = {}
    for W in [60, 10, 30, 120, 300]:
        C1 = candidates(f1, W)
        preds = []
        for k, typ in enumerate(order):
            tr, te = C1[C1.fold != k], C1[C1.fold == k]
            if len(te) == 0: continue
            syn_tr = synth(f1[f1.fold != k], TAU, 40, seed=500 + k)
            syn_te = synth(f1[f1.fold == k], TAU, 20, seed=900 + k)
            t, _ = evaluate(tr, syn_tr, pd.concat([te, syn_te])); t["held_out"] = typ; preds.append(t)
        PP = pd.concat(preds); P1 = PP[PP.source == "real"]; S1 = PP[PP.source == "synthetic"]
        # cross-day
        C2 = candidates(f2, W)
        syn_all = synth(f1, TAU, 40, seed=777)
        P2, ca_full = evaluate(C1, syn_all, C2)
        syn_sum = {m: dict(f1=score(S1.label.values, S1[m].values)["f1"],
                           acc_by_type={t: float(((g[m] >= 0.5).astype(int) == g.label).mean()) for t, g in S1.groupby("type")}) for m in METHODS}
        res[W] = dict(I1=summarise(P1, f1, with_ci=(W == 60)), I1_synthetic=syn_sum, I2=summarise(P2, f2, with_ci=(W == 60)),
                      n_cand_01_12=len(C1), n_cand_03_11=len(C2))
        print("W", W, "I1", {m: round(res[W]["I1"][m]["f1"], 4) for m in METHODS}, "FI", {m: res[W]["I1"][m]["false_incidents"] for m in METHODS},
              "| I2", {m: round(res[W]["I2"][m]["f1"], 4) for m in METHODS}, "FI", {m: res[W]["I2"][m]["false_incidents"] for m in METHODS}, flush=True)
        if W == 60:
            P1.to_csv(O2 / "incident_I1_predictions.csv", index=False); P2.to_csv(O2 / "incident_I2_predictions.csv", index=False)
            # SHAP for the full 01-12 augmented agent
            data = pd.concat([C1, syn_all]); sc, lr = ca_full[0], ca_full[1]
            Z = sc.transform(data[FEATURES]); sv = shap.LinearExplainer(lr, Z).shap_values(Z)
            real = (data.source == "real").values
            pd.DataFrame(dict(feature=FEATURES, pretty=[PRETTY[x] for x in FEATURES], mean_abs_shap_all=np.abs(sv).mean(0),
                              mean_abs_shap_real=np.abs(sv[real]).mean(0), coef_std=lr.coef_[0])).sort_values("mean_abs_shap_all", ascending=False).to_csv(O2 / "shap_global.csv", index=False)
        dump(res, "../results_ddos2019_local/incident_results.json")
    print("done")


if __name__ == "__main__":
    main()
