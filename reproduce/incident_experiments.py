"""Incident-level experiments (Correlation Agent), leakage-free.

Real incident candidates: flows are cut into windows of W consecutive records (record order is
used as a temporal proxy because the MachineLearningCSV release has no timestamps); inside each
window, flows are grouped by destination port and a (window, port) group becomes an incident
candidate if it contains >= 1 Monitoring-Agent alert (out-of-fold probability >= tau).
Label: ground-truth DDoS share of the group >= 0.5 (offline evaluation only).

Protocol: 5-fold blocked CV aligned with the flow-level blocks (so every probability is
out-of-fold). In fold k, synthetic training incidents are generated only from flows of the
training blocks, synthetic test incidents only from flows of block k (different seed).
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, roc_auc_score,
                             average_precision_score, confusion_matrix)
import shap
from common import load_clean, OUT, dump

FEATURES = ["log_n_flows", "log_n_alerts", "alert_fraction", "prob_mean_alerts", "prob_mean_all",
            "critical_fraction", "log_duration_mean", "log_pkts_per_s_mean", "log_bytes_per_s_mean",
            "log_iat_mean_mean"]
PRETTY = {"log_n_flows": "Flows in group (log)", "log_n_alerts": "Alerts in group (log)",
          "alert_fraction": "Alert fraction", "prob_mean_alerts": "Mean alert probability",
          "prob_mean_all": "Mean flow probability", "critical_fraction": "Critical-tier fraction",
          "log_duration_mean": "Mean flow duration (log)", "log_pkts_per_s_mean": "Mean packets/s (log)",
          "log_bytes_per_s_mean": "Mean bytes/s (log)", "log_iat_mean_mean": "Mean flow IAT (log)"}
CRIT = 0.95
K = 5


def aggregate(g, tau):
    """g: DataFrame of flows in one group with columns p, y, dur, pps, bps, iat."""
    a = g[g.p >= tau]
    n, na = len(g), len(a)
    lm = lambda s: float(np.log1p(np.clip(s.astype(float), 0, None)).mean())
    return dict(log_n_flows=np.log1p(n), log_n_alerts=np.log1p(na), alert_fraction=na / n,
                prob_mean_alerts=float(a.p.mean()) if na else 0.0, prob_mean_all=float(g.p.mean()),
                critical_fraction=float((a.p >= CRIT).mean()) if na else 0.0,
                log_duration_mean=lm(g["dur"]), log_pkts_per_s_mean=lm(g["pps"]),
                log_bytes_per_s_mean=lm(g["bps"]), log_iat_mean_mean=lm(g["iat"]),
                n_flows=n, n_alerts=na, ddos_share=float(g.y.mean()), label=int(g.y.mean() >= 0.5))


def build_flows():
    df, X, y, feats, _ = load_clean()
    oof = pd.read_parquet(OUT / "oof_flow_probs.parquet")
    f = pd.DataFrame(dict(row=np.arange(len(y)), port=df["Destination Port"].values, y=y,
                          p=oof.ddos_probability.values, fold=oof.fold.values,
                          dur=X["Flow Duration"].values, pps=X["Flow Packets/s"].values,
                          bps=X["Flow Bytes/s"].values, iat=X["Flow IAT Mean"].values))
    return f.fillna({"pps": 0, "bps": 0})


def real_candidates(f, W, tau):
    f = f.assign(win=f.row // W)
    rows = []
    for (w, port), g in f.groupby(["win", "port"], sort=True):
        if (g.p >= tau).any():
            r = aggregate(g, tau); r.update(win=int(w), port=int(port),
                                            fold=int(g.fold.mode().iloc[0]), source="real")
            rows.append(r)
    return pd.DataFrame(rows)


TYPES = {  # name: (n range, DDoS-share range, severity, benign port choices)
    "high_ddos": ((300, 1000), (0.90, 1.00), "high", [80]),
    "medium_ddos": ((50, 300), (0.60, 0.90), "medium", [80]),
    "lowrate_ddos": ((5, 50), (0.55, 1.00), "low", [80]),
    "benign_burst": ((300, 1000), (0.00, 0.05), "low", [80, 443, 53]),
    "ambiguous_mixed": ((20, 300), (0.10, 0.45), "low", [80, 443]),
}


def synth(f_pool, tau, n_per_type, seed):
    """Composition-controlled resampling of real flows (with their out-of-fold probabilities)."""
    rng = np.random.default_rng(seed)
    dd = f_pool[f_pool.y == 1]
    fp_pool = f_pool[(f_pool.y == 0) & (f_pool.p >= tau)]
    if len(fp_pool) == 0:
        fp_pool = f_pool[f_pool.y == 0].nlargest(50, "p")
    rows = []
    for t, ((nlo, nhi), (slo, shi), sev, ports) in TYPES.items():
        for _ in range(n_per_type):
            n = int(rng.integers(nlo, nhi + 1)); share = rng.uniform(slo, shi)
            nd = int(round(share * n)); port = int(rng.choice(ports))
            ben = f_pool[(f_pool.y == 0) & (f_pool.port == port)]
            if len(ben) == 0:
                ben = f_pool[f_pool.y == 0]
            parts = [dd.sample(nd, replace=True, random_state=int(rng.integers(1e9)))] if nd else []
            parts.append(ben.sample(n - nd, replace=True, random_state=int(rng.integers(1e9))))
            g = pd.concat(parts)
            if not (g.p >= tau).any():  # a candidate must contain >= 1 alert: inject 1-3 real false positives
                k = int(rng.integers(1, 4))
                g = pd.concat([g.iloc[k:], fp_pool.sample(k, replace=True, random_state=int(rng.integers(1e9)))])
            r = aggregate(g, tau); r.update(type=t, severity=sev, port=port, source="synthetic")
            rows.append(r)
    return pd.DataFrame(rows)


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))


def score(y, p, thr=0.5):
    yp = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yp, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    both = len(set(y)) > 1
    return dict(n=len(y), pos=int(sum(y)), accuracy=accuracy_score(y, yp), precision=pr, recall=rc, f1=f1,
                roc_auc=roc_auc_score(y, p) if both else np.nan,
                pr_auc=average_precision_score(y, p) if both else np.nan,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def boot_ci(y, p, key, B=2000, seed=0):
    rng = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p); vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2: continue
        vals.append(score(y[i], p[i])[key])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def run(W, tau, n_train=40, n_test=20, save=False):
    f = build_flows()
    real = real_candidates(f, W, tau)
    preds, syn_preds = [], []
    for k in range(K):
        tr_r, te_r = real[real.fold != k], real[real.fold == k]
        syn_tr = synth(f[f.fold != k], tau, n_train, seed=100 + k)
        syn_te = synth(f[f.fold == k], tau, n_test, seed=900 + k)
        # baseline rule: escalate if n_alerts >= kappa (kappa tuned for F1 on training real candidates)
        best = max(range(1, 51), key=lambda c: score(tr_r.label.values, (tr_r.n_alerts >= c).astype(float).values)["f1"])
        ca_real = model().fit(tr_r[FEATURES], tr_r.label)
        aug = pd.concat([tr_r, syn_tr])
        ca_aug = model().fit(aug[FEATURES], aug.label)
        for name, df_te, store in (("real", te_r, preds), ("synthetic", syn_te, syn_preds)):
            if len(df_te) == 0: continue
            d = df_te.copy(); d["test_fold"] = k
            d["p_escalate_all"] = 1.0
            d["p_rule"] = (d.n_alerts >= best).astype(float); d["rule_kappa"] = best
            d["p_ca_real"] = ca_real.predict_proba(d[FEATURES])[:, 1]
            d["p_ca_aug"] = ca_aug.predict_proba(d[FEATURES])[:, 1]
            store.append(d)
    R, S = pd.concat(preds), pd.concat(syn_preds)
    out = dict(W=W, tau=tau, n_real=len(real), real_pos=int(real.label.sum()),
               total_alerts=int((f.p >= tau).sum()), real={}, synthetic={}, synthetic_by_type={})
    for m in ["p_escalate_all", "p_rule", "p_ca_real", "p_ca_aug"]:
        out["real"][m] = score(R.label.values, R[m].values)
        out["synthetic"][m] = score(S.label.values, S[m].values)
        out["synthetic_by_type"][m] = {t: float(((g[m] >= 0.5).astype(int) == g.label).mean())
                                       for t, g in S.groupby("type")}
        if save:
            out["real"][m]["f1_ci"] = boot_ci(R.label.values, R[m].values, "f1")
            out["synthetic"][m]["f1_ci"] = boot_ci(S.label.values, S[m].values, "f1")
    # alert-to-incident compression for the augmented agent
    esc = R[R.p_ca_aug >= 0.5]
    out["compression"] = dict(alerts=int(R.n_alerts.sum()), candidates=len(R), escalated=len(esc),
                              escalated_true=int(esc.label.sum()),
                              alerts_in_rejected=int(R[R.p_ca_aug < 0.5].n_alerts.sum()),
                              fp_alerts_total=int(((f.p >= tau) & (f.y == 0)).sum()))
    if save:
        R.to_csv(OUT / "incident_real_predictions.csv", index=False)
        S.to_csv(OUT / "incident_synthetic_predictions.csv", index=False)
        real.to_csv(OUT / "incident_real_candidates.csv", index=False)
    return out, f, real


def explain(f, real, tau):
    syn = synth(f, tau, 40, seed=7)
    data = pd.concat([real, syn])
    m = model().fit(data[FEATURES], data.label)
    sc, lr = m[0], m[1]
    Z = sc.transform(data[FEATURES])
    expl = shap.LinearExplainer(lr, Z)
    sv = expl.shap_values(Z)
    is_real = (data.source == "real").values
    glob = pd.DataFrame(dict(feature=FEATURES, pretty=[PRETTY[x] for x in FEATURES],
                             mean_abs_shap_all=np.abs(sv).mean(0), mean_abs_shap_real=np.abs(sv[is_real]).mean(0),
                             mean_abs_shap_synth=np.abs(sv[~is_real]).mean(0), coef_std=lr.coef_[0]))
    glob = glob.sort_values("mean_abs_shap_all", ascending=False)
    glob.to_csv(OUT / "shap_global.csv", index=False)
    data = data.reset_index(drop=True).assign(p=m.predict_proba(data[FEATURES])[:, 1])
    picks = {}
    rr = data[data.source == "real"]
    picks["Real DDoS incident (port 80)"] = rr[(rr.label == 1)].sort_values("n_alerts").index[len(rr[rr.label == 1]) // 2]
    neg = rr[(rr.label == 0) & (rr.ddos_share == 0)]
    picks["Real false-alarm cluster"] = neg.sort_values(["n_alerts", "n_flows"]).index[-1] if len(neg) else None
    ss = data[data.source == "synthetic"]
    picks["Synthetic low-rate DDoS"] = ss[ss.type == "lowrate_ddos"].index[0]
    picks["Synthetic benign burst"] = ss[ss.type == "benign_burst"].index[0]
    local = {name: dict(index=int(i), p=float(data.loc[i, "p"]), label=int(data.loc[i, "label"]),
                        n_flows=int(data.loc[i, "n_flows"]), n_alerts=int(data.loc[i, "n_alerts"]),
                        port=int(data.loc[i, "port"]), base=float(expl.expected_value),
                        shap=dict(zip(FEATURES, sv[i].tolist())))
             for name, i in picks.items() if i is not None}
    dump(dict(global_=glob.to_dict(orient="records"), local=local, n_explained=len(data)), "shap_results.json")


if __name__ == "__main__":
    import json
    from common import load_clean
    summ = json.load(open(OUT / "flow_F2_oof_summary.json"))
    tau = summ["threshold_median"]
    main_out, f, real = run(1000, tau, save=True)
    dump(main_out, "incident_main_W1000.json")
    print(json.dumps({k: main_out[k] for k in ["n_real", "real_pos", "total_alerts", "compression"]}, indent=1))
    for m, v in main_out["real"].items(): print("real", m, {k: round(x, 4) if isinstance(x, float) else x for k, x in v.items()})
    for m, v in main_out["synthetic"].items(): print("syn ", m, {k: round(x, 4) if isinstance(x, float) else x for k, x in v.items()})
    print(json.dumps(main_out["synthetic_by_type"], indent=1))
    sens = {}
    for W in (250, 500, 2000, 5000):
        o, _, _ = run(W, tau)
        sens[W] = dict(n_real=o["n_real"], real_pos=o["real_pos"],
                       **{m: dict(f1=o["real"][m]["f1"], precision=o["real"][m]["precision"], recall=o["real"][m]["recall"],
                                  syn_f1=o["synthetic"][m]["f1"]) for m in o["real"]})
        print("W", W, sens[W], flush=True)
    dump(sens, "incident_sensitivity_W.json")
    explain(f, real, tau)
    print("done")
