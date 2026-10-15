"""Robustness experiments added before submission (step 8 and step 9).

R1 (CIC-IDS2017): repeat the out-of-fold Monitoring Agent (blocked 5-fold CV, E2) with five training
    seeds and rebuild the incident-level evaluation (E3, W = 1000) on each seed's scores. The data
    splits, validation blocks and synthetic-incident seeds are unchanged; only the detector's
    initialisation and mini-batch order change. Seed set "42" is the run reported in the main tables
    (its scores are reused from oof_flow_probs.parquet).
R2 (CIC-DDoS2019): the same for E4 (leave-one-attack-interval-out) and E5 (cross-day), W = 60 s.

Both parts also evaluate a non-linear Correlation Agent (LightGBM on the same ten features, trained
on real candidates only and with synthetic augmentation) next to the linear agents and baselines.
"""
import sys, json, time
import numpy as np, pandas as pd
import lightgbm as lgb
import common
from common import load_clean, fit_transform, train_mlp, predict_logits, fit_temperature, softmax_p1, select_threshold, metrics, OUT, dump, ROOT
import incident_experiments as IE
from incident_experiments import FEATURES, synth, score, model as lr_model, real_candidates, aggregate

SEED_SETS = [42, 1000, 2000, 3000, 4000]
QUICK = "--quick" in sys.argv          # smoke test only: two seed sets, few epochs
if QUICK:
    SEED_SETS = [42, 1000]; common.MAX_EPOCHS = 2
if "--base" in sys.argv:              # sanity check: reported run only (no training)
    SEED_SETS = [42]


def lgb_model():
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                              class_weight="balanced", random_state=0, verbose=-1, n_jobs=1)


def agents(train_real, train_syn):
    """Fit the four learned Correlation Agents on one training fold."""
    aug = pd.concat([train_real, train_syn])
    return {"p_ca_real": lr_model().fit(train_real[FEATURES], train_real.label),
            "p_ca_aug": lr_model().fit(aug[FEATURES], aug.label),
            "p_lgb_real": lgb_model().fit(train_real[FEATURES], train_real.label),
            "p_lgb_aug": lgb_model().fit(aug[FEATURES], aug.label)}


def apply(models, kappa, test):
    t = test.copy()
    t["p_escalate_all"] = 1.0
    t["p_rule"] = (t.n_alerts >= kappa).astype(float)
    for name, m in models.items():
        t[name] = m.predict_proba(t[FEATURES])[:, 1]
    return t


METHODS = ["p_escalate_all", "p_rule", "p_ca_real", "p_ca_aug", "p_lgb_real", "p_lgb_aug"]


def tune_kappa(tr):
    return max(range(1, 51), key=lambda c: score(tr.label.values, (tr.n_alerts >= c).astype(float).values)["f1"])


def summary(P, S=None):
    out = {}
    for m in METHODS:
        s = score(P.label.values, P[m].values)
        d = dict(f1=s["f1"], precision=s["precision"], recall=s["recall"], false_incidents=s["fp"], missed=s["fn"],
                 escalated=int((P[m] >= 0.5).sum()))
        if S is not None and len(S):
            d["syn_f1"] = score(S.label.values, S[m].values)["f1"]
            d["syn_acc_by_type"] = {t: float(((g[m] >= 0.5).astype(int) == g.label).mean()) for t, g in S.groupby("type")}
        out[m] = d
    return out


# ---------------------------------------------------------------- R1: CIC-IDS2017
def oof_for_seed(s, df, X, y):
    """Monitoring Agent only, E2 protocol; fold k trains with seed s + k (s = 42 reproduces the main run)."""
    K = 5; blocks = np.array_split(np.arange(len(y)), K)
    oof = np.full(len(y), np.nan); fold = np.zeros(len(y), int); thrs = []
    for k, te in enumerate(blocks):
        rest = np.setdiff1d(np.arange(len(y)), te)
        rng = np.random.default_rng(42 + k)                       # same validation block as E2
        nva = int(0.1 * len(rest)); start = rng.integers(0, len(rest) - nva)
        va = rest[start:start + nva]; tr = np.setdiff1d(rest, va)
        Xtr, Xva, Xte = fit_transform(X.values[tr], X.values[va], X.values[te])
        m, _ = train_mlp(Xtr, y[tr], Xva, y[va], s + k)
        lv = predict_logits(m, Xva); T = fit_temperature(lv, y[va])
        thrs.append(select_threshold(y[va], softmax_p1(lv, T)))
        oof[te] = softmax_p1(predict_logits(m, Xte), T); fold[te] = k
    return oof, fold, float(np.median(thrs))


def r1():
    df, X, y, feats, _ = load_clean()
    base = pd.read_parquet(OUT / "oof_flow_probs.parquet")
    tau42 = json.load(open(OUT / "flow_F2_oof_summary.json"))["threshold_median"]
    res = {}
    for s in SEED_SETS:
        t0 = time.time()
        if s == 42 and not QUICK:
            oof, fold, tau = base.ddos_probability.values, base.fold.values, tau42
        else:
            oof, fold, tau = oof_for_seed(s, df, X, y)
        fm = metrics(y, oof, tau)
        f = IE.build_flows().assign(p=oof, fold=fold)
        real = real_candidates(f, 1000, tau)
        preds, spreds = [], []
        for k in range(5):
            tr, te = real[real.fold != k], real[real.fold == k]
            syn_tr = synth(f[f.fold != k], tau, 40, seed=100 + k); syn_te = synth(f[f.fold == k], tau, 20, seed=900 + k)
            ms, kap = agents(tr, syn_tr), tune_kappa(tr)
            if len(te): preds.append(apply(ms, kap, te))
            spreds.append(apply(ms, kap, syn_te))
        P, S = pd.concat(preds), pd.concat(spreds)
        res[str(s)] = dict(tau=tau, flow=dict(f1=fm["f1"], fp=fm["fp"], fn=fm["fn"]), n_candidates=len(real),
                           n_pos=int(real.label.sum()), alerts=int((oof >= tau).sum()), incident=summary(P, S))
        r = res[str(s)]["incident"]
        print(f"seed set {s}: flow F1 {fm['f1']:.4f} FP {fm['fp']} | candidates {len(real)} ({int(real.label.sum())} DDoS) | real F1 "
              + " ".join(f"{m[2:]}={r[m]['f1']:.3f}/{r[m]['false_incidents']}" for m in METHODS)
              + f" | syn F1 aug={r['p_ca_aug']['syn_f1']:.3f} real={r['p_ca_real']['syn_f1']:.3f} | {time.time()-t0:.0f}s", flush=True)
        dump(res, "robustness_cicids2017.json")


# ---------------------------------------------------------------- R2: CIC-DDoS2019
def r2():
    from sklearn.model_selection import train_test_split
    import ddos2019_flow as DF
    from ddos2019_flow import load, intervals, cap_idx, O2
    from ddos2019_incidents import flows_table, candidates
    DF.common.MAX_EPOCHS, DF.common.PATIENCE = (2 if QUICK else 30), 5
    df, X = load()
    d1 = df.day.values == "01-12"; i1 = np.where(d1)[0]; i2 = np.where(~d1)[0]
    D1 = df.iloc[i1].reset_index(drop=True); X1 = X.values[i1]; y1 = D1.y.values; t1 = D1.type.values
    D2 = df.iloc[i2].reset_index(drop=True); y2 = D2.y.values; t2 = D2.type.values
    fold, order = intervals(D1)

    def ma(Xtr, ytr, Xte, split_seed, train_seed):
        itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.1, random_state=split_seed, stratify=ytr)
        A, V, T = fit_transform(Xtr[itr], Xtr[iva], Xte)
        m, _ = train_mlp(A, ytr[itr], V, ytr[iva], train_seed)
        Tt = fit_temperature(predict_logits(m, V), ytr[iva])
        return softmax_p1(predict_logits(m, T), Tt)

    baseL1 = pd.read_parquet(O2 / "oof_L1.parquet") if (O2 / "oof_L1.parquet").exists() else pd.read_pickle(O2 / "oof_L1.pkl")
    baseX1 = pd.read_parquet(O2 / "pred_X1.parquet") if (O2 / "pred_X1.parquet").exists() else pd.read_pickle(O2 / "pred_X1.pkl")
    res = {}
    for s in SEED_SETS:
        t0 = time.time()
        if s == 42 and not QUICK:
            pL1, pX1 = baseL1.p_MonitoringAgent.values, baseX1.p_MonitoringAgent.values
        else:
            rng = np.random.default_rng(0); pL1 = np.full(len(D1), np.nan)
            for k, typ in enumerate(order):                     # same training sets as E4 (same rng sequence)
                te = np.where(fold == k)[0]; tr = cap_idx(np.where(fold != k)[0], y1, rng)
                pL1[te] = ma(X1[tr], y1[tr], X1[te], 100 + k, s + 100 + k)
            rng = np.random.default_rng(0); tr = cap_idx(np.arange(len(D1)), y1, rng)
            pX1 = ma(X1[tr], y1[tr], X.values[i2], 300, s + 300)
        L1 = pd.DataFrame(dict(idx=i1, fold=fold, y=y1, type=t1, p_MonitoringAgent=pL1))
        X1p = pd.DataFrame(dict(idx=i2, y=y2, type=t2, p_MonitoringAgent=pX1))
        f1, f2 = flows_table(df, X, L1), flows_table(df, X, X1p)
        C1, C2 = candidates(f1, 60), candidates(f2, 60)
        preds = []
        for k, typ in enumerate(order):
            tr, te = C1[C1.fold != k], C1[C1.fold == k]
            if len(te) == 0: continue
            ms = agents(tr, synth(f1[f1.fold != k], 0.5, 40, seed=500 + k))
            preds.append(apply(ms, tune_kappa(tr), te))
        P1 = pd.concat(preds)
        P2 = apply(agents(C1, synth(f1, 0.5, 40, seed=777)), tune_kappa(C1), C2)
        res[str(s)] = dict(flow=dict(E4_benign_fpr=float((pL1[y1 == 0] >= 0.5).mean()), E4_recall=float((pL1[y1 == 1] >= 0.5).mean()),
                                     E5_benign_fpr=float((pX1[y2 == 0] >= 0.5).mean()), E5_recall=float((pX1[y2 == 1] >= 0.5).mean())),
                           E4=dict(n=len(P1), pos=int(P1.label.sum()), **summary(P1)),
                           E5=dict(n=len(P2), pos=int(P2.label.sum()), **summary(P2)))
        a, b = res[str(s)]["E4"], res[str(s)]["E5"]
        print(f"seed set {s}: E4 false incidents " + " ".join(f"{m[2:]}={a[m]['false_incidents']}" for m in METHODS)
              + " | E5 false incidents " + " ".join(f"{m[2:]}={b[m]['false_incidents']}" for m in METHODS)
              + f" | E4 F1 aug={a['p_ca_aug']['f1']:.3f} lgb_aug={a['p_lgb_aug']['f1']:.3f} | {time.time()-t0:.0f}s", flush=True)
        dump(res, "../results_ddos2019_local/robustness_cicddos2019.json")


if __name__ == "__main__":
    {"R1": r1, "R2": r2}[sys.argv[1]]()
    print("done")
