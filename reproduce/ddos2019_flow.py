"""CIC-DDoS2019 flow-level experiments.

L1 (unseen attack types): on the training day (01-12) the attacks run one after another, so each
   attack interval is a natural fold. Leave-one-interval-out: the detectors are trained on all
   other intervals and scored on the held-out interval, whose attack type they have never seen.
   The Monitoring Agent's held-out scores form out-of-fold (and attack-type-unseen) probabilities.
S1 (seen reference): stratified 5-fold random CV on 01-12 (attack types seen in training).
X1 (cross-day): train on all of 01-12, test on the testing day 03-11 (Portmap never seen).
"""
import sys, time, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb
import common
from common import *

common.MAX_EPOCHS, common.PATIENCE = 30, 5
FEATS = json.load(open(OUT / "dataset_info.json"))["features"]   # same 78 CICFlowMeter features as CIC-IDS2017
CAP = 150_000
D = ROOT / "data" / "ddos2019"
O2 = ROOT / "results_ddos2019_local"; O2.mkdir(exist_ok=True)


def load():
    df = pd.read_pickle(D / "all_sampled.pkl")
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "file"]).reset_index(drop=True)
    df["y"] = (df.Label != "BENIGN").astype(int)
    df["type"] = df.Label.replace({"WebDDoS": "UDP-lag", "UDPLag": "UDP-lag"})
    df = df.sort_values("ts").reset_index(drop=True)
    X = df[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df, X


def intervals(d):
    """Attack intervals of a day (start, end, type), benign-only time assigned to the nearest interval."""
    a = d[d.y == 1].groupby("type").ts.agg(["min", "max"]).sort_values("min")
    a = a[a.index != "WebDDoS"]
    fold = np.full(len(d), -1)
    starts = a["min"].values
    for i, (t, r) in enumerate(a.iterrows()):
        fold[((d.ts >= r["min"]) & (d.ts <= r["max"])).values] = i
    rest = np.where(fold < 0)[0]
    if len(rest):
        mids = (a["min"] + (a["max"] - a["min"]) / 2).values
        tt = d.ts.values[rest]
        fold[rest] = np.abs(tt[:, None] - mids[None, :]).argmin(1)
    return fold, list(a.index)


def cap_idx(idx, y, rng):
    if len(idx) <= CAP: return idx
    ben = idx[y[idx] == 0]; att = idx[y[idx] == 1]
    att = rng.choice(att, CAP - min(len(ben), CAP // 3), replace=False)
    ben = rng.choice(ben, min(len(ben), CAP // 3), replace=False)
    return np.sort(np.concatenate([ben, att]))


def fit_predict(Xtr, ytr, Xte, seed, rng):
    """Returns dict model -> test probabilities; MLP with temperature scaling; 10% random validation."""
    itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.1, random_state=seed, stratify=ytr)
    A, V, T = fit_transform(Xtr[itr], Xtr[iva], Xte)
    out = {}
    m = LogisticRegression(max_iter=2000, C=1.0).fit(A, ytr[itr]); out["LR"] = m.predict_proba(T)[:, 1]
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, random_state=seed, verbose=-1, n_jobs=-1).fit(A, ytr[itr])
    out["LightGBM"] = m.predict_proba(T)[:, 1]
    model, info = train_mlp(A, ytr[itr], V, ytr[iva], seed)
    lv = predict_logits(model, V); Tt = fit_temperature(lv, ytr[iva])
    out["MonitoringAgent"] = softmax_p1(predict_logits(model, T), Tt)
    out["_info"] = dict(best_epoch=info["best_epoch"], train_s=info["train_seconds"], T=Tt, n_train=len(itr))
    return out


def flow_report(y, p, types, thr=0.5):
    r = metrics(y, p, thr)
    r["recall_by_type"] = {t: float((p[(types == t) & (y == 1)] >= thr).mean()) for t in np.unique(types[y == 1])}
    r["benign_fpr"] = float((p[y == 0] >= thr).mean()) if (y == 0).any() else float("nan")
    return r


def main(part):
    df, X = load()
    rng = np.random.default_rng(0)
    d1 = df.day.values == "01-12"; i1 = np.where(d1)[0]
    D1 = df.iloc[i1].reset_index(drop=True); X1 = X.values[i1]; y1 = D1.y.values; t1 = D1.type.values
    if part == "L1":
        fold, order = intervals(D1)
        oof = {m: np.full(len(D1), np.nan) for m in ["LR", "LightGBM", "MonitoringAgent"]}
        res = {"order": order, "folds": {}}
        for k, typ in enumerate(order):
            te = np.where(fold == k)[0]; tr = cap_idx(np.where(fold != k)[0], y1, rng)
            t0 = time.time(); pr = fit_predict(X1[tr], y1[tr], X1[te], 100 + k, rng)
            for m in oof: oof[m][te] = pr[m]
            res["folds"][typ] = dict(info=pr["_info"], n_test=len(te), n_test_benign=int((y1[te] == 0).sum()),
                                     **{m: flow_report(y1[te], pr[m], t1[te]) for m in oof})
            print(typ, {m: round(res["folds"][typ][m]["recall_by_type"].get(typ, np.nan), 4) for m in oof},
                  "fpr", {m: round(res["folds"][typ][m]["benign_fpr"], 4) for m in oof}, f"{time.time()-t0:.0f}s", flush=True)
            dump(res, "../results_ddos2019_local/flow_L1_unseen.json")
        res["pooled"] = {m: flow_report(y1, oof[m], t1) for m in oof}
        dump(res, "../results_ddos2019_local/flow_L1_unseen.json")
        pd.DataFrame(dict(idx=i1, fold=fold, y=y1, type=t1, **{f"p_{m}": oof[m] for m in oof})).to_parquet(O2 / "oof_L1.parquet")
    elif part == "S1":
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        strat = np.where(y1 == 1, t1, "BENIGN")
        oof = {m: np.full(len(D1), np.nan) for m in ["LR", "LightGBM", "MonitoringAgent"]}
        for k, (tr, te) in enumerate(skf.split(X1, strat)):
            tr = cap_idx(tr, y1, rng); pr = fit_predict(X1[tr], y1[tr], X1[te], 200 + k, rng)
            for m in oof: oof[m][te] = pr[m]
            print("S1 fold", k, flush=True)
        dump({"pooled": {m: flow_report(y1, oof[m], t1) for m in oof}}, "../results_ddos2019_local/flow_S1_seen.json")
        pd.DataFrame(dict(idx=i1, y=y1, type=t1, **{f"p_{m}": oof[m] for m in oof})).to_parquet(O2 / "oof_S1.parquet")
    elif part == "X1":
        i2 = np.where(~d1)[0]; D2 = df.iloc[i2].reset_index(drop=True); y2 = D2.y.values; t2 = D2.type.values
        tr = cap_idx(np.arange(len(D1)), y1, rng)
        pr = fit_predict(X1[tr], y1[tr], X.values[i2], 300, rng)
        res = {m: flow_report(y2, pr[m], t2) for m in ["LR", "LightGBM", "MonitoringAgent"]}; res["info"] = pr["_info"]
        dump(res, "../results_ddos2019_local/flow_X1_crossday.json")
        pd.DataFrame(dict(idx=i2, y=y2, type=t2, **{f"p_{m}": pr[m] for m in ["LR", "LightGBM", "MonitoringAgent"]})).to_parquet(O2 / "pred_X1.parquet")
        print(json.dumps({m: {k: round(v, 4) if isinstance(v, float) else v for k, v in res[m].items() if k in ("f1", "recall", "benign_fpr", "recall_by_type")} for m in ["LR", "LightGBM", "MonitoringAgent"]}, indent=1, default=str))
    dinfo = dict(rows=len(df), by_day_label=df.groupby(["day", "Label"]).size().to_dict())
    dump({str(k): v for k, v in dinfo.items()} | {"by_day_label": {f"{a}|{b}": int(c) for (a, b), c in dinfo["by_day_label"].items()}}, "../results_ddos2019_local/dataset_info.json")


if __name__ == "__main__":
    main(sys.argv[1])
