"""Flow-level experiments.
F1: stratified random 70/15/15 split, 5 seeds; LR, LightGBM, Monitoring Agent; with/without Destination Port.
F2: blocked 5-fold CV over contiguous record blocks (temporal proxy); Monitoring Agent OOF
    probabilities are saved for incident construction.
"""
import sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from common import *

SEEDS = [42, 7, 123, 2024, 31337]


def run_models(Xtr, ytr, Xva, yva, Xte, yte, seed, which=("lr", "lgbm", "mlp")):
    res = {}
    if "lr" in which:
        t = time.perf_counter(); m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr); tt = time.perf_counter() - t
        pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        thr = select_threshold(yva, pv)
        res["LR"] = dict(test=metrics(yte, pt, thr), test_at_05=metrics(yte, pt), thr=thr, train_s=tt)
    if "lgbm" in which:
        t = time.perf_counter()
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, random_state=seed, verbose=-1, n_jobs=-1).fit(Xtr, ytr)
        tt = time.perf_counter() - t
        pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        thr = select_threshold(yva, pv)
        res["LightGBM"] = dict(test=metrics(yte, pt, thr), test_at_05=metrics(yte, pt), thr=thr, train_s=tt)
    if "mlp" in which:
        model, info = train_mlp(Xtr, ytr, Xva, yva, seed)
        lv, lt = predict_logits(model, Xva), predict_logits(model, Xte)
        T = fit_temperature(lv, yva)
        pv_raw, pt_raw = softmax_p1(lv), softmax_p1(lt)
        pv, pt = softmax_p1(lv, T), softmax_p1(lt, T)
        thr = select_threshold(yva, pv)
        t = time.perf_counter(); predict_logits(model, Xte); inf_s = time.perf_counter() - t
        res["MonitoringAgent"] = dict(test=metrics(yte, pt, thr), test_uncalibrated=metrics(yte, pt_raw, thr),
                                      thr=thr, temperature=T, train_s=info["train_seconds"],
                                      best_epoch=info["best_epoch"], epochs_run=info["epochs_run"],
                                      infer_s_per_1k=1000 * inf_s / len(Xte), n_params=n_params(model),
                                      history=info["history"])
        res["_mlp_test_probs"] = pt
    return res


def main(part):
    df, X, y, feats, info = load_clean()
    dump(dict(info, features=feats), "dataset_info.json")
    port_idx = feats.index("Destination Port")
    if part == "F1":
        out = {}
        for s in SEEDS:
            idx = np.arange(len(y))
            itr, itmp = train_test_split(idx, test_size=0.30, random_state=s, stratify=y)
            iva, ite = train_test_split(itmp, test_size=0.50, random_state=s, stratify=y[itmp])
            Xtr, Xva, Xte = fit_transform(X.values[itr], X.values[iva], X.values[ite])
            r = run_models(Xtr, y[itr], Xva, y[iva], Xte, y[ite], s)
            keep = [i for i in range(Xtr.shape[1]) if i != port_idx]
            r2 = run_models(Xtr[:, keep], y[itr], Xva[:, keep], y[iva], Xte[:, keep], y[ite], s, which=("lgbm", "mlp"))
            r.pop("_mlp_test_probs"); r2.pop("_mlp_test_probs")
            r["LightGBM_noPort"], r["MonitoringAgent_noPort"] = r2["LightGBM"], r2["MonitoringAgent"]
            r["sizes"] = dict(train=len(itr), val=len(iva), test=len(ite))
            out[str(s)] = r
            print("seed", s, {k: round(v["test"]["f1"], 6) for k, v in r.items() if k != "sizes"}, flush=True)
            dump(out, "flow_F1_random_split.json")
    elif part == "F2":
        K = 5
        blocks = np.array_split(np.arange(len(y)), K)
        oof = np.full(len(y), np.nan); oof_fold = np.zeros(len(y), int)
        out = {}
        for k, te in enumerate(blocks):
            rest = np.setdiff1d(np.arange(len(y)), te)
            rng = np.random.default_rng(42 + k)
            # validation = a contiguous block inside the remaining data (10%) to limit leakage
            nva = int(0.1 * len(rest)); start = rng.integers(0, len(rest) - nva)
            va = rest[start:start + nva]; tr = np.setdiff1d(rest, va)
            Xtr, Xva, Xte = fit_transform(X.values[tr], X.values[va], X.values[te])
            r = run_models(Xtr, y[tr], Xva, y[va], Xte, y[te], 42 + k)
            oof[te] = r.pop("_mlp_test_probs"); oof_fold[te] = k
            r["sizes"] = dict(train=len(tr), val=len(va), test=len(te), test_ddos_rate=float(y[te].mean()))
            out[str(k)] = r
            print("fold", k, {kk: round(v["test"]["f1"], 6) for kk, v in r.items() if kk != "sizes"}, flush=True)
            dump(out, "flow_F2_blocked_cv.json")
        thr = float(np.median([out[str(k)]["MonitoringAgent"]["thr"] for k in range(K)]))
        pd.DataFrame(dict(row_id=df["row_id"], fold=oof_fold, ddos_probability=oof, y=y)).to_parquet(OUT / "oof_flow_probs.parquet")
        dump(dict(threshold_median=thr, oof_metrics=metrics(y, oof, thr)), "flow_F2_oof_summary.json")


if __name__ == "__main__":
    main(sys.argv[1])
