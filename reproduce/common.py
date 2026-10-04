"""Shared utilities for the revised experiments (re-run, Oct 2026)."""
from pathlib import Path
import json, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
OUT = ROOT / "results_local"
OUT.mkdir(exist_ok=True)
torch.set_num_threads(max(1, (__import__("os").cpu_count() or 2) - 1))

BATCH_SIZE, MAX_EPOCHS, PATIENCE = 512, 50, 8
LR, WD, DROPOUT = 1e-3, 1e-4, 0.20


def load_clean():
    """Load CSV, strip names, drop exact duplicates, keep record order."""
    df = pd.read_csv(RAW, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()
    raw_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    y = (df["Label"] == "DDoS").astype(np.int64).values
    feats = [c for c in df.columns if c not in {"Label", "row_id"}]
    X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    info = dict(raw_rows=raw_rows, rows=len(df), duplicates_removed=raw_rows - len(df),
                n_features=len(feats), nonfinite=int(X.isna().sum().sum()),
                ddos_rate=float(y.mean()))
    return df, X, y, feats, info


def fit_transform(Xtr, *others):
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    A = sc.fit_transform(imp.fit_transform(Xtr)).astype(np.float32)
    rest = [sc.transform(imp.transform(o)).astype(np.float32) for o in others]
    return (A, *rest)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class ResidualBlock(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(p),
                                   nn.Linear(d, d), nn.BatchNorm1d(d))
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class MonitoringAgent(nn.Module):
    def __init__(self, d_in, p=DROPOUT):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p))
        self.residual_layers = nn.Sequential(ResidualBlock(256, p), ResidualBlock(256, p))
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p), nn.Linear(128, 2))

    def forward(self, x):
        return self.classifier(self.residual_layers(self.input_layer(x)))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def predict_logits(model, X, bs=8192):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[i:i + bs])).numpy())
    return np.concatenate(out)


def softmax_p1(logits, T=1.0):
    z = logits / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(1)


def train_mlp(Xtr, ytr, Xva, yva, seed):
    """Adam + ReduceLROnPlateau, early stopping on validation F1 (patience 8)."""
    set_seed(seed)
    model = MonitoringAgent(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    g = torch.Generator().manual_seed(seed)
    best, best_state, best_ep, wait = -1, None, 0, 0
    hist = dict(train_loss=[], val_loss=[], val_f1=[])
    t0 = time.perf_counter()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        lv = predict_logits(model, Xva)
        vloss = float(lossf(torch.from_numpy(lv), torch.from_numpy(yva)).item())
        f1 = precision_recall_fscore_support(yva, lv.argmax(1), average="binary", zero_division=0)[2]
        hist["train_loss"].append(tot / len(perm)); hist["val_loss"].append(vloss); hist["val_f1"].append(float(f1))
        sched.step(f1)
        if f1 > best + 1e-6:
            best, best_ep, wait = f1, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, dict(best_epoch=best_ep, epochs_run=ep, train_seconds=time.perf_counter() - t0, history=hist)


def fit_temperature(logits, y):
    """Temperature scaling on validation logits (Guo et al., 2017)."""
    lt, yt = torch.from_numpy(logits.astype(np.float64)), torch.from_numpy(y)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)
    lossf = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad(); l = lossf(lt / logT.exp(), yt); l.backward(); return l
    opt.step(closure)
    return float(logT.exp().item())


def ece(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def metrics(y, p, thr=0.5):
    yp = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yp, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, yp), precision=pr, recall=rc, f1=f1,
                roc_auc=roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
                pr_auc=average_precision_score(y, p) if len(set(y)) > 1 else float("nan"),
                fpr=fp / max(fp + tn, 1), fnr=fn / max(fn + tp, 1),
                brier=brier_score_loss(y, p), ece=ece(y, p),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def select_threshold(y, p, grid=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)):
    """Highest validation F1; ties broken by the higher threshold (original policy)."""
    best_t, best_f = None, -1
    for t in grid:
        f = metrics(y, p, t)["f1"]
        if f > best_f + 1e-12 or abs(f - best_f) <= 1e-12:
            best_t, best_f = t, f
    return best_t


def dump(obj, name):
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    with open(OUT / name, "w") as f:
        json.dump(obj, f, indent=2, default=conv)
