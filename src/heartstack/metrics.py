"""Clinical evaluation metrics: discrimination, operating points, calibration, decision curves, intervals."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def threshold_for_sensitivity(y, p, target: float) -> float:
    """The highest threshold whose sensitivity is at least ``target`` (found on training predictions)."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    t = float(thr[ok[0]]) if ok.size else float(thr[-1])
    return min(t, 1.0)


def operating_point(y, p, threshold: float) -> dict:
    y, pred = np.asarray(y), (np.asarray(p) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    def ratio(a, b):
        return round(a / b, 4) if b else float("nan")
    return {"threshold": round(threshold, 4), "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
            "ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def calibration(y, p, n_bins: int = 10) -> dict:
    """Expected calibration error, calibration intercept and slope, and the reliability table."""
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    table, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            table.append({"bin": b, "n": int(m.sum()), "mean_pred": round(float(p[m].mean()), 4),
                          "observed": round(float(y[m].mean()), 4)})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    slope = intercept = float("nan")
    if len(np.unique(y)) == 2:
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
        slope, intercept = float(lr.coef_[0][0]), float(lr.intercept_[0])
    return {"ece": round(float(ece), 4), "slope": round(slope, 3), "intercept": round(intercept, 3), "table": table}


def net_benefit(y, p, thresholds) -> list[dict]:
    """Decision curve: net benefit of the model, of 'treat all' and of 'treat none' at each threshold."""
    y, p = np.asarray(y), np.asarray(p)
    n, prev = len(y), float(np.mean(y))
    out = []
    for t in thresholds:
        pred = p >= t
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        w = t / (1 - t)
        out.append({"threshold": round(float(t), 3), "model": round(tp / n - fp / n * w, 4),
                    "treat_all": round(prev - (1 - prev) * w, 4), "treat_none": 0.0})
    return out


def discrimination(y, p) -> dict:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": round(float(brier_score_loss(y, p)), 4)}
    return {"auroc": round(float(roc_auc_score(y, p)), 4), "auprc": round(float(average_precision_score(y, p)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4)}


def bootstrap(y, p, *, threshold: float, n_boot: int, seed: int) -> dict:
    """Percentile 95% intervals for AUROC, AUPRC, sensitivity and specificity (stratified resampling)."""
    y, p = np.asarray(y), np.asarray(p)
    if n_boot == 0:
        return {}
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals: dict[str, list[float]] = {"auroc": [], "auprc": [], "sensitivity": [], "specificity": []}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yy, pp = y[idx], p[idx]
        vals["auroc"].append(roc_auc_score(yy, pp))
        vals["auprc"].append(average_precision_score(yy, pp))
        op = operating_point(yy, pp, threshold)
        vals["sensitivity"].append(op["sensitivity"])
        vals["specificity"].append(op["specificity"])
    return {k: [round(float(np.quantile(v, 0.025)), 4), round(float(np.quantile(v, 0.975)), 4)] for k, v in vals.items()}
