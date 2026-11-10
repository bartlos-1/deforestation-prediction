"""Weighted evaluation metrics. All functions take sample weights so results reflect the true forest population."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def pr_auc(y, p, w) -> float:
    return float(average_precision_score(y, p, sample_weight=w)) if y.sum() > 0 else float("nan")


def roc_auc(y, p, w) -> float:
    return float(roc_auc_score(y, p, sample_weight=w)) if 0 < y.sum() < len(y) else float("nan")


def top_k(y, p, w, ks) -> dict[str, dict[str, float]]:
    """Precision / recall when flagging the highest-scoring k share of forest area (by weight)."""
    order = np.argsort(-p, kind="stable")
    ys, ws = y[order], w[order]
    cw = np.cumsum(ws) / ws.sum()
    total_pos = (ys * ws).sum()
    out = {}
    for k in ks:
        sel = cw <= k
        if not sel.any():
            sel[0] = True
        tp = (ys[sel] * ws[sel]).sum()
        out[str(k)] = {
            "precision": float(tp / ws[sel].sum()),
            "recall": float(tp / total_pos) if total_pos > 0 else float("nan"),
            "lift": float((tp / ws[sel].sum()) / (total_pos / ws.sum())) if total_pos > 0 else float("nan"),
        }
    return out


def calibration_bins(y, p, w, n_bins: int = 10) -> list[dict[str, float]]:
    order = np.argsort(p)
    y, p, w = y[order], p[order], w[order]
    cw = np.cumsum(w) / w.sum()
    edges = np.searchsorted(cw, np.linspace(0, 1, n_bins + 1)[1:-1])
    out = []
    for idx in np.split(np.arange(len(p)), edges):
        if len(idx) == 0:
            continue
        ww = w[idx]
        out.append({"pred": float(np.average(p[idx], weights=ww)), "obs": float(np.average(y[idx], weights=ww)), "share": float(ww.sum() / w.sum())})
    return out


def summarize(y, p, w, ks, calibrated: bool = True) -> dict:
    y, p, w = np.asarray(y, float), np.asarray(p, float), np.asarray(w, float)
    out = {"pr_auc": pr_auc(y, p, w), "roc_auc": roc_auc(y, p, w), "top_k": top_k(y, p, w, ks)}
    if calibrated:
        out["brier"] = float(brier_score_loss(y, np.clip(p, 0, 1), sample_weight=w))
    return out
