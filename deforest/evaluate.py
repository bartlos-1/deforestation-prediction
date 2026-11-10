"""Score the eval models on held-out data and write metrics.json (reports/ and the web dashboard).

Primary test split: test years × held-out spatial blocks (never seen in training or tuning).
Secondary: test years × training blocks (temporal hold-out only) — shows how much spatial leakage would inflate scores.
Baselines on the same rows:
  * prevalence  — random ranking; PR-AUC equals the base rate
  * distance    — score = −distance to recent loss (the "contagion" rule of thumb)
  * logistic    — logistic regression on 4 contagion/spectral features

    python -m deforest.evaluate
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_config
from .metrics import calibration_bins, pr_auc, summarize, top_k
from .schema import BASELINE_FEATURES
from .train import load_panel, predict_raw, splits

HORIZON_LABEL = {1: "within 1 year", 2: "within 2 years", 3: "within 3 years"}


def load_model(cfg, h: int, kind: str):
    d = cfg.path("models") / f"h{h}"
    meta = json.loads((d / "meta.json").read_text())
    mdl = xgb.XGBClassifier()
    mdl.load_model(d / f"{kind}_model.json")
    iso = joblib.load(d / f"{kind}_calibrator.joblib")
    n_est = meta["n_estimators_eval" if kind == "eval" else "n_estimators_prod"]
    return mdl, iso, n_est, meta


def score(mdl, iso, n_est, df, feats) -> np.ndarray:
    return iso.predict(predict_raw(mdl, df[feats], n_est))


def baseline_scores(train: pd.DataFrame, test: pd.DataFrame, y: str) -> dict[str, np.ndarray]:
    dist = test["dist_recent_loss"].fillna(test["dist_recent_loss"].max()).to_numpy()
    logit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    logit.fit(train[BASELINE_FEATURES], train[y])
    return {"distance": -dist, "logistic": logit.predict_proba(test[BASELINE_FEATURES])[:, 1]}


def evaluate_horizon(panel, cfg, h) -> dict:
    ks = cfg["model"]["top_k"]
    y = f"y{h}"
    mdl, iso, n_est, meta = load_model(cfg, h, "eval")
    feats = meta["features"]
    s = splits(panel, cfg, h)
    out = {"h": h, "label": HORIZON_LABEL.get(h, f"within {h} years")}

    for split_name in ("test", "test_temporal"):
        df = s[split_name]
        if df.empty or df[y].sum() == 0:
            out[split_name] = None
            continue
        Y, W = df[y].to_numpy(), df["w"].to_numpy()
        p = score(mdl, iso, n_est, df, feats)
        base = baseline_scores(s["train"], df, y)
        prev = float(np.average(Y, weights=W))
        res = {
            "years": [int(df["year"].min()), int(df["year"].max())],
            "n": int(len(df)), "positives": int(Y.sum()), "prevalence": prev,
            "models": {
                "xgboost": summarize(Y, p, W, ks),
                "distance": summarize(Y, base["distance"], W, ks, calibrated=False),
                "logistic": summarize(Y, base["logistic"], W, ks, calibrated=False),
                "prevalence": {"pr_auc": prev, "roc_auc": 0.5, "top_k": {str(k): {"precision": prev, "recall": k, "lift": 1.0} for k in ks}},
            },
        }
        if split_name == "test":
            by_year = []
            for T, g in df.groupby("year"):
                mask = df.index.isin(g.index)
                by_year.append({
                    "year": int(T), "n": int(len(g)), "prevalence": float(np.average(g[y], weights=g["w"])),
                    "xgboost": pr_auc(g[y].to_numpy(), p[mask], g["w"].to_numpy()),
                    "distance": pr_auc(g[y].to_numpy(), base["distance"][mask], g["w"].to_numpy()),
                })
            res["by_year"] = by_year
            res["calibration"] = calibration_bins(Y, p, W)
            res["drivers"] = drivers(df, y, p, ks)
        out[split_name] = res
    return out


def drivers(df: pd.DataFrame, y: str, p: np.ndarray, ks) -> dict:
    """Recall at the top-10% (or largest k) for fire-driven vs other loss, using the same ranking."""
    k = 0.1 if 0.1 in ks else max(ks)
    pos = df[y].to_numpy() == 1
    out = {}
    for name, flag in (("fire", df["fire_loss"].to_numpy()), ("other", ~df["fire_loss"].to_numpy())):
        yy = (pos & flag).astype(float)
        if yy.sum() == 0:
            continue
        w = df["w"].to_numpy()
        out[name] = {
            "share_of_loss": float((yy * w).sum() / (pos * w).sum()),
            "pr_auc": pr_auc(yy, p, w),
            f"recall_at_{k}": top_k(yy, p, w, [k])[str(k)]["recall"],
        }
    return out


def importance(cfg, h: int, top: int = 15) -> list[dict]:
    mdl, _, _, meta = load_model(cfg, h, "eval")
    gain = mdl.get_booster().get_score(importance_type="gain")
    total = sum(gain.values()) or 1
    feats = meta["features"]
    rows = [{"feature": feats[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k, "gain": v / total} for k, v in gain.items()]
    return sorted(rows, key=lambda r: -r["gain"])[:top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    panel = load_panel(cfg)
    horizons = cfg["model"]["horizons"]
    result = {
        "schema": 2,
        "generated": dt.date.today().isoformat(),
        "region": cfg["region"]["name"],
        "target": "Hansen tree-cover loss in (T, T+h] for pixels still forested at the end of year T",
        "weighting": "Metrics are weighted to the full forest population (undoing the stratified sample).",
        "horizons": [evaluate_horizon(panel, cfg, h) for h in horizons],
        "importance": importance(cfg, horizons[0]),
        "model_meta": {h: json.loads((cfg.path("models") / f"h{h}" / "meta.json").read_text()) for h in horizons},
    }
    for path in (cfg.path("reports") / "metrics.json", cfg.path("web_data") / "metrics.json"):
        path.write_text(json.dumps(result, indent=2, default=float))
        print(f"wrote {path}")
    for hz in result["horizons"]:
        t = hz["test"]
        if t:
            m = t["models"]
            print(f"  h={hz['h']}: PR-AUC xgb {m['xgboost']['pr_auc']:.4f} | distance {m['distance']['pr_auc']:.4f} | "
                  f"logistic {m['logistic']['pr_auc']:.4f} | prevalence {t['prevalence']:.4f}")


if __name__ == "__main__":
    main()
