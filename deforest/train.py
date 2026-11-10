"""Train one XGBoost model per horizon.

Tree count comes from spatial-block CV (logloss early stopping); the val years are only used for calibration.

For each horizon h two models are saved:
  * eval — trained on train years / train blocks, Platt-calibrated on val years.
           Only this model is scored on the test split (see evaluate.py).
  * prod — same hyper-parameters, trained on all blocks and all years except the last two labelled years,
           which are used for calibration. This is the model used for the map (predict.py).

    python -m deforest.train [--trials 30] [--horizons 1 2 3]
"""

from __future__ import annotations

import argparse
import json
import subprocess

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from .calibration import PlattCalibrator
from .config import ROOT, load_config
from .metrics import pr_auc
from .schema import feature_columns

MAX_TUNE_ROWS = 150_000


def git_hash(cfg) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def load_panel(cfg) -> pd.DataFrame:
    return pd.read_parquet(cfg.path("processed") / "panel.parquet")


def available_features(panel: pd.DataFrame) -> list[str]:
    feats = feature_columns(["dist_road", "protected", "dist_built"])
    return [f for f in feats if f in panel.columns]


def splits(panel: pd.DataFrame, cfg, h: int) -> dict[str, pd.DataFrame]:
    m, last = cfg["model"], cfg["data"]["gfc_last_year"]
    y = f"y{h}"
    df = panel[panel["eligible"] & panel[y].notna()]
    T = df["year"]
    (tr0, tr1), (v0, v1), (te0, te1) = m["train_years"], m["val_years"], m["test_years"]
    te1 = min(te1, last - h)
    return {
        "train": df[T.between(tr0, tr1) & (T + h <= v0) & ~df["test_block"]],
        "val": df[T.between(v0, v1) & (T + h <= te0) & ~df["test_block"]],
        "test": df[T.between(te0, te1) & df["test_block"]],
        "test_temporal": df[T.between(te0, te1) & ~df["test_block"]],
    }


def prod_splits(panel: pd.DataFrame, cfg, h: int) -> dict[str, pd.DataFrame]:
    last = cfg["data"]["gfc_last_year"]
    y = f"y{h}"
    df = panel[panel["eligible"] & panel[y].notna()]
    cal0 = last - h - 1  # the last two T values with complete labels: last-h-1, last-h
    return {"train": df[df["year"] + h <= cal0], "cal": df[df["year"].between(cal0, last - h)]}


def base_params(seed: int) -> dict:
    return dict(tree_method="hist", objective="binary:logistic", eval_metric="logloss", n_jobs=-1, random_state=seed)


def tune(train: pd.DataFrame, feats: list[str], y: str, n_trials: int, seed: int) -> tuple[dict, int]:
    if len(train) > MAX_TUNE_ROWS:
        train = train.sample(MAX_TUNE_ROWS, random_state=seed)
    X, Y, W, G = train[feats], train[y].to_numpy(), train["w"].to_numpy(), train["block_id"].to_numpy()
    folds = list(GroupKFold(n_splits=3).split(X, Y, G))

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            max_depth=trial.suggest_int("max_depth", 3, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            gamma=trial.suggest_float("gamma", 0, 5),
        )
        scores, iters = [], []
        for tr, va in folds:
            mdl = xgb.XGBClassifier(**params, **base_params(seed), n_estimators=2000, early_stopping_rounds=50)
            mdl.fit(X.iloc[tr], Y[tr], eval_set=[(X.iloc[va], Y[va])], verbose=False)
            scores.append(pr_auc(Y[va], mdl.predict_proba(X.iloc[va])[:, 1], W[va]))
            iters.append(mdl.best_iteration + 1)
        trial.set_user_attr("n_estimators", int(np.mean(iters)))
        return float(np.nanmean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"    best CV PR-AUC {study.best_value:.4f}  {study.best_params}")
    return study.best_params, study.best_trial.user_attrs["n_estimators"]


def fit_calibrated(train, cal, feats, y, params, seed, n_estimators):
    """Fit with a fixed tree count (chosen by spatial CV), then calibrate on the separate `cal` rows."""
    mdl = xgb.XGBClassifier(**params, **base_params(seed), n_estimators=n_estimators)
    mdl.fit(train[feats], train[y], verbose=False)
    n_est = n_estimators
    raw = predict_raw(mdl, cal[feats], n_est)
    return mdl, PlattCalibrator().fit(raw, cal[y].to_numpy(), cal["w"].to_numpy()), n_est


def scaled_trees(n: int, rows_from: float, rows_to: float) -> int:
    return max(20, int(n * min(1.5, max(1.0, (rows_to / max(1.0, rows_from)) ** 0.5))))


def predict_raw(mdl: xgb.XGBClassifier, X: pd.DataFrame, n_est: int) -> np.ndarray:
    return mdl.predict_proba(X, iteration_range=(0, n_est))[:, 1]


def train_horizon(panel, cfg, h, n_trials):
    seed = cfg["data"]["seed"]
    feats = available_features(panel)
    y = f"y{h}"
    s = splits(panel, cfg, h)
    for k, v in s.items():
        print(f"    {k:<14} {len(v):>9,} rows  {int(v[y].sum()):>7,} positives  years {v['year'].min()}–{v['year'].max()}" if len(v) else f"    {k:<14} empty")
    if s["train"][y].sum() < 10 or s["val"][y].sum() < 5:
        raise SystemExit(f"Too few positives for h={h}; check years / data.")

    params, cv_trees = tune(s["train"], feats, y, n_trials, seed)
    # CV folds used ~2/3 of (at most MAX_TUNE_ROWS) train rows; grow the tree count gently with data size
    n_est = scaled_trees(cv_trees, min(len(s["train"]), MAX_TUNE_ROWS) * 2 / 3, len(s["train"]))
    eval_mdl, eval_iso, _ = fit_calibrated(s["train"], s["val"], feats, y, params, seed, n_est)

    p = prod_splits(panel, cfg, h)
    # scale tree count with data size (more rows → slightly more trees), but stay close to the early-stopped value
    n_prod = scaled_trees(n_est, len(s["train"]), len(p["train"]))
    prod_mdl, prod_iso, _ = fit_calibrated(p["train"], p["cal"], feats, y, params, seed, n_prod)

    out = cfg.path("models") / f"h{h}"
    out.mkdir(parents=True, exist_ok=True)
    eval_mdl.save_model(out / "eval_model.json")
    prod_mdl.save_model(out / "prod_model.json")
    joblib.dump(eval_iso, out / "eval_calibrator.joblib")
    joblib.dump(prod_iso, out / "prod_calibrator.joblib")
    meta = {
        "horizon": h, "features": feats, "params": params, "n_estimators_eval": n_est, "n_estimators_prod": n_prod,
        "rows": {k: len(v) for k, v in s.items()} | {"prod_train": len(p["train"]), "prod_cal": len(p["cal"])},
        "prod_years": {"train_max_T": int(p["train"]["year"].max()), "cal_T": sorted(p["cal"]["year"].unique().tolist())},
        "git": git_hash(cfg),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"    saved {out}  (eval trees {n_est}, prod trees {n_prod})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--horizons", type=int, nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    panel = load_panel(cfg)
    trials = args.trials or cfg["model"]["optuna_trials"]
    for h in args.horizons or cfg["model"]["horizons"]:
        print(f"horizon {h}:")
        train_horizon(panel, cfg, h, trials)


if __name__ == "__main__":
    main()
