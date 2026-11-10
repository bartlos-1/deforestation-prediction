"""Build the modelling panel from the raw Earth Engine CSVs.

One row per (training point, year T). Key rules:
  * eligible  — still forest at the end of T: treecover2000 ≥ threshold and no recorded loss in or before T
  * y{h}      — loss in (T, T+h]; NaN when T+h is beyond the last Hansen year
  * fire_loss — the point's loss year (or the year before) was burned per MODIS MCD64A1
  * w         — inverse-probability weight that undoes the stratified ("ever lost" / "never lost") design
  * block_id  — ~15 km spatial block; test_block marks blocks held out from training entirely

    python -m deforest.dataset
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import load_config
from .schema import NODATA


def _read(paths) -> pd.DataFrame:
    paths = sorted(paths)
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p, dtype={"point_id": str}) for p in paths], ignore_index=True)
    return df.replace(NODATA, np.nan)


def stratum_weights(points: pd.DataFrame, strata: pd.DataFrame) -> dict[int, float]:
    pop = {0: float(strata["count_0"].iloc[0]), 1: float(strata["count_1"].iloc[0])}
    samp = points["stratum"].value_counts().to_dict()
    n_pop, n_samp = sum(pop.values()), sum(samp.values())
    return {s: (pop[s] / n_pop) / (samp[s] / n_samp) for s in (0, 1)}


def add_labels(df: pd.DataFrame, horizons: list[int], last_year: int, treecover_min: float) -> pd.DataFrame:
    ly = df["lossyear"].fillna(0).astype(int)
    loss_cal_year = np.where(ly > 0, ly + 2000, 0)
    T = df["year"].to_numpy()
    df["eligible"] = (df["treecover2000"] >= treecover_min) & ((ly == 0) | (loss_cal_year > T))
    for h in horizons:
        y = ((ly > 0) & (loss_cal_year > T) & (loss_cal_year <= T + h)).astype(float)
        df[f"y{h}"] = np.where(T + h <= last_year, y, np.nan)
    return df


def add_fire_driver(df: pd.DataFrame) -> pd.DataFrame:
    """fire_loss = burned in the loss year or the year before (late-season fires show up a year later in Hansen)."""
    burned = df.pivot_table(index="point_id", columns="year", values="burned", aggfunc="max")
    loss_year = df.groupby("point_id")["lossyear"].first()
    fire = {}
    for pid, ly in loss_year.items():
        if not ly or np.isnan(ly) or pid not in burned.index:
            fire[pid] = False
            continue
        L = int(ly) + 2000
        row = burned.loc[pid]
        fire[pid] = bool(np.nanmax([row.get(L, np.nan), row.get(L - 1, np.nan), 0]) > 0)
    df["fire_loss"] = df["point_id"].map(fire).fillna(False).astype(bool)
    return df


def add_blocks(df: pd.DataFrame, bbox: list[float], block_deg: list[float], test_frac: float, seed: int) -> pd.DataFrame:
    bx = np.floor((df["lon"] - bbox[0]) / block_deg[0]).astype(int)
    by = np.floor((df["lat"] - bbox[1]) / block_deg[1]).astype(int)
    df["block_id"] = bx * 1000 + by
    blocks = np.sort(df["block_id"].unique())
    rng = np.random.default_rng(seed)
    test = set(rng.choice(blocks, size=max(1, int(round(test_frac * len(blocks)))), replace=False).tolist())
    df["test_block"] = df["block_id"].isin(test)
    return df


def build_panel(cfg) -> pd.DataFrame:
    raw, d, m = cfg.path("raw"), cfg["data"], cfg["model"]
    points = pd.read_csv(raw / "points.csv", dtype={"point_id": str})
    strata = pd.read_csv(raw / "strata.csv")
    feats = _read(raw.glob("features_*.csv"))
    if feats.empty:
        raise SystemExit(f"No features_*.csv in {raw}")

    df = feats.merge(points.drop(columns=["treecover2000"]), on="point_id", how="inner")
    df = add_labels(df, m["horizons"], d["gfc_last_year"], d["treecover_min"])
    df = add_fire_driver(df)
    w = stratum_weights(points, strata)
    df["w"] = df["stratum"].map(w)
    df = add_blocks(df, cfg.bbox, m["block_deg"], m["test_block_frac"], d["seed"])
    return df


def load_grid(cfg, year: int) -> pd.DataFrame:
    g = _read(cfg.path("raw").glob(f"grid_{year}_*.csv"))
    if g.empty:
        raise SystemExit(f"No grid_{year}_*.csv in {cfg.path('raw')}")
    ly = g["lossyear"].fillna(0).astype(int)
    forest_now = (g["treecover2000"] >= cfg["data"]["treecover_min"]) & ((ly == 0) | (ly + 2000 > year))
    return g[forest_now].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = build_panel(cfg)
    out = cfg.path("processed") / "panel.parquet"
    df.to_parquet(out, index=False)
    el = df[df["eligible"]]
    print(f"wrote {out}: {len(df):,} rows, {df['point_id'].nunique():,} points, years {df['year'].min()}–{df['year'].max()}")
    for h in cfg["model"]["horizons"]:
        y = el[f"y{h}"].dropna()
        wy = el.loc[y.index, "w"]
        print(f"  y{h}: {int(y.sum()):,} positives in sample, weighted prevalence {np.average(y, weights=wy):.4f}")


if __name__ == "__main__":
    main()
