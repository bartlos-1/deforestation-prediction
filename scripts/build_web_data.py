"""Aggregate pipeline outputs into compact JSON for the dashboard (web/data/).

    python scripts/build_web_data.py [--config config.yaml]

Inputs                                   → output
  data/raw/history_*.csv  (EE export)    → web/data/history.json   per-cell loss pixels per year
  data/processed/risk_<T>.parquet        → web/data/risk.json      per-cell mean probability per horizon
  web/data/metrics.json is written directly by `python -m deforest.evaluate`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deforest.config import load_config  # noqa: E402

RES = 0.04  # degrees; must match deforest.export.HISTORY_RES


def build_history(cfg) -> dict | None:
    files = sorted(cfg.path("raw").glob("history_*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ycols = sorted(c for c in df.columns if c.startswith("y") and c[1:].isdigit())
    years = [int(c[1:]) for c in ycols]
    df[ycols] = df[ycols].fillna(0).round().astype(int)
    df = df[df[ycols].sum(axis=1) > 0]
    return {
        "source": f"Hansen Global Forest Change ({cfg['data']['gfc_asset'].split('/')[-1]}), treecover2000 ≥ {cfg['data']['treecover_min']}%",
        "unit": "loss pixels (~30 m)",
        "years": years,
        "totals": df[ycols].sum().astype(int).tolist(),
        "cells": [[int(ix), int(iy), row.tolist()] for ix, iy, row in zip(df["ix"], df["iy"], df[ycols].to_numpy())],
    }


def build_risk(cfg) -> dict | None:
    files = sorted(cfg.path("processed").glob("risk_*.parquet"))
    if not files:
        return None
    path = files[-1]
    T = int(path.stem.split("_")[1])
    df = pd.read_parquet(path)
    horizons = [int(c[1:]) for c in df.columns if c.startswith("p") and c[1:].isdigit()]
    x0, y0 = cfg.bbox[0], cfg.bbox[1]
    df["ix"] = np.floor((df["lon"] - x0) / RES).astype(int)
    df["iy"] = np.floor((df["lat"] - y0) / RES).astype(int)
    g = df.groupby(["ix", "iy"]).agg(n=("lon", "size"), **{f"p{h}": (f"p{h}", "mean") for h in horizons}).reset_index()
    km2 = (cfg["data"]["grid_spacing"] / 1000) ** 2  # each grid point represents spacing² of forest
    return {
        "base_year": T,
        "horizons": horizons,
        "labels": {str(h): (f"{T + 1}" if h == 1 else f"{T + 1}–{T + h}") for h in horizons},
        "forest_points": int(len(df)),
        "km2_per_point": km2,
        "expected_loss_km2": {str(h): float(df[f"p{h}"].sum() * km2) for h in horizons},
        "mean_p": {str(h): float(df[f"p{h}"].mean()) for h in horizons},
        "quantiles": {str(h): {q: float(df[f"p{h}"].quantile(q)) for q in (0.5, 0.9, 0.99)} for h in horizons},
        "cells": [[int(r.ix), int(r.iy), int(r.n), *[round(float(getattr(r, f"p{h}")), 5) for h in horizons]] for r in g.itertuples(index=False)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = cfg.path("web_data")
    meta = {"roi": cfg.bbox, "res": RES, "region": cfg["region"]["name"]}
    for name, payload in (("history.json", build_history(cfg)), ("risk.json", build_risk(cfg))):
        if payload is None:
            print(f"skip {name}: inputs not found")
            continue
        p = out / name
        p.write_text(json.dumps({**meta, **payload}, separators=(",", ":")))
        print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB)")
    if not (out / "metrics.json").exists():
        print("note: metrics.json missing — run `python -m deforest.evaluate`")


if __name__ == "__main__":
    main()
