"""Score the wall-to-wall grid for the latest year with the production models.

Output: data/processed/risk_<T>.parquet with lon, lat, p1..pH — calibrated probability that the
pixel loses tree cover within h years after T (T = last Hansen year by default).

    python -m deforest.predict [--year 2023]
"""

from __future__ import annotations

import argparse

import pandas as pd

from .config import load_config
from .dataset import load_grid
from .evaluate import load_model, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--year", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    T = args.year or cfg["data"]["gfc_last_year"]
    grid = load_grid(cfg, T)
    out = grid[["lon", "lat"]].copy()
    for h in cfg["model"]["horizons"]:
        mdl, iso, n_est, meta = load_model(cfg, h, "prod")
        out[f"p{h}"] = score(mdl, iso, n_est, grid, meta["features"]).round(5)
    path = cfg.path("processed") / f"risk_{T}.parquet"
    out.attrs["base_year"] = T
    out.to_parquet(path, index=False)
    print(f"wrote {path}: {len(out):,} forest grid points")
    print(out.describe().T[["mean", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
