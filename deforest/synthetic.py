"""Synthetic stand-in for the Earth Engine exports, used by the tests and for a dry run.

Simulates a forest population with persistent harvest zones (loss clusters in space and continues for
several years) and random wildfires, then writes points.csv / strata.csv / features_T.csv / grid / history
CSVs with exactly the columns the real exports have.

    python -m deforest.synthetic --out data/synthetic/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import CORE_FEATURES, NODATA


def simulate(bbox, years, n_pop: int, seed: int):
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = bbox
    lon = rng.uniform(x0, x1, n_pop)
    lat = rng.uniform(y0, y1, n_pop)
    tree = rng.uniform(30, 100, n_pop)

    # harvest zones: centre, radius, active years
    K = 25
    zc = np.c_[rng.uniform(x0, x1, K), rng.uniform(y0, y1, K)]
    zr = rng.uniform(0.15, 0.4, K)
    zstart = rng.integers(years[0] - 3, years[-1], K)
    zlen = rng.integers(3, 9, K)
    zint = rng.uniform(0.05, 0.2, K)

    def zone_field(t):
        f = np.zeros(n_pop)
        for k in range(K):
            if zstart[k] <= t < zstart[k] + zlen[k]:
                d2 = (lon - zc[k, 0]) ** 2 + (lat - zc[k, 1]) ** 2
                f += zint[k] * np.exp(-d2 / (2 * zr[k] ** 2))
        return f

    alive = np.ones(n_pop, bool)
    lossyear = np.zeros(n_pop, int)
    burned, field = {}, {}
    for t in range(2001, years[-1] + 4):
        field[t] = zone_field(t)
        b = np.zeros(n_pop, bool)
        for _ in range(rng.integers(0, 4)):
            c = (rng.uniform(x0, x1), rng.uniform(y0, y1))
            r = rng.uniform(0.05, 0.25)
            b |= ((lon - c[0]) ** 2 + (lat - c[1]) ** 2 < r**2) & (rng.random(n_pop) < 0.8)
        burned[t] = b
        hazard = 0.002 + field[t] * (tree / 100)
        lost = alive & ((rng.random(n_pop) < hazard) | (b & (rng.random(n_pop) < 0.5)))
        lossyear[lost] = t - 2000
        alive &= ~lost
    return dict(lon=lon, lat=lat, tree=tree, lossyear=lossyear, burned=burned, field=field, rng=rng)


def features_for(sim, idx: np.ndarray, T: int) -> pd.DataFrame:
    rng = sim["rng"]
    n = len(idx)
    ly = sim["lossyear"][idx]
    tree = sim["tree"][idx]
    recent_field = np.mean([sim["field"][t][idx] for t in (T - 2, T - 1, T)], axis=0)
    past_field = np.mean([sim["field"][t][idx] for t in range(2001, T + 1)], axis=0)
    burned_recent = np.any([sim["burned"][t][idx] for t in (T - 2, T - 1, T)], axis=0)
    lost_now = (ly > 0) & (ly + 2000 <= T)
    f = pd.DataFrame({
        "blue": rng.normal(0.03, 0.01, n), "green": rng.normal(0.05, 0.01, n), "red": rng.normal(0.04, 0.015, n),
        "nir": rng.normal(0.25, 0.05, n), "swir1": rng.normal(0.12, 0.03, n), "swir2": rng.normal(0.06, 0.02, n),
        "ndvi": 0.55 + 0.25 * (tree - 65) / 35 - 0.3 * lost_now + rng.normal(0, 0.05, n),
        "nbr": 0.45 + 0.2 * (tree - 65) / 35 - 0.4 * burned_recent + rng.normal(0, 0.05, n),
        "ndmi": 0.25 + rng.normal(0, 0.05, n),
        "d1_ndvi": rng.normal(0, 0.03, n), "d1_nbr": -0.2 * sim["burned"][T][idx] + rng.normal(0, 0.03, n),
        "d1_ndmi": rng.normal(0, 0.03, n), "d3_ndvi": rng.normal(0, 0.05, n), "d3_nbr": rng.normal(0, 0.05, n),
        "elevation": rng.normal(1200, 400, n), "slope": np.abs(rng.normal(12, 8, n)),
        "treecover2000": tree.round(),
        "dist_recent_loss": np.clip(15000 * np.exp(-25 * recent_field) + rng.normal(0, 800, n), 0, 15000),
        "frac_recent_500m": np.clip(recent_field * 2 + rng.normal(0, 0.02, n), 0, 1),
        "frac_recent_2km": np.clip(recent_field * 1.5 + rng.normal(0, 0.01, n), 0, 1),
        "frac_past_2km": np.clip(past_field * 3 + rng.normal(0, 0.02, n), 0, 1),
        "burned_recent": burned_recent.astype(int),
        "frac_burned_2km": np.clip(burned_recent * 0.6 + rng.normal(0, 0.05, n), 0, 1),
        "burned": sim["burned"][T][idx].astype(int),
    })
    cloud = rng.random(n) < 0.05  # cloud gaps
    for c in ["blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "nbr", "ndmi", "d1_ndvi", "d1_nbr", "d1_ndmi"]:
        f.loc[cloud, c] = NODATA
    return f


def generate(out: Path, bbox, feature_years, last_year, n_pop=40000, n_per_stratum=3000, n_grid=6000, seed=7):
    out.mkdir(parents=True, exist_ok=True)
    years = list(range(feature_years[0], feature_years[1] + 1))
    sim = simulate(bbox, years, n_pop, seed)
    rng = sim["rng"]
    # Hansen only reports loss up to last_year
    sim["lossyear"][sim["lossyear"] + 2000 > last_year] = 0
    ever = sim["lossyear"] > 0

    pop_idx = np.arange(n_pop)
    grid_idx = rng.choice(pop_idx, n_grid, replace=False)
    pool = np.setdiff1d(pop_idx, grid_idx)
    s1 = rng.choice(pool[ever[pool]], min(n_per_stratum, ever[pool].sum()), replace=False)
    s0 = rng.choice(pool[~ever[pool]], min(n_per_stratum, (~ever[pool]).sum()), replace=False)
    idx = np.r_[s0, s1]
    points = pd.DataFrame({
        "point_id": [f"p{i}" for i in idx], "lon": sim["lon"][idx], "lat": sim["lat"][idx],
        "stratum": ever[idx].astype(int), "lossyear": sim["lossyear"][idx], "treecover2000": sim["tree"][idx].round(),
    })
    points.to_csv(out / "points.csv", index=False)
    pd.DataFrame({"count_0": [int((~ever).sum() * 1000)], "count_1": [int(ever.sum() * 1000)]}).to_csv(out / "strata.csv", index=False)

    for T in years:
        f = features_for(sim, idx, T)
        f.insert(0, "year", T)
        f.insert(0, "point_id", points["point_id"].to_numpy())
        f.to_csv(out / f"features_{T}.csv", index=False, columns=["point_id", "year"] + CORE_FEATURES + ["burned"])

    g = features_for(sim, grid_idx, last_year)
    g.insert(0, "year", last_year)
    g.insert(0, "lat", sim["lat"][grid_idx]); g.insert(0, "lon", sim["lon"][grid_idx])
    g.insert(0, "point_id", [f"g{i}" for i in grid_idx])
    g["lossyear"] = sim["lossyear"][grid_idx]
    g.to_csv(out / f"grid_{last_year}_00.csv", index=False)

    res = 0.04
    ix = np.floor((sim["lon"] - bbox[0]) / res).astype(int)
    iy = np.floor((sim["lat"] - bbox[1]) / res).astype(int)
    hist = pd.DataFrame({"ix": ix, "iy": iy, "ly": sim["lossyear"]})
    hist = hist[hist.ly > 0]
    pv = hist.groupby(["ix", "iy", "ly"]).size().unstack("ly", fill_value=0) * 50
    pv.columns = [f"y{2000 + c}" for c in pv.columns]
    for y in range(2001, last_year + 1):
        pv[f"y{y}"] = pv.get(f"y{y}", 0)
    pv = pv[[f"y{y}" for y in range(2001, last_year + 1)]].reset_index()
    pv.to_csv(out / "history_00.csv", index=False)
    return out


def main():
    from .config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="data/synthetic/raw")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = generate(Path(args.out), cfg.bbox, cfg["data"]["feature_years"], cfg["data"]["gfc_last_year"])
    print(f"wrote synthetic exports to {out}")


if __name__ == "__main__":
    main()
