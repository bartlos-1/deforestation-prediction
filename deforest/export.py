"""Launch Earth Engine export tasks (Google Drive).

    python -m deforest.export verify                 # check assets, record which optional features exist
    python -m deforest.export points                 # stratified training points → asset + Drive CSV, strata counts
    python -m deforest.export features [--years 2003-2023] [--parts 4]
    python -m deforest.export grid [--year 2023]     # wall-to-wall prediction grid
    python -m deforest.export history                # per-cell annual loss counts for the dashboard
    python -m deforest.export status                 # list recent tasks

After tasks finish, download the CSVs from the Drive folder in config.yaml into data/raw/.
"""

from __future__ import annotations

import argparse
import json
import math

import ee

from .config import load_config
from .ee_features import FeatureBuilder, roi_geometry

HISTORY_RES = 0.04  # degrees — must match scripts/build_web_data.py


def init(cfg) -> None:
    ee.Initialize(project=cfg["project"]["ee_project"])


def asset_exists(asset_id: str) -> bool:
    try:
        ee.data.getAsset(asset_id)
        return True
    except ee.EEException:
        return False


def _assets_file(cfg):
    return cfg.path("processed") / "ee_assets.json"


def load_available(cfg) -> list[str]:
    f = _assets_file(cfg)
    if not f.exists():
        raise SystemExit("Run `python -m deforest.export verify` first.")
    return json.loads(f.read_text())["available_optional"]


def _drive(cfg, fc, name: str, selectors: list[str]):
    task = ee.batch.Export.table.toDrive(
        collection=fc, description=name[:100], folder=cfg["project"]["drive_folder"],
        fileNamePrefix=name, fileFormat="CSV", selectors=selectors,
    )
    task.start()
    print(f"started {name}  ({task.id})")
    return task


def _tiles(bbox, nx: int, ny: int):
    x0, y0, x1, y1 = bbox
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    for j in range(ny):
        for i in range(nx):
            yield i + j * nx, [x0 + i * dx, y0 + j * dy, x0 + (i + 1) * dx, y0 + (j + 1) * dy]


# ---------------------------------------------------------------- commands
def cmd_verify(cfg, _args):
    d = cfg["data"]
    print(f"GFC asset {d['gfc_asset']}: {'ok' if asset_exists(d['gfc_asset']) else 'MISSING'}")
    for cand in ["UMD/hansen/global_forest_change_2025_v1_13", "UMD/hansen/global_forest_change_2024_v1_12"]:
        if asset_exists(cand):
            print(f"  newer GFC release available: {cand} — update data.gfc_asset and data.gfc_last_year")
            break
    from .schema import OPTIONAL

    available = []
    for feat, key in OPTIONAL.items():
        aid = d["optional_assets"].get(key)
        ok = bool(aid) and asset_exists(aid)
        print(f"optional {feat:<11} {aid}: {'ok' if ok else 'missing — feature skipped'}")
        if ok:
            available.append(feat)
    _assets_file(cfg).write_text(json.dumps({"available_optional": available}, indent=2))
    print(f"wrote {_assets_file(cfg)}")


def cmd_points(cfg, _args):
    fb = FeatureBuilder(cfg)
    root = cfg["project"]["asset_root"]
    if not asset_exists(root):
        ee.data.createAsset({"type": "FOLDER"}, root)
        print(f"created asset folder {root}")
    pts = fb.training_points()
    asset_id = f"{root}/training_points"
    t = ee.batch.Export.table.toAsset(collection=pts, description="training_points_asset", assetId=asset_id)
    t.start()
    print(f"started training_points_asset → {asset_id}  ({t.id})")
    _drive(cfg, pts, "points", ["point_id", "lon", "lat", "stratum", "lossyear", "treecover2000"])
    _drive(cfg, fb.strata_counts(), "strata", ["count_0", "count_1"])


def _parse_years(s: str, cfg) -> list[int]:
    if not s:
        a, b = cfg["data"]["feature_years"]
    else:
        a, b = (int(x) for x in s.split("-")) if "-" in s else (int(s), int(s))
    return list(range(a, b + 1))


def cmd_features(cfg, args):
    asset_id = f"{cfg['project']['asset_root']}/training_points"
    if not asset_exists(asset_id):
        raise SystemExit(f"{asset_id} not found — run `points` and wait for the asset task to finish.")
    fb = FeatureBuilder(cfg, load_available(cfg))
    pts = ee.FeatureCollection(asset_id)
    # --parts N splits the points into N longitude strips (one task each) for years that time out.
    x0, _, x1, _ = cfg.bbox
    edges = [x0 + (x1 - x0) * k / args.parts for k in range(args.parts + 1)]
    edges[0], edges[-1] = -181, 181
    for T in _parse_years(args.years, cfg):
        img = fb.image_for_year(T)
        only = [int(x) for x in args.only.split(",")] if args.only else range(args.parts)
        for k in only:
            sub = pts if args.parts == 1 else pts.filter(
                ee.Filter.And(ee.Filter.gte("lon", edges[k]), ee.Filter.lt("lon", edges[k + 1])))
            fc = img.sampleRegions(
                collection=sub, properties=["point_id"], scale=cfg["data"]["scale"], tileScale=16 if args.parts > 1 else 8
            ).map(lambda f, T=T: f.set("year", T))
            name = f"features_{T}" if args.parts == 1 else f"features_{T}_p{k}"
            _drive(cfg, fc, name, ["point_id", "year"] + fb.band_names())


def cmd_grid(cfg, args):
    fb = FeatureBuilder(cfg, load_available(cfg))
    T = args.year or cfg["data"]["gfc_last_year"]
    img = fb.image_for_year(T, with_labels=True)
    for k, tile in _tiles(cfg.bbox, args.nx, args.ny):
        pts = fb.grid_points(roi_geometry(tile), cfg["data"]["grid_spacing"])
        fc = img.sampleRegions(collection=pts, properties=["point_id", "lon", "lat"], scale=cfg["data"]["scale"], tileScale=8)
        fc = fc.map(lambda f, T=T: f.set("year", T))
        _drive(cfg, fc, f"grid_{T}_{k:02d}", ["point_id", "lon", "lat", "year"] + fb.band_names(with_labels=True))


def cmd_history(cfg, args):
    fb = FeatureBuilder(cfg)
    x0, y0, x1, y1 = cfg.bbox
    nx, ny = math.ceil((x1 - x0) / HISTORY_RES), math.ceil((y1 - y0) / HISTORY_RES)
    years = list(range(2001, cfg["data"]["gfc_last_year"] + 1))
    img = fb.history_image(years)
    step_x, step_y = math.ceil(nx / args.nx), math.ceil(ny / args.ny)
    k = 0
    for ix0 in range(0, nx, step_x):
        for iy0 in range(0, ny, step_y):
            cells = fb.history_cells(ix0, min(nx, ix0 + step_x), iy0, min(ny, iy0 + step_y), HISTORY_RES)
            fc = img.reduceRegions(collection=cells, reducer=ee.Reducer.sum(), scale=cfg["data"]["scale"], tileScale=16)
            _drive(cfg, fc, f"history_{k:02d}", ["ix", "iy"] + [f"y{y}" for y in years])
            k += 1


def cmd_status(_cfg, _args):
    for t in ee.batch.Task.list()[:40]:
        s = t.status()
        print(f"{s.get('state', '?'):<10} {s.get('description', ''):<28} {s.get('error_message', '')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    sub.add_parser("points")
    p = sub.add_parser("features"); p.add_argument("--years", default=""); p.add_argument("--parts", type=int, default=1); p.add_argument("--only", default="", help="comma-separated part indices to (re)run")
    p = sub.add_parser("grid"); p.add_argument("--year", type=int); p.add_argument("--nx", type=int, default=4); p.add_argument("--ny", type=int, default=4)
    p = sub.add_parser("history"); p.add_argument("--nx", type=int, default=3); p.add_argument("--ny", type=int, default=3)
    sub.add_parser("status")
    args = ap.parse_args()
    cfg = load_config(args.config)
    init(cfg)
    {"verify": cmd_verify, "points": cmd_points, "features": cmd_features, "grid": cmd_grid,
     "history": cmd_history, "status": cmd_status}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
