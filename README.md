# deforestation-prediction

Near-term tree-cover loss risk for the southern Rockies (BC / AB): Earth Engine features, XGBoost models with honest spatial and temporal validation, and a static web dashboard.

**Question the model answers:** for forest still standing at the end of year *T*, what is the probability that Hansen records tree-cover loss within the next 1, 2 or 3 years?

See [`docs/PIPELINE_REVIEW.md`](docs/PIPELINE_REVIEW.md) for why the original notebook (`legacy/`) was replaced.

## Layout

```
config.yaml            region, years, sample sizes, splits — the only file you normally edit
deforest/
  ee_features.py       Earth Engine feature stack (Landsat C2 SR, contagion, fire, terrain, optional roads/PA/built)
  export.py            CLI that launches Drive exports
  dataset.py           raw CSVs → panel.parquet (eligibility, labels, sampling weights, spatial blocks)
  train.py             Optuna + spatial-block CV, XGBoost per horizon, Platt calibration (eval + prod models)
  evaluate.py          held-out metrics vs baselines → reports/metrics.json + web/data/metrics.json
  predict.py           scores the 1 km grid for the latest year → data/processed/risk_<T>.parquet
  pipeline.py          runs dataset → train → evaluate → predict → web data
  synthetic.py         fake exports with the same schema (tests / dry runs)
scripts/build_web_data.py   aggregates outputs into web/data/*.json
web/                   static dashboard (MapLibre, no build step)
tests/                 unit tests + end-to-end run on synthetic data
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-train.txt
.venv/bin/earthengine authenticate          # one-time browser login
.venv/bin/python -m pytest                  # should pass in about 30 s
```

## Runbook

```bash
source .venv/bin/activate

# 1. Earth Engine (tasks run on Google's side; check with `status` or code.earthengine.google.com/tasks)
python -m deforest.export verify            # checks assets, decides which optional features exist
python -m deforest.export points            # training points → asset + Drive; wait until finished (~10–30 min)
python -m deforest.export features          # 21 tasks, one per year 2003–2023
python -m deforest.export features --years 2010 --parts 4   # rerun a year that timed out, as 4 smaller tasks
python -m deforest.export grid              # 16 tiles of the 1 km prediction grid for 2023
python -m deforest.export history           # 9 tiles of per-cell annual loss for the History tab
python -m deforest.export status

# 2. Download everything in Drive/deforest_v2/ into data/raw/
#    expected: points.csv, strata.csv, features_2003.csv … features_2023.csv, grid_2023_*.csv, history_*.csv

# 3. Local pipeline (≈ 10–40 min depending on optuna_trials)
python -m deforest.pipeline

# 4. Look at it
cd web && python -m http.server 8000        # http://localhost:8000
```

## Validation design

| Split | Years *T* | Blocks | Used for |
|---|---|---|---|
| train | 2003–2014, with *T + h* ≤ 2015 | 80% | fitting and Optuna CV (GroupKFold on ~15 km blocks) |
| val | 2015–2016, with *T + h* ≤ 2019 | 80% | probability calibration |
| **test** | 2019–2022, with *T + h* ≤ 2023 | **20%, never seen** | every reported number |
| prod | everything | all | the map; trained up to *T* = last − h − 2, calibrated on the last two labelled years |

Metrics are weighted back to the true forest population, which undoes the 50/50 "ever lost / never lost" sample. Baselines are reported alongside: the base rate, distance to recent loss, and a 4-feature logistic regression.

## Dashboard

`web/` is a static site: copy it to GitHub Pages, Netlify or any static host. It reads `web/data/{history,risk,metrics}.json`, so commit those after a real run.
