"""End-to-end run on synthetic exports: dataset → train → evaluate → predict → web data."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from deforest.config import ROOT
from deforest.synthetic import generate


def test_end_to_end(tmp_path: Path):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["model"]["optuna_trials"] = 2
    cfg["paths"] = {k: str(tmp_path / v) for k, v in cfg["paths"].items()}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    generate(Path(cfg["paths"]["raw"]), cfg["region"]["bbox"], cfg["data"]["feature_years"], cfg["data"]["gfc_last_year"],
             n_pop=15000, n_per_stratum=1200, n_grid=2000)

    subprocess.run([sys.executable, "-m", "deforest.pipeline", "--config", str(cfg_path), "--trials", "2"], check=True, cwd=ROOT)

    metrics = json.loads((Path(cfg["paths"]["web_data"]) / "metrics.json").read_text())
    assert metrics["schema"] == 2 and len(metrics["horizons"]) == 3
    for hz in metrics["horizons"]:
        m = hz["test"]["models"]
        # the synthetic signal is strong: the model must clearly beat a random ranking
        assert m["xgboost"]["pr_auc"] > 2 * hz["test"]["prevalence"]
    risk = json.loads((Path(cfg["paths"]["web_data"]) / "risk.json").read_text())
    assert risk["horizons"] == [1, 2, 3] and risk["cells"]
    # cumulative horizons must be monotone on average
    assert risk["mean_p"]["1"] <= risk["mean_p"]["2"] <= risk["mean_p"]["3"]
    assert (Path(cfg["paths"]["web_data"]) / "history.json").exists()
