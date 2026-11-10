import numpy as np
import pandas as pd
import pytest

from deforest.config import load_config
from deforest.dataset import add_fire_driver, add_labels, stratum_weights
from deforest.metrics import top_k
from deforest.train import prod_splits, splits


def _rows(lossyear, years):
    return pd.DataFrame({"point_id": "a", "year": years, "lossyear": lossyear, "treecover2000": 80})


def test_labels_and_eligibility():
    # loss in 2010 (lossyear 10)
    df = add_labels(_rows(10, [2007, 2008, 2009, 2010, 2011]), [1, 3], last_year=2023, treecover_min=30)
    assert df["eligible"].tolist() == [True, True, True, False, False]   # already lost by end of 2010
    assert df["y1"].tolist()[:3] == [0, 0, 1]                              # only T=2009 sees loss in T+1
    assert df["y3"].tolist()[:3] == [1, 1, 1]                              # 2010 ∈ (T, T+3] for T=2007..2009


def test_labels_are_nan_past_last_year():
    df = add_labels(_rows(0, [2021, 2022, 2023]), [1, 3], last_year=2023, treecover_min=30)
    assert df["y1"].isna().tolist() == [False, False, True]
    assert df["y3"].isna().tolist() == [True, True, True]


def test_never_lost_and_low_cover():
    df = add_labels(_rows(0, [2005]), [1], 2023, 30)
    assert df["eligible"].item() and df["y1"].item() == 0
    low = _rows(0, [2005]).assign(treecover2000=10)
    assert not add_labels(low, [1], 2023, 30)["eligible"].item()


def test_fire_driver_uses_loss_year_or_previous():
    df = pd.DataFrame({
        "point_id": ["a"] * 3 + ["b"] * 3, "year": [2009, 2010, 2011] * 2,
        "lossyear": [11] * 3 + [11] * 3, "burned": [0, 1, 0, 0, 0, 0],
    })
    out = add_fire_driver(df).groupby("point_id")["fire_loss"].first()
    assert out["a"] and not out["b"]   # a burned in 2010 = loss year − 1


def test_stratum_weights_recover_population_share():
    pts = pd.DataFrame({"stratum": [0] * 50 + [1] * 50})
    strata = pd.DataFrame({"count_0": [900], "count_1": [100]})
    w = stratum_weights(pts, strata)
    assert w[0] == pytest.approx(1.8) and w[1] == pytest.approx(0.2)


def test_splits_respect_embargo_and_blocks():
    cfg = load_config()
    years = np.arange(2003, 2023)
    panel = pd.DataFrame({
        "year": np.tile(years, 2), "eligible": True, "y3": 0.0, "w": 1.0,
        "test_block": np.repeat([False, True], len(years)), "block_id": np.repeat([1, 2], len(years)),
    })
    s = splits(panel, cfg, 3)
    v0 = cfg["model"]["val_years"][0]
    te0 = cfg["model"]["test_years"][0]
    assert (s["train"]["year"] + 3 <= v0).all()
    assert (s["val"]["year"] + 3 <= te0).all()
    assert not s["train"]["test_block"].any() and s["test"]["test_block"].all()
    assert (s["test"]["year"] + 3 <= cfg["data"]["gfc_last_year"]).all()
    p = prod_splits(panel.assign(y3=np.where(panel["year"] + 3 <= 2023, 0.0, np.nan)), cfg, 3)
    assert p["train"]["year"].max() + 3 <= p["cal"]["year"].min()


def test_top_k():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0], float)
    p = np.linspace(1, 0, 10)
    r = top_k(y, p, np.ones(10), [0.1, 0.5])
    assert r["0.1"]["precision"] == 1.0 and r["0.1"]["recall"] == 0.5
    assert r["0.5"]["recall"] == 1.0
