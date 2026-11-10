# Training pipeline review

Scope: `Deforestation_Model_Pipeline.ipynb` (Colab, latest version in this repo), the prediction CSVs, and `main.py`.
`model_notebook/…/Deforestation_Model_Pipeline.ipynb` is an older draft of the same notebook (Apr 6) and isn't covered separately.

## TL;DR

The current numbers don't show that either model can predict deforestation:

| | Reported | Trivial baseline ("always loss") | Notes |
|---|---|---|---|
| XGBoost | acc 0.747, F1 **0.803** | acc 0.680, F1 **0.809** | random split, prevalence 68% |
| LSTM | acc 0.674, F1 **0.722**, AUC 0.745 | acc 0.635, F1 **0.776** | evaluated on data it was trained on |

There are three root causes, and each one alone is enough to invalidate the results:

1. **The target is wrong for a forecasting task.** `loss` is Hansen's "lost at any time 2001–2023", while the features come from 2020–2023 imagery. The model learns to recognise land that has *already* been cleared. That's why `dw_shrub_and_scrub`, `dw_trees` and `dw_grass` top the importance chart. This is detection, not prediction.
2. **The dashboard map is computed incorrectly.** `predict_proba(df[classifier_bands])` was run on *raw* features, but the model was trained on `StandardScaler` output. As a result only 2.7% of mapped points score ≥ 0.5, even though the training data is 63% positive and test recall was 76%. The dashboard thresholds (0.04 / 0.19 / 0.29) were tuned to that broken distribution.
3. **The evaluation leaks in several ways.** The split is random by point while the features use 5×5-pixel windows (spatial autocorrelation). The test set is passed as `eval_set` and also used to choose the threshold. The LSTM is scored on all 39,389 sequences, including its own training data.

---

## Findings, ranked

### Critical

**C1 — Label/feature timing (classifier and LSTM).** `gfc.select("loss")` is a static band covering 2001–2023. Most positives lost their trees years before the 2020–2023 imagery was taken (e.g. `lossyear` 6, 14, 16 in the first rows). Neither the `treecover2000 ≥ 30` mask nor anything else removes pixels that were already cleared.
*Fix:* reframe as "given the state at end of year T, will this still-forested pixel lose tree cover in T+1 (or T+1…T+3)?" Keep only pixels with `lossyear == 0 or lossyear > T`, and set the label to `lossyear == T+1-2000`.

**C2 — Unscaled inference** (notebook cell 26 → `classifier_predictions.csv`). The model was trained on `scaler_class.fit_transform(...)` but used for prediction on raw `df[classifier_bands]`.
*Fix:* drop the scaler entirely (trees don't need it), or wrap scaler and model in a single `sklearn.pipeline.Pipeline` and save that.

**C3 — The LSTM label carries no temporal information.** Each pixel yields exactly one sequence (2020–2022 → 2023), and the "2023 label" is the same static `loss` band. The sequence can't encode anything about *when* loss happens.

**C4 — The 2024–2030 forecast is driven by simulation.** The future features are hand-coded drifts plus Gaussian noise added to MinMax-scaled values. The model was trained with `SEQ_LEN=3` but given 4-step sequences, and an arbitrary threshold of 0.98 was used (the tuned value was 0.44). The steady decline from 16.8% to 9.1% flagged comes from this setup, not from data. Year 2026 also has 373 more pixels than the other years.

**C5 — No baselines and the wrong headline metric.** Accuracy and F1 on a 63–68% positive set say little on their own. Neither model beats "predict loss everywhere" on F1.

### High

**H1 — Leaky splits.** The `train_test_split` is random over points, while `NDVI_stddev_5x5` and `slope_stddev_5x5` share neighbourhoods and nearby pixels are strongly correlated. The "0 overlapping coordinates" check doesn't rule this out.
*Fix:* use spatial blocks (`GroupKFold` on ~10–25 km tiles) plus a *temporal* hold-out (train T ≤ 2019, validate 2020–21, test 2022–23).

**H2 — Test set reused for model selection.** `eval_set=[(Xc_test, yc_test)]` and the precision/recall threshold plot both use the test set. `xgb.cv` runs afterwards but its result isn't used (no early stopping, so the full 1000 trees are kept).

**H3 — LSTM evaluated on training data.** `model.predict(X)` is run on the full set, and the best threshold is chosen on the same data. Its val_accuracy (≈0.58–0.64) is at the majority-class rate. Also, `validation_split=0.2` on data sorted by `lat` takes the northernmost strip, which is a reasonable spatial hold-out by accident, but its numbers were never reported.

**H4 — Broken road feature.** `TIGER/2016/Roads` is the **US** Census road network. The ROI is almost entirely in Canada, so `distance_to_roads` is effectively empty or constant.
*Fix:* use the National Road Network (Statistics Canada / NRCan), OSM, or GRIP global roads uploaded as an asset.

**H5 — Sentinel-2 setup.** `COPERNICUS/S2` is deprecated L1C top-of-atmosphere data. It's filtered only by scene cloud % (no pixel cloud masking), and a full-year median at 49–54°N mixes in snow. Sentinel-2 also starts mid-2015, so it can't provide history for 2001–2014.
*Fix:* use `COPERNICUS/S2_SR_HARMONIZED` with Cloud Score+ and a June–September composite. For long histories, use Landsat 5/7/8/9 Collection 2 SR (harmonised) from 2000 onward.

**H6 — Class balancing doesn't do what the comment says.** The "downsampling" samples negatives *with replacement* up to the positive count (an oversample). The lat/lon `isin` re-match then quietly de-duplicates them, so about 18% of negatives are dropped at random. `scale_pos_weight` is then computed on the result. With a correct label (C1), positives will be rare (~0.5–2% per year), so keep the natural prevalence and use `scale_pos_weight` or focal loss, but not both stacked.

### Medium

- **M1 — Leaky or redundant features.** `gain` comes from Hansen 2000–2012 and mostly marks regrowth after earlier loss, which is target-adjacent. `delta_ndvi` and `NDVI_2yr_delta` are identical (both are year−(year−2)). `NDVI_trend` is a 3-year mean, not a trend. The keys in `delta_dict` are swapped (`NDVI_trend_` ↔ `NDVI_2yr_delta_`); the band `rename`s save it, but it's fragile.
- **M2 — Missing the strongest known predictors.** These are distance to *recent* loss (contagion), the share of loss in the neighbourhood over the past 1–3 years, fire history (CWFIS / NBAC burned area), harvest tenure and cutblocks (BC consolidated cutblocks, Alberta AVI / harvest areas), protected areas (WDPA in GEE), forest age, and distance to settlements and well sites.
- **M3 — "Loss" ≠ "deforestation".** Most Hansen loss in BC/AB is wildfire or harvest followed by regrowth. Decide what you're predicting: if it's *any* stand-replacing disturbance, say so. For actual land-use change, use a drivers layer (e.g. WRI/Google tree-cover-loss drivers, or CanLaD / NTEMS disturbance-type data) to separate fire from harvest from conversion.
- **M4 — The ROI is mostly British Columbia**, not Alberta. It spans 122.8°W–113.5°W, while Alberta's western border follows the Rockies and then 120°W. Either rename the project or move the ROI to −120…−110°W.
- **M5 — Hansen version.** The notebook uses v1.11 (2023), but `main.py` says v1.10. A newer release covering 2024 should be in the GEE catalogue; use it, and you get one more labelled year for testing.
- **M6 — The LSTM loss setup conflicts with itself.** Focal loss with `alpha=0.25` *down*-weights positives, while `class_weight='balanced'` up-weights them. `MinMaxScaler` is fit on all data before the split.

### Reproducibility

- The notebook depends on Colab + Drive, and its exports were started by hand (`if year == 2023 and tile_idx == 63`). It reads from two Drive folders (`no-overlap-data` for the classifier, `test` for the LSTM) built by different sampling functions.
- `requirements.txt` covers only the dashboard. There's no `xgboost`, `tensorflow`, `scikit-learn` or `earthengine-api`.
- The metrics in `main.py` are hard-coded from different runs. The LSTM confusion matrix in the dashboard doesn't match the notebook.
- Artifacts (`xgb_classifier_model_2.pkl`, `scaler_classifier_2.pkl`, `lstm_forest_loss_model.keras`) aren't in the repo.

---

## Recommended pipeline

```
ee_export.py   → per-year feature stacks for T = 2005…2023 (Landsat SR, growing season)
                 sampled on a fixed grid of points, exported to GCS / Drive as Parquet
build_dataset.py → panel table: (pixel_id, block_id, T, features_T, lag features, label_{T+1})
                   filters: forested at end of T (treecover2000 ≥ 30 & (lossyear==0 | lossyear > T))
train.py       → XGBoost/LightGBM, spatial-block CV inside the train years, Optuna, early stopping on val
evaluate.py    → temporal test years, metrics.json (+ baselines, calibration, precision@k)
predict.py     → wall-to-wall risk raster for T = latest year → COG / tiles (or classify in-GEE)
```

### 1. Labels in Earth Engine (sketch)

```python
gfc = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")   # or the newest release
lossyear = gfc.select("lossyear")
def label_image(T):
    still_forest = gfc.select("treecover2000").gte(30).And(lossyear.eq(0).Or(lossyear.gt(T - 2000)))
    y_next = lossyear.eq(T + 1 - 2000).rename("y")
    recent = lossyear.gte(T - 2000 - 2).And(lossyear.lte(T - 2000)).And(lossyear.gt(0))
    # cost image of 1 per metre, sources = recent-loss pixels → distance in metres (capped at 20 km)
    dist_recent = ee.Image(1).cumulativeCost(source=recent.selfMask(), maxDistance=20000).unmask(20000).rename("dist_recent_loss")
    frac_recent = recent.reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.circle(1000, "meters")).rename("frac_recent_loss_1km")
    return y_next.addBands([dist_recent, frac_recent]).updateMask(still_forest)
```
Use `stratifiedSample` on `y` to keep enough positives, but **record the sampling rates** so you can re-weight metrics back to the true prevalence.

### 2. Splits and training

```python
from sklearn.model_selection import GroupKFold
import xgboost as xgb

train = df[df.T <= 2019]; val = df[df.T.between(2020, 2021)]; test = df[df.T >= 2022]
# block_id = floor(lon / 0.2), floor(lat / 0.15)  →  ~15 km blocks
model = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
                          early_stopping_rounds=100,
                          scale_pos_weight=(train.y == 0).sum() / (train.y == 1).sum())
model.fit(train[FEATS], train.y, eval_set=[(val[FEATS], val.y)])
# hyper-parameter search: Optuna over GroupKFold(groups=train.block_id) — never touch `test`
```

### 3. Evaluation that answers "is it useful?"

- **PR-AUC** (primary, given low prevalence), ROC-AUC, Brier score, and a calibration curve. Fit isotonic or Platt calibration on `val`.
- **Precision / recall at the top k% of area** (e.g. 1%, 5%, 10% of forest). That's how a land manager would use a risk map.
- **Baselines to beat:** (a) prevalence; (b) a score of `−distance to recent loss`; (c) logistic regression on 3–4 features. If XGBoost can't beat (b), the extra features aren't adding value.
- Report each test year separately, and by sub-region (BC vs AB, fire vs harvest if you have drivers).
- Write all of this to `metrics.json`. The new dashboard reads that file directly.

### 4. Should you keep the LSTM?

With honest yearly labels (T = 2005…2022) and Landsat history, you could build a real sequence model. In practice, gradient-boosted trees with lag features (Δ-NDVI over 1/2/3 years, recent-loss distance at T, T−1, T−2) usually match or beat an LSTM on tabular pixel data. If you want deep learning, a small U-Net or ConvLSTM on image patches has an actual advantage, because it uses spatial context. Either way, **forecast multiple horizons directly** (T+1, T+2, T+3 heads trained on real future labels) instead of feeding the model simulated future inputs.

### 5. Wall-to-wall predictions

Scoring 158k random points and smoothing them with a Gaussian blur gives a blurry map that's hard to interpret. Options:
- Train in Python, then build the features for the latest year in GEE, export them, score them, and write a GeoTIFF/COG.
- Or train an `ee.Classifier.smileGradientTreeBoost` directly in GEE and export the probability raster.

For the web, aggregate onto a grid (as `scripts/build_web_data.py` already does) or serve PMTiles.

### 6. Housekeeping

- Move the code out of the notebook into the scripts above with a `config.yaml` (ROI, years, scale, seeds). Keep the notebook only for exploration.
- Pin the training environment (`requirements-train.txt` or `pyproject` with `xgboost`, `scikit-learn`, `earthengine-api`, `pandas`, `pyarrow`, `optuna`).
- Save the whole pipeline object plus the feature list plus a git hash next to each model, and keep model files out of git (GCS/Drive, or Git LFS).
- Don't commit the 67 MB `loss_data.csv`. The web build only needs the aggregated JSON.
