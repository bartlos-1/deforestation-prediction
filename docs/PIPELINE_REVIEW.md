# From v1 to v2: what was wrong with my first model, and how I fixed it

The first version of this project (`legacy/`, April 2025) was a Colab notebook with an XGBoost classifier and an LSTM.
It reported 75% accuracy and an F1 of 0.80, and my resume described it as 65–70% accurate.
When I went back to it in 2026, I found that the numbers didn't mean what I thought they meant.
This document explains what I got wrong, how I checked it, and what the rebuilt pipeline does instead.

## The short version

v1 didn't beat a trivial baseline. On the same test sets, predicting "loss" everywhere scored the same or better:

| v1 model | Reported | "Always predict loss" | Why the comparison was unfair |
|---|---|---|---|
| XGBoost | accuracy 0.747, F1 **0.803** | accuracy 0.680, F1 **0.809** | random split; 68% of points were positive |
| LSTM | accuracy 0.674, F1 **0.722** | accuracy 0.635, F1 **0.776** | scored on data it was trained on |

There were three separate problems. Any one of them would have been enough to make the results meaningless.

### 1. I was detecting old loss, not predicting new loss

My label was Hansen's `loss` band, which marks any tree-cover loss between 2001 and 2023.
My features came from 2020–2023 imagery. Most positive pixels had lost their trees years before that imagery was taken,
so the model learned to recognise land that was *already* cleared. That's why land-cover bands such as
`dw_shrub_and_scrub` and `dw_trees` topped the feature-importance chart.

**v2:** the question is now "for forest still standing at the end of year T, will it lose tree cover within the next 1, 2 or 3 years?"
Each row only includes pixels that were still forested at the end of T, and the label comes from Hansen's `lossyear` in (T, T+h].

### 2. The map on the dashboard was computed wrongly

I trained XGBoost on standardised features but called `predict_proba` on the raw, unscaled features when I built the map.
Only 2.7% of mapped points scored above 0.5, even though 63% of the training data was positive.
I had then tuned the dashboard's colour thresholds to that broken distribution, which hid the bug.

**v2:** no scaler at all (tree models don't need one), and the same code path builds features for training and for the map.

### 3. My evaluation leaked in several ways

- **Spatial leakage.** I split points at random, but neighbouring pixels share 5×5 neighbourhood features and look almost identical.
  The test set was effectively made of near-copies of training points.
- **The test set was used for tuning.** It was passed to XGBoost as `eval_set` and also used to pick the decision threshold.
- **The LSTM was scored on its own training data,** and its forecast for 2024–2030 was driven by simulated future inputs
  (hand-coded drift plus noise), not real data.

**v2:** splits by space *and* time, with gaps so no label window overlaps the next split:

| Split | Years T | Spatial blocks (~15 km) | Used for |
|---|---|---|---|
| train | 2003–2014 | 80% | fitting, and Optuna tuning with GroupKFold on blocks |
| val | 2015–2016 | 80% | probability calibration only |
| test | 2019–2024 | the other 20%, never seen | every number reported below |

I dropped the LSTM. With honest labels, gradient-boosted trees on lag features are the stronger and simpler choice for tabular pixel data.

## Smaller things I fixed along the way

- **The road layer was American.** `TIGER/2016/Roads` is the US Census network, and the study area is almost entirely in Canada,
  so the "distance to roads" feature was effectively constant. Roads are now an optional input from a Canadian source.
- **Imagery.** I replaced deprecated Sentinel-2 top-of-atmosphere data (which only starts in 2015) with
  harmonised Landsat 5/7/8/9 Collection 2 surface reflectance from 2003 onward, using growing-season composites.
- **Missing predictors.** I added the features the literature says matter most: distance to recent loss,
  the share of recent loss nearby, and fire history.
- **Class balancing.** My "downsampling" actually sampled negatives *with replacement*, and a later de-duplication step
  silently dropped about 18% of them. v2 samples on purpose and records the sampling rates,
  so metrics can be re-weighted back to the real forest population.
- **Reproducibility.** Code moved from one notebook into a package with a single `config.yaml`, pinned training requirements,
  unit tests, and an end-to-end test on synthetic data.

## v2 results, and what they mean

Loss is rare (about 1.5% of standing forest per year), so accuracy is useless here.
I report precision-recall AUC and "how much loss do we catch if we flag the riskiest X% of forest", and compare against baselines.
All numbers are on held-out regions and years, weighted to the full forest population.

| Horizon | Base rate | XGBoost PR-AUC | Logistic baseline | Distance-to-recent-loss baseline |
|---|---|---|---|---|
| within 1 year | 1.5% | **0.074** | 0.047 | 0.035 |
| within 2 years | 3.3% | **0.078** | 0.060 | 0.051 |
| within 3 years | 5.2% | **0.083** | 0.071 | 0.066 |

For the 1-year horizon, the 1% of forest the model rates riskiest loses tree cover at **10.7×** the average rate
(the best simple baseline manages 2.3×), and flagging the top 10% catches 31% of the loss that follows.

A PR-AUC of 0.07 looks small, but the right comparison is the base rate of 0.015: the model ranks risk about 5× better than random.
The advantage over the baselines shrinks at longer horizons, which is what I'd expect: the further ahead you look, the less today's features tell you.

## What I took away from this

1. **Always compare against a trivial baseline first.** If v1 had been scored against "always predict loss", I would have caught all of this in five minutes.
2. **Ask whether the label can be known from the features.** If it can, you're doing detection, not prediction.
3. **Split the way the model will be used.** For a map that forecasts the future, that means unseen places and later years, not random points.
4. **A suspiciously clean result needs more checking, not less.**
