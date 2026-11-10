"""Probability calibration (kept in its own module so pickles load from any entry point)."""

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Logistic recalibration on logit(p), weighted to the population.

    Stable on small calibration sets (isotonic regression overfits the top bins) and corrects the
    prevalence shift introduced by the stratified sample.
    """

    def fit(self, p, y, w):
        self.lr = LogisticRegression(C=1e4, max_iter=1000).fit(_logit(p)[:, None], y, sample_weight=w)
        return self

    def predict(self, p):
        return self.lr.predict_proba(_logit(p)[:, None])[:, 1]


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
