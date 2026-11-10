"""Column names shared by the Earth Engine exporter and the Python side.

Every exported CSV uses these names; NaN is written by Earth Engine as NODATA and converted on load.
"""

NODATA = -9999

# Static per-point attributes (exported once with the point set).
POINT_COLS = ["point_id", "lon", "lat", "stratum", "lossyear", "treecover2000"]

SPECTRAL = ["blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "nbr", "ndmi"]
DELTAS = ["d1_ndvi", "d1_nbr", "d1_ndmi", "d3_ndvi", "d3_nbr"]
TERRAIN = ["elevation", "slope"]
CONTAGION = ["dist_recent_loss", "frac_recent_500m", "frac_recent_2km", "frac_past_2km"]
FIRE = ["burned_recent", "frac_burned_2km"]
OPTIONAL = {  # feature name → key in config.data.optional_assets
    "dist_road": "roads",
    "protected": "protected",
    "dist_built": "built",
}

CORE_FEATURES = SPECTRAL + DELTAS + TERRAIN + ["treecover2000"] + CONTAGION + FIRE

# Per-year, non-feature columns exported alongside the features.
YEAR_EXTRA = ["burned"]  # pixel burned (MODIS MCD64A1) during calendar year T — used for driver attribution only

# Baseline logistic regression uses only these.
BASELINE_FEATURES = ["dist_recent_loss", "frac_recent_2km", "frac_past_2km", "d1_nbr"]


def feature_columns(available_optional: list[str] | None = None) -> list[str]:
    """Model features: core + whichever optional features were exported."""
    return CORE_FEATURES + [f for f in OPTIONAL if available_optional and f in available_optional]
