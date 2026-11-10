"""Earth Engine feature construction.

Everything for year T uses only information available by the end of T's growing season, so a model
trained on (features_T → loss in (T, T+h]) can be applied to the latest year without leakage.
"""

from __future__ import annotations

import ee

from .schema import CORE_FEATURES, NODATA, OPTIONAL, YEAR_EXTRA

L57 = {"SR_B1": "blue", "SR_B2": "green", "SR_B3": "red", "SR_B4": "nir", "SR_B5": "swir1", "SR_B7": "swir2"}
L89 = {"SR_B2": "blue", "SR_B3": "green", "SR_B4": "red", "SR_B5": "nir", "SR_B6": "swir1", "SR_B7": "swir2"}
LANDSAT = [
    ("LANDSAT/LT05/C02/T1_L2", L57),
    ("LANDSAT/LE07/C02/T1_L2", L57),
    ("LANDSAT/LC08/C02/T1_L2", L89),
    ("LANDSAT/LC09/C02/T1_L2", L89),
]
# QA_PIXEL bits: 1 dilated cloud, 3 cloud, 4 cloud shadow, 5 snow
QA_MASK = (1 << 1) | (1 << 3) | (1 << 4) | (1 << 5)
MODIS_BA = "MODIS/061/MCD64A1"
DEM = "USGS/SRTMGL1_003"
GHSL_EPOCHS = list(range(1975, 2035, 5))


def roi_geometry(bbox: list[float]) -> ee.Geometry:
    return ee.Geometry.Rectangle(bbox, "EPSG:4326", False)


def _prep_landsat(img: ee.Image, mapping: dict[str, str]) -> ee.Image:
    clear = img.select("QA_PIXEL").bitwiseAnd(QA_MASK).eq(0)
    unsaturated = img.select("QA_RADSAT").eq(0)
    sr = img.select(list(mapping)).multiply(0.0000275).add(-0.2).rename(list(mapping.values()))
    return ee.Image(sr.updateMask(clear).updateMask(unsaturated).copyProperties(img, ["system:time_start"]))


class FeatureBuilder:
    def __init__(self, cfg, available_optional: list[str] | None = None):
        d = cfg["data"]
        self.cfg = cfg
        self.roi = roi_geometry(cfg.bbox)
        self.season = d["season"]
        self.treecover_min = d["treecover_min"]
        gfc = ee.Image(d["gfc_asset"])
        self.lossyear = gfc.select("lossyear").unmask(0)
        self.treecover = gfc.select("treecover2000")
        self.forest2000 = self.treecover.gte(self.treecover_min)
        # coarse grid for neighbourhood statistics (cheap and plenty for 500 m–2 km context)
        self.coarse = ee.Projection("EPSG:3005").atScale(150)
        self.available = set(available_optional or [])
        self.optional_assets = d.get("optional_assets", {})

        roi_buf = self.roi.buffer(20000)
        col = None
        for asset, mapping in LANDSAT:
            c = ee.ImageCollection(asset).filterBounds(roi_buf).map(lambda i, m=mapping: _prep_landsat(i, m))
            col = c if col is None else col.merge(c)
        self.landsat = col

    # ---------------------------------------------------------------- spectral
    def composite(self, year: int) -> ee.Image:
        s0, s1 = self.season
        img = self.landsat.filterDate(f"{year}-{s0}", f"{year}-{s1}").median()
        ndvi = img.normalizedDifference(["nir", "red"]).rename("ndvi")
        nbr = img.normalizedDifference(["nir", "swir2"]).rename("nbr")
        ndmi = img.normalizedDifference(["nir", "swir1"]).rename("ndmi")
        return img.addBands([ndvi, nbr, ndmi])

    # ---------------------------------------------------------------- helpers
    def _to_coarse_fraction(self, mask: ee.Image) -> ee.Image:
        return (
            mask.unmask(0).toFloat()
            .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
            .reproject(self.coarse)
        )

    def _neigh_mean(self, img: ee.Image, radius_m: int) -> ee.Image:
        return img.reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.circle(radius_m, "meters")).reproject(self.coarse)

    def _distance_to(self, mask_coarse: ee.Image, cap_px: int = 100) -> ee.Image:
        """Distance (m) on the 150 m grid to the nearest non-zero cell, capped at cap_px * 150 m."""
        cap_m = cap_px * 150
        d = (
            mask_coarse.gt(0).int()
            .fastDistanceTransform(cap_px, "pixels", "squared_euclidean")
            .sqrt().multiply(150)
            .reproject(self.coarse)
        )
        return d.min(cap_m).unmask(cap_m)

    def _burned(self, year: int) -> ee.Image:
        return (
            ee.ImageCollection(MODIS_BA).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .select("BurnDate").max().gt(0).unmask(0)
        )

    # ---------------------------------------------------------------- feature groups
    def contagion(self, T: int) -> ee.Image:
        ly, t = self.lossyear, T - 2000
        recent = ly.gte(t - 2).And(ly.lte(t)).And(ly.gt(0))
        past = ly.gt(0).And(ly.lte(t))
        rf = self._to_coarse_fraction(recent)
        pf = self._to_coarse_fraction(past)
        return ee.Image.cat([
            self._distance_to(rf).rename("dist_recent_loss"),
            self._neigh_mean(rf, 500).rename("frac_recent_500m"),
            self._neigh_mean(rf, 2000).rename("frac_recent_2km"),
            self._neigh_mean(pf, 2000).rename("frac_past_2km"),
        ])

    def fire(self, T: int) -> ee.Image:
        recent = self._burned(T).Or(self._burned(T - 1)).Or(self._burned(T - 2))
        frac = recent.toFloat().reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.circle(2000, "meters"))
        return ee.Image.cat([recent.rename("burned_recent"), frac.rename("frac_burned_2km"), self._burned(T).rename("burned")])

    def terrain(self) -> ee.Image:
        dem = ee.Image(DEM).select("elevation")
        return dem.addBands(ee.Terrain.slope(dem).rename("slope"))

    def optional(self, T: int) -> list[ee.Image]:
        out = []
        a = self.optional_assets
        if "dist_road" in self.available:
            roads = ee.FeatureCollection(a["roads"]).filterBounds(self.roi.buffer(20000))
            raster = ee.Image(0).byte().paint(roads, 1).reproject(self.coarse)
            out.append(self._distance_to(raster, cap_px=200).rename("dist_road"))
        if "protected" in self.available:
            pa = ee.FeatureCollection(a["protected"]).filterBounds(self.roi).filter(ee.Filter.lte("STATUS_YR", T))
            out.append(ee.Image(0).byte().paint(pa, 1).rename("protected"))
        if "dist_built" in self.available:
            epoch = max(e for e in GHSL_EPOCHS if e <= T)
            built = (
                ee.ImageCollection(a["built"]).filterDate(f"{epoch}-01-01", f"{epoch}-12-31").first()
                .select("built_surface").gt(500)
            )
            out.append(self._distance_to(built.reproject(self.coarse), cap_px=200).rename("dist_built"))
        return out

    # ---------------------------------------------------------------- full stack
    def image_for_year(self, T: int, with_labels: bool = False) -> ee.Image:
        c, c1, c3 = self.composite(T), self.composite(T - 1), self.composite(T - 3)
        deltas = ee.Image.cat([
            c.select("ndvi").subtract(c1.select("ndvi")).rename("d1_ndvi"),
            c.select("nbr").subtract(c1.select("nbr")).rename("d1_nbr"),
            c.select("ndmi").subtract(c1.select("ndmi")).rename("d1_ndmi"),
            c.select("ndvi").subtract(c3.select("ndvi")).rename("d3_ndvi"),
            c.select("nbr").subtract(c3.select("nbr")).rename("d3_nbr"),
        ])
        img = ee.Image.cat([
            c, deltas, self.terrain(), self.treecover.rename("treecover2000"),
            self.contagion(T), self.fire(T), *self.optional(T),
        ])
        if with_labels:
            img = img.addBands(self.lossyear.rename("lossyear"))
        return img.unmask(NODATA)

    def band_names(self, with_labels: bool = False) -> list[str]:
        opt = [f for f in OPTIONAL if f in self.available]
        return CORE_FEATURES + opt + YEAR_EXTRA + (["lossyear"] if with_labels else [])

    # ---------------------------------------------------------------- sampling designs
    def training_points(self) -> ee.FeatureCollection:
        d = self.cfg["data"]
        stratum = self.lossyear.gt(0).rename("stratum").updateMask(self.forest2000).int()
        img = stratum.addBands(self.lossyear.rename("lossyear")).addBands(self.treecover.rename("treecover2000"))
        pts = img.stratifiedSample(
            numPoints=0, classBand="stratum", region=self.roi, scale=d["scale"], seed=d["seed"],
            classValues=[0, 1], classPoints=[d["n_points_never_loss"], d["n_points_ever_loss"]],
            geometries=True, tileScale=16,
        )
        return pts.map(_with_lonlat)

    def strata_counts(self) -> ee.FeatureCollection:
        stratum = self.lossyear.gt(0).rename("stratum").updateMask(self.forest2000).int()
        hist = ee.Dictionary(
            stratum.reduceRegion(ee.Reducer.frequencyHistogram(), self.roi, 90, maxPixels=1e11, tileScale=16).get("stratum")
        )
        return ee.FeatureCollection([ee.Feature(None, {"count_0": hist.get("0"), "count_1": hist.get("1")})])

    def grid_points(self, tile: ee.Geometry, spacing: int) -> ee.FeatureCollection:
        pts = ee.Image.constant(1).rename("one").sample(
            region=tile, scale=spacing, projection=ee.Projection("EPSG:3005"), geometries=True
        )
        return pts.map(_with_lonlat)

    def history_cells(self, ix0: int, ix1: int, iy0: int, iy1: int, res: float) -> ee.FeatureCollection:
        lon0, lat0 = self.cfg.bbox[0], self.cfg.bbox[1]
        ys = ee.List.sequence(iy0, iy1 - 1)

        def per_x(ix):
            ix = ee.Number(ix)

            def per_y(iy):
                iy = ee.Number(iy)
                x0 = ee.Number(lon0).add(ix.multiply(res))
                y0 = ee.Number(lat0).add(iy.multiply(res))
                rect = ee.Geometry.Rectangle(ee.List([x0, y0, x0.add(res), y0.add(res)]), "EPSG:4326", False)
                return ee.Feature(rect, {"ix": ix, "iy": iy})

            return ys.map(per_y)

        return ee.FeatureCollection(ee.List.sequence(ix0, ix1 - 1).map(per_x).flatten())

    def history_image(self, years: list[int]) -> ee.Image:
        bands = [self.lossyear.eq(y - 2000).And(self.forest2000).rename(f"y{y}") for y in years]
        return ee.Image.cat(bands)


def _with_lonlat(f):
    f = ee.Feature(f)
    xy = f.geometry().coordinates()
    return f.set({"lon": xy.get(0), "lat": xy.get(1), "point_id": f.get("system:index")})
