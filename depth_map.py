"""
Depth / shoreline map.

Builds a lake-bottom + surrounding-land elevation grid and works out, for any lake level,
where the water is, how deep it is, and where the shoreline falls.

Where the data comes from
  * Land and shoreline: USGS 3D Elevation Program (3DEP) lidar terrain, exported from the
    National Map's public ImageServer. Lidar can't see through water, so the lake shows up
    as a flat "hydro-flattened" surface at whatever level the lake was when it was flown.
    Everything ABOVE that surface (exposed banks, flats, coves, the flood pool) is real
    measured terrain.
  * Below that surface: no public survey of the whole lake bed exists, so depths are
    ESTIMATED from distance to shore and distance from the dam, scaled to the lake's known
    maximum depth. If you have real soundings (sonar log, survey points), upload a CSV with
    lon, lat, bottom_elev_ft and they replace the estimate wherever you have coverage.

Datums: 3DEP is NAVD88 metres; most Corps lakes report NGVD29 feet. `datum_offset_ft`
(lake gauge datum minus NAVD88) converts between them.
"""
from __future__ import annotations

import base64
import io
import math

import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy import ndimage as ndi

M_TO_FT = 3.28084
R_EARTH = 6378137.0
DEM_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"

ESRI_IMG = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
BASEMAPS = {
    # one cloud-free mosaic for the whole country, so VA and NC match (10 m detail)
    "Satellite – even color (Sentinel-2)": ["https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/g/{z}/{y}/{x}.jpg"],
    # sharpest, but stitched from different photo sources per area, so colors can change at state lines
    "Satellite – high detail (Esri)": [ESRI_IMG],
    "Satellite – high detail + labels": [ESRI_IMG],
    "Aerial photo (USGS / NAIP)": ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"],
    "USGS topo": ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}"],
}
BASEMAP_CREDIT = {
    "Satellite – even color (Sentinel-2)": "Sentinel-2 cloudless 2023 by EOX IT Services (contains modified Copernicus Sentinel data)",
    "Aerial photo (USGS / NAIP)": "USGS The National Map / USDA NAIP",
    "USGS topo": "USGS The National Map",
}
ATTRIBUTION = "Imagery © Esri, Maxar, Earthstar Geographics"
NHD_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/12/query"


# ----------------------------------------------------------------------------- projections
def ll_to_merc(lon, lat):
    x = np.radians(lon) * R_EARTH
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R_EARTH
    return x, y


def merc_to_ll(x, y):
    lon = np.degrees(x / R_EARTH)
    lat = np.degrees(2 * np.arctan(np.exp(y / R_EARTH)) - np.pi / 2)
    return lon, lat


def default_bbox(lake):
    if lake.get("map_bbox"):
        return tuple(lake["map_bbox"])
    lat, lon = lake["lat"], lake["lon"]
    return (lon - 0.22, lat - 0.15, lon + 0.22, lat + 0.15)


# ----------------------------------------------------------------------------- terrain
def fetch_dem(bbox, width=1600):
    """bbox = (lon_min, lat_min, lon_max, lat_max). Returns dict with elevation (m, NAVD88)."""
    x0, y0 = ll_to_merc(bbox[0], bbox[1])
    x1, y1 = ll_to_merc(bbox[2], bbox[3])
    height = int(round(width * (y1 - y0) / (x1 - x0)))
    r = requests.get(DEM_URL, params={
        "bbox": f"{x0},{y0},{x1},{y1}", "bboxSR": 3857, "imageSR": 3857,
        "size": f"{width},{height}", "format": "tiff", "pixelType": "F32",
        "noData": -9999, "interpolation": "RSP_BilinearInterpolation", "f": "image",
    }, headers={"User-Agent": "LakeLevelTracker/1.0"}, timeout=90)
    r.raise_for_status()
    if not r.content[:4] in (b"II*\x00", b"MM\x00*"):
        raise RuntimeError(f"Elevation service returned no image: {r.content[:200]!r}")
    dem = np.array(Image.open(io.BytesIO(r.content)), dtype=np.float32)
    dem[(dem < -1000) | ~np.isfinite(dem)] = np.nan
    return {"elev_m": dem, "merc": (x0, y0, x1, y1), "bbox": bbox}


def fetch_lake_outlines(bbox, min_km2=2.0):
    """Lake/reservoir polygons (USGS National Hydrography Dataset) in Web-Mercator metres."""
    r = requests.get(NHD_URL, params={
        "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}", "geometryType": "esriGeometryEnvelope",
        "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "where": "1=1", "outFields": "*",
        "returnGeometry": "true", "outSR": 3857, "maxAllowableOffset": 15, "f": "geojson",
    }, headers={"User-Agent": "LakeLevelTracker/1.0"}, timeout=90)
    r.raise_for_status()
    polys = []
    for f in r.json().get("features", []):
        props = {k.lower(): v for k, v in (f.get("properties") or {}).items()}
        area = props.get("areasqkm")
        g = f.get("geometry") or {}
        rings_sets = [g["coordinates"]] if g.get("type") == "Polygon" else g.get("coordinates", []) if g.get("type") == "MultiPolygon" else []
        for rings in rings_sets:
            if area is None:
                xy = np.asarray(rings[0])
                a = 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1))) / 1e6
            else:
                a = area
            if a and a >= min_km2 * 0.5:
                polys.append(rings)
    return polys


# ----------------------------------------------------------------------------- lake bed
class LakeBed:
    def __init__(self, dem: dict, lake: dict, datum_offset_ft: float, survey: pd.DataFrame | None = None,
                 outlines: list | None = None):
        self.lake = lake
        self.offset = datum_offset_ft
        self.x0, self.y0, self.x1, self.y1 = dem["merc"]
        self.bbox = dem["bbox"]
        e = dem["elev_m"]
        self.H, self.W = e.shape
        self.res_merc = (self.x1 - self.x0) / self.W
        self.scale = math.cos(math.radians((self.bbox[1] + self.bbox[3]) / 2))  # mercator -> ground
        self.px_m = self.res_merc * self.scale
        self.px_acres = self.px_m ** 2 / 4046.856

        land_ft = e * M_TO_FT + datum_offset_ft
        self.outline_mask = self._rasterize(outlines) if outlines else None
        self._find_lake_surface(land_ft)
        self._estimate_depths(land_ft)
        self.survey_used = 0
        if survey is not None and len(survey):
            self._apply_survey(survey)
        self.bottom_ft[np.isnan(self.bottom_ft)] = np.nanmax(self.bottom_ft)
        self.seed = np.unravel_index(np.argmin(np.where(self.lake_mask, self.bottom_ft, np.inf)), self.bottom_ft.shape)
        self._cache = {}

    def _rasterize(self, polys):
        from PIL import ImageDraw
        img = Image.new("L", (self.W, self.H), 0)
        d = ImageDraw.Draw(img)
        for rings in polys:
            for k, ring in enumerate(rings):
                xy = np.asarray(ring, dtype=float)
                px = [((x - self.x0) / self.res_merc, (self.y1 - y) / self.res_merc) for x, y in xy]
                if len(px) >= 3:
                    d.polygon(px, fill=1 if k == 0 else 0)
        return np.array(img, dtype=bool)

    # Lidar shows the lake as perfectly flat ("hydro-flattened") water. A lake that spans
    # several lidar projects (e.g. the VA and NC halves of Kerr) has one flat surface per
    # project, each at the level the lake was when that project was flown. Find them all.
    def _find_lake_surface(self, land_ft):
        norm = self.lake["normal_pool"]
        lo = (self.lake.get("min_pool") or norm - 30) - 5
        hi = max(self.lake.get("flood_top") or norm + 10, norm) + 2
        filled = np.where(np.isfinite(land_ft), land_ft, -9999.0)
        local_range = ndi.maximum_filter(filled, 3) - ndi.minimum_filter(filled, 3)
        flat = (local_range < 0.1) & (filled > lo) & (filled < hi)
        lab, n = ndi.label(flat)
        if n == 0 and self.outline_mask is None:
            raise RuntimeError("Couldn't find the lake surface in the elevation data for this map area.")
        idx = np.arange(1, n + 1)
        sizes = ndi.sum(flat, lab, index=idx) if n else np.array([0.0])
        keep = idx[(sizes >= max(300, 0.02 * sizes.max()))] if n else idx
        surf = np.isin(lab, keep)
        levels = list(ndi.mean(filled, lab, index=keep)) if len(keep) else []
        level_of_label = np.zeros(n + 1)
        level_of_label[keep] = levels
        w_flat = np.where(surf, level_of_label[lab], np.nan)

        mask = ndi.binary_fill_holes(ndi.binary_closing(surf, iterations=3))
        if self.outline_mask is not None:
            om = self.outline_mask
            # drop outlines that sit well below the lake (river / lake below the dam)
            lab_o, no = ndi.label(om)
            if no:
                med = ndi.median(filled, lab_o, index=np.arange(1, no + 1))
                ref = min(levels) if levels else lo
                good = np.arange(1, no + 1)[np.asarray(med) > ref - 15]
                om = np.isin(lab_o, good)
            mask |= om
        # keep only water connected to the main lake surface
        lab_m, nm = ndi.label(mask)
        if nm > 1:
            anchor = surf if surf.any() else mask
            ids = np.unique(lab_m[anchor & (lab_m > 0)])
            if len(ids):
                sz = ndi.sum(anchor, lab_m, index=ids)
                main = ids[np.asarray(sz) >= 0.05 * np.max(sz)]
                mask = np.isin(lab_m, main)

        # water-surface level for every lake pixel: the flat lidar level where we found one,
        # otherwise the terrain value itself (smoothed), limited to a plausible pool range
        smooth = ndi.median_filter(filled, size=5)
        w_pix = np.where(np.isfinite(w_flat), w_flat, np.clip(smooth, lo, hi))
        if surf.any():
            _, (ni, nj) = ndi.distance_transform_edt(~surf, return_indices=True)
            near = w_flat[ni, nj]
            # pixels in the outline without a flat surface: use their own value if it is in range,
            # else the nearest flat surface
            bad = ~np.isfinite(w_flat) & ((smooth < lo) | (smooth > hi))
            w_pix = np.where(bad, near, w_pix)
        if not levels:
            levels = [float(np.nanmedian(w_pix[mask]))]
            sizes = np.array([float(mask.sum())]); keep = np.array([1])
        self.flat_found = surf
        mask &= ~(np.isfinite(land_ft) & (land_ft > w_pix + 1.5))
        self.lake_mask = mask
        self.w_pix = w_pix
        wts = np.asarray(sizes)[np.asarray(keep) - 1] if len(sizes) >= len(keep) else np.ones(len(levels))
        self.lidar_levels = sorted({round(float(v), 1) for v, s_ in zip(levels, wts) if s_ >= 0.05 * max(wts)})
        self.lidar_level_ft = float(np.average(levels, weights=wts))
        # Water well below the lake surface outside the lake (e.g. the river below the dam)
        # must never join the lake, even if the dam is broken up in the terrain data.
        self.barrier = ~mask & np.isfinite(land_ft) & (land_ft < min(levels) - 15)

    def _estimate_depths(self, land_ft):
        m = self.lake_mask
        dist_shore = ndi.distance_transform_edt(m) * self.px_m              # metres to nearest shore
        dx, dy = ll_to_merc(self.lake["lon"], self.lake["lat"])
        jj, ii = np.meshgrid(np.arange(self.W), np.arange(self.H))
        cx = self.x0 + (jj + 0.5) * self.res_merc
        cy = self.y1 - (ii + 0.5) * self.res_merc
        dist_dam = np.hypot(cx - dx, cy - dy) * self.scale
        L = max(np.percentile(dist_dam[m], 90), 1000.0)
        shape = (dist_shore ** 0.6) * (0.3 + 0.7 * np.exp(-dist_dam / (0.6 * L)))
        # max depth is quoted at normal pool: deepest bottom elevation is the same everywhere
        deepest_bottom = self.lake["normal_pool"] - self.lake.get("max_depth_ft", 60)
        ref = np.percentile(shape[m], 99.7) if m.any() else 1.0
        frac = np.clip(shape / ref, 0, 1.05)
        bottom = self.w_pix - frac * np.maximum(self.w_pix - deepest_bottom, 5)
        bottom = ndi.gaussian_filter(np.where(m, bottom, np.nan_to_num(self.w_pix, nan=self.lidar_level_ft)), 1.0)
        bottom = np.minimum(bottom, self.w_pix - 0.5)
        self.bottom_ft = np.where(m, bottom, land_ft)
        self.estimated = m.copy()

    def _apply_survey(self, df):
        cols = {c.lower().strip(): c for c in df.columns}
        need = ["lon", "lat", "bottom_elev_ft"]
        if not all(k in cols for k in need):
            raise ValueError("Survey CSV needs columns: lon, lat, bottom_elev_ft")
        lon = df[cols["lon"]].astype(float).values
        lat = df[cols["lat"]].astype(float).values
        z = df[cols["bottom_elev_ft"]].astype(float).values
        x, y = ll_to_merc(lon, lat)
        j = ((x - self.x0) / self.res_merc).astype(int)
        i = ((self.y1 - y) / self.res_merc).astype(int)
        ok = (i >= 0) & (i < self.H) & (j >= 0) & (j < self.W) & np.isfinite(z)
        if not ok.any():
            return
        s = np.zeros((self.H, self.W)); c = np.zeros((self.H, self.W))
        np.add.at(s, (i[ok], j[ok]), z[ok]); np.add.at(c, (i[ok], j[ok]), 1)
        has = c > 0
        grid = np.where(has, s / np.maximum(c, 1), np.nan)
        d, (ni, nj) = ndi.distance_transform_edt(~has, return_indices=True)
        filled = grid[ni, nj]
        use = (d * self.px_m <= 150) & self.lake_mask
        self.bottom_ft = np.where(use, filled, self.bottom_ft)
        self.estimated &= ~use
        self.survey_used = int(ok.sum())

    # ------------------------------------------------------------------ queries
    def water(self, level_ft: float) -> np.ndarray:
        key = round(level_ft, 2)
        if key not in self._cache:
            wet = (self.bottom_ft < level_ft) & ~self.barrier
            lab, _ = ndi.label(wet)
            s = lab[self.seed]
            self._cache[key] = (lab == s) if s else np.zeros_like(wet)
            if len(self._cache) > 12:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def stats(self, level_ft):
        w = self.water(level_ft)
        depth = np.where(w, level_ft - self.bottom_ft, 0)
        return {"area_acres": float(w.sum() * self.px_acres),
                "volume_acft": float(depth.sum() * self.px_acres),
                "max_depth_ft": float(depth.max()) if w.any() else 0.0,
                "mean_depth_ft": float(depth[w].mean()) if w.any() else 0.0}

    def corners_ll(self):
        lon0, lat1 = merc_to_ll(self.x0, self.y1)
        lon1, lat0 = merc_to_ll(self.x1, self.y0)
        lon0, lon1, lat0, lat1 = map(float, (lon0, lon1, lat0, lat1))
        return [[lon0, lat1], [lon1, lat1], [lon1, lat0], [lon0, lat0]]

    def pixel_ll(self, i, j):
        return merc_to_ll(self.x0 + (j + 0.5) * self.res_merc, self.y1 - (i + 0.5) * self.res_merc)

    # ------------------------------------------------------------------ rendering
    def render_depth(self, level_ft, max_color_depth=None, contour_ft=10):
        """RGBA image: blue depth shading, exposed lakebed (tan), flooded land above normal pool (orange)."""
        w = self.water(level_ft)
        depth = np.where(w, level_ft - self.bottom_ft, np.nan)
        mx = max_color_depth or max(np.nanpercentile(depth, 99) if w.any() else 10, 10)
        t = np.clip(np.nan_to_num(depth) / mx, 0, 1) ** 0.7
        light, dark = np.array([198, 236, 255]), np.array([4, 30, 90])
        rgb = light[None, None, :] * (1 - t[..., None]) + dark[None, None, :] * t[..., None]
        a = np.where(w, 205, 0).astype(np.uint8)
        img = np.dstack([rgb.astype(np.uint8), a])

        normal = self.water(self.lake["normal_pool"])
        exposed = normal & ~w
        img[exposed] = [222, 184, 112, 215]
        flooded = w & (self.bottom_ft >= self.lake["normal_pool"])
        img[flooded] = [255, 120, 30, 175]

        if contour_ft:
            band = np.floor(np.nan_to_num(depth, nan=-1) / contour_ft)
            edge = np.zeros_like(w)
            edge[:, 1:] |= band[:, 1:] != band[:, :-1]
            edge[1:, :] |= band[1:, :] != band[:-1, :]
            edge &= ndi.binary_erosion(w, iterations=1)  # interior only - shoreline drawn separately
            img[edge] = [255, 255, 255, 90]
        return img

    def shoreline(self, level_ft, thickness=1):
        w = self.water(level_ft)
        edge = w & ~ndi.binary_erosion(w)
        if thickness > 1:
            edge = ndi.binary_dilation(edge, iterations=thickness - 1)
        return edge

    def render_lines(self, lines: list[tuple[float, tuple, bool]], thickness=2):
        """lines: [(level_ft, (r,g,b), dashed)] drawn as shorelines."""
        img = np.zeros((self.H, self.W, 4), dtype=np.uint8)
        ii, jj = np.indices((self.H, self.W))
        dash = ((ii + jj) // 6) % 2 == 0
        for level, color, dashed in lines:
            e = self.shoreline(level, thickness)
            if dashed:
                e &= dash
            img[e] = [*color, 255]
        return img

    def hover_points(self, level_ft, max_points=4000):
        w = self.water(level_ft)
        step = max(1, int(math.sqrt(w.sum() / max_points)) + 1)
        sub = np.zeros_like(w)
        sub[::step, ::step] = True
        i, j = np.nonzero(w & sub)
        lon, lat = self.pixel_ll(i, j)
        depth = level_ft - self.bottom_ft[i, j]
        est = self.estimated[i, j]
        return pd.DataFrame({"lon": lon, "lat": lat, "depth": depth, "bottom": self.bottom_ft[i, j], "estimated": est})


def to_data_url(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
