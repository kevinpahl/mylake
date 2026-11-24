"""
Fetchers for every outside data source the app uses. All are free and need no key
(a USGS API key is optional but recommended - see README).

  USGS Water Data (OGC API, with legacy NWIS fallback) ... lake elevation history, river gauges
  USACE CWMS Data API .................................... Corps elevation / inflow / outflow / storage
  Open-Meteo ............................................. rainfall history, 16-day forecast, seasonal (7-month) outlook
  NWS api.weather.gov .................................... local forecast text and active alerts
  NWS National Water Prediction Service .................. river forecast points near the lake
  NOAA Climate Prediction Center ......................... 6-10 day, 8-14 day, monthly and seasonal outlook maps
  US Drought Monitor ..................................... drought status of nearby counties
  Google News RSS ........................................ news about the lake

Every function returns empty results instead of raising, so one source being down
never breaks the page. Errors are collected in SOURCE_ERRORS for the "Data sources" tab.
"""
from __future__ import annotations

import datetime as dt
import re
import urllib.parse
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import requests

USER_AGENT = "LakeLevelTracker/1.0 (personal Streamlit app)"
TIMEOUT = 30
SOURCE_ERRORS: dict[str, str] = {}

USGS_OGC_BASES = [
    "https://api.waterdata.usgs.gov/ogcapi/v1",
    "https://api.waterdata.usgs.gov/ogcapi/v0",
]
USGS_LEGACY = "https://waterservices.usgs.gov/nwis"
CWMS_BASE = "https://cwms-data.usace.army.mil/cwms-data"

# Lake-surface elevation parameter codes, in order of preference
ELEV_PARAMS = ["62614", "00062", "62615"]   # NGVD29, above gauge datum, NAVD88
FLOW_PARAM = "00060"                         # discharge, cfs
GAGE_PARAM = "00065"                         # gage height, ft


def _note(source: str, err: Exception | str):
    SOURCE_ERRORS[source] = str(err)[:300]


def _get(url, params=None, headers=None, timeout=TIMEOUT):
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    r = requests.get(url, params=params, headers=h, timeout=timeout)
    r.raise_for_status()
    return r


# --------------------------------------------------------------------------------------
# USGS
# --------------------------------------------------------------------------------------
def _usgs_ogc_items(collection, params, api_key=None, max_pages=20):
    """Page through a USGS OGC collection; returns a list of feature 'properties' dicts."""
    headers = {"Accept": "application/geo+json, application/json"}
    if api_key:
        headers["X-Api-Key"] = api_key
    last_err = None
    for base in USGS_OGC_BASES:
        url = f"{base}/collections/{collection}/items"
        p = dict(params, f="json", limit=10000)
        rows = []
        try:
            for _ in range(max_pages):
                r = _get(url, params=p, headers=headers)
                js = r.json()
                rows += [f.get("properties", {}) for f in js.get("features", [])]
                nxt = next((l["href"] for l in js.get("links", []) if l.get("rel") == "next"), None)
                if not nxt:
                    break
                url, p = nxt, None
            return rows
        except Exception as e:  # try next base
            last_err = e
    raise RuntimeError(last_err)


def _to_series(rows, time_key="time", value_key="value"):
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    if time_key not in df or value_key not in df:
        return pd.Series(dtype=float)
    t = pd.to_datetime(df[time_key], utc=True, errors="coerce")
    v = pd.to_numeric(df[value_key], errors="coerce")
    s = pd.Series(v.values, index=t).dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def usgs_daily(site: str, param_codes, start: dt.date, end: dt.date, api_key=None):
    """Daily mean values. Returns (series indexed by date, parameter code used)."""
    if isinstance(param_codes, str):
        param_codes = [param_codes]
    for pc in param_codes:
        # 1) modern OGC API
        try:
            rows = _usgs_ogc_items("daily", {
                "monitoring_location_id": f"USGS-{site}",
                "parameter_code": pc,
                "time": f"{start.isoformat()}/{end.isoformat()}",
            }, api_key)
            if rows:
                df = pd.DataFrame(rows)
                if "statistic_id" in df and (df["statistic_id"] == "00003").any():
                    df = df[df["statistic_id"] == "00003"]
                s = _to_series(df.to_dict("records"))
                if len(s):
                    s.index = s.index.tz_convert(None).normalize()
                    return s.groupby(level=0).mean(), pc
        except Exception as e:
            _note(f"USGS daily {site}/{pc} (OGC)", e)
        # 2) legacy NWIS daily values
        try:
            r = _get(f"{USGS_LEGACY}/dv/", params={
                "format": "json", "sites": site, "parameterCd": pc,
                "startDT": start.isoformat(), "endDT": end.isoformat(),
            })
            ts = r.json().get("value", {}).get("timeSeries", [])
            best = None
            for t in ts:
                vals = t["values"][0]["value"]
                if vals and (best is None or len(vals) > len(best)):
                    best = vals
            if best:
                s = pd.Series({pd.Timestamp(v["dateTime"][:10]): float(v["value"]) for v in best
                               if v["value"] not in ("", None) and float(v["value"]) > -999})
                if len(s):
                    return s.sort_index(), pc
        except Exception as e:
            _note(f"USGS daily {site}/{pc} (legacy)", e)
    return pd.Series(dtype=float), None


def usgs_recent(site: str, param_codes, days=7, api_key=None):
    """15-minute (instantaneous) values for the last few days."""
    if isinstance(param_codes, str):
        param_codes = [param_codes]
    for pc in param_codes:
        try:
            rows = _usgs_ogc_items("continuous", {
                "monitoring_location_id": f"USGS-{site}",
                "parameter_code": pc, "time": f"P{days}D",
            }, api_key, max_pages=5)
            s = _to_series(rows)
            if len(s):
                return s, pc
        except Exception as e:
            _note(f"USGS recent {site}/{pc} (OGC)", e)
        try:
            r = _get(f"{USGS_LEGACY}/iv/", params={
                "format": "json", "sites": site, "parameterCd": pc, "period": f"P{days}D"})
            ts = r.json().get("value", {}).get("timeSeries", [])
            if ts and ts[0]["values"][0]["value"]:
                vals = ts[0]["values"][0]["value"]
                s = pd.Series({pd.Timestamp(v["dateTime"]): float(v["value"]) for v in vals
                               if v["value"] not in ("", None) and float(v["value"]) > -999})
                if len(s):
                    return s.sort_index(), pc
        except Exception as e:
            _note(f"USGS recent {site}/{pc} (legacy)", e)
    return pd.Series(dtype=float), None


# --------------------------------------------------------------------------------------
# US Army Corps of Engineers - CWMS Data API
# --------------------------------------------------------------------------------------
CWMS_KINDS = {
    "Elevation": r"\.Elev\.",
    "Inflow": r"\.Flow-In\.",
    "Outflow": r"\.Flow-(Out|Res Out)\.",
    "Storage": r"\.Stor\.",
    "Precip": r"\.Precip",
}


def cwms_catalog(office: str, pattern: str):
    """Find the lake's time series names in the Corps database."""
    rx = re.compile(pattern)

    def fetch(like, max_pages):
        out, page = [], None
        for _ in range(max_pages):
            params = {"office": office, "like": like, "page-size": 1000}
            if page:
                params["page"] = page
            js = _get(f"{CWMS_BASE}/catalog/TIMESERIES", params=params,
                      headers={"Accept": "application/json;version=2"}).json()
            out += [e.get("name") for e in js.get("entries", []) if e.get("name")]
            page = js.get("next-page")
            if not page:
                break
        return [n for n in out if rx.search(n.split(".")[0])]

    try:
        # CDA's `like` is a regular expression on the full time-series id
        like = pattern.replace("(?i)", "")
        names = fetch(f"({like}).*", 3)
        if not names:  # server regex may be case-sensitive: scan the office catalog
            names = fetch(".*", 15)
        return names
    except Exception as e:
        _note(f"USACE CWMS catalog ({office})", e)
        return []


def _rank_tsid(name: str) -> tuple:
    """Prefer daily/hourly, computed/revised ('Rev', 'Ccp') series over raw ones."""
    n = name.lower()
    interval = 0 if "1day" in n else 1 if "1hour" in n else 2 if "15minutes" in n else 3
    rev = 0 if ("rev" in n or "ccp" in n) else 1
    ave = 0 if ".ave." in n or ".inst." in n else 1
    return (rev, interval, ave, len(name))


def cwms_pick(names, kind):
    rx = re.compile(CWMS_KINDS[kind], re.I)
    hits = [n for n in names if rx.search(n)]
    if kind == "Storage":
        hits = [h for h in hits if "flood" not in h.lower()] or hits
    return sorted(hits, key=_rank_tsid)[0] if hits else None


def cwms_timeseries(office: str, tsid: str, days: int = 120):
    end = dt.datetime.now(dt.timezone.utc)
    begin = end - dt.timedelta(days=days)
    try:
        r = _get(f"{CWMS_BASE}/timeseries", params={
            "office": office, "name": tsid, "unit": "EN",
            "begin": begin.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "page-size": 50000,
        }, headers={"Accept": "application/json;version=2"})
        js = r.json()
        vals = js.get("values") or []
        units = js.get("units", "")
        s = pd.Series({pd.Timestamp(v[0], unit="ms", tz="UTC"): v[1] for v in vals if v[1] is not None},
                      dtype=float).sort_index()
        return s, units
    except Exception as e:
        _note(f"USACE CWMS {tsid}", e)
        return pd.Series(dtype=float), ""


# --------------------------------------------------------------------------------------
# Open-Meteo weather (rain history, forecast, seasonal)
# --------------------------------------------------------------------------------------
def _om_multi(url, points, params):
    lats = ",".join(f"{p[0]:.4f}" for p in points)
    lons = ",".join(f"{p[1]:.4f}" for p in points)
    r = _get(url, params=dict(params, latitude=lats, longitude=lons))
    js = r.json()
    return js if isinstance(js, list) else [js]


def _om_basin_mean(blocks, var):
    frames = []
    for b in blocks:
        d = b.get("daily", {})
        if var in d:
            frames.append(pd.Series(d[var], index=pd.to_datetime(d["time"]), dtype=float))
    if not frames:
        return pd.Series(dtype=float)
    return pd.concat(frames, axis=1).mean(axis=1, skipna=True)


def basin_weather_history(points, start: dt.date, end: dt.date):
    """Daily basin-average precipitation and reference evapotranspiration (inches)."""
    out = pd.DataFrame()
    try:
        blocks = _om_multi("https://archive-api.open-meteo.com/v1/archive", points, {
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "daily": "precipitation_sum,et0_fao_evapotranspiration,temperature_2m_mean",
            "precipitation_unit": "inch", "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
        })
        out = pd.DataFrame({
            "precip": _om_basin_mean(blocks, "precipitation_sum"),
            "et0": _om_basin_mean(blocks, "et0_fao_evapotranspiration") / 25.4,  # mm -> in
            "tmean": _om_basin_mean(blocks, "temperature_2m_mean"),
        })
    except Exception as e:
        _note("Open-Meteo archive", e)
    return out


def basin_weather_forecast(points, past_days=14, forecast_days=16):
    """Recent days + 16-day forecast, basin average. Fills the archive's ~5-day lag."""
    try:
        blocks = _om_multi("https://api.open-meteo.com/v1/forecast", points, {
            "daily": "precipitation_sum,et0_fao_evapotranspiration,temperature_2m_max,"
                     "temperature_2m_min,precipitation_probability_max",
            "precipitation_unit": "inch", "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
            "past_days": past_days, "forecast_days": forecast_days,
        })
        return pd.DataFrame({
            "precip": _om_basin_mean(blocks, "precipitation_sum"),
            "et0": _om_basin_mean(blocks, "et0_fao_evapotranspiration") / 25.4,
            "tmax": _om_basin_mean(blocks, "temperature_2m_max"),
            "tmin": _om_basin_mean(blocks, "temperature_2m_min"),
            "pop": _om_basin_mean(blocks, "precipitation_probability_max"),
        })
    except Exception as e:
        _note("Open-Meteo forecast", e)
        return pd.DataFrame()


def model_comparison(lat, lon, forecast_days=16):
    """Daily rain from several weather models at the lake, so you can see agreement."""
    models = {"gfs_seamless": "GFS (NOAA)", "ecmwf_ifs025": "ECMWF", "icon_seamless": "ICON (DWD)",
              "gem_seamless": "GEM (Canada)"}
    try:
        r = _get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "daily": "precipitation_sum",
            "models": ",".join(models), "precipitation_unit": "inch",
            "timezone": "America/New_York", "forecast_days": forecast_days})
        d = r.json()["daily"]
        idx = pd.to_datetime(d["time"])
        cols = {label: d.get(f"precipitation_sum_{m}") for m, label in models.items()}
        return pd.DataFrame({k: v for k, v in cols.items() if v is not None}, index=idx)
    except Exception as e:
        _note("Open-Meteo model comparison", e)
        return pd.DataFrame()


def seasonal_outlook(lat, lon, days=210):
    """ECMWF seasonal (EC46 + SEAS5) daily precipitation, ensemble-mean, inches/day."""
    base = {"latitude": lat, "longitude": lon, "daily": "precipitation_sum",
            "forecast_days": days, "timezone": "America/New_York"}
    for extra, factor in (({"precipitation_unit": "inch"}, 1.0), ({}, 1 / 25.4)):
        try:
            r = _get("https://seasonal-api.open-meteo.com/v1/seasonal", params=dict(base, **extra))
            d = r.json().get("daily", {})
            idx = pd.to_datetime(d.get("time", []))
            members = [k for k in d if k.startswith("precipitation_sum")]
            if not members:
                continue
            arr = np.array([[np.nan if x is None else x for x in d[k]] for k in members], dtype=float)
            return pd.Series(np.nanmean(arr, axis=0) * factor, index=idx).dropna()
        except Exception as e:
            _note("Open-Meteo seasonal", e)
    return pd.Series(dtype=float)


# --------------------------------------------------------------------------------------
# National Weather Service
# --------------------------------------------------------------------------------------
def nws_point_forecast(lat, lon):
    try:
        meta = _get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}",
                    headers={"Accept": "application/geo+json"}).json()["properties"]
        fc = _get(meta["forecast"], headers={"Accept": "application/geo+json"}).json()
        periods = fc["properties"]["periods"]
        return {"office": meta.get("cwa"), "city": meta.get("relativeLocation", {})
                .get("properties", {}).get("city"), "periods": periods}
    except Exception as e:
        _note("NWS forecast", e)
        return {"periods": []}


def nws_alerts(lat, lon):
    try:
        js = _get("https://api.weather.gov/alerts/active", params={"point": f"{lat:.4f},{lon:.4f}"},
                  headers={"Accept": "application/geo+json"}).json()
        return [f["properties"] for f in js.get("features", [])]
    except Exception as e:
        _note("NWS alerts", e)
        return []


def nwps_gauges_near(lat, lon, radius_deg=0.9):
    """River forecast points (NWS National Water Prediction Service) around the lake."""
    try:
        js = _get("https://api.water.noaa.gov/nwps/v1/gauges", params={
            "bbox.xmin": lon - radius_deg, "bbox.ymin": lat - radius_deg,
            "bbox.xmax": lon + radius_deg, "bbox.ymax": lat + radius_deg, "srid": "EPSG_4326"}).json()
        rows = []
        for g in js.get("gauges", []):
            st = g.get("status") or {}
            ob, fc = st.get("observed") or {}, st.get("forecast") or {}
            rows.append({
                "ID": g.get("lid"), "Name": g.get("name"),
                "Observed": ob.get("primary"), "Unit": ob.get("primaryUnit"),
                "Observed flow": ob.get("secondary"), "Flow unit": ob.get("secondaryUnit"),
                "Flood category (obs)": ob.get("floodCategory"),
                "Forecast": fc.get("primary"), "Flood category (fcst)": fc.get("floodCategory"),
                "Link": f"https://water.noaa.gov/gauges/{g.get('lid')}",
            })
        return pd.DataFrame(rows)
    except Exception as e:
        _note("NWS NWPS gauges", e)
        return pd.DataFrame()


CPC_MAPS = {
    "6–10 day precipitation": "https://www.cpc.ncep.noaa.gov/products/predictions/610day/610prcp.new.gif",
    "8–14 day precipitation": "https://www.cpc.ncep.noaa.gov/products/predictions/814day/814prcp.new.gif",
    "6–10 day temperature": "https://www.cpc.ncep.noaa.gov/products/predictions/610day/610temp.new.gif",
    "8–14 day temperature": "https://www.cpc.ncep.noaa.gov/products/predictions/814day/814temp.new.gif",
    "Next month precipitation": "https://www.cpc.ncep.noaa.gov/products/predictions/30day/off15_prcp.gif",
    "Next 3 months precipitation": "https://www.cpc.ncep.noaa.gov/products/predictions/long_range/lead01/off01_prcp.gif",
    "Next 3 months temperature": "https://www.cpc.ncep.noaa.gov/products/predictions/long_range/lead01/off01_temp.gif",
    "Seasonal drought outlook": "https://www.cpc.ncep.noaa.gov/products/expert_assessment/sdo_summary.png",
}


# --------------------------------------------------------------------------------------
# US Drought Monitor
# --------------------------------------------------------------------------------------
def drought_status(fips_map: dict, weeks=12):
    end = dt.date.today()
    start = end - dt.timedelta(weeks=weeks)
    rows = []
    for fips, label in fips_map.items():
        try:
            js = _get("https://usdmdataservices.unl.edu/api/CountyStatistics/"
                      "GetDroughtSeverityStatisticsByAreaPercent",
                      params={"aoi": fips, "startdate": start.strftime("%m/%d/%Y"),
                              "enddate": end.strftime("%m/%d/%Y"), "statisticsType": 1},
                      headers={"Accept": "application/json"}).json()
            for rec in js:
                rec = {k.lower(): v for k, v in rec.items()}
                rows.append({"County": label, "Week": pd.to_datetime(rec.get("mapdate") or rec.get("validstart")),
                             **{k.upper(): float(rec.get(k.lower(), 0) or 0)
                                for k in ["None", "D0", "D1", "D2", "D3", "D4"]}})
        except Exception as e:
            _note(f"US Drought Monitor {fips}", e)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# News
# --------------------------------------------------------------------------------------
IMPACT_KEYWORDS = {
    "Drought / low water": r"drought|low water|low lake|dry|below normal|water shortage|conservation",
    "Flooding / high water": r"flood|high water|rising|spillway|release|overflow|heavy rain|hurricane|tropical",
    "Corps operations": r"corps|usace|army corps|dam|guide curve|water control|discharge|generation",
    "Water quality": r"algae|bloom|bacteria|advisory|contamina|water quality|e\. coli",
    "Recreation access": r"boat ramp|ramp|campground|closure|closed|park|marina|swim",
}


def news(terms, extra=None, max_items=40):
    q = " OR ".join(terms)
    queries = [q, f"({q}) (level OR drought OR flood OR Corps OR dam OR release)"]
    if extra:
        queries.append(extra)
    items, seen = [], set()
    for query in queries:
        url = ("https://news.google.com/rss/search?" +
               urllib.parse.urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}))
        try:
            root = ET.fromstring(_get(url).content)
            for it in root.iter("item"):
                title = (it.findtext("title") or "").strip()
                key = re.sub(r"\W+", "", title.lower())[:80]
                if not title or key in seen:
                    continue
                seen.add(key)
                try:
                    when = parsedate_to_datetime(it.findtext("pubDate"))
                except Exception:
                    when = None
                desc = re.sub("<[^>]+>", " ", it.findtext("description") or "")
                text = f"{title} {desc}".lower()
                tags = [k for k, rx in IMPACT_KEYWORDS.items() if re.search(rx, text)]
                src = it.find("source")
                items.append({"title": title, "link": it.findtext("link"), "published": when,
                              "source": src.text if src is not None else "", "tags": tags})
        except Exception as e:
            _note("Google News RSS", e)
    items.sort(key=lambda x: x["published"] or dt.datetime.min.replace(tzinfo=dt.timezone.utc),
               reverse=True)
    return items[:max_items]
