"""
Lake Level Tracker - Streamlit app
Run:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import math
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_sources as ds
import depth_map as dm
import folium
from streamlit_folium import st_folium
from lakes import DEFAULT_LAKE, LAKES, custom_lake
from model import LakeModel, climatology_bands, guide_curve_series, outlook_ratio_by_month

st.set_page_config(page_title="Lake Level Tracker", page_icon="🌊", layout="wide",
                   initial_sidebar_state="collapsed")


def _user_agent():
    try:
        return st.context.headers.get("User-Agent", "") or ""
    except Exception:
        return ""


_UA = _user_agent()
IS_PHONE = ("Mobi" in _UA or "iPhone" in _UA) and "iPad" not in _UA
IS_TABLET = ("iPad" in _UA or "Tablet" in _UA or ("Android" in _UA and "Mobi" not in _UA))

st.markdown("""
<style>
/* ---------- navigation bar: big, always at the top ---------- */
.st-key-topnav {
    position: sticky; top: 3.6rem; z-index: 990;
    background: #ffffff; padding: 0.35rem 0 0.45rem 0;
    border-bottom: 1px solid rgba(128,128,128,0.25);
    box-shadow: 0 4px 8px -6px rgba(0,0,0,0.25);
}
@media (prefers-color-scheme: dark) { .st-key-topnav { background: #0e1117; } }
.st-key-topnav [data-testid="stButtonGroup"] > div,
.st-key-topnav [role="radiogroup"] { flex-wrap: nowrap !important; overflow-x: auto; gap: 0.35rem;
    scrollbar-width: thin; -webkit-overflow-scrolling: touch; padding-bottom: 2px; }
.st-key-topnav button, .st-key-topnav label {
    min-height: 44px; font-size: 1.02rem !important; font-weight: 600; white-space: nowrap; flex: 0 0 auto;
}
.st-key-pager button { min-height: 44px; font-weight: 600; }
/* ---------- general touch friendliness ---------- */
button, [data-baseweb="select"] > div, input { min-height: 40px; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }
/* ---------- tablets ---------- */
@media (max-width: 1024px) {
  .block-container { padding-left: 1.2rem !important; padding-right: 1.2rem !important; }
}
/* ---------- phones ---------- */
@media (max-width: 640px) {
  .block-container { padding: 0.6rem 0.6rem 3rem 0.6rem !important; }
  h1 { font-size: 1.45rem !important; }
  h4 { font-size: 1.05rem !important; }
  [data-testid="stMetricValue"] { font-size: 1.25rem; }
  [data-testid="stMetricLabel"] p { font-size: 0.8rem; }
  [data-testid="stMetric"] { padding: 0.2rem 0; }
  .st-key-topnav { top: 3.3rem; }
  .st-key-topnav button { font-size: 0.95rem !important; padding: 0 0.7rem; }
  /* keep pairs of metrics side by side instead of one long column */
  .st-key-kpis [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 0.4rem; }
  .st-key-kpis [data-testid="stColumn"] { min-width: calc(50% - 0.4rem) !important; flex: 1 1 calc(50% - 0.4rem) !important; }
}
</style>
""", unsafe_allow_html=True)

TODAY = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)

# Streamlit renamed use_container_width -> width="stretch" in newer releases
try:
    _v = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _v = (1, 40)
W = {"width": "stretch"} if _v >= (1, 48) else {"use_container_width": True}


def _secret(name, default=None):
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except Exception:
        return os.environ.get(name, default)


USGS_KEY = _secret("USGS_API_KEY")

# ------------------------------------------------------------------------------------
# Cached fetchers (ttl in seconds)
# ------------------------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_level_history(site, years):
    start = (TODAY - pd.DateOffset(years=years)).date()
    return ds.usgs_daily(site, ds.ELEV_PARAMS, start, TODAY.date(), USGS_KEY)


@st.cache_data(ttl=15 * 60, show_spinner=False)
def get_level_recent(site):
    return ds.usgs_recent(site, ds.ELEV_PARAMS, days=10, api_key=USGS_KEY)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_flow_history(site, days=120):
    start = (TODAY - pd.Timedelta(days=days)).date()
    return ds.usgs_daily(site, [ds.FLOW_PARAM], start, TODAY.date(), USGS_KEY)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_weather_history(points, years):
    start = (TODAY - pd.DateOffset(years=years) - pd.Timedelta(days=40)).date()
    return ds.basin_weather_history(list(points), start, (TODAY - pd.Timedelta(days=1)).date())


@st.cache_data(ttl=2 * 3600, show_spinner=False)
def get_weather_forecast(points):
    return ds.basin_weather_forecast(list(points))


@st.cache_data(ttl=2 * 3600, show_spinner=False)
def get_model_comparison(lat, lon):
    return ds.model_comparison(lat, lon)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_seasonal(lat, lon):
    return ds.seasonal_outlook(lat, lon)


@st.cache_data(ttl=30 * 60, show_spinner=False)
def get_nws(lat, lon):
    return ds.nws_point_forecast(lat, lon), ds.nws_alerts(lat, lon)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_nwps(lat, lon):
    return ds.nwps_gauges_near(lat, lon)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_cwms_catalog(office, pattern):
    return ds.cwms_catalog(office, pattern)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_cwms_series(office, tsid, days):
    return ds.cwms_timeseries(office, tsid, days)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_drought(fips_items):
    return ds.drought_status(dict(fips_items))


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_news(terms):
    return ds.news(list(terms))


@st.cache_resource(show_spinner=False, max_entries=20)
def fit_model(lake_key, guide_bp, level_hash, _level, _weather):
    return LakeModel(guide_bp).fit(_level, _weather)


# ------------------------------------------------------------------------------------
# Top bar: lake picker + settings (no hidden sidebar - easier on phones and tablets)
# ------------------------------------------------------------------------------------
names = list(LAKES) + ["➕ Other lake (enter a USGS gauge)"]
t1, t2 = st.columns([3, 1], vertical_alignment="bottom")
choice = t1.selectbox("🌊 Lake or reservoir", names, index=names.index(DEFAULT_LAKE))
_settings = t2.popover("⚙️ Settings", **W)

if choice.startswith("➕"):
    with st.expander("Custom lake", expanded=True):
        c_name = st.text_input("Name", "My Lake")
        c_site = st.text_input("USGS site number (lake elevation gauge)", "")
        cc1, cc2 = st.columns(2)
        c_lat = cc1.number_input("Latitude", value=36.0, format="%.4f")
        c_lon = cc2.number_input("Longitude", value=-79.0, format="%.4f")
        c_norm = cc1.number_input("Normal / full pool (ft)", value=300.0)
        c_flood = cc2.number_input("Top of flood pool (ft)", value=320.0)
    if not c_site:
        st.info("Enter the lake's USGS site number above to load it.")
        st.stop()
    lake = custom_lake(c_name, c_site.strip(), c_lat, c_lon, c_norm, c_flood)
else:
    lake = LAKES[choice]

with _settings:
    st.markdown("**Projection settings**")
    horizon = st.slider("Projection length (days)", 14, 180, 90, step=7)
    use_outlook = st.checkbox("Adjust for seasonal rainfall outlook", value=True,
                              help="Scales rainfall after the 16-day forecast by the ECMWF "
                                   "seasonal outlook vs. normal for each month.")
    scenario = st.slider("Rainfall scenario (% of normal after 16 days)", 50, 150, 100, step=5,
                         help="Try 'what if it stays dry' (e.g. 60%) or 'wet winter' (130%).")
    years = st.slider("Years of history to learn from", 5, 25, 15)
    show_members = st.checkbox("Show individual scenarios", value=False)
    st.markdown("**Display**")
    COMPACT = st.toggle("Compact layout (phones)", value=IS_PHONE,
                        help="Shorter charts and simpler toolbars. Turned on automatically on phones.")
    if st.button("🔄 Refresh all data", **W):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.caption("Data refreshes automatically (levels every 15 min, weather every 2 h). "
               + ("USGS API key loaded." if USGS_KEY else "Tip: add a free USGS API key (see README)."))


def show_chart(fig, height=None):
    """Plotly chart sized for the device: shorter, no toolbar, no accidental zoom on phones."""
    h = height or fig.layout.height or 400
    if COMPACT:
        fig.update_layout(height=min(h, 360), margin=dict(l=4, r=4, t=36, b=4),
                          legend=dict(orientation="h", y=-0.25, font=dict(size=10)), font=dict(size=11))
        cfg = {"displayModeBar": False, "scrollZoom": False, "responsive": True}
    else:
        cfg = {"displaylogo": False, "responsive": True}
    st.plotly_chart(fig, **W, config=cfg)


# ------------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------------
ds.SOURCE_ERRORS.clear()
points = tuple(tuple(p) for p in lake["basin_points"])

with st.spinner(f"Loading {lake['short']} data from USGS, the Corps of Engineers and weather services…"):
    level_hist, elev_param = get_level_history(lake["usgs_site"], years) if lake.get("usgs_site") else (pd.Series(dtype=float), None)
    level_recent, _ = get_level_recent(lake["usgs_site"]) if lake.get("usgs_site") else (pd.Series(dtype=float), None)
    wx_hist = get_weather_history(points, years)
    wx_fc = get_weather_forecast(points)

    cwms_names, cwms = [], {}
    if lake.get("cwms_office") and lake.get("cwms_pattern"):
        cwms_names = get_cwms_catalog(lake["cwms_office"], lake["cwms_pattern"])
        for kind in ds.CWMS_KINDS:
            tsid = ds.cwms_pick(cwms_names, kind)
            if tsid:
                days = 365 * years if (kind == "Elevation" and level_hist.empty) else 120
                s, units = get_cwms_series(lake["cwms_office"], tsid, days)
                if len(s):
                    cwms[kind] = {"tsid": tsid, "series": s, "units": units}

# Fall back to the Corps' elevation record if USGS had nothing
level_source = f"USGS {lake.get('usgs_site')} (parameter {elev_param})"
if level_hist.empty and "Elevation" in cwms:
    s = cwms["Elevation"]["series"]
    level_hist = s.tz_convert("America/New_York").tz_localize(None).resample("D").mean().dropna()
    level_source = f"USACE CWMS {cwms['Elevation']['tsid']}"
if level_recent.empty and "Elevation" in cwms:
    level_recent = cwms["Elevation"]["series"].tail(96 * 7)

if level_hist.empty:
    st.error(f"Couldn't load lake-level data for {lake['short']}. The USGS and Corps services may "
             "be down or rate-limited. Try Refresh in a few minutes, or add a USGS API key.")
    with st.expander("Error details"):
        st.json(ds.SOURCE_ERRORS)
    st.stop()

# Bring the daily record up to today using the 15-minute readings
if len(level_recent):
    rec = level_recent.copy()
    if rec.index.tz is not None:
        rec.index = rec.index.tz_convert("America/New_York").tz_localize(None)
    rec_daily = rec.resample("D").mean().dropna()
    level = level_hist.combine_first(rec_daily) if len(level_hist) else rec_daily
    current = float(rec.iloc[-1])
    current_time = rec.index[-1]
else:
    level = level_hist
    current = float(level.iloc[-1])
    current_time = level.index[-1]
level = level.sort_index()
level.index = pd.DatetimeIndex(level.index).normalize()
level = level[~level.index.duplicated(keep="last")]

guide_today = float(guide_curve_series(pd.DatetimeIndex([TODAY]), lake["guide_curve"]).iloc[0])
clim = climatology_bands(level_hist if len(level_hist) > 730 else level)

# ------------------------------------------------------------------------------------
# Fit model + project
# ------------------------------------------------------------------------------------
proj = members = None
n_fc = 0
model = None
model_error = None
outlook = {}
seasonal = pd.Series(dtype=float)
if not wx_hist.empty:
    try:
        model = fit_model(choice, tuple(lake["guide_curve"]),
                          (len(level_hist), float(level_hist.iloc[-1]), str(level_hist.index[-1])),
                          level_hist, wx_hist)
        if use_outlook:
            seasonal = get_seasonal(lake["lat"], lake["lon"])
            outlook = outlook_ratio_by_month(seasonal, wx_hist)
        lo = (lake.get("min_pool") or level.min()) - 5
        hi = lake.get("flood_top", level.max()) + 3
        proj, members, central, n_fc = model.project(
            level, wx_hist, wx_fc, horizon_days=horizon,
            outlook_ratio=outlook if use_outlook else None,
            manual_pct=scenario, bounds=(lo, hi))
    except Exception as e:
        model_error = str(e)
else:
    model_error = "Rainfall history (Open-Meteo) could not be loaded."

# ------------------------------------------------------------------------------------
# Header + navigation
# ------------------------------------------------------------------------------------
st.markdown("<div id='lake-top'></div>", unsafe_allow_html=True)
st.title(f"{lake['short']} — {current:.2f} ft")
st.caption(f"{current - lake['normal_pool']:+.2f} ft vs. normal pool ({lake['normal_pool']:g}) · "
           f"last reading {current_time:%b %d, %I:%M %p} · elevations in feet ({lake.get('datum', '')}) · "
           f"source: {level_source}")

PAGES = {
    "level": ("📈", "Level & projection"),
    "map": ("🗺️", "Depth map"),
    "weather": ("🌧️", "Weather"),
    "outlook": ("🗓️", "Long-range"),
    "inflows": ("🏞️", "Inflows & Corps"),
    "news": ("📰", "News"),
    "info": ("ℹ️", "Lake info"),
    "model": ("⚙️", "Model & sources"),
}
PAGE_KEYS = list(PAGES)
if "page" not in st.session_state:
    qp = st.query_params.get("page")
    st.session_state.page = qp if qp in PAGES else "level"


def _label(k):
    icon, name = PAGES[k]
    return f"{icon} {name}"


def _go(k):
    st.session_state.page = k
    st.session_state.nav = k


def _nav_changed():
    if st.session_state.nav is None:          # tapped the active button again: stay put
        st.session_state.nav = st.session_state.page
    else:
        st.session_state.page = st.session_state.nav


if "nav" not in st.session_state:
    st.session_state.nav = st.session_state.page
with st.container(key="topnav"):
    if hasattr(st, "segmented_control"):
        st.segmented_control("Go to", PAGE_KEYS, format_func=_label, key="nav", on_change=_nav_changed,
                             label_visibility="collapsed")
    else:
        st.radio("Go to", PAGE_KEYS, format_func=_label, key="nav", horizontal=True,
                 on_change=_nav_changed, label_visibility="collapsed")
page = st.session_state.page
st.query_params["page"] = page   # page survives refresh and can be bookmarked / shared

fc_nws, alerts = get_nws(lake["lat"], lake["lon"])
for a in alerts:
    sev = (a.get("severity") or "").lower()
    box = st.error if sev in ("extreme", "severe") else st.warning
    box(f"**{a.get('event')}** — {a.get('headline', '')}")


def _delta(days):
    past = level.loc[:TODAY - pd.Timedelta(days=days)]
    return current - float(past.iloc[-1]) if len(past) else None


def pager():
    """Previous / next buttons at the bottom of each page (handy on phones)."""
    i = PAGE_KEYS.index(page)
    st.divider()
    with st.container(key="pager"):
        b1, b2, b3 = st.columns([1, 1, 1])
        if i > 0:
            b1.button(f"← {_label(PAGE_KEYS[i - 1])}", on_click=_go, args=(PAGE_KEYS[i - 1],), **W)
        b2.markdown("<div style='text-align:center;padding-top:0.6rem'><a href='#lake-top'>⬆ Top</a></div>",
                    unsafe_allow_html=True)
        if i < len(PAGE_KEYS) - 1:
            b3.button(f"{_label(PAGE_KEYS[i + 1])} →", on_click=_go, args=(PAGE_KEYS[i + 1],), **W)


# ------------------------------------------------------------------------------------
# Tab 1: level & projection
# ------------------------------------------------------------------------------------
if page == "level":
    with st.container(key="kpis"):
        c = st.columns(3)
        c[0].metric("Current level", f"{current:.2f} ft",
                    f"{_delta(1):+.2f} ft (24h)" if _delta(1) is not None else None)
        c[1].metric("vs. normal pool", f"{current - lake['normal_pool']:+.2f} ft", help=f"Normal pool {lake['normal_pool']} ft")
        c[2].metric("vs. guide curve today", f"{current - guide_today:+.2f} ft", help=f"Guide curve ≈ {guide_today:.1f} ft today")
        c = st.columns(3)
        c[0].metric("7-day change", f"{_delta(7):+.2f} ft" if _delta(7) is not None else "—")
        c[1].metric("30-day change", f"{_delta(30):+.2f} ft" if _delta(30) is not None else "—")
        doy = TODAY.dayofyear
        if doy in clim.index:
            med = clim.loc[doy, "median"]
            c[2].metric("vs. typical for date", f"{current - med:+.2f} ft", help=f"Median for this date ≈ {med:.1f} ft")

        if proj is not None:
            st.markdown("#### Projected level")
            marks = [d for d in [14, 28, 56, 84, 120, 180] if d <= horizon]
            for k in range(0, len(marks), 3):
                cols = st.columns(3)
                for col, d in zip(cols, marks[k:k + 3]):
                    row = proj.iloc[d - 1]
                    rng = f"{row['p10']:.1f}–{row['p90']:.1f}" if "p10" in row else ""
                    col.metric(f"In {d // 7} wks ({proj.index[d - 1]:%b %d})", f"{row['p50']:.1f} ft",
                               f"{row['p50'] - current:+.1f} ft · likely {rng}", delta_color="off")
    lookback = st.select_slider("Show history", ["90 days", "6 months", "1 year", "2 years", "5 years"],
                                value="1 year")
    lb_days = {"90 days": 90, "6 months": 182, "1 year": 365, "2 years": 730, "5 years": 1826}[lookback]
    x0 = TODAY - pd.Timedelta(days=lb_days)
    x1 = TODAY + pd.Timedelta(days=horizon)
    span = pd.date_range(x0, x1, freq="D")

    fig = go.Figure()
    cb = clim.reindex(span.dayofyear)
    cb.index = span
    fig.add_trace(go.Scatter(x=span, y=cb["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=span, y=cb["p10"], fill="tonexty", fillcolor="rgba(150,150,150,0.15)",
                             line=dict(width=0), name="Historical 10–90% range"))
    fig.add_trace(go.Scatter(x=span, y=cb["median"], line=dict(color="gray", dash="dot", width=1),
                             name="Historical median"))
    fig.add_trace(go.Scatter(x=span, y=guide_curve_series(span, lake["guide_curve"]),
                             line=dict(color="#2ca02c", dash="dash", width=1.5), name="Guide curve (approx.)"))

    if proj is not None:
        if show_members and members is not None:
            for col in members.columns[: min(60, members.shape[1])]:
                fig.add_trace(go.Scatter(x=members.index, y=members[col], line=dict(width=0.5, color="rgba(31,119,180,0.15)"),
                                         showlegend=False, hoverinfo="skip"))
        if "p10" in proj:
            fig.add_trace(go.Scatter(x=proj.index, y=proj["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=proj.index, y=proj["p10"], fill="tonexty", fillcolor="rgba(255,127,14,0.15)",
                                     line=dict(width=0), name="Projection 10–90%"))
            fig.add_trace(go.Scatter(x=proj.index, y=proj["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=proj.index, y=proj["p25"], fill="tonexty", fillcolor="rgba(255,127,14,0.3)",
                                     line=dict(width=0), name="Projection 25–75%"))
        fig.add_trace(go.Scatter(x=proj.index, y=proj["p50"], line=dict(color="#ff7f0e", width=2.5),
                                 name="Projected (median)"))
        fig.add_trace(go.Scatter(x=proj.index, y=proj["normal_rain"], line=dict(color="#ff7f0e", width=1, dash="dot"),
                                 name="Projected with normal rain"))
        if n_fc:
            fig.add_vrect(x0=TODAY, x1=TODAY + pd.Timedelta(days=n_fc), fillcolor="rgba(31,119,180,0.06)",
                          line_width=0, annotation_text="rain forecast", annotation_position="top left")

    obs = level.loc[x0:]
    fig.add_trace(go.Scatter(x=obs.index, y=obs.values, line=dict(color="#1f77b4", width=2.5), name="Observed"))
    for label, elev in lake["pools"].items():
        if elev is None:
            continue
        ymin = obs.min() if len(obs) else elev
        ymax = obs.max() if len(obs) else elev
        if ymin - 8 <= elev <= ymax + 8 or "Normal" in label or "full" in label.lower():
            fig.add_hline(y=elev, line=dict(color="#d62728" if "flood" in label.lower() else "#555", width=1),
                          annotation_text=f"{label} {elev:g}", annotation_position="bottom right")
    fig.add_vline(x=TODAY, line=dict(color="black", width=1))
    fig.update_layout(height=560, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title=f"Elevation (ft {lake.get('datum', '')})",
                      legend=dict(orientation="h", y=-0.12))
    show_chart(fig)

    if model_error:
        st.warning(f"Projection unavailable: {model_error}")
    elif proj is not None:
        st.markdown("#### Weekly outlook")
        wk = proj.iloc[6::7].copy()
        tbl = pd.DataFrame({"Date": wk.index.strftime("%a %b %d"),
                            "Median (ft)": wk["p50"].round(2),
                            "Likely range (ft)": [f"{a:.1f} – {b:.1f}" for a, b in zip(wk.get("p10", wk["p50"]), wk.get("p90", wk["p50"]))],
                            "Normal-rain path (ft)": wk["normal_rain"].round(2),
                            "Guide curve (ft)": guide_curve_series(wk.index, lake["guide_curve"]).round(1).values})
        if members is not None and not members.empty:
            m = members.loc[wk.index]
            tbl[f"Chance above {lake['normal_pool']:g} ft"] = (m.gt(lake["normal_pool"]).mean(axis=1) * 100).round(0).astype(int).astype(str).values + "%"
            g = guide_curve_series(wk.index, lake["guide_curve"]).values
            tbl["Chance >2 ft below guide"] = (m.lt(g[:, None] - 2).mean(axis=1) * 100).round(0).astype(int).astype(str).values + "%"
        st.dataframe(tbl, hide_index=True, **W)
        rain_note = (f"Seasonal outlook multipliers by month: " +
                     ", ".join(f"{pd.Timestamp(2001, k, 1):%b} {v:.0%}" for k, v in sorted(outlook.items()))
                     ) if outlook else "No seasonal adjustment applied."
        st.caption(f"First {n_fc} days use the forecast basin rainfall; after that, "
                   f"{members.shape[1] if members is not None else 0} scenarios built from past years' rainfall "
                   f"× {scenario}% scenario setting. {rain_note} This is a statistical estimate, not an official "
                   "Corps of Engineers forecast — planned releases or drawdowns can change the outcome.")

# ------------------------------------------------------------------------------------
# Tab: depth map
# ------------------------------------------------------------------------------------
@st.cache_data(ttl=30 * 24 * 3600, show_spinner=False, max_entries=6)
def get_dem(bbox, width):
    return dm.fetch_dem(bbox, width)


@st.cache_data(ttl=30 * 24 * 3600, show_spinner=False, max_entries=6)
def get_outlines(bbox):
    try:
        return dm.fetch_lake_outlines(bbox)
    except Exception as e:
        ds._note("USGS lake outline (NHD)", e)
        return []


@st.cache_resource(show_spinner=False, max_entries=6)
def get_bed(lake_key, bbox, width, offset, survey_key, n_outlines, _dem, _lake, _survey, _outlines):
    return dm.LakeBed(_dem, _lake, offset, _survey, _outlines)


@st.cache_data(ttl=30 * 24 * 3600, show_spinner=False, max_entries=40)
def get_dem_view(view_bbox, width):
    return dm.fetch_dem(view_bbox, width)


_fragment = getattr(st, "fragment", None) or (lambda f: f)


@_fragment
def render_map(lake, bed, bbox, basemap, shade_level, opacity, contour_ft, line_specs, sel_date):
    """Interactive map. Re-runs on its own when you pan/zoom, fetching sharper terrain for the view."""
    key = f"depthmap_{lake['short']}"
    view = st.session_state.get(key) or {}
    center = view.get("center") or {"lat": (bbox[1] + bbox[3]) / 2, "lng": (bbox[0] + bbox[2]) / 2}
    zoom = view.get("zoom") or int(round(math.log2(900 * 360 / (256 * (bbox[2] - bbox[0])))))
    b = view.get("bounds") or {}
    grid, detail_note = bed, "whole-lake view"
    try:
        sw, ne = b.get("_southWest"), b.get("_northEast")
        if sw and ne and zoom >= 13:
            # pad the view a little so small pans don't leave blank edges; round so the cache is reused
            pad = 0.15
            dlon, dlat = ne["lng"] - sw["lng"], ne["lat"] - sw["lat"]
            vb = (sw["lng"] - pad * dlon, sw["lat"] - pad * dlat, ne["lng"] + pad * dlon, ne["lat"] + pad * dlat)
            q = 10 ** max(2, min(4, zoom - 11))
            vb = tuple(round(v * q) / q for v in vb)
            vb = (max(vb[0], bbox[0]), max(vb[1], bbox[1]), min(vb[2], bbox[2]), min(vb[3], bbox[3]))
            if vb[2] > vb[0] and vb[3] > vb[1]:
                with st.spinner("Loading sharper terrain for this view…"):
                    fine = get_dem_view(vb, 1400)
                grid = dm.DetailBed(bed, fine)
                detail_note = f"detail view · {grid.px_m:.1f} m terrain pixels"
    except Exception as e:
        detail_note = f"detail unavailable ({e})"
        grid = bed

    m = folium.Map(location=[center["lat"], center["lng"]], zoom_start=zoom, tiles=None,
                   max_zoom=19, control_scale=True, prefer_canvas=True)
    for name, urls in dm.BASEMAPS.items():
        folium.TileLayer(tiles=urls[0], attr=dm.BASEMAP_CREDIT.get(name, dm.ATTRIBUTION), name=name,
                         max_zoom=19, max_native_zoom=dm.BASEMAP_MAX_NATIVE.get(name, 19),
                         overlay=False, control=True, show=(name == basemap)).add_to(m)
    if basemap.endswith("+ labels"):
        folium.TileLayer(tiles=dm.LABELS_URL, attr=dm.LABELS_ATTR, name="Labels", overlay=True,
                         max_zoom=19).add_to(m)

    img = grid.render_depth(shade_level, contour_ft=contour_ft)
    folium.raster_layers.ImageOverlay(dm.to_data_url(img), bounds=grid.bounds_latlon(), opacity=opacity,
                                      name=f"Depth at {shade_level:.1f} ft", interactive=False,
                                      zindex=2).add_to(m)
    step = 1 if grid is not bed else 2
    for level, color, dash, label in line_specs:
        fg = folium.FeatureGroup(name=label)
        for lat, lon in grid.shoreline_paths(level, step=step):
            folium.PolyLine(list(zip(lat.round(6), lon.round(6))), color=color, weight=3 if dash is None else 2,
                            dash_array=dash, opacity=0.95, tooltip=label).add_to(fg)
        fg.add_to(m)
    folium.Marker([lake["lat"], lake["lon"]], tooltip="Dam / lake gauge",
                  icon=folium.Icon(color="red", icon="tint")).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    out = st_folium(m, key=key, height=480 if COMPACT else 680, use_container_width=True,
                    returned_objects=["bounds", "zoom", "center", "last_clicked"])

    clicked = (out or {}).get("last_clicked")
    c1, c2 = st.columns([3, 2])
    c1.caption(f"Map: {detail_note}. Use the layers button (top right) to switch imagery or hide lines.")
    if clicked:
        info = grid.depth_at(clicked["lng"], clicked["lat"], shade_level) or bed.depth_at(clicked["lng"], clicked["lat"], shade_level)
        if info and info["wet"]:
            c2.markdown(f"📍 **{info['depth']:.0f} ft deep** at {shade_level:.1f} ft "
                        f"(bottom {info['bottom']:.0f} ft, {'estimated' if info['estimated'] else 'measured'})")
        elif info:
            c2.markdown(f"📍 Dry at {shade_level:.1f} ft — ground at {info['bottom']:.0f} ft")
    st.markdown(
        "<div style='font-size:0.9em'>"
        "<span style='display:inline-block;width:120px;height:12px;background:linear-gradient(to right,#c6ecff,#041e5a);"
        "vertical-align:middle'></span> shallow → deep &nbsp; "
        "<span style='background:#deb870;padding:0 8px'>&nbsp;</span> lakebed exposed (below normal pool) &nbsp; "
        "<span style='background:#ff781e;padding:0 8px'>&nbsp;</span> flooded land above normal pool &nbsp; "
        "<span style='color:#e6c800'>━━</span> current &nbsp; <span style='color:#ff00ff'>━━</span> projected &nbsp; "
        "<span style='color:#ff69b4'>╍╍</span> likely range &nbsp; <span style='color:#999'>╍╍</span> normal pool (white)"
        "</div>", unsafe_allow_html=True)


if page == "map":
    st.markdown("Water depth and shoreline on satellite imagery. **Zoom in and the map re-draws in more detail** "
                "(down to ~1–3 m from USGS lidar). Click anywhere on the water to read the depth.")
    o1, o2, o3, o4 = st.columns([1.3, 1, 1, 1])
    basemap = o1.selectbox("Base map", list(dm.BASEMAPS), index=1)
    shade_at = o2.selectbox("Shade depths at", ["Projected level", "Current level"], index=0)
    opacity = o3.slider("Depth shading opacity", 0.0, 1.0, 0.6, 0.05)
    contour_ft = o4.selectbox("Depth contours every", [None, 5, 10, 20], index=2,
                              format_func=lambda v: "off" if v is None else f"{v} ft")

    if proj is not None:
        opts = [TODAY] + list(proj.index[6::7])
        sel_date = st.select_slider("Projection date", options=opts, value=opts[min(4, len(opts) - 1)],
                                    format_func=lambda d: "Today" if d == TODAY else d.strftime("%b %d, %Y"))
    else:
        sel_date = TODAY
    if sel_date == TODAY or proj is None:
        p_mid = p_lo = p_hi = current
    else:
        r = proj.loc[sel_date]
        p_mid, p_lo, p_hi = float(r["p50"]), float(r.get("p10", r["p50"])), float(r.get("p90", r["p50"]))

    l1, l2, l3, l4 = st.columns(4)
    show_cur = l1.checkbox("Current shoreline", True)
    show_proj = l2.checkbox("Projected shoreline", proj is not None)
    show_rng = l3.checkbox("Projected likely range", False)
    show_norm = l4.checkbox("Normal-pool shoreline", True)

    with st.expander("Datum, real depth soundings, and map area"):
        offset = st.number_input(f"Lake datum ({lake.get('datum')}) minus NAVD88, ft", value=float(lake.get("datum_offset_ft", 0.0)),
                                 step=0.1, format="%.2f",
                                 help="The terrain data is in NAVD88. Around Kerr Lake NGVD29 ≈ NAVD88 + ~0.9 ft. "
                                      "Nudge this if the normal-pool shoreline doesn't match the satellite shoreline.")
        up = st.file_uploader("Optional: CSV of real depth soundings (columns: lon, lat, bottom_elev_ft)", type="csv")
        survey = None
        if up is not None:
            try:
                survey = pd.read_csv(up)
            except Exception as e:
                st.error(f"Couldn't read CSV: {e}")
        bbox = dm.default_bbox(lake)
        st.caption(f"Map area: {bbox[0]:.3f}, {bbox[1]:.3f} to {bbox[2]:.3f}, {bbox[3]:.3f} (set `map_bbox` in lakes.py).")

    bed = None
    try:
        with st.spinner("Loading USGS lidar terrain for the whole lake (first time takes ~10–30 s)…"):
            dem = get_dem(bbox, 1600)
            skey = (len(survey), float(pd.util.hash_pandas_object(survey).sum())) if survey is not None else None
            outlines = get_outlines(bbox)
            bed = get_bed(choice, bbox, 1600, round(offset, 2), skey, len(outlines), dem, lake, survey, outlines)
    except Exception as e:
        st.error(f"Couldn't build the depth map: {e}")

    if bed is not None:
        shade_level = p_mid if shade_at == "Projected level" else current
        line_specs = []
        if show_norm:
            line_specs.append((lake["normal_pool"], "#ffffff", "8 6", f"Normal pool {lake['normal_pool']:g} ft"))
        if show_rng and proj is not None and sel_date != TODAY:
            line_specs += [(p_lo, "#ff69b4", "6 6", f"Projected low (10%) {p_lo:.1f} ft"),
                           (p_hi, "#ff69b4", "6 6", f"Projected high (90%) {p_hi:.1f} ft")]
        if show_proj and proj is not None and sel_date != TODAY:
            line_specs.append((p_mid, "#ff00ff", None, f"Projected {sel_date:%b %d}: {p_mid:.1f} ft"))
        if show_cur:
            line_specs.append((current, "#ffe600", None, f"Current: {current:.1f} ft"))
        render_map(lake, bed, bbox, basemap, shade_level, opacity, contour_ft, line_specs, sel_date)

        s_now, s_sel, s_norm = bed.stats(current), bed.stats(p_mid), bed.stats(lake["normal_pool"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Surface area now", f"{s_now['area_acres']:,.0f} ac",
                  f"{s_now['area_acres'] / s_norm['area_acres'] - 1:+.1%} vs normal pool" if s_norm["area_acres"] else None)
        m2.metric(f"Surface area {'projected ' + sel_date.strftime('%b %d') if sel_date != TODAY else 'today'}",
                  f"{s_sel['area_acres']:,.0f} ac", f"{s_sel['area_acres'] - s_now['area_acres']:+,.0f} ac vs now")
        m3.metric("Volume now (est.)", f"{s_now['volume_acft'] / 1000:,.0f}k ac-ft",
                  f"{(s_sel['volume_acft'] - s_now['volume_acft']) / 1000:+,.0f}k ac-ft by selected date" if sel_date != TODAY else None)
        m4.metric("Deepest point now (est.)", f"{s_now['max_depth_ft']:.0f} ft")

        est_pct = bed.estimated.sum() / max(bed.lake_mask.sum(), 1)
        st.caption(
            f"Terrain: USGS 3DEP lidar. The lidar was flown with the lake at about "
            f"**{' / '.join(f'{v:.1f}' for v in bed.lidar_levels)} ft**"
            f"{' (separate surveys for different parts of the lake)' if len(bed.lidar_levels) > 1 else ''}, "
            "so ground above that level — banks, flats, coves and the flood pool — is "
            f"measured. Depths below it are **estimated** ({est_pct:.0%} of the lake bed) from distance to shore and to "
            f"the dam, scaled to a ~{lake.get('max_depth_ft', '?')} ft maximum depth"
            + (f"; {bed.survey_used:,} uploaded soundings replace the estimate where they reach." if bed.survey_used else ".")
            + " Not for navigation. Satellite imagery shows the lake on the day it was photographed, which may be a different level."
            + (" Lake outline from the USGS National Hydrography Dataset." if bed.outline_mask is not None else ""))


# ------------------------------------------------------------------------------------
# Tab 2: weather
# ------------------------------------------------------------------------------------
if page == "weather":
    if wx_fc.empty:
        st.warning("Weather forecast unavailable right now.")
    else:
        fut = wx_fc.loc[wx_fc.index >= TODAY]
        past14 = wx_fc.loc[(wx_fc.index < TODAY) & (wx_fc.index >= TODAY - pd.Timedelta(days=14))]
        normal_daily = (wx_hist["precip"].groupby(wx_hist.index.dayofyear).mean()
                        .rolling(15, center=True, min_periods=1).mean()) if not wx_hist.empty else None
        a, b, cc, d = st.columns(4)
        a.metric("Basin rain, last 14 days", f"{past14['precip'].sum():.2f} in")
        b.metric("Basin rain, next 16 days", f"{fut['precip'].sum():.2f} in")
        if normal_daily is not None:
            nf = normal_daily.reindex(fut.index.dayofyear).sum()
            npast = normal_daily.reindex(past14.index.dayofyear).sum()
            cc.metric("Next 16 days vs. normal", f"{fut['precip'].sum() / nf:.0%}" if nf else "—")
            d.metric("Last 14 days vs. normal", f"{past14['precip'].sum() / npast:.0%}" if npast else "—")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=wx_fc.index, y=wx_fc["precip"], name="Basin-average rain (in)",
                             marker_color=["#9ecae1" if x < TODAY else "#3182bd" for x in wx_fc.index]))
        if normal_daily is not None:
            fig.add_trace(go.Scatter(x=wx_fc.index, y=normal_daily.reindex(wx_fc.index.dayofyear).values,
                                     name="Normal daily rain", line=dict(color="gray", dash="dot")))
        fig.add_trace(go.Scatter(x=wx_fc.index, y=wx_fc["tmax"], name="High °F", yaxis="y2", line=dict(color="#d62728")))
        fig.add_trace(go.Scatter(x=wx_fc.index, y=wx_fc["tmin"], name="Low °F", yaxis="y2", line=dict(color="#9467bd")))
        fig.add_vline(x=TODAY, line=dict(color="black", width=1))
        fig.update_layout(height=420, yaxis=dict(title="Rain (in)"),
                          yaxis2=dict(title="°F", overlaying="y", side="right"),
                          hovermode="x unified", legend=dict(orientation="h", y=-0.2),
                          margin=dict(l=10, r=10, t=30, b=10),
                          title="Watershed rainfall & temperature (average of points across the drainage basin)")
        show_chart(fig)

        mc = get_model_comparison(lake["lat"], lake["lon"])
        if not mc.empty:
            st.markdown("#### Do the weather models agree? (rain at the lake, cumulative)")
            fig = go.Figure()
            for col in mc.columns:
                fig.add_trace(go.Scatter(x=mc.index, y=mc[col].fillna(0).cumsum(), name=col, mode="lines"))
            fig.update_layout(height=320, yaxis_title="Cumulative rain (in)", hovermode="x unified",
                              margin=dict(l=10, r=10, t=10, b=10))
            show_chart(fig)

    st.markdown(f"#### National Weather Service forecast{' — ' + fc_nws.get('city', '') if fc_nws.get('city') else ''}")
    periods = fc_nws.get("periods", [])
    if periods:
        for p in periods[:10]:
            st.markdown(f"**{p['name']}** — {p['temperature']}°{p['temperatureUnit']}, "
                        f"{p.get('shortForecast', '')}. {p.get('detailedForecast', '')}")
    else:
        st.info("NWS forecast unavailable right now.")

# ------------------------------------------------------------------------------------
# Tab 3: long range
# ------------------------------------------------------------------------------------
if page == "outlook":
    if seasonal.empty:
        seasonal = get_seasonal(lake["lat"], lake["lon"])
    if not seasonal.empty and not wx_hist.empty:
        st.markdown("#### ECMWF seasonal rainfall outlook vs. normal (at the lake)")
        mon = seasonal.resample("MS").sum()
        days_in = seasonal.resample("MS").count()
        norm = wx_hist["precip"].groupby(wx_hist.index.month).mean()
        df = pd.DataFrame({"Outlook (in)": mon, "Days covered": days_in})
        df["Normal (in)"] = [norm.get(i.month, np.nan) * n for i, n in zip(df.index, df["Days covered"])]
        df = df[df["Days covered"] >= 10]
        df["% of normal"] = (df["Outlook (in)"] / df["Normal (in)"] * 100).round(0)
        fig = go.Figure([go.Bar(x=df.index.strftime("%b %Y"), y=df["Normal (in)"], name="Normal", marker_color="lightgray"),
                         go.Bar(x=df.index.strftime("%b %Y"), y=df["Outlook (in)"], name="Seasonal outlook", marker_color="#3182bd")])
        fig.update_layout(barmode="group", height=340, yaxis_title="Rain (in)", margin=dict(l=10, r=10, t=10, b=10))
        show_chart(fig)
        st.caption("Ensemble mean of the ECMWF EC46 / SEAS5 seasonal models via Open-Meteo. Partial months are "
                   "compared against normal for the same number of days. Seasonal rainfall forecasts have modest skill — "
                   "treat them as a tilt in the odds, not a prediction.")
    else:
        st.info("Seasonal outlook data unavailable right now.")

    st.markdown("#### NOAA Climate Prediction Center outlooks")
    keys = list(ds.CPC_MAPS)
    for i in range(0, len(keys), 2):
        cols = st.columns(2)
        for col, k in zip(cols, keys[i:i + 2]):
            col.image(ds.CPC_MAPS[k], caption=k, **W)

    if lake.get("drought_fips"):
        st.markdown("#### U.S. Drought Monitor — counties around the lake")
        dr = get_drought(tuple(lake["drought_fips"].items()))
        if dr.empty:
            st.info("Drought Monitor data unavailable right now — see https://droughtmonitor.unl.edu")
        else:
            latest = dr.sort_values("Week").groupby("County").tail(1)
            show = latest[["County", "Week", "NONE", "D0", "D1", "D2", "D3", "D4"]].copy()
            show["Week"] = show["Week"].dt.strftime("%b %d, %Y")
            st.dataframe(show.rename(columns={"NONE": "% no drought", "D0": "% D0+ abnormally dry",
                                              "D1": "% D1+ moderate", "D2": "% D2+ severe",
                                              "D3": "% D3+ extreme", "D4": "% D4 exceptional"}),
                         hide_index=True, **W)
            fig = go.Figure()
            for county, g in dr.groupby("County"):
                fig.add_trace(go.Scatter(x=g["Week"], y=g["D1"], name=county, mode="lines+markers"))
            fig.update_layout(height=280, yaxis_title="% of county in moderate drought or worse",
                              margin=dict(l=10, r=10, t=10, b=10))
            show_chart(fig)

# ------------------------------------------------------------------------------------
# Tab 4: inflows & Corps data
# ------------------------------------------------------------------------------------
if page == "inflows":
    if lake.get("upstream_usgs"):
        st.markdown("#### Rivers flowing into the lake (USGS)")
        fig = go.Figure()
        for site, label in lake["upstream_usgs"].items():
            s, _ = get_flow_history(site, 120)
            if len(s):
                fig.add_trace(go.Scatter(x=s.index, y=s.values, name=f"{label} ({site})"))
        if fig.data:
            fig.update_layout(height=340, yaxis_title="Flow (cfs)", yaxis_type="log", hovermode="x unified",
                              margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2))
            show_chart(fig)
        else:
            st.info("Upstream river data unavailable right now.")

    st.markdown("#### U.S. Army Corps of Engineers (CWMS) data")
    if cwms:
        for kind, info in cwms.items():
            s = info["series"].tail(24 * 120 * 4)
            fig = go.Figure(go.Scatter(x=s.index, y=s.values, name=kind))
            fig.update_layout(height=260, title=f"{kind} — {info['tsid']} ({info['units']})",
                              margin=dict(l=10, r=10, t=40, b=10))
            show_chart(fig)
        if "Inflow" in cwms and "Outflow" in cwms:
            i = cwms["Inflow"]["series"].resample("D").mean()
            o = cwms["Outflow"]["series"].resample("D").mean()
            net = (i - o).dropna().tail(30)
            st.metric("Avg. net inflow, last 7 days (inflow − release)", f"{net.tail(7).mean():,.0f} cfs")
    else:
        st.info("No Corps time series were found for this lake through the public CWMS Data API "
                "(or the service is unavailable). Levels still come from the USGS gauge.")
    with st.expander("Corps time series found for this lake"):
        st.write(cwms_names or "None found.")

    st.markdown("#### NWS river forecast points nearby")
    nw = get_nwps(lake["lat"], lake["lon"])
    if nw.empty:
        st.info("No NWS river forecast points returned.")
    else:
        st.dataframe(nw, hide_index=True, **W,
                     column_config={"Link": st.column_config.LinkColumn("Link", display_text="open")})

# ------------------------------------------------------------------------------------
# Tab 5: news
# ------------------------------------------------------------------------------------
if page == "news":
    items = get_news(tuple(lake["news_terms"]))
    if not items:
        st.info("No news found right now.")
    else:
        all_tags = sorted({t for it in items for t in it["tags"]})
        pick = st.multiselect("Filter by topic that could affect the lake", all_tags)
        only_impact = st.checkbox("Only show stories tagged with a possible lake impact", value=False)
        for it in items:
            if pick and not set(pick) & set(it["tags"]):
                continue
            if only_impact and not it["tags"]:
                continue
            when = it["published"].strftime("%b %d, %Y") if it["published"] else ""
            tags = " ".join(f"`{t}`" for t in it["tags"])
            st.markdown(f"**[{it['title']}]({it['link']})**  \n{it['source']} · {when} {tags}")
        st.caption("Headlines from Google News. Topic tags are keyword-based, so give them a quick sanity check.")

# ------------------------------------------------------------------------------------
# Tab 6: lake info
# ------------------------------------------------------------------------------------
if page == "info":
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### About the lake")
        for f in lake.get("facts", []):
            st.markdown(f"- {f}")
        st.markdown("#### Key elevations")
        st.dataframe(pd.DataFrame({"Level": list(lake["pools"]), "Elevation (ft)": list(lake["pools"].values())}),
                     hide_index=True)
        rec = level_hist.dropna()
        if len(rec):
            st.markdown(f"#### In the record shown here ({rec.index[0]:%Y}–{rec.index[-1]:%Y})")
            st.markdown(f"- Highest daily level: **{rec.max():.2f} ft** on {rec.idxmax():%b %d, %Y}\n"
                        f"- Lowest daily level: **{rec.min():.2f} ft** on {rec.idxmin():%b %d, %Y}")
        st.markdown("#### Links")
        for k, v in lake.get("links", {}).items():
            st.markdown(f"- [{k}]({v})")
    with right:
        pts = pd.DataFrame([{"lat": lake["lat"], "lon": lake["lon"], "size": 400, "color": "#d62728"}] +
                           [{"lat": a, "lon": b, "size": 150, "color": "#1f77b4"} for a, b in lake["basin_points"]])
        st.map(pts, latitude="lat", longitude="lon", size="size", color="color")
        st.caption("Red: the dam/gauge. Blue: watershed points averaged for rainfall.")

# ------------------------------------------------------------------------------------
# Tab 7: model & sources
# ------------------------------------------------------------------------------------
if page == "model":
    st.markdown("#### How the projection works")
    st.markdown(
        "1. **Learn the lake's behavior** from its own daily record: how the level responds to watershed rain "
        "(same day through ~3 weeks earlier), evaporation, the season, and how far it sits above or below the guide "
        "curve (the Corps releases more when high and holds water when low).\n"
        "2. **Next 16 days:** drive that model with the forecast basin rainfall.\n"
        "3. **After that:** run scenarios using the rain that actually fell over the same dates in each past year, "
        "tilted by the seasonal outlook and your scenario slider.\n"
        "4. The spread of scenarios gives the shaded likely range.")
    if model is not None:
        a, b, cc = st.columns(3)
        a.metric("Days of history used", f"{model.n:,}")
        b.metric("Day-to-day fit (R²)", f"{model.r2:.2f}")
        hc, stats = model.hindcast(level_hist, wx_hist, lead=30)
        if stats:
            cc.metric("30-day hindcast error (model vs. 'no change')",
                      f"{stats['model_mae']:.2f} ft vs {stats['persistence_mae']:.2f} ft",
                      help="Average miss when the model is started on past dates and given the rain that actually fell.")
            fig = go.Figure([go.Scatter(x=hc["start"] + pd.Timedelta(days=30), y=hc["observed"], mode="markers", name="Observed"),
                             go.Scatter(x=hc["start"] + pd.Timedelta(days=30), y=hc["model"], mode="markers", name="Model (30 days ahead)")])
            fig.update_layout(height=300, yaxis_title="Elevation (ft)", margin=dict(l=10, r=10, t=10, b=10))
            show_chart(fig)
            st.caption("Hindcasts use observed rain, so real-world error is larger because the rain forecast is uncertain.")
        with st.expander("Model coefficients"):
            st.dataframe(model.coefficients(), hide_index=True)

    st.markdown("#### Data sources")
    st.markdown(
        "- **USGS Water Data** (lake elevation & river flows) — api.waterdata.usgs.gov, with legacy NWIS fallback\n"
        "- **U.S. Army Corps of Engineers CWMS Data API** (elevation, inflow, release, storage) — cwms-data.usace.army.mil\n"
        "- **Open-Meteo** (ERA5 rainfall history, GFS/ECMWF/ICON/GEM forecasts, ECMWF seasonal outlook)\n"
        "- **National Weather Service** (forecast text, active alerts) — api.weather.gov\n"
        "- **NWS National Water Prediction Service** (river forecast points) — api.water.noaa.gov\n"
        "- **NOAA Climate Prediction Center** (6–10 day, 8–14 day, monthly, seasonal outlooks)\n"
        "- **U.S. Drought Monitor** (county drought statistics)\n"
        "- **Google News RSS** (headlines)")
    if ds.SOURCE_ERRORS:
        with st.expander(f"⚠️ {len(ds.SOURCE_ERRORS)} source(s) had problems on this load"):
            st.json(ds.SOURCE_ERRORS)
    else:
        st.success("No source problems reported on this load (cached results are re-checked when they expire).")

pager()
