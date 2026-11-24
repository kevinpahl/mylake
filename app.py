"""
Lake Level Tracker - Streamlit app
Run:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_sources as ds
from lakes import DEFAULT_LAKE, LAKES, custom_lake
from model import LakeModel, climatology_bands, guide_curve_series, outlook_ratio_by_month

st.set_page_config(page_title="Lake Level Tracker", page_icon="🌊", layout="wide")

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
# Sidebar
# ------------------------------------------------------------------------------------
st.sidebar.title("🌊 Lake Level Tracker")
names = list(LAKES) + ["➕ Other lake (enter a USGS gauge)"]
choice = st.sidebar.selectbox("Lake or reservoir", names, index=names.index(DEFAULT_LAKE))

if choice.startswith("➕"):
    with st.sidebar.expander("Custom lake", expanded=True):
        c_name = st.text_input("Name", "My Lake")
        c_site = st.text_input("USGS site number (lake elevation gauge)", "")
        c_lat = st.number_input("Latitude", value=36.0, format="%.4f")
        c_lon = st.number_input("Longitude", value=-79.0, format="%.4f")
        c_norm = st.number_input("Normal / full pool (ft)", value=300.0)
        c_flood = st.number_input("Top of flood pool (ft)", value=320.0)
    if not c_site:
        st.info("Enter a USGS site number in the sidebar to load a custom lake.")
        st.stop()
    lake = custom_lake(c_name, c_site.strip(), c_lat, c_lon, c_norm, c_flood)
else:
    lake = LAKES[choice]

st.sidebar.markdown("### Projection settings")
horizon = st.sidebar.slider("Projection length (days)", 14, 180, 90, step=7)
use_outlook = st.sidebar.checkbox("Adjust for seasonal rainfall outlook", value=True,
                                  help="Scales rainfall after the 16-day forecast by the ECMWF "
                                       "seasonal outlook vs. normal for each month.")
scenario = st.sidebar.slider("Rainfall scenario (% of normal after 16 days)", 50, 150, 100, step=5,
                             help="Try 'what if it stays dry' (e.g. 60%) or 'wet winter' (130%).")
years = st.sidebar.slider("Years of history to learn from", 5, 25, 15)
show_members = st.sidebar.checkbox("Show individual scenarios", value=False)

if st.sidebar.button("🔄 Refresh all data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
st.sidebar.caption(f"Data refreshes automatically (levels every 15 min, weather every 2 h). "
                   f"{'USGS API key loaded.' if USGS_KEY else 'Tip: add a free USGS API key (see README).'}")

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
# Header
# ------------------------------------------------------------------------------------
st.title(f"{lake['short']} — level tracker & outlook")
st.caption(f"Elevations in feet ({lake.get('datum', '')}). Lake level from {level_source}. "
           f"Last reading {current_time:%b %d, %Y %I:%M %p}.")

fc_nws, alerts = get_nws(lake["lat"], lake["lon"])
for a in alerts:
    sev = (a.get("severity") or "").lower()
    box = st.error if sev in ("extreme", "severe") else st.warning
    box(f"**{a.get('event')}** — {a.get('headline', '')}")


def _delta(days):
    past = level.loc[:TODAY - pd.Timedelta(days=days)]
    return current - float(past.iloc[-1]) if len(past) else None


c = st.columns(6)
c[0].metric("Current level", f"{current:.2f} ft",
            f"{_delta(1):+.2f} ft (24h)" if _delta(1) is not None else None)
c[1].metric("vs. normal pool", f"{current - lake['normal_pool']:+.2f} ft", help=f"Normal pool {lake['normal_pool']} ft")
c[2].metric("vs. guide curve today", f"{current - guide_today:+.2f} ft", help=f"Guide curve ≈ {guide_today:.1f} ft today")
c[3].metric("7-day change", f"{_delta(7):+.2f} ft" if _delta(7) is not None else "—")
c[4].metric("30-day change", f"{_delta(30):+.2f} ft" if _delta(30) is not None else "—")
doy = TODAY.dayofyear
if doy in clim.index:
    med = clim.loc[doy, "median"]
    c[5].metric("vs. typical for date", f"{current - med:+.2f} ft", help=f"Median for this date ≈ {med:.1f} ft")

if proj is not None:
    marks = [d for d in [14, 28, 56, 84, 120, 180] if d <= horizon]
    cols = st.columns(len(marks))
    for col, d in zip(cols, marks):
        row = proj.iloc[d - 1]
        rng = f"{row['p10']:.1f}–{row['p90']:.1f} ft" if "p10" in row else ""
        col.metric(f"Projected in {d // 7} wks ({proj.index[d - 1]:%b %d})", f"{row['p50']:.1f} ft",
                   f"{row['p50'] - current:+.1f} ft · range {rng}", delta_color="off")

tabs = st.tabs(["📈 Level & projection", "🌧️ Weather (next 16 days)", "🗓️ Long-range outlook",
                "🏞️ Inflows & Corps data", "📰 News", "ℹ️ Lake info", "⚙️ Model & sources"])

# ------------------------------------------------------------------------------------
# Tab 1: level & projection
# ------------------------------------------------------------------------------------
with tabs[0]:
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
    st.plotly_chart(fig, **W)

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
# Tab 2: weather
# ------------------------------------------------------------------------------------
with tabs[1]:
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
        st.plotly_chart(fig, **W)

        mc = get_model_comparison(lake["lat"], lake["lon"])
        if not mc.empty:
            st.markdown("#### Do the weather models agree? (rain at the lake, cumulative)")
            fig = go.Figure()
            for col in mc.columns:
                fig.add_trace(go.Scatter(x=mc.index, y=mc[col].fillna(0).cumsum(), name=col, mode="lines"))
            fig.update_layout(height=320, yaxis_title="Cumulative rain (in)", hovermode="x unified",
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, **W)

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
with tabs[2]:
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
        st.plotly_chart(fig, **W)
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
            st.plotly_chart(fig, **W)

# ------------------------------------------------------------------------------------
# Tab 4: inflows & Corps data
# ------------------------------------------------------------------------------------
with tabs[3]:
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
            st.plotly_chart(fig, **W)
        else:
            st.info("Upstream river data unavailable right now.")

    st.markdown("#### U.S. Army Corps of Engineers (CWMS) data")
    if cwms:
        for kind, info in cwms.items():
            s = info["series"].tail(24 * 120 * 4)
            fig = go.Figure(go.Scatter(x=s.index, y=s.values, name=kind))
            fig.update_layout(height=260, title=f"{kind} — {info['tsid']} ({info['units']})",
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, **W)
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
with tabs[4]:
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
with tabs[5]:
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
with tabs[6]:
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
            st.plotly_chart(fig, **W)
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
