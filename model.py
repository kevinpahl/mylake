"""
Lake-level projection model.

How it works (plain English):
  1. From the lake's own history we learn how the level changes day to day in response to
     basin rainfall (recent and lagged up to ~3 weeks), evaporation, the time of year, and
     how far the lake sits above or below the Corps' guide curve (the Corps releases water
     when the lake is high and holds it when the lake is low).
  2. For the first ~16 days we drive that model with the actual rainfall forecast.
  3. Beyond the forecast we run many scenarios ("ensemble members"), each using the rainfall
     that actually fell over the same calendar dates in a past year. Those rain amounts are
     scaled by the seasonal outlook (or your manual wetter/drier setting).
  4. The spread of the scenarios gives a likely range (10th-90th percentile) for each day.

This is a statistical model, not the Corps' operational forecast. Corps operations
(e.g. a planned drawdown, hydropower scheduling, or emergency releases) can override it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [(0, 1), (2, 4), (5, 10), (11, 24)]  # precipitation windows, days before the step


def guide_curve_series(index: pd.DatetimeIndex, breakpoints) -> pd.Series:
    """Interpolate (month, day, elevation) breakpoints onto any date index (repeats every year)."""
    doy_pts = np.array([pd.Timestamp(2001, m, d).dayofyear for m, d, _ in breakpoints], dtype=float)
    elev_pts = np.array([e for _, _, e in breakpoints], dtype=float)
    order = np.argsort(doy_pts)
    doy_pts, elev_pts = doy_pts[order], elev_pts[order]
    # wrap around the year so Dec->Jan interpolates smoothly
    x = np.concatenate([doy_pts - 365, doy_pts, doy_pts + 365])
    y = np.concatenate([elev_pts, elev_pts, elev_pts])
    doy = np.asarray(index.dayofyear, dtype=float)
    return pd.Series(np.interp(doy, x, y), index=index)


def _lagged(precip: np.ndarray, t: int) -> list[float]:
    """Precipitation sums in each lag window ending at day t (inclusive)."""
    out = []
    for a, b in LAGS:
        lo, hi = max(t - b, 0), t - a + 1
        out.append(float(np.nansum(precip[lo:hi])) if hi > lo else 0.0)
    return out


def _features(precip, et0, t, h, guide_next, doy):
    p = _lagged(precip, t)
    et7 = float(np.nansum(et0[max(t - 6, 0): t + 1]))
    dev = h - guide_next
    ang = 2 * np.pi * doy / 365.25
    return [1.0, *p, et7, max(dev, 0.0), min(dev, 0.0), max(dev, 0.0) ** 2,
            np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)]


FEATURE_NAMES = ["intercept", "rain 0-1d", "rain 2-4d", "rain 5-10d", "rain 11-24d",
                 "evap 7d", "above guide", "below guide", "above guide²",
                 "sin(season)", "cos(season)", "sin(2·season)", "cos(2·season)"]


class LakeModel:
    def __init__(self, guide_breakpoints, ridge=1.0):
        self.guide_bp = guide_breakpoints
        self.ridge = ridge
        self.coef = None
        self.resid = None
        self.resid_month = {}
        self.r2 = np.nan
        self.n = 0

    # ------------------------------------------------------------------ fitting
    def fit(self, level: pd.Series, weather: pd.DataFrame):
        """level: daily elevation (ft). weather: daily 'precip' and 'et0' (inches)."""
        df = pd.DataFrame({"h": level}).join(weather[["precip", "et0"]], how="inner")
        df = df.asfreq("D")
        df["precip"] = df["precip"].fillna(0.0)
        df["et0"] = df["et0"].fillna(df["et0"].mean() if df["et0"].notna().any() else 0.1)
        guide = guide_curve_series(df.index, self.guide_bp).values
        P, E, H = df["precip"].values, df["et0"].values, df["h"].values
        X, y, months = [], [], []
        for t in range(24, len(df) - 1):
            if np.isnan(H[t]) or np.isnan(H[t + 1]):
                continue
            X.append(_features(P, E, t + 1, H[t], guide[t + 1], df.index[t + 1].dayofyear))
            y.append(H[t + 1] - H[t])
            months.append(df.index[t + 1].month)
        X, y = np.asarray(X), np.asarray(y)
        if len(y) < 200:
            raise ValueError(f"Only {len(y)} usable days of history - need at least 200 to fit.")
        # robust-ish: drop extreme jumps (gauge glitches) before fitting
        keep = np.abs(y - np.median(y)) < 8 * (np.percentile(np.abs(y - np.median(y)), 90) + 1e-6)
        X, y, months = X[keep], y[keep], np.asarray(months)[keep]
        mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-9
        Xs = np.column_stack([X[:, 0], (X[:, 1:] - mu) / sd])
        A = Xs.T @ Xs + self.ridge * np.diag([0] + [1] * (Xs.shape[1] - 1))
        beta = np.linalg.solve(A, Xs.T @ y)
        # back to unscaled coefficients
        coef = np.empty_like(beta)
        coef[1:] = beta[1:] / sd
        coef[0] = beta[0] - np.sum(beta[1:] * mu / sd)
        self.coef = coef
        pred = X @ coef
        self.resid = y - pred
        self.resid_month = {m: self.resid[months == m] for m in range(1, 13)}
        self.r2 = 1 - np.var(self.resid) / np.var(y)
        self.n = len(y)
        return self

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": FEATURE_NAMES, "coefficient": self.coef})

    # ------------------------------------------------------------------ simulation
    def _step(self, P, E, t, h, guide_next, doy):
        return float(np.dot(_features(P, E, t, h, guide_next, doy), self.coef))

    def simulate(self, h0, start_date, precip_path, et_path, past_precip, past_et,
                 noise=None, bounds=(None, None)):
        """Run one path. precip_path/et_path: arrays for days 1..N after start_date.
        past_precip/past_et: the last >=24 days ending on start_date (for lag features)."""
        n = len(precip_path)
        P = np.concatenate([past_precip, precip_path])
        E = np.concatenate([past_et, et_path])
        off = len(past_precip) - 1  # index of start_date in P
        dates = pd.date_range(start_date + pd.Timedelta(days=1), periods=n, freq="D")
        guide = guide_curve_series(dates, self.guide_bp).values
        lo, hi = bounds
        h, out = h0, np.empty(n)
        for i in range(n):
            dh = self._step(P, E, off + i + 1, h, guide[i], dates[i].dayofyear)
            if noise is not None:
                dh += noise[i]
            h = h + dh
            if lo is not None:
                h = max(h, lo)
            if hi is not None:
                h = min(h, hi)
            out[i] = h
        return pd.Series(out, index=dates)

    def _noise(self, rng, dates, block=7):
        """Block-bootstrap residuals from the matching month (keeps realistic persistence)."""
        out = np.empty(len(dates))
        i = 0
        while i < len(dates):
            pool = self.resid_month.get(dates[i].month)
            if pool is None or len(pool) < block:
                pool = self.resid
            j = rng.integers(0, len(pool) - block)
            k = min(block, len(dates) - i)
            out[i:i + k] = pool[j:j + k]
            i += k
        return out * 0.6  # residuals partly reflect gauge noise, damp so bands aren't overstated

    def project(self, level, weather_hist, forecast, horizon_days=120, outlook_ratio=None,
                manual_pct=100, n_noise=3, seed=7, bounds=(None, None)):
        """
        level         observed daily elevation (to today)
        weather_hist  daily precip/et0 history (long - provides analog years)
        forecast      daily precip/et0 for the coming days (Open-Meteo 16-day)
        outlook_ratio dict month -> precip multiplier from the seasonal outlook (or None)
        manual_pct    user's scenario multiplier (100 = as outlook / normal)
        Returns (percentile DataFrame, members DataFrame, deterministic "normal-rain" path)
        """
        rng = np.random.default_rng(seed)
        last = level.dropna()
        start = last.index[-1]
        h0 = float(last.iloc[-1])
        dates = pd.date_range(start + pd.Timedelta(days=1), periods=horizon_days, freq="D")

        # reanalysis history first; the forecast feed's "past days" fills the archive's lag
        wx = weather_hist[["precip", "et0"]].copy()
        if forecast is not None and len(forecast):
            wx = wx.combine_first(forecast[["precip", "et0"]]) if len(wx) else forecast[["precip", "et0"]].copy()
        past = wx.loc[:start].tail(30).asfreq("D").fillna({"precip": 0.0})
        past["et0"] = past["et0"].fillna(0.1)
        pastP, pastE = np.array(past["precip"].values, dtype=float), np.array(past["et0"].values, dtype=float)

        fc = forecast.loc[forecast.index > start, ["precip", "et0"]] if forecast is not None and len(forecast) else pd.DataFrame()
        fc = fc.reindex(dates).dropna(how="all")
        n_fc = len(fc)

        hist = weather_hist[["precip", "et0"]].dropna(how="all")
        years = sorted({d.year for d in hist.index if d.year < dates[0].year})
        mult = np.ones(horizon_days) * manual_pct / 100.0
        if outlook_ratio:
            mult *= np.array([outlook_ratio.get(d.month, 1.0) for d in dates])

        # climatological daily mean by day-of-year (for the "normal rain" path)
        clim = hist.groupby(hist.index.dayofyear).mean()
        climP = np.array(clim["precip"].reindex(dates.dayofyear).fillna(hist["precip"].mean()).values, dtype=float)
        climE = np.array(clim["et0"].reindex(dates.dayofyear).fillna(hist["et0"].mean()).values, dtype=float)

        members = {}
        for y in years:
            shift = dates[0].year - y
            try:
                analog_dates = dates - pd.DateOffset(years=shift)
            except Exception:
                continue
            a = hist.reindex(analog_dates)
            if a["precip"].isna().mean() > 0.2:
                continue
            P = np.array(a["precip"].fillna(0.0).values, dtype=float) * mult
            E = np.array(a["et0"].fillna(np.nanmean(climE)).values, dtype=float)
            if n_fc:
                P[:n_fc] = fc["precip"].fillna(0.0).values
                E[:n_fc] = fc["et0"].fillna(np.nanmean(climE)).values
            for k in range(n_noise):
                members[f"{y}-{k}"] = self.simulate(h0, start, P, E, pastP, pastE,
                                                    noise=self._noise(rng, dates), bounds=bounds).values

        Pn, En = climP * mult, climE.copy()
        if n_fc:
            Pn[:n_fc] = fc["precip"].fillna(0.0).values
            En[:n_fc] = fc["et0"].fillna(np.nanmean(climE)).values
        central = self.simulate(h0, start, Pn, En, pastP, pastE, bounds=bounds)

        mem = pd.DataFrame(members, index=dates)
        if mem.empty:
            pct = pd.DataFrame({"p50": central})
        else:
            q = mem.quantile([0.1, 0.25, 0.5, 0.75, 0.9], axis=1).T
            q.columns = ["p10", "p25", "p50", "p75", "p90"]
            pct = q
        pct["normal_rain"] = central
        return pct, mem, central, n_fc

    # ------------------------------------------------------------------ skill check
    def hindcast(self, level, weather, lead=30, every=10, years_back=3):
        """Start the model on past dates using the rain that actually fell, and compare
        the projected level `lead` days later to what really happened."""
        lvl = level.dropna().asfreq("D").interpolate(limit=3)
        wx = weather[["precip", "et0"]].asfreq("D")
        wx["precip"] = wx["precip"].fillna(0.0)
        wx["et0"] = wx["et0"].fillna(wx["et0"].mean())
        end = min(lvl.index[-1], wx.index[-1]) - pd.Timedelta(days=lead)
        start = end - pd.Timedelta(days=365 * years_back)
        rows = []
        for d in pd.date_range(start, end, freq=f"{every}D"):
            if d not in lvl.index or np.isnan(lvl.get(d, np.nan)):
                continue
            past = wx.loc[:d].tail(30)
            fut = wx.loc[d + pd.Timedelta(days=1): d + pd.Timedelta(days=lead)]
            if len(past) < 25 or len(fut) < lead:
                continue
            sim = self.simulate(float(lvl[d]), d, fut["precip"].values, fut["et0"].values,
                                past["precip"].values, past["et0"].values)
            obs = lvl.get(sim.index[-1], np.nan)
            if np.isnan(obs):
                continue
            rows.append({"start": d, "start_level": float(lvl[d]), "observed": obs,
                         "model": sim.iloc[-1], "persistence": float(lvl[d])})
        df = pd.DataFrame(rows)
        if df.empty:
            return df, {}
        stats = {
            "model_mae": float(np.mean(np.abs(df["model"] - df["observed"]))),
            "persistence_mae": float(np.mean(np.abs(df["persistence"] - df["observed"]))),
            "n": len(df), "lead": lead,
        }
        return df, stats


def outlook_ratio_by_month(seasonal: pd.Series, weather_hist: pd.DataFrame, lo=0.6, hi=1.5):
    """Seasonal-model rainfall vs. the basin's normal rainfall, per calendar month."""
    if seasonal is None or seasonal.empty or weather_hist.empty:
        return {}
    hist = weather_hist["precip"].dropna()
    clim_daily = hist.groupby(hist.index.month).mean()
    out = {}
    for m, s in seasonal.groupby(seasonal.index.month):
        if len(s) < 10 or m not in clim_daily or clim_daily[m] <= 0:
            continue
        out[int(m)] = float(np.clip(s.mean() / clim_daily[m], lo, hi))
    return out


def climatology_bands(level: pd.Series) -> pd.DataFrame:
    """Percentiles of the historical level for each day of the year."""
    s = level.dropna()
    g = s.groupby(s.index.dayofyear)
    out = pd.DataFrame({
        "min": g.min(), "p10": g.quantile(0.1), "p25": g.quantile(0.25), "median": g.median(),
        "p75": g.quantile(0.75), "p90": g.quantile(0.9), "max": g.max(),
    })
    return out.rolling(7, center=True, min_periods=1).mean()
