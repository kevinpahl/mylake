# Lake Level Tracker (Streamlit)

Tracks lake/reservoir levels, the weather that drives them, and projects the level
weeks to months ahead. Defaults to **John H. Kerr Lake (Buggs Island)**; pick other lakes
from the drop-down, or choose "Other lake" and type any USGS lake-gauge number.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional but recommended: get a free USGS API key at https://api.waterdata.usgs.gov/signup/
and put it in `.streamlit/secrets.toml` (see `secrets.toml.example`) or set the
environment variable `USGS_API_KEY`. Without a key, USGS limits how many requests you can make
per hour. The app caches everything, so normal use is fine either way.

## What's on each tab

| Tab | What it shows |
|---|---|
| Level & projection | Observed level, historical normal range for each date, the Corps guide curve, pool elevations, and the projected level with a likely range. The weekly table shows the chance of being above full pool, or more than 2 ft below the guide curve. |
| Weather | Rainfall averaged over the whole watershed for the past 14 and next 16 days vs. normal, highs and lows, how 4 weather models compare, and the NWS forecast text. |
| Long-range outlook | ECMWF seasonal rainfall by month vs. normal, NOAA CPC 6–10 day, 8–14 day, monthly and seasonal outlook maps, and U.S. Drought Monitor status of nearby counties. |
| Inflows & Corps data | Upstream river flows (USGS), Army Corps CWMS elevation, inflow, release and storage, and NWS river forecast points nearby. |
| News | Headlines about the lake, tagged by possible impact (drought, flooding, Corps operations, water quality, recreation access). |
| Lake info | Facts, key elevations, record high and low in the data, links, and a map of the watershed points. |
| Model & sources | How the projection works, how well it fits, a 30-day hindcast compared with "no change", and any data source that failed. |

Sidebar: projection length (2–26 weeks), seasonal-outlook adjustment on/off, a rainfall
scenario slider (e.g. 60% of normal = a dry spell), and years of history to learn from.

## How the projection works

`model.py` learns the lake's day-to-day level change from its own history:
basin rainfall in four windows (0–1, 2–4, 5–10 and 11–24 days back), 7-day evaporation, the season, and
how far the lake is above or below the guide curve. That last term captures Corps operations: water
gets released when the lake is high and held when it's low.

* Days 1–16 use the forecast rainfall (Open-Meteo, basin-averaged).
* After that, one scenario per past year uses the rain that fell over the same dates, scaled by
  the ECMWF seasonal outlook and the scenario slider. Random noise from the model's own errors is added.
* The shaded band is the 10th–90th percentile of the scenarios.

It's a statistical estimate, not the Corps' official forecast. Planned drawdowns, hydropower
scheduling or emergency releases can override it. The Model tab's hindcast shows how far off it
has been in the past.

## Adding a lake

Copy an entry in `lakes.py`. The essentials are `usgs_site`, `lat`/`lon`, `normal_pool`,
`flood_top`, `guide_curve` and a few `basin_points` spread over the upstream watershed. For Corps
data, set `cwms_office` (district code such as SAW, SAM or SAS) and `cwms_pattern` (a regex matching the
CWMS location name). The app finds the matching time series automatically.

## Notes

* The guide curves in `lakes.py` are approximate seasonal targets. Kerr's comes from the 2016 Water
  Control Plan (winter ~295.5 ft, spring ~302 ft, summer ~299.5 ft); the breakpoint dates are estimates.
* USGS is retiring its old NWIS web services. The app uses the new Water Data API first and
  falls back to the old one.
* If the Corps' public CWMS API doesn't list a lake's series, the app still works from the USGS gauge.
