"""
Lake / reservoir catalog.

Each entry tells the app where to find data for one lake:
  usgs_site      USGS monitoring-location number for the lake-level gauge at the dam
  cwms_office    Army Corps district office code used by the CWMS Data API (SAW = Wilmington)
  cwms_pattern   regex used to discover the lake's CWMS time series (elevation, inflow, outflow, storage)
  pools          key elevations in feet (same datum as the USGS gauge, normally NGVD29)
  guide_curve    (month, day, elevation) breakpoints of the operating target ("rule curve").
                 Values between breakpoints are interpolated. Approximate - check the Corps'
                 water control manual for the official curve.
  basin_points   (lat, lon) points spread over the upstream watershed; rainfall is averaged over these
  upstream_usgs  river gauges that feed the lake (shown as "inflow indicators")
  drought_fips   county FIPS codes around the lake for the US Drought Monitor
  news_terms     search terms for the news feed

Add your own lakes by copying an entry. Anything left as None is simply skipped.
"""

LAKES = {
    "John H. Kerr Lake (Buggs Island), VA/NC": {
        "short": "Kerr Lake",
        "usgs_site": "02079490",
        "lat": 36.5989,
        "lon": -78.3017,
        "cwms_office": "SAW",
        "cwms_pattern": r"(?i)^(J\.?\s?H\.?\s?)?Kerr|^JHK",
        "datum": "NGVD29",
        "pools": {
            "Top of flood pool": 320.0,
            "Normal (full) pool": 300.0,
            "Min. power pool": 293.0,
            "Bottom of conservation pool": 268.0,
        },
        "normal_pool": 300.0,
        "flood_top": 320.0,
        "min_pool": 268.0,
        # Winter 295.5, spring rise to 302 for striped-bass spawning storage,
        # summer pool 299.5, fall drawdown back to 295.5 (2016 Water Control Plan).
        # Breakpoint dates are approximate.
        "guide_curve": [
            (1, 1, 295.5), (3, 1, 295.5), (4, 30, 302.0), (5, 31, 302.0),
            (6, 30, 299.5), (10, 15, 299.5), (12, 1, 295.5), (12, 31, 295.5),
        ],
        "basin_points": [
            (37.27, -79.94),  # Roanoke, VA (upper Roanoke River)
            (37.05, -79.55),  # Smith Mountain Lake area
            (36.69, -79.87),  # Martinsville (Smith River)
            (36.59, -79.40),  # Danville (Dan River)
            (36.70, -78.90),  # South Boston (Dan / Banister)
            (36.39, -78.98),  # Roxboro (Hyco River)
            (36.62, -78.56),  # Clarksville (lake)
            (36.95, -78.45),  # Upper Roanoke (Staunton) tributaries
        ],
        "upstream_usgs": {
            "02066000": "Roanoke (Staunton) River at Randolph, VA",
            "02075500": "Dan River at Paces, VA",
        },
        "drought_fips": {"51117": "Mecklenburg Co., VA", "51083": "Halifax Co., VA",
                         "37181": "Vance Co., NC", "37077": "Granville Co., NC"},
        "news_terms": ['"Kerr Lake"', '"Buggs Island Lake"', '"Kerr Reservoir"', '"Kerr Dam"'],
        "links": {
            "USACE Kerr project page": "https://www.saw.usace.army.mil/Locations/District-Lakes-and-Dams/John-H-Kerr/",
            "USACE Access to Water": "https://water.usace.army.mil/",
            "USGS gauge 02079490": "https://waterdata.usgs.gov/monitoring-location/USGS-02079490/",
            "Kerr Water Control Plan (2016)": "https://deq.nc.gov/water-resources/planning-section/water-supply-planning/kerr-lake-water-control-plan-2016/open",
        },
        "facts": [
            "Built by the Army Corps of Engineers on the Roanoke River; storage began in 1950.",
            "Conservation pool 268–300 ft; controlled flood-storage pool 300–320 ft (NGVD29).",
            "Guide curve: ~295.5 ft in winter, raised to ~302 ft in spring for striped-bass spawning releases, ~299.5 ft in summer.",
            "Below 300 ft, releases follow hydropower operations; between 300 and 320 ft the Corps releases inflow up to ~35,000 cfs at Roanoke Rapids.",
            "Roughly 50,000 surface acres and about 800 miles of shoreline at full pool.",
            "Releases flow downstream into Lake Gaston and Roanoke Rapids Lake (Dominion Energy).",
        ],
    },
    "Falls Lake, NC": {
        "short": "Falls Lake",
        "usgs_site": "02087182",
        "lat": 35.9408, "lon": -78.5825,
        "cwms_office": "SAW",
        "cwms_pattern": r"(?i)^Falls",
        "datum": "NGVD29",
        "pools": {"Top of flood pool": 264.8, "Normal pool": 251.5},
        "normal_pool": 251.5, "flood_top": 264.8, "min_pool": None,
        "guide_curve": [(1, 1, 251.5), (12, 31, 251.5)],
        "basin_points": [(36.07, -78.90), (36.15, -78.75), (36.25, -79.00), (36.03, -78.70), (35.98, -78.60)],
        "upstream_usgs": {"02085500": "Flat River at Bahama, NC"},
        "drought_fips": {"37183": "Wake Co., NC", "37063": "Durham Co., NC", "37077": "Granville Co., NC"},
        "news_terms": ['"Falls Lake"'],
        "links": {"USGS gauge 02087182": "https://waterdata.usgs.gov/monitoring-location/USGS-02087182/"},
        "facts": ["Corps of Engineers reservoir on the Neuse River; Raleigh's main water supply."],
    },
    "B. Everett Jordan Lake, NC": {
        "short": "Jordan Lake",
        "usgs_site": "02098197",
        "lat": 35.6547, "lon": -79.0683,
        "cwms_office": "SAW",
        "cwms_pattern": r"(?i)^(B\.?\s?E\.?\s?)?Jordan",
        "datum": "NGVD29",
        "pools": {"Top of flood pool": 240.0, "Normal pool": 216.0},
        "normal_pool": 216.0, "flood_top": 240.0, "min_pool": None,
        "guide_curve": [(1, 1, 216.0), (12, 31, 216.0)],
        "basin_points": [(36.07, -79.79), (36.10, -79.45), (35.95, -79.05), (35.80, -79.20), (36.20, -79.60)],
        "upstream_usgs": {"02096960": "Haw River near Bynum, NC"},
        "drought_fips": {"37037": "Chatham Co., NC", "37135": "Orange Co., NC", "37001": "Alamance Co., NC"},
        "news_terms": ['"Jordan Lake"'],
        "links": {"USGS gauge 02098197": "https://waterdata.usgs.gov/monitoring-location/USGS-02098197/"},
        "facts": ["Corps of Engineers reservoir on the Haw and New Hope rivers (Cape Fear basin)."],
    },
    "Philpott Lake, VA": {
        "short": "Philpott Lake",
        "usgs_site": "02071900",
        "lat": 36.7812, "lon": -80.0278,
        "cwms_office": "SAW",
        "cwms_pattern": r"(?i)^Philpott",
        "datum": "NGVD29",
        "pools": {"Spillway crest": 985.0, "Normal pool": 974.0},
        "normal_pool": 974.0, "flood_top": 985.0, "min_pool": 951.0,
        "guide_curve": [(1, 1, 974.0), (12, 31, 974.0)],
        "basin_points": [(36.85, -80.30), (36.80, -80.15), (36.70, -80.25)],
        "upstream_usgs": {},
        "drought_fips": {"51089": "Henry Co., VA", "51141": "Patrick Co., VA", "51067": "Franklin Co., VA"},
        "news_terms": ['"Philpott Lake"'],
        "links": {"USGS gauge 02071900": "https://waterdata.usgs.gov/monitoring-location/USGS-02071900/"},
        "facts": ["Corps of Engineers reservoir on the Smith River, upstream of Kerr Lake in the Roanoke basin."],
    },
    "Lake Sidney Lanier, GA": {
        "short": "Lake Lanier",
        "usgs_site": "02334400",
        "lat": 34.16, "lon": -84.07,
        "cwms_office": "SAM",
        "cwms_pattern": r"(?i)^Buford",
        "datum": "NGVD29",
        "pools": {"Top of flood pool": 1085.0, "Full summer pool": 1071.0},
        "normal_pool": 1071.0, "flood_top": 1085.0, "min_pool": 1035.0,
        "guide_curve": [(1, 1, 1070.0), (5, 31, 1071.0), (9, 30, 1071.0), (12, 31, 1070.0)],
        "basin_points": [(34.55, -83.85), (34.45, -83.95), (34.30, -84.00), (34.65, -83.75)],
        "upstream_usgs": {"02331600": "Chattahoochee River near Cornelia, GA"},
        "drought_fips": {"13139": "Hall Co., GA", "13117": "Forsyth Co., GA"},
        "news_terms": ['"Lake Lanier"'],
        "links": {"USGS gauge 02334400": "https://waterdata.usgs.gov/monitoring-location/USGS-02334400/"},
        "facts": ["Corps of Engineers (Mobile District) reservoir behind Buford Dam on the Chattahoochee River."],
    },
    "Hartwell Lake, SC/GA": {
        "short": "Hartwell Lake",
        "usgs_site": "02187010",
        "lat": 34.475, "lon": -82.816,
        "cwms_office": "SAS",
        "cwms_pattern": r"(?i)^Hartwell",
        "datum": "NGVD29",
        "pools": {"Top of flood pool": 665.0, "Full pool": 660.0},
        "normal_pool": 660.0, "flood_top": 665.0, "min_pool": 625.0,
        "guide_curve": [(1, 1, 656.0), (4, 1, 660.0), (10, 15, 660.0), (12, 15, 656.0), (12, 31, 656.0)],
        "basin_points": [(34.85, -83.05), (34.70, -82.95), (34.60, -83.20), (34.90, -82.80)],
        "upstream_usgs": {},
        "drought_fips": {"45007": "Anderson Co., SC", "13147": "Hart Co., GA"},
        "news_terms": ['"Hartwell Lake"', '"Lake Hartwell"'],
        "links": {"USGS gauge 02187010": "https://waterdata.usgs.gov/monitoring-location/USGS-02187010/"},
        "facts": ["Corps of Engineers (Savannah District) reservoir on the Savannah River."],
    },
    "J. Strom Thurmond Lake (Clarks Hill), SC/GA": {
        "short": "Thurmond Lake",
        "usgs_site": "02193900",
        "lat": 33.840, "lon": -82.350,
        "cwms_office": "SAS",
        "cwms_pattern": r"(?i)^Thurmond|^JST",
        "datum": "NGVD29",
        "pools": {"Top of flood pool": 335.0, "Full pool": 330.0},
        "normal_pool": 330.0, "flood_top": 335.0, "min_pool": 312.0,
        "guide_curve": [(1, 1, 326.0), (4, 1, 330.0), (10, 15, 330.0), (12, 15, 326.0), (12, 31, 326.0)],
        "basin_points": [(34.10, -82.60), (33.95, -82.45), (34.20, -82.80), (33.80, -82.50)],
        "upstream_usgs": {},
        "drought_fips": {"45065": "McCormick Co., SC", "13073": "Columbia Co., GA", "13181": "Lincoln Co., GA"},
        "news_terms": ['"Thurmond Lake"', '"Clarks Hill Lake"'],
        "links": {"USGS gauge 02193900": "https://waterdata.usgs.gov/monitoring-location/USGS-02193900/"},
        "facts": ["Corps of Engineers (Savannah District) reservoir on the Savannah River."],
    },
}

DEFAULT_LAKE = "John H. Kerr Lake (Buggs Island), VA/NC"


def custom_lake(name, usgs_site, lat, lon, normal_pool, flood_top):
    """Build a minimal catalog entry for a lake the user types in."""
    return {
        "short": name, "usgs_site": usgs_site, "lat": lat, "lon": lon,
        "cwms_office": None, "cwms_pattern": None, "datum": "gauge datum",
        "pools": {"Top of flood pool": flood_top, "Normal pool": normal_pool},
        "normal_pool": normal_pool, "flood_top": flood_top, "min_pool": None,
        "guide_curve": [(1, 1, normal_pool), (12, 31, normal_pool)],
        "basin_points": [(lat, lon), (lat + 0.2, lon - 0.2), (lat + 0.2, lon + 0.2),
                         (lat - 0.15, lon - 0.25)],
        "upstream_usgs": {}, "drought_fips": {}, "news_terms": [f'"{name}"'],
        "links": {f"USGS gauge {usgs_site}": f"https://waterdata.usgs.gov/monitoring-location/USGS-{usgs_site}/"},
        "facts": [],
    }
