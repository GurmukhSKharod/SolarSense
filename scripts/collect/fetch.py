import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries



# This script fetches GOES XRS data and saves to a CSV file.
# The data is for a specific date range, defined by START_TIME and END_TIME.

# You can modify START_TIME and END_TIME to fetch data for different dates.
# You can also change the SAVE_PATH to save the output CSV file to a different location 
# with a different name depending on the date/timeframe being fetched.

# This fetch file is dynamic and can fetch from multiple sources.
# It fetches GOES XRS data from SunPy for a specific date range and also fetches recent 7-day data from NOAA SWPC.
# If user requests a date range that includes the last 7 days, it will fetch from NOAA SWPC - live.
# If the date range is older than 7 days, it will fetch from SunPy.
# If includes 7 days and older than 7 days, it will fetch both sources and combine them.

# The above steps are done becuase SunPy does not have data for the last 7 days, it only has historical data.
# and NOAA SWPC has the last 7 days of data, but not historical data.

# Save path - change to your desired output file - training data, seed data, or actual data.
SAVE_PATH = "data/processed/prediction_seed.csv"

# Adjust these for any desired interval (UTC)
# If you want training data, set start time and end time to something large like greater than 7 days.
# If you want a seed (used later for prediction), just fetch a single day. 
# If you want actual data, set to the actual date you want to compare against. (1 day after seed date).
# Be sure to rename the above SAVE_PATH depending on training or seed or actual.
# For example, training data could be "data/processed/training_data.csv" 
# and seed data could be "data/processed/prediction_seed.csv".
# and actual data could be "data/processed/actual.csv".

START_TIME = datetime(2025, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
END_TIME   = datetime(2025, 8, 1, 23, 59, 0, tzinfo=timezone.utc)

SWPC_7DAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

CLASS_THRESH = [
    (1e-4, "X"),
    (1e-5, "M"),
    (1e-6, "C"),
    (1e-7, "B"),
    (0.0,  "A"),
]

def flux_to_class(f):
    for thr, label in CLASS_THRESH:
        if f >= thr:
            return label
    return "A"

def fetch_sunpy(start_dt, end_dt):
    res = Fido.search(a.Time(start_dt, end_dt), a.Instrument("XRS"))
    if len(res) == 0:
        return []
    return Fido.fetch(res)

def parse_sunpy(files):
    if not files:
        return pd.DataFrame(columns=["timestamp", "long_flux", "short_flux"])
    dfs = []
    for fp in files:
        try:
            ts = TimeSeries(fp)
            df = (ts.to_dataframe()
                    .reset_index()
                    .rename(columns={"index": "timestamp", "xrsb": "long_flux", "xrsa": "short_flux"}))
            dfs.append(df[["timestamp", "long_flux", "short_flux"]])
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame(columns=["timestamp", "long_flux", "short_flux"])
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("min")
    return df

def fetch_recent_json():
    r = requests.get(SWPC_7DAY_URL, timeout=15)
    r.raise_for_status()
    raw = r.json()
    if not raw:
        return pd.DataFrame(columns=["timestamp", "long_flux", "short_flux"])
    df = pd.DataFrame(raw)
    short_df = df[df.energy == "0.05-0.4nm"][["time_tag", "flux"]].rename(columns={"flux": "short_flux"})
    long_df  = df[df.energy == "0.1-0.8nm"][["time_tag", "flux"]].rename(columns={"flux": "long_flux"})
    merged = long_df.merge(short_df, on="time_tag")
    merged["timestamp"] = pd.to_datetime(merged["time_tag"], utc=True).dt.floor("min")
    return merged[["timestamp", "long_flux", "short_flux"]]

def combine_and_save(parts, start_dt, end_dt, out_path):
    if not parts:
        print("No data gathered.")
        return
    df = pd.concat(parts, ignore_index=True)
    df = (df.groupby("timestamp", as_index=False)
            .mean(numeric_only=True)
            .sort_values("timestamp"))
    df = df[(df.timestamp >= start_dt) & (df.timestamp <= end_dt)]
    if df.empty:
        print("No rows in requested window.")
        return
    df["goes_class"] = df["long_flux"].apply(flux_to_class)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} rows)")

def main():
    start = START_TIME
    end = END_TIME
    if end < start:
        raise ValueError("END_TIME must be >= START_TIME")
    now = datetime.now(timezone.utc)
    if end > now:
        end = now
    recent_cutoff = now - timedelta(days=7)

    parts = []

    # Historical part (SunPy) if any portion is older than 7 days ago
    if start < recent_cutoff:
        sunpy_end = min(end, recent_cutoff - timedelta(seconds=1))
        if sunpy_end >= start:
            parts.append(parse_sunpy(fetch_sunpy(start, sunpy_end)))

    # Recent 7-day JSON part if any portion touches recent window
    if end >= recent_cutoff:
        json_start = max(start, recent_cutoff)
        recent_df = fetch_recent_json()
        recent_df = recent_df[(recent_df.timestamp >= json_start) & (recent_df.timestamp <= end)]
        parts.append(recent_df)

    combine_and_save(parts, start, end, SAVE_PATH)

if __name__ == "__main__":
    main()
