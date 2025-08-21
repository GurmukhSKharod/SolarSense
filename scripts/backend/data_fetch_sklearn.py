# scripts/backend/data_fetch.py
import requests, pandas as pd, datetime as dt

# this file is a stripped down version of the original fetch.py
# be sure to look at that for more details on how to fetch data
# this is specifically made for the FastAPI for the simpler frontend.

URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
THRESH = [(1e-4,"X"), (1e-5,"M"), (1e-6,"C"), (1e-7,"B"), (0,"A")]

def flux_to_class(x):
    for thr, lab in THRESH:
        if x >= thr:       
            return lab
    return "A"

def fetch_day_minute(date_iso: str) -> pd.DataFrame:
    """Return minute-level GOES frame for *exact* UTC date."""
    r = requests.get(URL, timeout=20)
    r.raise_for_status()
    raw = pd.DataFrame(r.json())

    # reshape to long/short columns
    short = raw[raw.energy=="0.05-0.4nm"][["time_tag","flux"]]\
              .rename(columns={"flux":"short_flux"})
    long  = raw[raw.energy=="0.1-0.8nm"][["time_tag","flux"]]\
              .rename(columns={"flux":"long_flux"})
    df = pd.merge(long, short, on="time_tag")
    df["timestamp"] = pd.to_datetime(df.time_tag, utc=True)
    start = pd.Timestamp(f"{date_iso} 00:00:00+00:00")
    end   = start + dt.timedelta(days=1) - dt.timedelta(seconds=1)
    df = df[(df.timestamp>=start)&(df.timestamp<=end)]

    df["goes_class"] = df.long_flux.map(flux_to_class)
    return df[["timestamp","long_flux","short_flux","goes_class"]]
