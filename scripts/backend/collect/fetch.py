# scripts/backend/collect/fetch.py
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries

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

# ----------------- helpers (defined BEFORE main functions) -----------------

def _fetch_sunpy(start_dt, end_dt):
    res = Fido.search(a.Time(start_dt, end_dt), a.Instrument("XRS"))
    if len(res) == 0:
        return []
    return Fido.fetch(res)

def _parse_sunpy(files):
    if not files:
        return pd.DataFrame(columns=["timestamp", "long_flux", "short_flux"])
    dfs = []
    for fp in files:
        try:
            ts = TimeSeries(fp)
            df = (
                ts.to_dataframe()
                  .reset_index()
                  .rename(columns={"index": "timestamp", "xrsb": "long_flux", "xrsa": "short_flux"})
            )
            dfs.append(df[["timestamp", "long_flux", "short_flux"]])
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame(columns=["timestamp", "long_flux", "short_flux"])
    out = pd.concat(dfs, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.floor("min")
    return out

def _fetch_recent_json():
    """NOAA SWPC last-7-days JSON → minute dataframe."""
    r = requests.get(SWPC_7DAY_URL, timeout=20)
    r.raise_for_status()
    raw = r.json()
    if not raw:
        return pd.DataFrame(columns=["timestamp", "long_flux", "short_flux"])
    df = pd.DataFrame(raw)

    short_df = df[df["energy"] == "0.05-0.4nm"][["time_tag", "flux"]].rename(columns={"flux": "short_flux"})
    long_df  = df[df["energy"] == "0.1-0.8nm"][["time_tag", "flux"]].rename(columns={"flux": "long_flux"})

    merged = long_df.merge(short_df, on="time_tag", how="inner")
    merged["timestamp"] = pd.to_datetime(merged["time_tag"], utc=True).dt.floor("min")
    return merged[["timestamp", "long_flux", "short_flux"]]

# ----------------- public functions -----------------

def fetch_range_minute(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Minute-cadence long/short flux in [start_dt, end_dt] **UTC** (inclusive),
    returned on a continuous 1-minute UTC grid.
    """
    # Ensure tz-aware UTC
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    if end_dt > now:
        end_dt = now
    recent_cutoff = now - timedelta(days=7)

    parts = []

    # SunPy for historical portion
    if start_dt < recent_cutoff:
        sunpy_end = min(end_dt, recent_cutoff - timedelta(seconds=1))
        if sunpy_end >= start_dt:
            parts.append(_parse_sunpy(_fetch_sunpy(start_dt, sunpy_end)))

    # NOAA JSON for recent portion
    if end_dt >= recent_cutoff:
        json_start = max(start_dt, recent_cutoff)
        rec = _fetch_recent_json()
        if not rec.empty:
            rec = rec[(rec["timestamp"] >= json_start) & (rec["timestamp"] <= end_dt)]
            parts.append(rec)

    if not parts:
        return pd.DataFrame(columns=["timestamp", "long_flux", "short_flux", "goes_class"])

    df = (
        pd.concat(parts, ignore_index=True)
          .groupby("timestamp", as_index=False).mean(numeric_only=True)
          .sort_values("timestamp")
    )

    # Restrict to window
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "long_flux", "short_flux", "goes_class"])

    # Build continuous 1-minute grid and fill
    grid = pd.date_range(start=start_dt, end=end_dt, freq="1min", tz=timezone.utc)
    df = (
        df.set_index("timestamp")
          .reindex(grid)
          .interpolate(method="time")
          .ffill()
          .bfill()
          .rename_axis("timestamp")
          .reset_index()
    )

    df["goes_class"] = df["long_flux"].apply(flux_to_class)
    return df[["timestamp", "long_flux", "short_flux", "goes_class"]]

def fetch_day_minute(date_iso: str) -> pd.DataFrame:
    """Exactly one UTC day (00:00–23:59)."""
    day_start = datetime.strptime(date_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    day_end   = day_start + timedelta(days=1) - timedelta(minutes=1)
    return fetch_range_minute(day_start, day_end)
