from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

try:
    from sunpy.net import Fido, attrs as a
    from astropy import units as u
    from scipy.ndimage import gaussian_filter, label, find_objects
    _HAS_SUNPY = True
except Exception:
    _HAS_SUNPY = False

SDO_BASE = "https://sdo.gsfc.nasa.gov/assets/img/dailymov"

@dataclass
class Region:
    x: float
    y: float
    r: float
    score: float

def _as_date_utc(date_iso: str) -> datetime.date:
    return datetime.strptime(date_iso, "%Y-%m-%d").date()

@lru_cache(maxsize=256)
def _url_exists(url: str) -> bool:
    """Some SDO/CDNs reject HEAD; use a tiny streaming GET."""
    try:
        r = requests.get(url, timeout=8, stream=True, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False

def _movie_url_for(d) -> str:
    stamp = f"{d:%Y%m%d}_1024_0171"
    # include YEAR / MONTH / DAY folders
    rel = f"{d:%Y/%m/%d}/{stamp}"
    mp4 = f"{SDO_BASE}/{rel}.mp4"
    ogv = f"{SDO_BASE}/{rel}.ogv"
    if _url_exists(mp4):
        return mp4
    if _url_exists(ogv):
        return ogv
    return ""

def _closest_available_movie(date_iso: str) -> (str, str):
    d_req = _as_date_utc(date_iso)
    today_utc = datetime.now(timezone.utc).date()
    # If user selects today, SDO daily movie usually isn’t published yet → try yesterday.
    candidates = [d_req] if d_req < today_utc else [today_utc - timedelta(days=1)]
    # Try a couple more days back just in case the archive is late.
    for d in candidates + [d_req - timedelta(days=1), d_req - timedelta(days=2)]:
        url = _movie_url_for(d)
        if url:
            return (d.isoformat(), url)
    return (d_req.isoformat(), "")

def _bright_regions(date_iso: str, max_regions: int = 6) -> List[Region]:
    if not _HAS_SUNPY:
        return []
    try:
        from sunpy.net import Fido, attrs as a
        from astropy import units as u
        t0, t1 = f"{date_iso} 11:55", f"{date_iso} 12:05"
        res = Fido.search(a.Time(t0, t1), a.Instrument('AIA'), a.Wavelength(171*u.angstrom))
        if len(res) == 0:
            return []
        files = Fido.fetch(res[0, :1])
        import sunpy.map
        amap = sunpy.map.Map(files[0])
        data = np.array(amap.data, dtype=float)
        sm = gaussian_filter(data, 2.0)
        thr = np.percentile(sm, 99.5)
        lab, n = label(sm > thr)
        if n == 0:
            return []
        boxes = find_objects(lab)
        h, w = sm.shape
        regs: List[Region] = []
        for sl in boxes[: max_regions * 2]:
            if sl is None:
                continue
            y0, y1 = sl[0].start, sl[0].stop
            x0, x1 = sl[1].start, sl[1].stop
            cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
            ry, rx = (y1 - y0) / 2, (x1 - x0) / 2
            r = float(max(rx, ry)) / max(h, w)
            peak = float(sm[y0:y1, x0:x1].max())
            regs.append(Region(x=float(cx)/w, y=float(cy)/h, r=max(0.01, min(0.12, r*1.2)), score=peak))
        regs.sort(key=lambda r: r.score, reverse=True)
        return regs[:max_regions]
    except Exception:
        return []

def _summarize_flux(minute_pred: pd.DataFrame, minute_act: Optional[pd.DataFrame]) -> str:
    def _top(df, col):
        if df is None or df.empty:
            return None, None
        idx = df[col].idxmax()
        return df.at[idx, col], df.at[idx, "timestamp"]

    pval, ptime = _top(minute_pred, "long_flux_pred")
    aval, atime = _top(minute_act, "long_flux") if minute_act is not None else (None, None)

    def cls(x):
        if x is None: return "—"
        return "X" if x>=1e-4 else "M" if x>=1e-5 else "C" if x>=1e-6 else "B" if x>=1e-7 else "A"

    parts = []
    if pval is not None:
        parts.append(f"Predicted peak {cls(pval)} at ~{pd.to_datetime(ptime).strftime('%H:%M')} UTC.")
    if aval is not None:
        parts.append(f"Observed peak {cls(aval)} at ~{pd.to_datetime(atime).strftime('%H:%M')} UTC.")
    try:
        s = minute_pred["long_flux_pred"].rolling(120, min_periods=30).mean()
        trend = "rising" if s.iloc[-1] > s.iloc[0]*1.05 else "falling" if s.iloc[-1] < s.iloc[0]*0.95 else "steady"
        parts.append(f"Predicted trend is {trend} across the day.")
    except Exception:
        pass
    return " ".join(parts) if parts else "No summary available."

def build_sdo_payload(date_iso: str,
                      minute_pred: Optional[pd.DataFrame] = None,
                      minute_act: Optional[pd.DataFrame] = None) -> dict:
    used, url = _closest_available_movie(date_iso)
    summary = _summarize_flux(minute_pred, minute_act) if minute_pred is not None else ""
    regions = _bright_regions(used)
    return {
        "date_used": used,
        "movie_url": url,
        "summary": summary,
        "regions": [dict(x=r.x, y=r.y, r=r.r, score=r.score) for r in regions],
    }
