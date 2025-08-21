# scripts/backend/collect/sdo.py
from __future__ import annotations
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Cloud guard: disable SunPy-based work when set, or when SunPy stack is unavailable
NO_SUNPY = os.getenv("NO_SUNPY", "0") == "1"

try:
    from sunpy.net import Fido, attrs as a
    from astropy import units as u
    from scipy.ndimage import gaussian_filter, label, find_objects
    _HAS_SUNPY_BASE = True
except Exception:
    _HAS_SUNPY_BASE = False

_HAS_SUNPY = _HAS_SUNPY_BASE and not NO_SUNPY

SDO_BASE = "https://sdo.gsfc.nasa.gov/assets/img/dailymov"


@dataclass
class Region:
    x: float   # center x in [0..1]
    y: float   # center y in [0..1]
    r: float   # radius as fraction of frame (0..~0.2)
    score: float  # brightness score


def _as_date_utc(date_iso: str) -> datetime.date:
    return datetime.strptime(date_iso, "%Y-%m-%d").date()


@lru_cache(maxsize=256)
def _url_exists(url: str) -> bool:
    """Some SDO/CDNs reject HEAD; use a tiny streaming GET with redirects allowed."""
    try:
        r = requests.get(url, timeout=8, stream=True, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False


def _movie_url_for(d) -> str:
    # dailymov uses YEAR/MONTH/DAY + "<YYYYMMDD>_1024_0171"
    stamp = f"{d:%Y%m%d}_1024_0171"
    rel = f"{d:%Y/%m/%d}/{stamp}"
    mp4 = f"{SDO_BASE}/{rel}.mp4"
    ogv = f"{SDO_BASE}/{rel}.ogv"
    if _url_exists(mp4):
        return mp4
    if _url_exists(ogv):
        return ogv
    return ""


def _closest_available_movie(date_iso: str) -> Tuple[str, str]:
    """Return (date_used_iso, movie_url). If requested day missing, try nearby days."""
    d_req = _as_date_utc(date_iso)
    today_utc = datetime.now(timezone.utc).date()
    # If user selects *today*, the daily movie usually isn’t ready → prefer yesterday
    candidates = [d_req] if d_req < today_utc else [today_utc - timedelta(days=1)]
    # Try a couple more days back in case the archive is late
    for d in candidates + [d_req - timedelta(days=1), d_req - timedelta(days=2)]:
        url = _movie_url_for(d)
        if url:
            return (d.isoformat(), url)
    return (d_req.isoformat(), "")


def _class_of(flux: Optional[float]) -> str:
    if flux is None:
        return "—"
    return (
        "X" if flux >= 1e-4 else
        "M" if flux >= 1e-5 else
        "C" if flux >= 1e-6 else
        "B" if flux >= 1e-7 else
        "A"
    )


def _peak_from_df(df: Optional[pd.DataFrame], col: str) -> Optional[dict]:
    """Return {class, utc, value} for the max of 'col' (minute resolution), else None."""
    if df is None or df.empty or col not in df.columns:
        return None
    # Ensure datetime indexable series
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    vals = pd.to_numeric(df[col], errors="coerce")
    ok = ts.notna() & vals.notna()
    if not ok.any():
        return None
    idxmax = vals[ok].idxmax()
    peak_val = float(vals.iloc[idxmax])
    peak_ts = ts.iloc[idxmax]
    return {
        "class": f"{_class_of(peak_val)}-Class",
        "utc": peak_ts.strftime("%H:%M"),
        "value": peak_val,
    }


def _summarize_flux(minute_pred: Optional[pd.DataFrame],
                    minute_act: Optional[pd.DataFrame]) -> str:
    p = _peak_from_df(minute_pred, "long_flux_pred")
    a = _peak_from_df(minute_act, "long_flux")
    parts = []
    if p:
        parts.append(f"Predicted peak {p['class'][0]} at ~{p['utc']} UTC.")
    if a:
        parts.append(f"Observed peak {a['class'][0]} at ~{a['utc']} UTC.")
    # Simple trend (pred only), using a 120-min smoothing window
    try:
        if minute_pred is not None and not minute_pred.empty:
            s = pd.to_numeric(minute_pred["long_flux_pred"], errors="coerce").rolling(120, min_periods=30).mean()
            if s.notna().sum() >= 2:
                trend = (
                    "rising" if s.iloc[-1] > s.iloc[0] * 1.05 else
                    "falling" if s.iloc[-1] < s.iloc[0] * 0.95 else
                    "steady"
                )
                parts.append(f"Predicted trend is {trend} across the day.")
    except Exception:
        pass
    return " ".join(parts) if parts else "No summary available."


def _bright_regions(date_iso: str, max_regions: int = 6) -> List[Region]:
    """Heuristic bright region hints using AIA-171 around noon UTC.
       Returns [] if SunPy is disabled/unavailable (cloud mode)."""
    if not _HAS_SUNPY:
        return []

    try:
        t0, t1 = f"{date_iso} 11:55", f"{date_iso} 12:05"
        res = Fido.search(a.Time(t0, t1), a.Instrument('AIA'), a.Wavelength(171*u.angstrom))
        if len(res) == 0:
            return []
        files = Fido.fetch(res[0, :1])
        import sunpy.map
        amap = sunpy.map.Map(files[0])
        data = np.array(amap.data, dtype=float)

        # Smooth + threshold top 0.5% brightest pixels
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
            # normalize to [0..1] frame; limit r for nicer UI circles
            r = max(rx, ry) / max(h, w)
            peak = float(sm[y0:y1, x0:x1].max())
            regs.append(
                Region(
                    x=float(cx) / w,
                    y=float(cy) / h,
                    r=max(0.01, min(0.12, r * 1.2)),
                    score=peak,
                )
            )

        regs.sort(key=lambda r: r.score, reverse=True)
        return regs[:max_regions]
    except Exception:
        return []


def build_sdo_payload(date_iso: str,
                      minute_pred: Optional[pd.DataFrame] = None,
                      minute_act: Optional[pd.DataFrame] = None) -> dict:
    """
    Assemble a compact payload for the UI:
      - date_used: ISO date of the movie actually used
      - movie_url: AIA-171 daily MP4/OGV (or empty string if not found)
      - summary:   short natural-language summary
      - regions:   [{x,y,r,score}, …] bright hints (empty in cloud mode)
      - pred_peak: {class, utc, value} or None
      - obs_peak:  {class, utc, value} or None
    """
    used, url = _closest_available_movie(date_iso)

    pred_peak = _peak_from_df(minute_pred, "long_flux_pred")
    obs_peak  = _peak_from_df(minute_act,  "long_flux") if minute_act is not None else None
    summary   = _summarize_flux(minute_pred, minute_act) if minute_pred is not None else ""

    # Bright regions (skipped in cloud mode)
    regions = _bright_regions(used) if _HAS_SUNPY else []

    return {
        "date_used": used,
        "movie_url": url,
        "summary": summary,
        "regions": [dict(x=r.x, y=r.y, r=r.r, score=r.score) for r in regions],
        "pred_peak": pred_peak,
        "obs_peak": obs_peak,
    }
