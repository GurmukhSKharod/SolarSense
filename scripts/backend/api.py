# scripts/backend/api.py
import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta, timezone

import pandas as pd

from scripts.backend.collect.fetch import fetch_range_minute, fetch_day_minute
from scripts.backend.collect.sdo   import build_sdo_payload
from scripts.backend.model.predict_pytorch import predict_from_seed_df, WINDOW
from starlette.middleware.gzip import GZipMiddleware

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=500)

FRONTEND = os.getenv("FRONTEND_ORIGIN", "https://solarsense.netlify.app")
EXTRA    = os.getenv("EXTRA_ORIGINS", "")  # comma-separated if you want more

allowed = [FRONTEND.strip(), "http://localhost:8080", "http://localhost:5173",
           "http://localhost:3000", "http://127.0.0.1:8080"]
allowed += [o.strip() for o in EXTRA.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed,
    allow_credentials=False,
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)

def _utc_day_window(date_iso: str):
    """Return UTC start/end for a YYYY-MM-DD."""
    try:
        day_start = datetime.strptime(date_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(400, detail="Date must be YYYY-MM-DD")
    day_end = day_start + timedelta(days=1) - timedelta(minutes=1)
    return day_start, day_end

def _avg_hourly(df: pd.DataFrame, flux_col: str, class_col: str):
    out = (
        df.set_index("timestamp")[flux_col]
          .resample("1h")         # lower-case 'h' (pandas deprecation-proof)
          .mean()
          .reset_index()
    )
    out["hour"] = out.timestamp.dt.hour
    out["class"] = out[flux_col].apply(
        lambda x: "X" if x>=1e-4 else
                  "M" if x>=1e-5 else
                  "C" if x>=1e-6 else
                  "B" if x>=1e-7 else "A"
    )
    return out[["hour", flux_col, "class"]]

async def _bg_warm():
    try:
        # import here to avoid circulars
        from scripts.backend.model.predict_pytorch import _loaded_assets
        # run heavy model load in a worker thread so we don't block the loop
        await asyncio.to_thread(_loaded_assets)
        print("Warmup completed")
    except Exception as e:
        print("Warmup failed:", e)

@app.on_event("startup")
async def schedule_warm():
    # schedule and return immediately so Uvicorn can finish startup & bind
    asyncio.create_task(_bg_warm())


@app.get("/health")
def health():
    return {"ok": True}

@app.head("/health")
def health_head():
    # Render pings via HEAD; keep it green instead of 405.
    return Response(status_code=200)


@app.get("/forecast/{date_iso}")
def forecast(date_iso: str):
    # 00:00–23:59 UTC of the requested day
    day_start, day_end = _utc_day_window(date_iso)

    # Seed = previous UTC day, exactly WINDOW minutes (24h) if your model expects that
    seed_end   = day_start - timedelta(minutes=1)
    seed_start = seed_end - timedelta(minutes=WINDOW - 1)

    print("WINDOW (seed minutes) =", WINDOW)

    try:
        seed_df = fetch_range_minute(seed_start, seed_end)[["timestamp","long_flux","short_flux"]]
    except RuntimeError as e:
        # Clear message for the UI
        raise HTTPException(status_code=422, detail=str(e))
    
    if len(seed_df) < WINDOW:
        raise HTTPException(500, detail="Not enough seed data for prediction window")

    try:
        actual_df = fetch_range_minute(day_start, day_end)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    pred_df   = predict_from_seed_df(seed_df)                # minute cadence

    # Make absolutely sure we only return the requested UTC day
    pred_df  = pred_df[(pred_df["timestamp"] >= day_start) & (pred_df["timestamp"] <= day_end)]
    actual_df= actual_df[(actual_df["timestamp"] >= day_start) & (actual_df["timestamp"] <= day_end)]

    # Hourly means (for the strip)
    act_hour  = _avg_hourly(actual_df, "long_flux", "goes_class")
    pred_hour = _avg_hourly(pred_df,   "long_flux_pred", "goes_class_pred")

    # Convert timestamps to ISO UTC strings with 'Z' so the frontend never shifts them
    def _iso(df, col="timestamp"):
        df = df.copy()
        df[col] = pd.to_datetime(df[col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return df

    return {
        "date": date_iso,
        "hourly_actual": act_hour.to_dict(orient="records"),
        "hourly_pred":   pred_hour.to_dict(orient="records"),
        "minute_actual": _iso(actual_df)[["timestamp", "long_flux"]].to_dict(orient="records"),
        "minute_pred":   _iso(pred_df)[["timestamp", "long_flux_pred"]].to_dict(orient="records"),
    }


@app.get("/sdo/{date_iso}")
def sdo(date_iso: str):
    """
    Daily SDO AIA-171 movie + auto summary + optional bright-region hints.
    If movie for requested day isn't published yet, it falls back to yesterday.
    """
    # Reuse what we already compute so the summary can mention peaks/trend.
    day_start, day_end = _utc_day_window(date_iso)

    # Try to reuse actual & predicted minute series; if they fail, still return movie.
    try:
        seed_start = day_start - timedelta(minutes=WINDOW)
        seed_end   = day_start - timedelta(minutes=1)
        seed_df    = fetch_range_minute(seed_start, seed_end)
        pred_df    = predict_from_seed_df(seed_df)
        pred_df    = pred_df[(pred_df["timestamp"] >= day_start) & (pred_df["timestamp"] <= day_end)]
    except Exception:
        pred_df = pd.DataFrame(columns=["timestamp","long_flux_pred"])

    try:
        actual_df  = fetch_range_minute(day_start, day_end)
    except Exception:
        actual_df = pd.DataFrame(columns=["timestamp","long_flux"])

    payload = build_sdo_payload(
        date_iso,
        minute_pred=pred_df if not pred_df.empty else None,
        minute_act=actual_df if not actual_df.empty else None,
    )
    return payload
