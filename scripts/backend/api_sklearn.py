# scripts/backend/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import pandas as pd
from scripts.backend.paths_sklearn import DATA_DIR  

from scripts.backend.data_fetch_sklearn import fetch_day_minute
from predict     import predict_day, WINDOW

# The backend folder and all files within it are used to serve the API.
# This API provides endpoints to fetch solar flare data and predictions.
# It is only used for the frontend in react-tailwind-vanilla, and acts as a 


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

def avg_hourly(df: pd.DataFrame, flux_col: str, class_col: str):
    hourly = df.set_index("timestamp")[flux_col]\
               .resample("1H").mean().reset_index()
    hourly["hour"] = hourly.timestamp.dt.hour
    hourly["class"] = hourly[flux_col].apply(lambda x: "X" if x>=1e-4
                                             else "M" if x>=1e-5
                                             else "C" if x>=1e-6
                                             else "B" if x>=1e-7
                                             else "A")
    return hourly[["hour", flux_col, "class"]]

@app.get("/forecast/{date_iso}")
def forecast(date_iso: str):
    """Return hourly actual & sklearn-predicted flux for given UTC date."""
    try:
        datetime.strptime(date_iso, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, detail="Date must be YYYY-MM-DD")

    # 1) fetch minute data for *previous* day for seeding
    prev_day = (datetime.strptime(date_iso,"%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    seed_df  = fetch_day_minute(prev_day)
    if len(seed_df) < WINDOW:
        raise HTTPException(500, detail="Not enough seed data")

    # 2) fetch actual minutes for the requested day
    actual_df = fetch_day_minute(date_iso)

    # 3) run prediction
    pred_df   = predict_day(date_iso, seed_df)

    # 4) average both to hourly
    act_hour  = avg_hourly(actual_df, "long_flux", "goes_class")
    pred_hour = avg_hourly(pred_df, "long_flux_pred", "goes_class_pred")

    return {
        "date": date_iso,
        "hourly_actual": act_hour.to_dict(orient="records"),
        "hourly_pred":   pred_hour.to_dict(orient="records")
    }
