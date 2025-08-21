# scripts/backend/predict.py
import joblib, numpy as np, pandas as pd, datetime as dt, os
from pathlib import Path
from scripts.backend.paths_sklearn import MODELS_DIR

# This script predicts the next 24 hours of solar flux using sklearn trained model.
# NOTE: This is a stripped down version of the original predict_24h_sklearn.py.
# The frontend only runs the sklearn pipeline, so this is used for the FastAPI.

MODEL_PATH = MODELS_DIR / "flux_hgbr_1step.pkl"
XS_PATH    = MODELS_DIR / "x_scaler_1step.pkl"
YS_PATH    = MODELS_DIR / "y_scaler_1step.pkl"

MODEL = joblib.load(MODEL_PATH)
XS    = joblib.load(XS_PATH)
YS    = joblib.load(YS_PATH)

WINDOW = 720     # 12 h history
THRESH = [(1e-4,"X"),(1e-5,"M"),(1e-6,"C"),(1e-7,"B"),(0,"A")]

def f2c(x):
    for t,l in THRESH:
        if x>=t: return l
    return "A"

def one_step(buf_log):
    X = XS.transform(buf_log.reshape(1,-1))
    y_scaled = MODEL.predict(X).reshape(-1,1)
    y_log = YS.inverse_transform(y_scaled).ravel()[0]
    return 10**y_log        # back to linear flux

def predict_day(date_iso: str, seed_df: pd.DataFrame) -> pd.DataFrame:
    """
    seed_df must contain at least WINDOW minutes **ending 23:59 of previous day**.
    """
    lin = seed_df[["long_flux","short_flux"]].to_numpy(dtype=float)[-WINDOW:]
    buf_log = np.log10(np.maximum(lin, 1e-12))

    preds = []
    start_ts = pd.Timestamp(f"{date_iso} 00:00:00+00:00")
    for _ in range(60*24):          # 1 440 minutes -> 24 h
        y_lin = one_step(buf_log)
        preds.append(y_lin)
        next_row = np.array([[y_lin, buf_log[-1,0]]])  # reuse last short flux
        buf_log = np.vstack([buf_log[1:], np.log10(np.maximum(next_row,1e-12))])

    minute_idx = pd.date_range(start=start_ts, periods=1440, freq="1min", tz="UTC")
    out = pd.DataFrame({
        "timestamp": minute_idx,
        "long_flux_pred": preds,
    })
    out["goes_class_pred"] = out.long_flux_pred.map(f2c)
    return out
