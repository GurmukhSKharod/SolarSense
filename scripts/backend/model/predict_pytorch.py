# scripts/backend/model/predict_pytorch.py
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import joblib
import os
import torch
import torch.nn as nn

ROOT       = Path(__file__).resolve().parents[3]           # .../SolarSense
MODEL_DIR  = ROOT / "models" / "regressors"
XSC_PATH    = MODEL_DIR / "x_scaler_60step.pkl"
YS_PATH     = MODEL_DIR / "y_scaler_60step.pkl"
MODEL_PATH  = MODEL_DIR / "flux_lstm_60step.pth"

# Window (seed length) & horizon must match your training setup
SEQ_LEN = int(os.getenv("SEQ_LEN", 60))   # minutes of history; 60 matches your *60step* model
WINDOW  = 1440                       # exported for api.py
HORIZON  = 1440          # minutes to forecast (24h)
STEP     = 60            # model outputs 60 minutes at a time

INPUT_FEATURES = 2
HIDDEN_SIZE    = 256
NUM_LAYERS     = 2
OUTPUT_SIZE    = STEP
DROPOUT_RATE   = 0.2

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

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.linear = nn.Linear(hidden_size * WINDOW, output_size)
        self.relu   = nn.ReLU()

    def forward(self, x):
        # x: (batch, WINDOW, input_features)
        lstm_out, _ = self.lstm(x)                   # (batch, WINDOW, hidden)
        flat = lstm_out.reshape(lstm_out.shape[0], -1)
        act  = self.relu(flat)
        out  = self.linear(act)                      # (batch, OUTPUT_SIZE)
        return out

@lru_cache(maxsize=1)
def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@lru_cache(maxsize=1)
def _loaded_assets():
    """Load model + scalers once (cached)."""
    device = _device()

    model = LSTMRegressor(
        INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT_RATE
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    x_scaler = joblib.load(XSC_PATH)
    y_scaler = joblib.load(YS_PATH)
    return model, x_scaler, y_scaler, device

def _clean_seed_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure minute cadence, sorted, no NaNs/inf, log-safe."""
    df = df.copy()
    df = df.sort_values("timestamp")
    for c in ("long_flux", "short_flux"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = (
        df.set_index("timestamp")
          .asfreq("1min")
          .interpolate(method="time")
          .ffill()
          .bfill()
          .reset_index()
    )
    return df

def predict_from_seed_df(seed_df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    In-memory PyTorch forecast.
    Input: seed_df with columns ['timestamp','long_flux','short_flux'] covering WINDOW minutes.
    Output: DataFrame with minute-by-minute prediction columns:
            ['timestamp','long_flux_pred','goes_class_pred']
    """
    if len(seed_df) < WINDOW:
        raise RuntimeError(f"Need at least {WINDOW} seed minutes; got {len(seed_df)}")

    model, x_scaler, y_scaler, device = _loaded_assets()

    seed_df  = _clean_seed_df(seed_df)
    seed_df = seed_df.sort_values("timestamp").tail(SEQ_LEN)
    # rolling buffer in log10 space
    buf      = seed_df[["long_flux", "short_flux"]].iloc[-WINDOW:].to_numpy(dtype=float)
    buf_log  = np.log10(np.maximum(buf, 1e-12))

    preds    = []
    t0       = seed_df["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)

    with torch.no_grad():
        for _ in range(horizon // STEP):
            X_scaled = x_scaler.transform(buf_log)                 # (WINDOW, 2)
            X_tensor = torch.from_numpy(X_scaled).float().unsqueeze(0).to(device)
            y_log_scaled = model(X_tensor).cpu().numpy()           # (1, 60)
            y_log = y_scaler.inverse_transform(y_log_scaled).flatten()
            y_lin = 10.0 ** y_log
            preds.extend(y_lin)

            # Update buffer with predicted long_flux; reuse last short_flux log
            last_short_log = np.log10(max(buf[-1, 1], 1e-12))
            new_short_log  = np.full(STEP, last_short_log)
            new_rows_log   = np.column_stack([y_log, new_short_log])
            buf_log        = np.vstack([buf_log[STEP:], new_rows_log])

    idx = pd.date_range(start=t0, periods=horizon, freq="1min")
    out = pd.DataFrame({
        "timestamp": idx,
        "long_flux_pred": preds
    })
    out["goes_class_pred"] = out["long_flux_pred"].apply(flux_to_class)
    return out
