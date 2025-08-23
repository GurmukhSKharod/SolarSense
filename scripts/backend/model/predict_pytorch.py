# scripts/backend/model/predict_pytorch.py
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import pandas as pd
import joblib
import os
import torch
import torch.nn as nn

ROOT       = Path(__file__).resolve().parents[3]
MODEL_DIR  = ROOT / "models" / "regressors"
XSC_PATH   = MODEL_DIR / "x_scaler_60step.pkl"
YS_PATH    = MODEL_DIR / "y_scaler_60step.pkl"
MODEL_PATH = MODEL_DIR / "flux_lstm_60step.pth"
META_PATH  = MODEL_DIR / "flux_lstm_60step.meta.json"  # optional (if you have it)

# Defaults (will be overwritten by checkpoint/meta)
WINDOW_DEFAULT  = int(os.getenv("SEQ_LEN", 60))
STEP_DEFAULT    = 60

CLASS_THRESH = [(1e-4,"X"),(1e-5,"M"),(1e-6,"C"),(1e-7,"B"),(0.0,"A")]
def flux_to_class(f):
    for thr, label in CLASS_THRESH:
        if f >= thr: return label
    return "A"

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate, window):
        super().__init__()
        self.window = window
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.linear = nn.Linear(hidden_size * self.window, output_size)
        self.relu   = nn.ReLU()

    def forward(self, x):
        # x: (batch, window, input_features)
        lstm_out, _ = self.lstm(x)
        flat = lstm_out.reshape(lstm_out.shape[0], -1)
        act  = self.relu(flat)
        out  = self.linear(act)
        return out

def _infer_from_state_dict(sd):
    """Infer window, hidden_size, num_layers, step directly from checkpoint weights."""
    step = sd["linear.weight"].shape[0]                    # output_size
    in_features = sd["linear.weight"].shape[1]            # hidden_size * window
    # LSTM shapes: weight_hh_l0 is (4*hidden, hidden)
    hidden = sd["lstm.weight_hh_l0"].shape[1]
    window = in_features // hidden
    # num_layers = count lstm layers in keys
    num_layers = len([k for k in sd.keys() if k.startswith("lstm.weight_ih_l")])
    return {"window": int(window), "hidden_size": int(hidden), "num_layers": int(num_layers), "step": int(step)}

def _load_hparams(sd):
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text())
        except Exception:
            pass
    return _infer_from_state_dict(sd)

@lru_cache(maxsize=1)
def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# These two are exported for api.py (seed window + horizon)
WINDOW  = WINDOW_DEFAULT
HORIZON = 1440

@lru_cache(maxsize=1)
def _loaded_assets():
    """Load checkpoint, infer hyperparams, then build the exact same net."""
    device = _device()
    state = torch.load(MODEL_PATH, map_location="cpu")     # state_dict

    hp = _load_hparams(state)
    global WINDOW
    WINDOW = int(hp.get("window", WINDOW_DEFAULT))         # export for api
    hidden_size = int(hp.get("hidden_size", 256))
    num_layers  = int(hp.get("num_layers", 2))
    step        = int(hp.get("step", STEP_DEFAULT))

    model = LSTMRegressor(
        input_size=2,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=step,
        dropout_rate=0.2,
        window=WINDOW,
    ).to(device)

    model.load_state_dict(state)  # will succeed now because shapes match
    model.eval()

    x_scaler = joblib.load(XSC_PATH)
    y_scaler = joblib.load(YS_PATH)
    return model, x_scaler, y_scaler, device, step

def _clean_seed_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp")
    for c in ("long_flux","short_flux"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf,-np.inf], np.nan)
    df = (df.set_index("timestamp").asfreq("1min").interpolate("time").ffill().bfill().reset_index())
    return df

def predict_from_seed_df(seed_df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    model, x_scaler, y_scaler, device, step = _loaded_assets()

    seed_df  = _clean_seed_df(seed_df).sort_values("timestamp").tail(WINDOW)
    if len(seed_df) < WINDOW:
        raise RuntimeError(f"Need at least {WINDOW} minutes of seed data; got {len(seed_df)}")

    buf     = seed_df[["long_flux","short_flux"]].to_numpy(dtype=float)
    buf_log = np.log10(np.maximum(buf, 1e-12))

    preds = []
    t0    = seed_df["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)

    with torch.no_grad():
        steps = max(1, horizon // step)
        for _ in range(steps):
            X_scaled = x_scaler.transform(buf_log)              # (window, 2)
            X_tensor = torch.from_numpy(X_scaled).float().unsqueeze(0).to(device)
            y_log_scaled = model(X_tensor).cpu().numpy()        # (1, step)
            y_log = y_scaler.inverse_transform(y_log_scaled).ravel()
            y_lin = 10.0 ** y_log
            preds.extend(y_lin.tolist())

            last_short_log = np.log10(max(buf[-1, 1], 1e-12))
            new_short_log  = np.full(step, last_short_log)
            new_rows_log   = np.column_stack([y_log, new_short_log])
            buf_log        = np.vstack([buf_log[step:], new_rows_log])

    idx = pd.date_range(start=t0, periods=len(preds), freq="1min")
    out = pd.DataFrame({"timestamp": idx, "long_flux_pred": preds})
    out["goes_class_pred"] = out["long_flux_pred"].apply(flux_to_class)
    return out
