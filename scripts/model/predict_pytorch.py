import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

MODEL_DIR = Path("models/regressors")
XSC_PATH  = MODEL_DIR / "x_scaler_60step.pkl"
YS_PATH   = MODEL_DIR / "y_scaler_60step.pkl"
MODEL_PATH= MODEL_DIR / "flux_lstm_60step.pth" # Use PyTorch model


# This script predicts the next x hours of solar flux using pytorch trained model.
# Change to different seed csv path if needed at bottom.
# Seed is typically last day of the trained data. Eg. July 24-31 of training, seed is July 31.
# Pytorch looks through entire window at once from seed - 24hrs, so we predict 60 minutes at a time.
# Modify hyperparameters to match the training script.
# The model predicts x hours after seed date, so be sure to have actual data to compare against.
# Currently set to use 60-step prediction (1 hour at a time).
# Attempts to predict 60 minutes at a time, using previous 24 hours (1440 minutes) of data.

# You can change HORIZON to predict greater or less than 24 hours.

WINDOW  = 1440
HORIZON = 1440

#  Model Hyperparameters (must match training) 
INPUT_FEATURES = 2
HIDDEN_SIZE = 256
NUM_LAYERS = 2
OUTPUT_SIZE = 60
DROPOUT_RATE = 0.2 # Must match training

CLASS_THRESH = [
    (1e-4, "X"),
    (1e-5, "M"),
    (1e-6, "C"),
    (1e-7, "B"),
    (0.0,  "A"),
]

# --- PyTorch Model Definition (must match training script) ---
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size * WINDOW, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x) # lstm_out shape: (batch, window, hidden_size)
        flattened_out = lstm_out.reshape(lstm_out.shape[0], -1)
        activated_out = self.relu(flattened_out)
        out = self.linear(activated_out)
        return out

def flux_to_class(f):
    for thr, label in CLASS_THRESH:
        if f >= thr:
            return label
    return "A"

def load_clean(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp")
    for c in ("long_flux", "short_flux"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = (df.set_index("timestamp")
            .asfreq("1min")
            .interpolate(method="time")
            .ffill()
            .bfill()
            .reset_index())
    return df

def main(csv_path, out_dir):
    df = load_clean(csv_path)
    if len(df) < WINDOW:
        raise RuntimeError(f"Need at least {WINDOW} minutes of data")

    # --- Load Model and Scalers ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = LSTMRegressor(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT_RATE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode

    x_scaler = joblib.load(XSC_PATH)
    y_scaler = joblib.load(YS_PATH)

    # rolling buffer, log10 scale
    buf = df[["long_flux", "short_flux"]].iloc[-WINDOW:].to_numpy(dtype=float)
    buf_log = np.log10(np.maximum(buf, 1e-12))

    preds = []
    ts_start = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)

    with torch.no_grad(): # Disable gradient calculation for inference
        # Loop 24 times to generate a 24-hour forecast (24 * 60 minutes = 1440)
        for _ in range(HORIZON // OUTPUT_SIZE):
            # Scale the entire window, which has shape (WINDOW, FEATURES)
            X_scaled = x_scaler.transform(buf_log)

            # Reshape for LSTM: (batch, sequence, features) and convert to tensor
            X_tensor = torch.from_numpy(X_scaled).float().unsqueeze(0).to(device)

            # Predict the next 60 minutes in one shot
            y_log_scaled = model(X_tensor).cpu().numpy()
            
            # The y_scaler was fit on data of shape (samples, 60), so it expects that here.
            y_log = y_scaler.inverse_transform(y_log_scaled).flatten()
            y_lin = 10**y_log

            preds.extend(y_lin)

            # --- Update the buffer for the next hour's prediction ---
            # We need to create 60 new rows for the buffer.
            # We have the predicted long_flux, but need to generate short_flux.
            # A simple strategy is to reuse the last known short_flux value.
            last_short_flux = buf[-1, 1]
            new_short_flux_log = np.full(OUTPUT_SIZE, np.log10(np.maximum(last_short_flux, 1e-12)))
            
            # Combine predicted long_flux and generated short_flux
            new_rows_log = np.column_stack([y_log, new_short_flux_log])
            
            # Append the 60 new minutes and drop the 60 oldest minutes
            buf_log = np.vstack([buf_log[OUTPUT_SIZE:], new_rows_log])

    idx = pd.date_range(start=ts_start, periods=HORIZON, freq="1min")
    out = pd.DataFrame({
        "timestamp": idx,
        "long_flux_pred": preds,
        "goes_class_pred": [flux_to_class(x) for x in preds]
    })

    # make output name the date range of the forecast
    start_date = idx[0].strftime("%Y_%m_%d")
    end_date = idx[-1].strftime("%Y_%m_%d")
    out_csv = Path(out_dir) / f"forecast_pytorch_{start_date}-{end_date}.csv"

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved forecast -> {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/prediction_seed.csv")
    ap.add_argument("--out_dir", default="data/processed")
    args = ap.parse_args()
    main(args.csv, args.out_dir)
    args = ap.parse_args()
    main(args.csv, args.out_dir)
