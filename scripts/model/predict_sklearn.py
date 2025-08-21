import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = Path("models/regressors")
XSC_PATH  = MODEL_DIR / "x_scaler_1step.pkl"
YS_PATH   = MODEL_DIR / "y_scaler_1step.pkl"
MODEL_PATH= MODEL_DIR / "flux_hgbr_1step.pkl"


# This script predicts the next x hours of solar flux using sklearn trained model.
# Change to different seed csv path if needed at bottom.
# Seed is typically last day of the trained data. Eg. July 24-31 of training, seed is July 31.
# Sklearn looks through half of seed at once - 12hrs.
# The model predicts 24 hours after seed date, so be sure to have actual data to compare against.
# Currently set to use 1-step prediction (1 minute).
# Attempts to predict 1 minute at a time, using previous 12 hours (720 minutes) of data.

# You can change HORIZON to predict greater than 24 hours.

WINDOW  = 720
HORIZON = 1440

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

    model = joblib.load(MODEL_PATH)
    x_scaler = joblib.load(XSC_PATH)
    y_scaler = joblib.load(YS_PATH)

    # rolling buffer, log10 scale
    buf = df[["long_flux", "short_flux"]].iloc[-WINDOW:].to_numpy(dtype=float)
    buf_log = np.log10(np.maximum(buf, 1e-12))

    preds = []
    ts_start = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)

    for _ in range(HORIZON):
        X = buf_log.reshape(1, -1)
        Xs = x_scaler.transform(X)
        y_log_scaled = model.predict(Xs).reshape(-1, 1)
        y_log = y_scaler.inverse_transform(y_log_scaled).ravel()[0]
        y_lin = 10**y_log

        preds.append(y_lin)

        # feed the prediction back (teacher forcing style)
        next_row = np.array([[y_lin, buf_log[-1, 0]]])  # long_flux_pred, reuse last short (cheap approx)
        # better: predict short flux too with a second 1-step model; or keep measured short flux
        buf_log = np.vstack([buf_log[1:], np.log10(np.maximum(next_row, 1e-12))])

    idx = pd.date_range(start=ts_start, periods=HORIZON, freq="1min")
    out = pd.DataFrame({
        "timestamp": idx,
        "long_flux_pred": preds,
        "goes_class_pred": [flux_to_class(x) for x in preds]
    })

    # make output name the date range of the forecast
    start_date = idx[0].strftime("%Y_%m_%d")
    end_date = idx[-1].strftime("%Y_%m_%d")
    out_csv = Path(out_dir) / f"forecast_sklearn_{start_date}-{end_date}.csv"

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved forecast -> {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/prediction_seed.csv") # seed - 1 day preferred
    ap.add_argument("--out_dir", default="data/processed")
    args = ap.parse_args()
    main(args.csv, args.out_dir)
