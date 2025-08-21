import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path("models/regressors")
XSC_PATH  = MODEL_DIR / "x_scaler_1step.pkl"
YS_PATH   = MODEL_DIR / "y_scaler_1step.pkl"
MODEL_PATH= MODEL_DIR / "flux_hgbr_1step.pkl"

# This script trains the sklearn model using the last x days of data.
# Uses sklearn for training.
# Currently set to use 1-step prediction (1 minute at a time).
# Currently uses all available data for training.
# be sure to change training csv path at bottom if needed.

# HistGradientBoostingRegressor(
#    max_depth=6,      # Limits each tree to 6 levels to control complexity and overfitting
#    max_iter=500,     # Builds up to 500 boosting trees
#    learning_rate=0.05, # Shrinks each tree’s contribution for slower, more accurate learning
#    random_state=42   # Fixes randomization so results are reproducible
# )

WINDOW = 720  

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

def make_supervised_1step(df, window):
    vals = df[["long_flux", "short_flux"]].to_numpy(dtype=float)
    log_vals = np.log10(np.maximum(vals, 1e-12))
    y = np.log10(np.maximum(df["long_flux"].shift(-1).to_numpy(dtype=float), 1e-12))

    X_list, y_list = [], []
    for i in range(window, len(df) - 1):
        X_list.append(log_vals[i - window:i, :].reshape(-1))
        y_list.append(y[i])
    return np.asarray(X_list), np.asarray(y_list)

def main(csv_path):
    df = load_clean(csv_path)
    X, y = make_supervised_1step(df, WINDOW)

    # scalers
    x_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)

    y = y.reshape(-1, 1)
    y_scaler = StandardScaler()
    ys = y_scaler.fit_transform(y).ravel()

    model = HistGradientBoostingRegressor(
        max_depth=6,
        max_iter=500,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(Xs, ys)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(x_scaler, XSC_PATH)
    joblib.dump(y_scaler, YS_PATH)
    print(f"Saved: {MODEL_PATH}, {XSC_PATH}, {YS_PATH}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/training_data.csv")
    args = ap.parse_args()
    main(args.csv)
