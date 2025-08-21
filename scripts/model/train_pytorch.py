import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR

MODEL_DIR = Path("models/regressors")
XSC_PATH  = MODEL_DIR / "x_scaler_60step.pkl"
YS_PATH   = MODEL_DIR / "y_scaler_60step.pkl"
MODEL_PATH= MODEL_DIR / "flux_lstm_60step.pth" # Changed model extension

# This script retrains the LSTM model using the last 90 days of data.
# Uses PyTorch for training.
# Currently set to use 60-step prediction (1 hour at a time).
# Currently uses 2 weeks of data for training.
# be sure to change training csv path at bottom if needed.


# --- Model & Training Hyperparameters ---
WINDOW = 1440          # Input sequence length (12h of 1-min data)
HORIZON = 60
INPUT_FEATURES = 2    # Number of features per time step (long_flux, short_flux)
HIDDEN_SIZE = 256      # Increased model capacity
NUM_LAYERS = 2        # Number of stacked LSTM layers
OUTPUT_SIZE = HORIZON      # Number of output values to predict (just long_flux)
EPOCHS = 50           # Number of full passes through the training data
BATCH_SIZE = 128       # Number of samples to process in one batch
LEARNING_RATE = 0.001 # Step size for the optimizer
DROPOUT_RATE = 0.2    # Dropout rate for regularization

# --- PyTorch Model Definition ---
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMRegressor, self).__init__()
        # LSTM layer processes sequences of input data. batch_first=True means
        # the input tensor shape is (batch, sequence_length, features).
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        # A standard fully-connected layer to map the LSTM output to the desired output size.
        # The input to this layer is now the flattened output of the LSTM.
        self.linear = nn.Linear(hidden_size * WINDOW, output_size)
        self.relu = nn.ReLU()

    # forward pass
    def forward(self, x):
        # The LSTM returns its output and the final hidden and cell states.
        # We now use the entire output sequence from the LSTM.
        lstm_out, _ = self.lstm(x) # lstm_out shape: (batch, window, hidden_size)
        
        # Flatten the LSTM output to feed into the linear layer
        # This allows the model to use information from all input time steps.
        flattened_out = lstm_out.reshape(lstm_out.shape[0], -1)
        
        # Apply ReLU activation
        activated_out = self.relu(flattened_out)
        
        # Pass the flattened output through the linear layer.
        predictions = self.linear(activated_out)
        return predictions

def weighted_mse_loss(inputs, targets):
    """
    Calculates a weighted Mean Squared Error, giving more importance
    to samples with higher target values using an exponential weight.
    """
    # Weights are exponential of the target values. Since targets are standardized,
    # this gives much higher weight to values above the mean (flares).
    weights = torch.exp(targets)
    loss = torch.mean(weights * (inputs - targets) ** 2)
    return loss

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

def make_supervised_60step(df, window, horizon):
    """Prepares data for multi-step LSTM. X shape: (samples, window, features), y shape: (samples, horizon)"""
    X_vals = df[["long_flux", "short_flux"]].to_numpy(dtype=float)
    y_vals = df["long_flux"].to_numpy(dtype=float)

    log_X = np.log10(np.maximum(X_vals, 1e-12))
    log_y = np.log10(np.maximum(y_vals, 1e-12))

    X_list, y_list = [], []
    for i in range(window, len(df) - horizon):
        X_list.append(log_X[i - window:i, :])
        y_list.append(log_y[i:i + horizon])
    return np.asarray(X_list), np.asarray(y_list)

def main(csv_path):
    df = load_clean(csv_path)
    X, y = make_supervised_60step(df, WINDOW, HORIZON)

    # Scaling 
    # StandardScaler expects 2D input [samples, features]. Our input X is 3D
    # [samples, timesteps, features]. We reshape to 2D, fit/transform, then reshape back.
    x_scaler = StandardScaler()
    X_shape = X.shape
    Xs = x_scaler.fit_transform(X.reshape(-1, X_shape[-1])).reshape(X_shape)

    # y = y.reshape(-1, 1)
    y_scaler = StandardScaler()
    ys = y_scaler.fit_transform(y)

    # PyTorch DataLoader 
    # Convert numpy arrays to PyTorch tensors.
    X_tensor = torch.from_numpy(Xs).float()
    y_tensor = torch.from_numpy(ys).float()
    # Create a tensor dataset and a loader to handle batching and shuffling.
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Model, Loss, Optimizer ---
    # NVIDIA GPU's
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # Apple Silicon
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model = LSTMRegressor(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT_RATE).to(device)
    # use custom weighted loss function
    # Adam is an optimizer that adapts learning rates
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5) # Gentler learning rate scheduler

    # --- Training Loop ---
    print("Starting model training...")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        # Iterate the data in batches.
        for i, (X_batch, y_batch) in enumerate(loader):
            # Move data for the current batch to the active device.
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass: compute predicted y by passing x to the model.
            outputs = model(X_batch)
            loss = weighted_mse_loss(outputs, y_batch)

            # Backward pass and optimize
            optimizer.zero_grad() # Clear gradients from previous step.
            loss.backward()       # Compute gradients of the loss w.r.t. model parameters.
            optimizer.step()      # Update the model parameters.
            epoch_loss += loss.item()

        scheduler.step() # Update learning rate
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]}")

    # --- Save Model and Scalers ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(x_scaler, XSC_PATH)
    joblib.dump(y_scaler, YS_PATH)
    print(f"Saved: {MODEL_PATH}, {XSC_PATH}, {YS_PATH}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/training_data.csv") 
    args = ap.parse_args()
    main(args.csv)