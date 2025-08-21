import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# This script plots the predicted solar flux from a CSV file.
# Specifically for sklearn model predictions and actual.
# You can modify the file paths to point to your specific CSV files and output image path.
# Make sure actual and sklearn predictons are for the same date range.

def plot_predictions(pytorch_csv_path, validation_csv_path, output_path):
    df_pytorch = pd.read_csv(pytorch_csv_path, parse_dates=["timestamp"])
    df_validation = pd.read_csv(validation_csv_path, parse_dates=["timestamp"])
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
 
    ax1.plot(df_pytorch["timestamp"], df_pytorch["long_flux_pred"], color="blue", label="sklearn Forecast")
    ax1.plot(df_validation["timestamp"], df_validation["long_flux"], color="green", label="Actual Flux", linestyle='--')

    ax1.set_yscale("log")
    ax1.set_ylabel("Watts · m⁻²")
    ax1.set_xlabel("Universal Time")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%b %d"))
    ax1.set_title("GOES X-Ray Flux: Forecast vs. Actual")
    ax1.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(output_path)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    pytorch_csv_path = os.path.join(script_dir, "../data/processed/forecast_sklearn_2025_08_02-2025_08_02.csv")
    validation_csv_path = os.path.join(script_dir, "../data/processed/actual.csv")
    output_path = os.path.join(script_dir, "forecast_validation_sklearn.png")
    
    plot_predictions(pytorch_csv_path, validation_csv_path, output_path)