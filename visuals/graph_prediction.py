import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# This script plots the predicted solar flux from a CSV file.
# You can modify the file paths to point to your specific CSV file and output image path.

def plot_predictions(csv_path, output_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(df["timestamp"], df["long_flux_pred"], color="orange", label="Predicted Long Flux")
    ax1.set_yscale("log")
    ax1.set_ylabel("Watts · m⁻²")
    ax1.set_xlabel("Universal Time")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%b %d"))
    ax1.set_title("GOES X-Ray Flux Predictions (1-minute data)")
    ax1.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(output_path)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    csv_path = os.path.join(script_dir, "../data/processed/forecast_sklearn_2025_08_02-2025_08_02.csv")
    output_path = os.path.join(script_dir, "forecast_plot.png")
    
    plot_predictions(csv_path, output_path)