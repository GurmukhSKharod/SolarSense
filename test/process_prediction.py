import pandas as pd
import os
import sys

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

# Processes minute-level predictions from a CSV file and aggregates them into hourly predictions.
# This is done for verification purposes to compare 24 entries instead of 1440.
# THIS FILE IS OPTIONAL, and only used for smaller predictions.

def process_forecast_hourly(input_file_path, output_file_path):
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        return

    try:
        df = pd.read_csv(input_file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Resample to hourly frequency, calculating the mean for 'long_flux_pred'
    hourly_df = df['long_flux_pred'].resample('H').mean().to_frame()

    # Determine the GOES class from the averaged flux
    hourly_df['goes_class_pred'] = hourly_df['long_flux_pred'].apply(flux_to_class)

    hourly_df.reset_index(inplace=True)
    hourly_df['timestamp'] = hourly_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    hourly_df.to_csv(output_file_path, index=False)
    
    print(f"Processed {len(df)} minute-level predictions into {len(hourly_df)} hourly predictions.")
    print(f"Saved hourly forecast to: {output_file_path}")

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    input_filename = 'forecast_sklearn_2025_08_02-2025_08_02.csv' # chnage to pytorch or sklearn as needed
    output_filename = f"processed_{input_filename}"
    input_path = os.path.join(project_root, 'data', 'processed', input_filename)
    output_path = os.path.join(project_root, 'data', 'processed', output_filename)
    print(f"Usage: python {sys.argv[0]} [input_file] [output_file]")
    print(f"Using default files:\n  Input: {input_path}\n  Output: {output_path}")

    process_forecast_hourly(input_path, output_path)
