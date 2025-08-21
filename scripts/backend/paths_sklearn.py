from pathlib import Path

# project root = two levels up from this file (…/SolarSense)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models" / "regressors"
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
