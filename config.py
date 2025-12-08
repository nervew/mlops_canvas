import os
from pathlib import Path

API_TITLE = os.getenv("API_TITLE", "ML New")
API_VERSION = os.getenv("API_VERSION", "3.0.0")

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
MODEL_PATH = ARTIFACTS_DIR / os.getenv("MODEL_FILE", "model.pkl")
DATA_PATH = ARTIFACTS_DIR / os.getenv("DATA_FILE", "data.parquet")
REQUIREMENTS_PATH = ARTIFACTS_DIR / os.getenv("REQUIREMENTS_FILE", "requirements.txt")

THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.65"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "0.0.2-dev.1")
MODEL_NAME = os.getenv("MODEL_NAME", "version2")

OUTPUT_PREDICT_PROBA_KEY = os.getenv("OUTPUT_PREDICT_PROBA_KEY", "predict_proba")
OUTPUT_PREDICT_KEY = os.getenv("OUTPUT_PREDICT_KEY", "predict")
OUTPUT_THRESHOLD_KEY = os.getenv("OUTPUT_THRESHOLD_KEY", "threshold")

ERROR_PREDICT_PROBA_VALUE = float(os.getenv("ERROR_PREDICT_PROBA_VALUE", "0.0"))
ERROR_PREDICT_VALUE = int(os.getenv("ERROR_PREDICT_VALUE", "0"))
ERROR_THRESHOLD_VALUE = float(os.getenv("ERROR_THRESHOLD_VALUE", "0.0"))

SHOW_THRESHOLD = os.getenv("SHOW_THRESHOLD", "true").lower() == "true"

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
BUILD_ID = os.getenv("BUILD_ID", os.getenv("BUILD_BUILDID", "unknown"))

