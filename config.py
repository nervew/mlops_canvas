import os
from pathlib import Path

API_TITLE = os.getenv("API_TITLE", "ML Inference API")
API_VERSION = os.getenv("API_VERSION", "1.0.0")

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
MODEL_PATH = ARTIFACTS_DIR / os.getenv("MODEL_FILE", "model.pkl")
DATA_PATH = ARTIFACTS_DIR / os.getenv("DATA_FILE", "data.parquet")
REQUIREMENTS_PATH = ARTIFACTS_DIR / os.getenv("REQUIREMENTS_FILE", "requirements.txt")

THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.65"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
MODEL_NAME = os.getenv("MODEL_NAME", "gompertz-logreg-v1")

OUTPUT_PREDICT_PROBA_KEY = os.getenv("OUTPUT_PREDICT_PROBA_KEY", "predict_proba")
OUTPUT_PREDICT_KEY = os.getenv("OUTPUT_PREDICT_KEY", "predict")
OUTPUT_THRESHOLD_KEY = os.getenv("OUTPUT_THRESHOLD_KEY", "threshold")

ERROR_PREDICT_PROBA_VALUE = float(os.getenv("ERROR_PREDICT_PROBA_VALUE", "1.0"))
ERROR_PREDICT_VALUE = int(os.getenv("ERROR_PREDICT_VALUE", "1"))
ERROR_THRESHOLD_VALUE = float(os.getenv("ERROR_THRESHOLD_VALUE", "0.0"))

SHOW_THRESHOLD = os.getenv("SHOW_THRESHOLD", "true").lower() == "true"

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
BUILD_ID = os.getenv("BUILD_ID", os.getenv("BUILD_BUILDID", "unknown"))

