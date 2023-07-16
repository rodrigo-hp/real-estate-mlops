import json
import logging
import joblib
import gcsfs
from config import get_settings
from google.oauth2 import credentials


# Create a logger
def get_logger(logger_name="model_logger", log_file="model.log"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


# Load the model
def load_joblib(bucket_name, file_name, credentials_obj):
    fs = gcsfs.GCSFileSystem(token=credentials_obj)
    with fs.open(f"{bucket_name}/{file_name}", "rb") as f:
        return joblib.load(f)


def get_model_from_gcs():
    settings = get_settings()
    gcp_json_credentials_dict = json.loads(settings.GOOGLE_APPLICATION_CREDENTIALS)
    credentials_obj = credentials.Credentials.from_authorized_user_info(
        gcp_json_credentials_dict
    )
    try:
        model = load_joblib(
            settings.STORAGE_BUCKET_NAME, settings.MODEL_PATH, credentials_obj
        )
    except OSError:
        raise OSError("Model not found in cloud storage!")
    return model
