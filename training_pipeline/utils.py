import joblib
import gcsfs
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from google.cloud import storage
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)


def print_metrics(metrics_dict):
    print("RMSE: ", metrics_dict["rmse"])
    print("MAPE: ", metrics_dict["mape"])
    print("MAE : ", metrics_dict["mae"])


def get_metrics(predictions, target):
    rmse = np.sqrt(mean_squared_error(target, predictions))
    mape = mean_absolute_percentage_error(target, predictions)
    mae = mean_absolute_error(target, predictions)
    metrics = {
        "rmse": rmse,
        "mape": mape,
        "mae": mae,
    }
    return metrics


def blob_exists(project_name, bucket_name, file_name, credentials_obj):
    storage_client = storage.Client(project=project_name, credentials=credentials_obj)
    bucket = storage_client.bucket(bucket_name)
    stats = storage.Blob(bucket=bucket, name=file_name).exists()
    return stats


def load_joblib(bucket_name, file_name, credentials_obj):
    fs = gcsfs.GCSFileSystem(token=credentials_obj)
    with fs.open(f"{bucket_name}/{file_name}", "rb") as f:
        return joblib.load(f)


def dump_joblib(bucket_name, file_name, model, credentials_obj):
    fs = gcsfs.GCSFileSystem(token=credentials_obj)
    with fs.open(f"{bucket_name}/{file_name}", "wb") as f:
        return joblib.dump(model, f)


def get_data_from_postgres(credentials, table_name):
    # Extract credentials from environment variables
    db_url = credentials.get("DB_URL")
    db_user = credentials.get("DB_USER")
    db_password = credentials.get("DB_PASSWORD")
    db_name = credentials.get("DB_NAME")

    # Create a database connection string
    db_connection_string = f"postgresql://{db_user}:{db_password}@{db_url}/{db_name}"

    # Create a SQLAlchemy engine
    engine = create_engine(db_connection_string)

    # Execute SELECT query to fetch all rows from the table
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, engine)

    # Split the data into train and test DataFrames
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df, test_df
