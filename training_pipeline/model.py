import os
import json
import pandas as pd

from dotenv import load_dotenv
from google.oauth2 import credentials
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from utils import (
    get_metrics,
    print_metrics,
    blob_exists,
    dump_joblib,
    load_joblib,
    get_data_from_postgres,
)


def train_model(train, train_cols, target):
    categorical_cols = ["type", "sector"]

    categorical_transformer = TargetEncoder()

    # Define preprocessor step to transform categorical columns
    preprocessor = ColumnTransformer(
        transformers=[("categorical", categorical_transformer, categorical_cols)]
    )

    steps = [
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingRegressor(
                **{
                    "learning_rate": 0.01,
                    "n_estimators": 300,
                    "max_depth": 5,
                    "loss": "absolute_error",
                }
            ),
        ),
    ]

    pipeline = Pipeline(steps)

    # Define the parameter grid for GridSearchCV
    param_grid = {
        "model__learning_rate": [0.01, 0.1],
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7],
    }

    # Perform GridSearchCV to find the best model
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(train[train_cols], train[target])

    return grid_search.best_estimator_


def main():
    # Try to load .env file or already defined env variables
    load_dotenv()

    google_application_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    gcp_project_name = os.getenv("GCP_PROJECT_NAME")
    storage_bucket_name = os.getenv("STORAGE_BUCKET_NAME")
    train_set_path = os.getenv("TRAIN_SET_PATH")
    test_set_path = os.getenv("TEST_SET_PATH")
    model_path = os.getenv("MODEL_PATH")
    dataset_source = os.getenv("DATASET_SOURCE")

    gcp_json_credentials_dict = json.loads(google_application_credentials)
    credentials_obj = credentials.Credentials.from_authorized_user_info(
        gcp_json_credentials_dict
    )

    # Try to load datasets
    if dataset_source == "files":
        try:
            train_df = pd.read_csv(
                f"gs://{storage_bucket_name}/{train_set_path}",
                storage_options={"token": credentials_obj},
            )
            test_df = pd.read_csv(
                f"gs://{storage_bucket_name}/{test_set_path}",
                storage_options={"token": credentials_obj},
            )
        except OSError:
            raise OSError("Datasets not found!")
    elif dataset_source == "database":
        DB_CREDENTIALS = os.getenv("DB_CREDENTIALS")
        table_name = "properties"
        train_df, test_df = get_data_from_postgres(DB_CREDENTIALS, table_name)
    else:
        raise OSError(
            "Invalid env variable for dataset source! It must be 'files' or 'database'"
        )

    # Set target and train model
    remove_cols = ["price"]
    train_cols = [col for col in train_df.columns if col not in remove_cols]
    target = "price"
    model = train_model(train_df, train_cols, target)

    # Evaluate the model on the test set
    model_predictions = model.predict(test_df[train_cols])
    test_target = test_df[target].values
    new_model_metrics = get_metrics(model_predictions, test_target)
    print("New model metrics:")
    print_metrics(new_model_metrics)

    # If there is no model already in the bucket, then upload new one
    if not blob_exists(
        gcp_project_name,
        storage_bucket_name,
        model_path,
        credentials_obj,
    ):
        dump_joblib(storage_bucket_name, model_path, model, credentials_obj)
        print("New trained model uploaded to Google Cloud Storage!")
    # If there is a model already, compare evaluation metrics between both
    else:
        old_model = load_joblib(storage_bucket_name, model_path, credentials_obj)
        old_model_predictions = old_model.predict(test_df[train_cols])
        old_model_metrics = get_metrics(old_model_predictions, test_target)
        print("Old model metrics:")
        print_metrics(old_model_metrics)
        if (
            (new_model_metrics["rmse"] < old_model_metrics["rmse"])
            & (new_model_metrics["mape"] < old_model_metrics["mape"])
            & (new_model_metrics["mae"] < old_model_metrics["mae"])
        ):
            dump_joblib(storage_bucket_name, model_path, model, credentials_obj)
            print("New trained model replaced old model in Google Cloud Storage!")
        else:
            print("New model did not outperform the old model. Discarding new model.")


if __name__ == "__main__":
    main()
