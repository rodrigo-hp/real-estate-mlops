from pydantic import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    API_KEY: str
    GCP_PROJECT_NAME: str
    GOOGLE_APPLICATION_CREDENTIALS: str
    STORAGE_BUCKET_NAME: str
    MODEL_PATH: str

    class Config:
        env_file = ".env"


# New decorator for cache
@lru_cache()
def get_settings():
    return Settings()
