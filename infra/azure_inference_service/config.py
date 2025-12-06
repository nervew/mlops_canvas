from functools import lru_cache
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Configuration for the inference service."""

    azure_storage_connection_string: str = Field(..., env="AZURE_STORAGE_CONNECTION_STRING")
    model_blob_container: str = Field(..., env="MODEL_BLOB_CONTAINER")
    model_name: str = Field("demo-model", env="MODEL_NAME")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
