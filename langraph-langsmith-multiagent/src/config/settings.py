from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API Configuration
    OPENAI_API_KEY: str
    UOC_API_KEY: str
    UOC_ENDPOINT: str
    UOC_MODEL_NAME: str
    UOC_API_VERSION: str

    # Database Configuration
    DB_URL: str = "sqlite:///./chinook.db"
    
    # Application Configuration
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
