import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
  app_name: str = "contextcraft"
  api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
  model_name: str = "llama-3.1-8b-instant"
  db_path: str = "./data/chroma"
  host: str = "0.0.0.0"
  port: int = 8000
  top_k: int = 3

  class Config:
    env_file = ".env"
    extra = "ignore"


settings = Settings()
Path(settings.db_path).mkdir(parents=True, exist_ok=True)