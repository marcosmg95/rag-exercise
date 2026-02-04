from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """System settings management using Pydantic."""

    # API Keys
    GROQ_API_KEY: str
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_TRACING: bool = False
    LANGSMITH_ENDPOINT: str = "https://eu.api.smith.langchain.com"
    LANGSMITH_PROJECT: str = "rag-exercise"

    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_PERSIST_DIR: str = str(DATA_DIR / "chroma_db")

    # PDF Config
    DEFAULT_PDF_PATH: str = str(
        DATA_DIR / "Clase0_NLP_especificaciones_actividad_1.pdf"
    )
    COLLECTION_NAME: str = "rag_collection"

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
