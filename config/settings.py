"""
Configuration management for Meditron3-Qwen2.5 project
"""
import os
from pathlib import Path
from typing import Optional

from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class ModelConfig(BaseSettings):
    """Model configuration settings"""
    model_config = ConfigDict(extra='ignore')

    model_name: str = Field("OpenMeditron/Meditron3-Qwen2.5-7B", env="MODEL_NAME")
    model_path: str = Field("./models/Meditron3-Qwen2.5-7B", env="MODEL_PATH")
    max_length: int = Field(4096, env="MAX_LENGTH")
    temperature: float = Field(0.7, env="TEMPERATURE")
    top_p: float = Field(0.9, env="TOP_P")
    max_new_tokens: int = Field(512, env="MAX_NEW_TOKENS")
    repetition_penalty: float = Field(1.1, env="REPETITION_PENALTY")
    dtype: str = Field("bfloat16", env="DTYPE")
    device_map: str = Field("auto", env="DEVICE_MAP")
    load_in_4bit: bool = Field(False, env="LOAD_IN_4BIT")
    load_in_8bit: bool = Field(False, env="LOAD_IN_8BIT")


class APIConfig(BaseSettings):
    """API configuration settings"""
    model_config = ConfigDict(extra='ignore')

    host: str = Field("0.0.0.0", env="API_HOST")
    port: int = Field(8000, env="API_PORT")
    workers: int = Field(1, env="API_WORKERS")
    reload: bool = Field(False, env="API_RELOAD")


class KnowledgeGraphConfig(BaseSettings):
    """Knowledge Graph configuration"""
    model_config = ConfigDict(extra='ignore')

    neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field("neo4j", env="NEO4J_USER")
    neo4j_password: str = Field("your_password", env="NEO4J_PASSWORD")


class VectorDBConfig(BaseSettings):
    """Vector database configuration"""
    model_config = ConfigDict(extra='ignore')

    chroma_persist_directory: str = Field("./data/embeddings/chroma", env="CHROMA_PERSIST_DIRECTORY")
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    collection_name: str = Field("medical_documents", env="COLLECTION_NAME")


class DataConfig(BaseSettings):
    """Data paths configuration"""
    model_config = ConfigDict(extra='ignore')

    raw_data_path: str = Field("./data/raw", env="RAW_DATA_PATH")
    processed_data_path: str = Field("./data/processed", env="PROCESSED_DATA_PATH")
    embeddings_path: str = Field("./data/embeddings", env="EMBEDDINGS_PATH")


class LoggingConfig(BaseSettings):
    """Logging configuration"""
    model_config = ConfigDict(extra='ignore')

    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("./logs/app.log", env="LOG_FILE")


class Settings(BaseSettings):
    """Main settings class"""
    model_config = ConfigDict(extra='ignore', env_file=".env", env_file_encoding="utf-8")

    environment: str = Field("development", env="ENVIRONMENT")

    # Sub-configurations
    model: ModelConfig = ModelConfig()
    api: APIConfig = APIConfig()
    kg: KnowledgeGraphConfig = KnowledgeGraphConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig()


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent


# Global settings instance
settings = Settings()

# Resolve relative paths to absolute paths based on project root
project_root = get_project_root()
settings.model.model_path = str(project_root / settings.model.model_path)
settings.vector_db.chroma_persist_directory = str(project_root / settings.vector_db.chroma_persist_directory)
settings.data.raw_data_path = str(project_root / settings.data.raw_data_path)
settings.data.processed_data_path = str(project_root / settings.data.processed_data_path)
settings.data.embeddings_path = str(project_root / settings.data.embeddings_path)
settings.logging.log_file = str(project_root / "logs" / "app.log")  # Ensure log file is in a subdir


def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        os.path.dirname(settings.logging.log_file),
        settings.data.raw_data_path,
        settings.data.processed_data_path,
        settings.data.embeddings_path,
        settings.vector_db.chroma_persist_directory,
        settings.model.model_path,
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test configuration loading
    print("Configuration loaded successfully:")
    print(f"Model: {settings.model.model_name}")
    print(f"API Port: {settings.api.port}")
    print(f"Environment: {settings.environment}")
    ensure_directories()
    print("Directories created successfully")