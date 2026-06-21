"""Application configuration settings."""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings configuration."""

    host: str = "0.0.0.0"
    port: int = 8000

    aitomia_log_dir: str = "~/.aitomia"

    # OpenAI Configuration
    openai_api_key: str
    openai_api_base: str
    openai_model: str

    # deepseek_model: str
    # deepseek_api_key: str
    # deepseek_api_base: str

    reload: bool = False

    dev_mode: bool = False
    dev_workfolder: Optional[str] = None
    molecule_container: Optional[str] = None
    dev_file: Optional[str] = None
    # Backend Service
    file_service_url: str

    slurm_service_url: str

    debug: bool = True
    log_level: str = "DEBUG"
    log_file: Optional[str] = None

    memory_saver: Optional[str] = None
    db_dsn: Optional[str] = None
    statistic_database_dsn: Optional[str] = None
    db_pool_max_size: Optional[int] = 20

    molecule_service_url: Optional[str] = None
    pcp_service_url: Optional[str] = None

    jwt_secret: Optional[str] = None
    job_submit_command: Optional[str] = None

    result_dir: str

    white_list_hosts: Optional[str] = None

    use_input: bool = False

    recursion_limit: int = 100

    workers: int = 1

    # NATS Configuration
    nats_enabled: bool = False
    nats_url: str = "nats://localhost:4222"
    nats_subject_prefix: str = "aitomia.chat"

    # Redis Configuration
    # redis_host: str
    # redis_port: int
    # redis_password: Optional[str]
    # redis_db: Optional[int]

    # Qdrant Configuration
    # qdrant_url: str
    # qdrant_collection_name: str

    # model path
    # embedding_model_path:str
    # crossencoder_model_path: str

    # Application Configuration
    # app_host: str
    # app_port: int
    # app_reload: bool

    model_config = {
        "env_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        "env_file_encoding": "utf-8",
        "arbitrary_types_allowed": True,
    }

    # class Config:
    #     dir_path = os.path.dirname(os.path.abspath(__file__))
    #     env_file = os.path.join(dir_path, '.env')
    #     arbitrary_types_allowed = True


# Global settings instance
settings = Settings()
