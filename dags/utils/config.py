from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class MLflowConfig:
    """Конфигурация MLflow"""
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "wine-quality")

@dataclass
class DVCConfig:
    """Конфигурация DVC"""
    data_file: str = os.getenv("DVC_DATA_FILE", "winequality-red.csv")
    data_file_dvc: str = os.getenv("DVC_DATA_FILE_DVC", "winequality-red.csv.dvc")
    artifacts_dir: str = os.getenv("DVC_ARTIFACTS_DIR", "/app/artifacts")
    best_model_path: str = os.getenv("DVC_BEST_MODEL_PATH", "/app/artifacts/best_model.pkl")

@dataclass
class ModelConfig:
    """Конфигурация моделей"""
    test_size: float = float(os.getenv("MODEL_TEST_SIZE", "0.2"))
    random_state: int = int(os.getenv("MODEL_RANDOM_STATE", "42"))
    target_column: str = os.getenv("MODEL_TARGET_COLUMN", "quality")

@dataclass
class AirflowConfig:
    """Конфигурация Airflow"""
    owner: str = os.getenv("AIRFLOW_OWNER", "airflow")
    retries: int = int(os.getenv("AIRFLOW_RETRIES", "1"))
    retry_delay_minutes: int = int(os.getenv("AIRFLOW_RETRY_DELAY_MINUTES", "1"))
    email_on_failure: bool = os.getenv("AIRFLOW_EMAIL_ON_FAILURE", "True").lower() == "true"
    email_on_retry: bool = os.getenv("AIRFLOW_EMAIL_ON_RETRY", "True").lower() == "true"
    alert_email: Optional[str] = os.getenv("AIRFLOW_ALERT_EMAIL")

# Создаем экземпляры конфигураций
mlflow_config = MLflowConfig()
dvc_config = DVCConfig()
model_config = ModelConfig()
airflow_config = AirflowConfig() 