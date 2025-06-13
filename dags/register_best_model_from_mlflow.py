from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow
import logging
import os
import shutil
from typing import Dict, Any

from utils.config import mlflow_config, dvc_config, airflow_config
from utils.notifications import send_failure_notification

logger = logging.getLogger(__name__)

def register_best_model(**context) -> None:
    """
    Находит лучшую модель в MLflow и регистрирует её
    
    Args:
        **context: Контекст выполнения задачи Airflow
        
    Raises:
        RuntimeError: При ошибках в процессе регистрации модели
    """
    try:
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        experiment = mlflow.get_experiment_by_name(mlflow_config.experiment_name)
        
        if experiment is None:
            raise RuntimeError(f"Эксперимент {mlflow_config.experiment_name} не найден")
            
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="metrics.f1_score IS NOT NULL",
            order_by=["metrics.f1_score DESC"]
        )
        
        if runs.empty:
            raise RuntimeError("Не найдены запуски с метрикой f1_score")
            
        best_run = runs.iloc[0]
        best_run_id = best_run["run_id"]
        best_f1_score = best_run["metrics.f1_score"]
        
        logger.info(f"🏆 Лучшая модель: run_id={best_run_id}, f1_score={best_f1_score}")
        
        # Копируем модель в директорию для лучших моделей
        os.makedirs(os.path.dirname(dvc_config.best_model_path), exist_ok=True)
        
        # Находим путь к модели в артефактах
        artifacts = mlflow.artifacts.list_artifacts(best_run_id)
        model_artifact = next((a for a in artifacts if a.path.endswith(".pkl")), None)
        
        if model_artifact is None:
            raise RuntimeError("Модель не найдена в артефактах")
            
        # Скачиваем модель
        local_path = mlflow.artifacts.download_artifacts(
            run_id=best_run_id,
            artifact_path=model_artifact.path
        )
        
        # Копируем в нужную директорию
        shutil.copy2(local_path, dvc_config.best_model_path)
        logger.info(f"✅ Модель скопирована в {dvc_config.best_model_path}")
        
        # Регистрируем модель в MLflow Model Registry
        model_uri = f"runs:/{best_run_id}/{model_artifact.path}"
        model_details = mlflow.register_model(
            model_uri=model_uri,
            name="wine-quality-model"
        )
        
        logger.info(f"✅ Модель зарегистрирована в MLflow Model Registry: {model_details.name}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при регистрации лучшей модели: {str(e)}")
        if airflow_config.email_on_failure:
            send_failure_notification(
                task_instance=context['task_instance'],
                error=e,
                context=context,
                email_to=airflow_config.alert_email
            )
        raise RuntimeError(f"Не удалось зарегистрировать лучшую модель: {str(e)}")

default_args = {
    "owner": airflow_config.owner,
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": airflow_config.retries,
    "retry_delay": timedelta(minutes=airflow_config.retry_delay_minutes),
    "email_on_failure": airflow_config.email_on_failure,
    "email_on_retry": airflow_config.email_on_retry,
}

with DAG(
    dag_id="register_best_model_from_mlflow",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    tags=["ml", "wine"],
    description="Регистрация лучшей модели из MLflow в Model Registry",
) as dag:
    register_task = PythonOperator(
        task_id="register_best_model",
        python_callable=register_best_model,
    )

    register_task
