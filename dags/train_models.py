from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow
from mlflow.data import from_pandas
from typing import Dict, Any
import logging

from utils.config import mlflow_config, dvc_config, model_config, airflow_config
from utils.notifications import send_failure_notification

logger = logging.getLogger(__name__)

def load_data_from_dvc() -> pd.DataFrame:
    """
    Загружает данные из DVC
    
    Returns:
        pd.DataFrame: Загруженный датасет
        
    Raises:
        RuntimeError: Если не удалось загрузить данные
    """
    try:
        result = subprocess.run(
            ["dvc", "pull", dvc_config.data_file_dvc],
            cwd="/app",
            capture_output=True,
            text=True
        )
        logger.info("📦 Выполняем dvc pull...")
        logger.info(f"STDOUT: {result.stdout}")
        logger.info(f"STDERR: {result.stderr}")
        result.check_returncode()
        return pd.read_csv(f"/app/{dvc_config.data_file}")
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке данных: {str(e)}")
        raise RuntimeError(f"Не удалось загрузить данные: {str(e)}")

def get_dataset_version() -> str:
    """
    Получает версию датасета из git
    
    Returns:
        str: Хеш коммита
        
    Raises:
        RuntimeError: Если не удалось получить версию
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd="/app",
            capture_output=True,
            text=True
        )
        result.check_returncode()
        return result.stdout.strip()
    except Exception as e:
        logger.error(f"❌ Ошибка при получении версии датасета: {str(e)}")
        raise RuntimeError(f"Не удалось получить версию датасета: {str(e)}")

def train_and_log_models(**context) -> None:
    """
    Обучает модели и логирует их в MLflow
    
    Args:
        **context: Контекст выполнения задачи Airflow
        
    Raises:
        RuntimeError: При ошибках в процессе обучения
    """
    try:
        os.makedirs(dvc_config.artifacts_dir, exist_ok=True)

        df = load_data_from_dvc()
        X = df.drop(model_config.target_column, axis=1)
        y = df[model_config.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=model_config.test_size, 
            random_state=model_config.random_state
        )

        models = {
            "LogisticRegression_l1_C_0.1": LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=1000),
            "LogisticRegression_l2_C_1.0": LogisticRegression(penalty="l2", solver="liblinear", C=1.0, max_iter=1000),
            "LogisticRegression_l2_C_10.0": LogisticRegression(penalty="l2", solver="liblinear", C=10.0, max_iter=1000),
            "DecisionTree_depth_3": DecisionTreeClassifier(max_depth=3),
            "DecisionTree_depth_5": DecisionTreeClassifier(max_depth=5),
            "DecisionTree_gini_depth_7": DecisionTreeClassifier(criterion="gini", max_depth=7),
            "DecisionTree_entropy_depth_7": DecisionTreeClassifier(criterion="entropy", max_depth=7),
            "RandomForest_50": RandomForestClassifier(n_estimators=50, max_depth=5),
            "RandomForest_100": RandomForestClassifier(n_estimators=100, max_depth=7),
            "RandomForest_200_depth_None": RandomForestClassifier(n_estimators=200, max_depth=None),
        }

        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        mlflow.set_experiment(mlflow_config.experiment_name)

        dataset_version = get_dataset_version()

        for name, model in models.items():
            logger.info(f"📦 Logging to MLflow at: {mlflow.get_tracking_uri()}")
            with mlflow.start_run(run_name=name):
                mlflow.set_tag("model_class", model.__class__.__name__)
                mlflow.set_tag("dataset_name", dvc_config.data_file)
                mlflow.log_param("dataset_path", f"/app/{dvc_config.data_file}")
                mlflow.log_param("dataset_git_version", dataset_version)

                mlflow.log_param("model_name", name)
                mlflow.log_params(model.get_params())

                dataset_input = from_pandas(df, source=f"/app/{dvc_config.data_file}", name=dvc_config.data_file)
                mlflow.log_input(dataset_input)

                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
                mlflow.log_metric("f1_score", f1_score(y_test, preds, average="weighted"))
                mlflow.log_metric("precision", precision_score(y_test, preds, average="weighted"))
                mlflow.log_metric("recall", recall_score(y_test, preds, average="weighted"))

                report = classification_report(y_test, preds)
                report_path = f"{dvc_config.artifacts_dir}/{name}_report.txt"
                with open(report_path, "w") as f:
                    f.write(report)
                mlflow.log_artifact(report_path)

                model_path = f"{dvc_config.artifacts_dir}/{name}.pkl"
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path)

                logger.info(f"✅ Модель {name} сохранена и залогирована: {model_path}")

    except Exception as e:
        logger.error(f"❌ Ошибка при обучении моделей: {str(e)}")
        if airflow_config.email_on_failure:
            send_failure_notification(
                task_instance=context['task_instance'],
                error=e,
                context=context,
                email_to=airflow_config.alert_email
            )
        raise RuntimeError(f"Не удалось обучить модели: {str(e)}")

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
    dag_id="train_ml_models",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    tags=["ml", "wine"],
    description="Обучение и логирование моделей в MLflow",
) as dag:
    train_task = PythonOperator(
        task_id="train_and_log_models",
        python_callable=train_and_log_models,
    )

    train_task
