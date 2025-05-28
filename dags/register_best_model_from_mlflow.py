
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient
import os
import shutil
import subprocess

def select_best_model_from_mlflow():
    print("🔄 Using updated DAG version — logging active.")

    tracking_uri = "file:/app/mlruns"
    artifacts_dir = "/app/artifacts"
    best_model_path = os.path.join(artifacts_dir, "best_model.pkl")

    os.makedirs(artifacts_dir, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name("wine-quality")
    if not experiment:
        raise ValueError("Эксперимент 'wine-quality' не найден")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("Нет завершённых запусков для выбора модели")

    best_run = runs[0]
    print(f"🏆 Лучшая модель: run_id={best_run.info.run_id}, f1_score={best_run.data.metrics['f1_score']}")

    artifact_root = best_run.info.artifact_uri.replace("file:/", "")

    model_path = None
    for root, dirs, files in os.walk(artifact_root):
        for file in files:
            if file.endswith(".pkl"):
                model_path = os.path.join(root, file)
                break
        if model_path:
            break

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError("Не найден .pkl файл в артефактах run-а")

    shutil.copy(model_path, best_model_path)
    print(f"✅ Модель скопирована: {best_model_path}")

    subprocess.run(["dvc", "add", best_model_path], cwd="/app", check=True)
    subprocess.run(["git", "add", f"{best_model_path}.dvc", ".gitignore"], cwd="/app", check=True)

    status = subprocess.run(["git", "status", "--porcelain"], cwd="/app", capture_output=True, text=True)
    if status.stdout.strip():
        print("📌 Выполняем git commit...")
        commit = subprocess.run(["git", "commit", "-m", "Register best model from MLflow"],
                                cwd="/app",
                                capture_output=True,
                                text=True
        )
        print("🔧 Git commit stdout:", commit.stdout)
        print("🔧 Git commit stderr:", commit.stderr)
        commit.check_returncode()

    else:
        print("ℹ️ Git чист — нечего коммитить.")

    subprocess.run(["dvc", "push"], cwd="/app", check=True)
    print("🚀 Модель добавлена в DVC и запушена")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="register_best_model_from_mlflow",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["ml", "mlflow", "register"],
) as dag:

    register_task = PythonOperator(
        task_id="select_best_model_from_mlflow",
        python_callable=select_best_model_from_mlflow,
    )

    register_task
