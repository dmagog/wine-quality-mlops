
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
from sklearn.metrics import accuracy_score, f1_score
import mlflow

def load_data_from_dvc() -> pd.DataFrame:
    dvc_file = "winequality-red.csv.dvc"
    csv_file = "winequality-red.csv"
    result = subprocess.run(
        ["dvc", "pull", dvc_file],
        cwd="/app",
        capture_output=True,
        text=True
    )
    print("🧪 Выполняем dvc pull...")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    result.check_returncode()
    return pd.read_csv(f"/app/{csv_file}")

def train_and_log_models():
    df = load_data_from_dvc()
    X = df.drop("quality", axis=1)
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(max_depth=5),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=7),
    }

    mlflow.set_tracking_uri("file:/app/mlruns")
    mlflow.set_experiment("wine-quality")

    os.makedirs("/app/artifacts", exist_ok=True)

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")

            mlflow.log_param("model_name", name)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            # Сохраняем и логируем модель
            model_path = f"/app/artifacts/{name}.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)
            print(f"✅ Модель {name} сохранена и залогирована: {model_path}")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="train_ml_models_old",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    tags=["ml", "wine"],
) as dag:
    train_task = PythonOperator(
        task_id="train_and_log_models_old",
        python_callable=train_and_log_models,
    )

    train_task
