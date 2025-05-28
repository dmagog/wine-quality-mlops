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

def load_data_from_dvc() -> pd.DataFrame:
    dvc_file = "winequality-red.csv.dvc"
    csv_file = "winequality-red.csv"
    result = subprocess.run(
        ["dvc", "pull", dvc_file],
        cwd="/app",
        capture_output=True,
        text=True
    )
    print("📦 Выполняем dvc pull...")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    result.check_returncode()
    return pd.read_csv(f"/app/{csv_file}")

def get_dataset_version() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd="/app",
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def train_and_log_models():
    dataset_name = "winequality-red.csv"
    dataset_path = f"/app/{dataset_name}"

    df = load_data_from_dvc()
    X = df.drop("quality", axis=1)
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("wine-quality")

    os.makedirs("/app/artifacts", exist_ok=True)

    dataset_version = get_dataset_version()

    for name, model in models.items():
        print(f"📦 Logging to MLflow at: {mlflow.get_tracking_uri()}")
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("model_class", model.__class__.__name__)
            mlflow.set_tag("dataset_name", dataset_name)
            mlflow.log_param("dataset_path", dataset_path)
            mlflow.log_param("dataset_git_version", dataset_version)

            mlflow.log_param("model_name", name)
            mlflow.log_params(model.get_params())

            # Логируем датасет как input dataset
            dataset_input = from_pandas(df, source=dataset_path, name=dataset_name)
            mlflow.log_input(dataset_input)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
            mlflow.log_metric("f1_score", f1_score(y_test, preds, average="weighted"))
            mlflow.log_metric("precision", precision_score(y_test, preds, average="weighted"))
            mlflow.log_metric("recall", recall_score(y_test, preds, average="weighted"))

            report = classification_report(y_test, preds)
            report_path = f"/app/artifacts/{name}_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path)

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
    dag_id="train_ml_models",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    tags=["ml", "wine"],
) as dag:
    train_task = PythonOperator(
        task_id="train_and_log_models",
        python_callable=train_and_log_models,
    )

    train_task
