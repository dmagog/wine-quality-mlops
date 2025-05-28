from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="full_training_pipeline",
    default_args=default_args,
    description="Full training pipeline: train -> select best -> register",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
) as dag:

    trigger_train = TriggerDagRunOperator(
        task_id="trigger_train_ml_models",
        trigger_dag_id="train_ml_models",
        wait_for_completion=True,
    )

    trigger_register = TriggerDagRunOperator(
        task_id="trigger_register_best_model",
        trigger_dag_id="register_best_model_from_mlflow",
        wait_for_completion=True,
    )

    trigger_train >> trigger_register
