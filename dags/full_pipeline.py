from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import logging

from utils.config import airflow_config
from utils.notifications import send_failure_notification

logger = logging.getLogger(__name__)

def on_failure_callback(context):
    """
    Callback для обработки ошибок в DAG
    
    Args:
        context: Контекст выполнения DAG
    """
    task_instance = context['task_instance']
    error = context.get('exception')
    
    logger.error(f"❌ Ошибка в DAG {context['dag'].dag_id}: {str(error)}")
    
    if airflow_config.email_on_failure:
        send_failure_notification(
            task_instance=task_instance,
            error=error,
            context=context,
            email_to=airflow_config.alert_email
        )

default_args = {
    "owner": airflow_config.owner,
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": airflow_config.retries,
    "retry_delay": timedelta(minutes=airflow_config.retry_delay_minutes),
    "email_on_failure": airflow_config.email_on_failure,
    "email_on_retry": airflow_config.email_on_retry,
    "on_failure_callback": on_failure_callback,
}

with DAG(
    dag_id="full_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    tags=["ml", "wine"],
    description="Полный пайплайн обучения и регистрации моделей",
) as dag:
    train_models = TriggerDagRunOperator(
        task_id="train_models",
        trigger_dag_id="train_ml_models",
        wait_for_completion=True,
        poke_interval=60,
        timeout=3600,
    )

    register_best_model = TriggerDagRunOperator(
        task_id="register_best_model",
        trigger_dag_id="register_best_model_from_mlflow",
        wait_for_completion=True,
        poke_interval=60,
        timeout=3600,
    )

    train_models >> register_best_model
