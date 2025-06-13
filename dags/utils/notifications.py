from airflow.models import TaskInstance
from airflow.utils.email import send_email
from typing import Optional
import os

def send_failure_notification(
    task_instance: TaskInstance,
    error: Exception,
    context: dict,
    email_to: Optional[str] = None
) -> None:
    """
    Отправляет уведомление о сбое задачи
    
    Args:
        task_instance: Экземпляр задачи Airflow
        error: Исключение, вызвавшее сбой
        context: Контекст выполнения задачи
        email_to: Email для отправки уведомления (если None, берется из конфигурации)
    """
    if email_to is None:
        email_to = os.getenv("AIRFLOW_ALERT_EMAIL", "admin@example.com")
    
    dag_id = task_instance.dag_id
    task_id = task_instance.task_id
    execution_date = context.get('execution_date')
    
    subject = f'Airflow Alert: {dag_id}.{task_id} Failed'
    
    html_content = f"""
    <h3>Airflow Task Failure</h3>
    <p><b>DAG:</b> {dag_id}</p>
    <p><b>Task:</b> {task_id}</p>
    <p><b>Execution Date:</b> {execution_date}</p>
    <p><b>Error:</b> {str(error)}</p>
    <p><b>Log URL:</b> {task_instance.log_url}</p>
    """
    
    send_email(
        to=email_to,
        subject=subject,
        html_content=html_content
    ) 