FROM apache/airflow:2.8.3

USER root
RUN apt-get update && apt-get install -y git && apt-get clean && git config --global user.email "airflow@example.com" && git config --global user.name "AirflowBot"
# RUN apt-get update && apt-get install -y git && apt-get clean && git config --global user.email "airflow@example.com" && git config --global user.name "AirflowBot"

USER airflow
RUN pip install --no-cache-dir pandas sqlalchemy
RUN pip install --upgrade numpy setuptools mlflow
RUN pip install scikit-learn==0.24.2 --no-cache-dir
RUN pip install --no-cache-dir dvc[s3]

RUN git config --global user.email "airflow@example.com" && git config --global user.name "AirflowBot"

ENV PYTHONPATH="${PYTHONPATH}:/app/src"

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r /app/requirements.txt