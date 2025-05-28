# Wine Quality MLOps Project

## 🌍 Описание

Проект посвящён построению полного MLOps-пайплайна для задачи предсказания качества вина. Используются современные инструменты: Airflow, MLflow, DVC, FastAPI, MinIO, GitHub Actions.

Цель: отследить эксперименты, выбрать лучшую модель, развернуть API для инференса и настроить CI/CD.

---

## 🗂️ Структура проекта

```
wine-quality-mlops/
├── .github/workflows/          # GitHub Actions workflows (CI/CD)
├── api/                        # FastAPI-приложение
│   └── main.py
├── dags/                       # Airflow DAGs
│   ├── train_models.py
│   ├── register_best_model_from_mlflow.py
│   └── full_training_pipeline.py
├── mlruns/                     # MLflow метаданные и артефакты (монтируется в контейнер)
├── models/                     # Локально сохранённые модели (через DVC)
├── src/                        # Основной код проекта
│   ├── train.py                # Обучение и логирование моделей
│   └── utils.py                # Загрузка данных, утилиты
├── tests/                      # Тесты FastAPI
│   └── test_api.py
├── Dockerfile                  # Общий Dockerfile
├── docker-compose.yml          # Сборка и запуск всех сервисов
├── requirements.txt
└── README.md                   # Описание проекта
```

---

## 🔢 Компоненты и запуск

### 1. Сборка и запуск всех сервисов:

```bash
docker-compose up --build
```

После запуска доступны:

* Airflow: [http://localhost:8081](http://localhost:8081)
* MLflow: [http://localhost:5001](http://localhost:5001)
* FastAPI: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. DVC

Для загрузки данных:

```bash
dvc pull
```

Для добавления новой модели:

```bash
dvc add models/best_model.pkl
```

### 3. FastAPI endpoints

* `GET /healthcheck` — статус сервиса
* `POST /predict` — предсказание по параметрам вина

### 4. CI/CD

При push в репозиторий автоматически запускаются:

* линтер (`ruff`)
* unit-тесты (`pytest`)
* проверка API (`FastAPI TestClient`)

---

## 🎓 Эксперименты

Обучаются несколько моделей (LogisticRegression, DecisionTree и RandomForest) с разными гиперпараметрами. Каждая модель логируется в MLflow:

* accuracy, f1, AUC-ROC, confusion matrix
* теги: название модели, версия датасета, путь к артефакту

Выбор лучшей модели происходит автоматически на основе f1-метрики. Победитель сохраняется в `models/best_model.pkl` и фиксируется в DVC.

## 🔍 Как выбирается лучшая модель

### 1. Обучение и логирование нескольких моделей

В DAG `train_ml_models` реализовано обучение сразу нескольких моделей с несколькими наборами гиперпараметров:

Для каждого варианта модели создаётся **отдельный эксперимент** (`mlflow.start_run()`), где логируются:
- Все гиперпараметры (`mlflow.log_params()`).
- Метрики качества: accuracy, F1, ROC-AUC, confusion matrix.
- Артефакты: сериализованная модель (`mlflow.log_artifact()`).
- Версия и путь датасета, загруженного из DVC (`mlflow.set_tag()`).

Таким образом, **каждое обучение — это отдельный `run` в MLflow**, и все они связаны с одним именем эксперимента, например, `"wine_quality_experiment"`.

### 2. DAG для выбора лучшей модели

Следующий DAG — `register_best_model_from_mlflow.py` — анализирует все `runs` внутри указанного эксперимента.

#### Что происходит:
- Через `mlflow.search_runs()` загружаются все выполненные `runs` из MLflow.
- Фильтрация по статусу: используются только `FINISHED` эксперименты.
- Все метрики собираются в DataFrame, далее сортируются по целевой метрике (у вас это `f1_score`).
- Лучший `run` определяется как тот, у которого `f1_score` максимален.

```python
runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string="attributes.status = 'FINISHED'")
best_run = runs_df.sort_values("metrics.f1_score", ascending=False).iloc[0]
```

### 3. Регистрация и сохранение модели

После определения лучшего `run`:
- Из MLflow извлекается путь к модели.
- Модель копируется в `artifacts/best_model.pkl`.
- Файл добавляется в DVC (`dvc add`, `dvc push`) и фиксируется в Git (`git commit`, `git push`).