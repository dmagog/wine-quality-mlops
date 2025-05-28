# Wine Quality MLOps Project

## 📁 Структура проекта

```
wine-quality-mlops/
│
├── .github/workflows/          # CI/CD пайплайны GitHub Actions
├── dags/                       # DAG-файлы Airflow
├── data/                       # Оригинальные данные (DVC управляет содержимым)
├── mlruns/                     # Каталог логов MLflow
├── models/                     # Финальная модель (через DVC)
├── src/                        # Исходный код (тренировка, предсказание и utils)
├── tests/                      # Тесты FastAPI
├── .dvc/                       # Конфигурация DVC
├── Dockerfile                  # Docker-файл для Airflow/FastAPI
├── docker-compose.yml          # Основной docker-compose
├── requirements.txt            # Зависимости
├── api/                        # FastAPI-приложение
│   └── main.py
└── README.md                   # Описание проекта
```

## ⚙️ Запуск проекта

### Подготовка

1. Убедитесь, что установлены `Docker`, `docker-compose`, `make`.
2. Клонируйте репозиторий:
```bash
git clone https://github.com/<ваш-профиль>/wine-quality-mlops.git
cd wine-quality-mlops
```

3. Инициализация DVC и загрузка данных:
```bash
make dvc-init
make dvc-pull
```

### Запуск инфраструктуры

```bash
make up
```

Проверьте доступность:
- Airflow: http://localhost:8081
- MLflow: http://localhost:5001
- FastAPI: http://localhost:8000/docs

## 🔬 Эксперименты

- Обучение моделей LogisticRegression, DecisionTree и RandomForest с разными гиперпараметрами.
- Метрики логируются в MLflow (accuracy, F1, ROC AUC, confusion matrix).
- Каждая модель логируется отдельно через MLflow run.
- Артефакты сохраняются через DVC.

### Примеры записей в MLflow:

- Название эксперимента: `wine_quality_experiment`
- Теги: `model_name`, `dataset_version`, `dataset_path`
- Артефакты: `confusion_matrix.png`, `model.pkl`

## 🏆 Выбор лучшей модели

- DAG `register_best_model_from_mlflow.py` ищет run с максимальной `f1_score`.
- Лучшая модель копируется в `models/best_model.pkl` и отслеживается в DVC.

## 🌐 FastAPI

Предоставляет endpoint `/predict` для предсказания качества вина.
Пример запроса:
```json
POST /predict
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  ...
}
```

## ✅ CI/CD

Настроены следующие проверки:
- `ruff` — линтер
- `pytest` — тесты API
- Проверка запуска сервиса

CI/CD на GitHub Actions: `.github/workflows/api-ci.yml`

## 🔗 Merge Request

[Ссылка на Merge Request](https://github.com/<ваш-профиль>/wine-quality-mlops/pull/X)

## 🔍 Как выбирается лучшая модель

### 1. Обучение и логирование нескольких моделей

В DAG `train_ml_models` реализовано обучение сразу нескольких моделей:
- `RandomForestClassifier` с несколькими наборами гиперпараметров.
- `DecisionTreeClassifier` также с разными параметрами.

Для каждого варианта модели создаётся **отдельный эксперимент** (`mlflow.start_run()`), где логируются:
- Все гиперпараметры (`mlflow.log_params()`).
- Метрики качества: accuracy, F1, ROC-AUC, confusion matrix (`mlflow.log_metrics()` и `mlflow.log_figure()`).
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

### 4. Почему это обоснованный подход

- ✅ **Автоматизация**: процесс полностью автоматизирован, не требует ручного сравнения моделей.
- ✅ **Прозрачность**: вся история экспериментов сохраняется в MLflow — можно вернуться, изучить, воспроизвести.
- ✅ **Честное сравнение**: выбор делается по согласованной метрике (`f1_score`), на одной и той же выборке и конфигурации.
- ✅ **Воспроизводимость**: всё сохраняется в MLflow и DVC, включая модели и данные.