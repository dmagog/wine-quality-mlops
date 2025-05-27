
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import subprocess
from pathlib import Path

app = FastAPI(title="Wine Quality Predictor")

MODEL_DVC_PATH = "artifacts/best_model.pkl.dvc"
MODEL_PATH = "artifacts/best_model.pkl"

# Входные параметры
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# Переименование признаков
feature_mapping = {
    "fixed_acidity": "fixed acidity",
    "volatile_acidity": "volatile acidity",
    "citric_acid": "citric acid",
    "residual_sugar": "residual sugar",
    "chlorides": "chlorides",
    "free_sulfur_dioxide": "free sulfur dioxide",
    "total_sulfur_dioxide": "total sulfur dioxide",
    "density": "density",
    "pH": "pH",
    "sulphates": "sulphates",
    "alcohol": "alcohol"
}

# Хранилище модели
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        print("📦 DVC pull модели...")
        result = subprocess.run(["dvc", "pull", MODEL_DVC_PATH], capture_output=True, text=True)
        print("📦 DVC pull stdout:", result.stdout)
        print("📦 DVC pull stderr:", result.stderr)
        result.check_returncode()

        model_file = Path(MODEL_PATH)
        if not model_file.exists():
            raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")

        model = joblib.load(model_file)
        print("✅ Модель загружена из файла")

    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        model = None

@app.get("/healthcheck")
def healthcheck():
    if model is None:
        return {"status": "error", "reason": "model not loaded"}
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return {
        "model_type": type(model).__name__,
        "features": list(WineFeatures.schema()["properties"].keys())
    }

@app.post("/predict")
def predict(features: WineFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    df = pd.DataFrame([features.dict()])
    df.rename(columns=feature_mapping, inplace=True)
    prediction = model.predict(df)[0]
    return {"predicted_quality": float(prediction)}
