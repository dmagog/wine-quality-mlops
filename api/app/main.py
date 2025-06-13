from fastapi import FastAPI
from app.api.endpoints import health, model, predict
from app.core.model import model_manager

app = FastAPI(title="Wine Quality Predictor")

# Регистрация роутеров
app.include_router(health.router, tags=["health"])
app.include_router(model.router, tags=["model"])
app.include_router(predict.router, tags=["predict"])

@app.on_event("startup")
def startup_event():
    model_manager.load_model() 