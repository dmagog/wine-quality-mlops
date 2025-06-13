from fastapi import APIRouter, HTTPException
from app.schemas.wine import WineFeatures
from app.services.prediction import prediction_service

router = APIRouter()

@router.post("/predict")
def predict(features: WineFeatures):
    try:
        prediction = prediction_service.predict_quality(features)
        return {"predicted_quality": prediction}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) 