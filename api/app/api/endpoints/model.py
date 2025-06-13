from fastapi import APIRouter, HTTPException
from app.services.model_info import model_info_service

router = APIRouter()

@router.get("/model-info")
def model_info():
    try:
        return model_info_service.get_model_info()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) 