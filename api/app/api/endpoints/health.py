from fastapi import APIRouter
from app.services.health import health_service

router = APIRouter()

@router.get("/healthcheck")
def healthcheck():
    return health_service.check_health() 