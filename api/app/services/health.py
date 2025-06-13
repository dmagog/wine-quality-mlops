from app.core.model import model_manager

class HealthService:
    @staticmethod
    def check_health() -> dict:
        """
        Проверка состояния сервиса
        
        Returns:
            dict: Статус сервиса
        """
        if not model_manager.is_model_loaded():
            return {"status": "error", "reason": "model not loaded"}
        return {"status": "ok"}

health_service = HealthService() 