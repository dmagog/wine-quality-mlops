from app.core.model import model_manager
from app.schemas.wine import WineFeatures

class ModelInfoService:
    @staticmethod
    def get_model_info() -> dict:
        """
        Получение информации о модели
        
        Returns:
            dict: Информация о модели
            
        Raises:
            RuntimeError: Если модель не загружена
        """
        if not model_manager.is_model_loaded():
            raise RuntimeError("Модель не загружена")
            
        return {
            "model_type": type(model_manager.get_model()).__name__,
            "features": list(WineFeatures.schema()["properties"].keys())
        }

model_info_service = ModelInfoService() 