import pandas as pd
from app.core.model import model_manager
from app.schemas.wine import WineFeatures, feature_mapping

class PredictionService:
    @staticmethod
    def predict_quality(features: WineFeatures) -> float:
        """
        Предсказание качества вина на основе входных параметров
        
        Args:
            features: Параметры вина
            
        Returns:
            float: Предсказанное качество вина
            
        Raises:
            RuntimeError: Если модель не загружена
        """
        if not model_manager.is_model_loaded():
            raise RuntimeError("Модель не загружена")
            
        df = pd.DataFrame([features.dict()])
        df.rename(columns=feature_mapping, inplace=True)
        prediction = model_manager.get_model().predict(df)[0]
        return float(prediction)

prediction_service = PredictionService() 