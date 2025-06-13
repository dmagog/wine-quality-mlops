import joblib
import subprocess
from pathlib import Path
from .config import MODEL_DVC_PATH, MODEL_PATH

class ModelManager:
    def __init__(self):
        self.model = None

    def load_model(self):
        try:
            print("📦 DVC pull модели...")
            result = subprocess.run(["dvc", "pull", MODEL_DVC_PATH], capture_output=True, text=True)
            print("📦 DVC pull stdout:", result.stdout)
            print("📦 DVC pull stderr:", result.stderr)
            result.check_returncode()

            model_file = Path(MODEL_PATH)
            if not model_file.exists():
                raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")

            self.model = joblib.load(model_file)
            print("✅ Модель загружена из файла")
            return True

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            self.model = None
            return False

    def get_model(self):
        return self.model

    def is_model_loaded(self):
        return self.model is not None

model_manager = ModelManager() 