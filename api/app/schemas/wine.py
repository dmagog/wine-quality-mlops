from pydantic import BaseModel, Field

class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., ge=0, le=20)
    volatile_acidity: float = Field(..., ge=0, le=2)
    citric_acid: float = Field(..., ge=0, le=1)
    residual_sugar: float = Field(..., ge=0, le=15)
    chlorides: float = Field(..., ge=0, le=1)
    free_sulfur_dioxide: float = Field(..., ge=0, le=100)
    total_sulfur_dioxide: float = Field(..., ge=0, le=250)
    density: float = Field(..., ge=0.990, le=1.005)
    pH: float = Field(..., ge=0, le=14)
    sulphates: float = Field(..., ge=0, le=2)
    alcohol: float = Field(..., ge=0, le=20)

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