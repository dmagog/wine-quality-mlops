import pytest
from httpx import AsyncClient
from api.main import app

@pytest.mark.asyncio
async def test_healthcheck():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/healthcheck")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_predict():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/predict", json={
            "fixed_acidity": 7.4,
            "volatile_acidity": 0.7,
            "citric_acid": 0.0,
            "residual_sugar": 1.9,
            "chlorides": 0.076,
            "free_sulfur_dioxide": 11.0,
            "total_sulfur_dioxide": 34.0,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4
        })
    assert response.status_code == 200
    json_response = response.json()
    assert "predicted_quality" in json_response
    assert isinstance(json_response["predicted_quality"], float)