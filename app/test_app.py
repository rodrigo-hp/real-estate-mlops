from fastapi.testclient import TestClient
from config import get_settings
from app import app


client = TestClient(app)
settings = get_settings()


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Property Valuation API",
        "model_version": 0.1,
    }


def test_predict_property_valuation_with_valid_api_key():
    payload = {
        "type": "casa",
        "sector": "vitacura",
        "net_usable_area": 152.0,
        "net_area": 257.0,
        "n_rooms": 3.0,
        "n_bathroom": 3.0,
        "latitude": -33.3794,
        "longitude": -70.5447,
    }
    headers = {"x-api-key": settings.API_KEY}
    response = client.post("/predict", json=payload, headers=headers)
    assert response.status_code == 200
    assert "predicted_price" in response.json()


def test_predict_property_valuation_with_invalid_api_key():
    payload = {
        "type": "casa",
        "sector": "vitacura",
        "net_usable_area": 152.0,
        "net_area": 257.0,
        "n_rooms": 3.0,
        "n_bathroom": 3.0,
        "latitude": -33.3794,
        "longitude": -70.5447,
    }
    headers = {"x-api-key": "your_invalid_api_key"}
    response = client.post("/predict", json=payload, headers=headers)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API Key"


def test_predict_property_valuation_with_missing_api_key():
    payload = {
        "type": "house",
        "sector": "residential",
        "net_usable_area": 100.0,
        "net_area": 150.0,
        "n_rooms": 3.0,
        "n_bathroom": 2.0,
        "latitude": 51.5,
        "longitude": -0.1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API Key"


def test_predict_property_valuation_with_missing_parameters():
    payload = {
        "net_usable_area": 152.0,
        "net_area": 257.0,
        "n_rooms": 3.0,
        "n_bathroom": 3.0,
        "latitude": -33.3794,
        "longitude": -70.5447,
    }
    headers = {"x-api-key": settings.API_KEY}
    response = client.post("/predict", json=payload, headers=headers)
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "field required"
