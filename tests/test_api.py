from fastapi.testclient import TestClient
import pytest
from unittest.mock import patch, MagicMock
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

# def test_predict_valid_input():
#     payload = {
#         "question": "ગુજરાતની રાજધાની કઈ છે?",
#         "context": "ગાંધીનગર ગુજરાતની રાજધાની છે. ગુજરાત ભારતનું એક રાજ્ય છે."
#     }
#     response = client.post("/predict", json=payload)
#     assert response.status_code == 200
#     data = response.json()
#     assert "answer" in data
#     assert "confidence" in data

def test_predict_valid_input():
    # Mock GujaratiQAModel so no real model loads
    mock_model = MagicMock()
    mock_model.answer.return_value = {
        "answer": "ગાંધીનગર",
        "confidence": 0.95,
        "answer_start": 0
    }

    with patch("src.api.main.model", mock_model):
        payload = {
            "question": "ગુજરાતની રાજધાની કઈ છે?",
            "context": "ગાંધીનગર ગુજરાતની રાજધાની છે. ગુજરાત ભારતનું એક રાજ્ય છે."
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "confidence" in data

def test_predict_missing_question():
    payload = {"context": "Some context"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error

def test_predict_short_question():
    payload = {"question": "Hi", "context": "Some long context here for testing purposes"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # min_length=5 not met