# test_api.py
"""
Unit tests for ModelFloat MVP api.py using FastAPI TestClient.
"""

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_upload_and_list_models():
    # Upload a model
    response = client.post(
        "/models/upload",
        files={"file": ("testmodel.bin", b"abcde")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    model_id = data["model_id"]

    # List models
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()["models"]
    assert any(m["model_id"] == model_id for m in models)

def test_inference_success():
    # Upload a model
    response = client.post(
        "/models/upload",
        files={"file": ("testmodel2.bin", b"xyz")}
    )
    model_id = response.json()["model_id"]

    # Run inference
    response = client.post(
        f"/models/{model_id}/inference",
        json={"input": "test"}
    )
    assert response.status_code == 200
    assert "Mocked inference result" in response.json()["result"]

def test_inference_model_not_found():
    response = client.post(
        "/models/nonexistent/inference",
        json={"input": "test"}
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Model not found"
