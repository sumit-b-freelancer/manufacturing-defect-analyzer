import torch
import pytest
from src.models.model import SentinelModel
from fastapi.testclient import TestClient
from src.api import app

def test_model_creation():
    model = SentinelModel(model_name='tf_efficientnetv2_s', num_classes=2, pretrained=False)
    assert model is not None
    print("Model created successfully")

def test_model_forward_pass():
    model = SentinelModel(model_name='tf_efficientnetv2_s', num_classes=2, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 2)
    print("Forward pass successful")

def test_api_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    print("API health check passed")
