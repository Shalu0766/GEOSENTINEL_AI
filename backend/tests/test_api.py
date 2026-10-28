from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_reports_configured_services():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "online"
    assert isinstance(payload["model_available"], bool)
    assert payload["max_upload_mb"] > 0


def test_upload_rejects_unsupported_extension():
    response = client.post(
        "/api/v1/analyses",
        files={"file": ("payload.exe", b"not executable", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_upload_rejects_invalid_thresholds():
    response = client.post(
        "/api/v1/analyses",
        data={"confidence": "1.5", "iou": "0.45"},
        files={"file": ("image.jpg", b"placeholder", "image/jpeg")},
    )
    assert response.status_code == 422


def test_zone_requires_three_points():
    response = client.post(
        "/api/v1/zones",
        json={"name": "Invalid", "mode": "pixel", "geometry": [[1, 2], [3, 4]]},
    )
    assert response.status_code == 422
