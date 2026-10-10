from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_top_200_requires_batch():
    response = TestClient(app).get("/api/v1/alerts/top-200")
    assert response.status_code == 400


def test_artifact_endpoint_only_serves_whitelisted_files():
    client = TestClient(app)
    assert client.get("/api/v1/artifacts/model.joblib").status_code == 404
    assert client.get("/api/v1/artifacts/metadata.json").status_code == 404
    assert client.get("/api/v1/artifacts/..%2Fmetadata.json").status_code == 404


def test_missing_report_returns_clear_unavailable_error(tmp_path, monkeypatch):
    import backend.app.main as main

    monkeypatch.setattr(main, "ARTIFACT_DIR", tmp_path)
    response = TestClient(app).get("/api/v1/evaluation")
    assert response.status_code == 503
    assert "scripts/train.py" in response.json()["detail"]
