"""API contract: endpoints respond, errors are explicit, alerts respect the 5% cap."""
import pytest
from fastapi.testclient import TestClient

from api.main import app

DATE = "2025-10-15"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["alert_cap_per_run"] == 0.05


def test_fleet_alerts_within_cap_and_explained(client):
    body = client.get("/fleet/risk", params={"date": DATE, "level": "ALERTS", "limit": 2000}).json()
    assert body["alert_rate"] <= 0.05
    assert body["returned"] == body["alerts"]
    for c in body["components"]:
        assert c["risk_level"] in ("HIGH", "MEDIUM")
        assert 1 <= len(c["top_factors"]) <= 3 and all(f["reason"] for f in c["top_factors"])


def test_predict_by_serial_and_position_agree(client):
    top = client.get("/fleet/risk", params={"date": DATE, "limit": 1}).json()["components"][0]
    a = client.post("/predict", json={"as_of_date": DATE, "components": [{"serial_no": top["serial_no"]}]}).json()[0]
    b = client.post("/predict", json={"as_of_date": DATE, "components": [
        {"tail_no": top["tail_no"], "position": top["position"]}]}).json()[0]
    assert a["risk_score"] == b["risk_score"] == top["risk_score"]


def test_errors(client):
    assert client.post("/predict", json={"as_of_date": DATE, "components": [{"serial_no": "XXX"}]}).status_code == 404
    assert client.post("/predict", json={"as_of_date": DATE, "components": [{}]}).status_code == 422
    assert client.get("/fleet/risk", params={"date": "2030-01-01"}).status_code == 422
