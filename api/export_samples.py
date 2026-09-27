"""Call every API endpoint in-process and save the JSON responses to docs/sample_responses/.

Usage:
    python -m api.export_samples
"""
import json

from fastapi.testclient import TestClient

from api.main import app
from src.config import SAMPLES_DIR

DATE = "2025-10-15"


def save(name: str, resp):
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    body = resp.json()
    (SAMPLES_DIR / f"{name}.json").write_text(json.dumps(body, indent=2, ensure_ascii=False))
    print(f"{resp.status_code}  {name}.json")
    return body


def main():
    c = TestClient(app)
    save("health", c.get("/health"))
    fleet = save("fleet_risk_high", c.get("/fleet/risk", params={"date": DATE, "level": "HIGH", "limit": 3}))
    top = fleet["components"][0]
    save("predict_by_serial", c.post("/predict", json={"as_of_date": DATE, "components": [
        {"serial_no": top["serial_no"]}]}))
    low = c.get("/fleet/risk", params={"date": DATE, "level": "LOW", "limit": 2000}).json()["components"]
    new = next((x for x in low if x["data_quality"]["low_history"]), low[0])
    save("predict_by_position_new_component", c.post("/predict", json={"as_of_date": DATE, "components": [
        {"tail_no": new["tail_no"], "position": new["position"]}]}))
    save("error_not_installed", c.post("/predict", json={"as_of_date": DATE, "components": [
        {"serial_no": "XXX-99999"}]}))
    save("error_date_out_of_range", c.get("/fleet/risk", params={"date": "2030-01-01"}))


if __name__ == "__main__":
    main()
