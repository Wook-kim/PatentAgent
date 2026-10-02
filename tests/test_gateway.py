import importlib
import json
import sys

from fastapi.testclient import TestClient


def test_gateway_native_runtime_and_review(tmp_path, monkeypatch):
    monkeypatch.setenv("PATENTAGENT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PATENTAGENT_EXAMPLES_DIR", str(tmp_path / "examples"))
    sys.modules.pop("gateway", None)
    gateway = importlib.import_module("gateway")
    command = gateway.JM._build_cmd(tmp_path / "job", {
        "auto_pages": True, "structure_pages": "1-2",
        "markush": False, "assay_names": "IC50",
    })
    assert command[:4] == [sys.executable, "-m", "patentagent", "extract"]
    assert "--structure-pages" in command and "--no-markush" in command
    assert not any(flag in command for flag in ("--mg-service-url", "--engine", "--gpu"))
    job = gateway.JOBS_DIR / "contract-test"
    job.mkdir()
    (job / "progress.json").write_text(json.dumps([
        {"key": "activity", "label": "화합물 ID·활성값 추출", "status": "running"},
    ]))
    assert gateway._compute_steps("contract-test", {"status": "running"})[0]["state"] == "active"
    assert gateway._compute_steps("contract-test", {"status": "cancelled"})[0]["state"] == "pending"
    with TestClient(gateway.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "MolCoref" not in response.text
        assert "Markush 구조 인식" in response.text
        assert gateway.review_app.KETCHER_DIR.joinpath("index.html").is_file()
