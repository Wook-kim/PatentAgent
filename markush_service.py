#!/usr/bin/env python3
"""
MarkushGrapher REST 마이크로서비스
==================================

MarkushGrapher 를 HTTP 서비스로 격리한다. 통합 파이프라인(integrate_prototype.py)이
subprocess 로 inference.sh 를 매번 호출(모델 매번 로드, GPU 경합)하는 대신,
이 서비스에 이미지를 POST 하면 CXSMILES 를 받는다.

설계 노트:
- MarkushGrapher 는 venv 2개(chemicalocr-env: vllm / markushgrapher-env: fork)가
  분리돼 있어 한 프로세스에 두 모델을 동시에 올릴 수 없다.
- 따라서 이 서비스는 "요청을 받아 inference.sh 를 호출하고 predictions JSONL 을
  파싱해 반환"하는 격리 래퍼다. 환경 충돌·GPU 경합을 서비스 단에서 직렬화(lock)한다.
- 향후 모델 상주(warm) 최적화는 두 venv 를 각각 별도 서버로 쪼개 RPC 하는 방식으로 확장 가능.

실행 (chemicalocr-env 에 fastapi/uvicorn 있음):
    cd /data1/wook_workspace/PatentAgent/MarkushGrapher
    PATH=/home/wkim/.local/bin:$PATH CUDA_VISIBLE_DEVICES=1 \
        chemicalocr-env/bin/python ../markush_service.py --port 8100

API:
    GET  /health                       -> {"status": "ok"}
    POST /predict  (multipart: files)  -> {"results": {"<stem>": {"cxsmiles","cxsmiles_opt"}}}
    POST /predict_dir {"image_dir": ".."} -> 동일
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn

MG_DIR = Path("/data1/wook_workspace/PatentAgent/MarkushGrapher")
PA_ROOT = MG_DIR.parent.resolve()
INFERENCE_SH = MG_DIR / "scripts" / "inference" / "inference.sh"
EXTRA_ALLOWED_IMAGE_ROOTS = [
    Path(p).resolve()
    for p in os.environ.get("MG_ALLOWED_IMAGE_ROOTS", "").split(os.pathsep)
    if p.strip()
]

app = FastAPI(title="MarkushGrapher Service", version="0.1.0")
_lock = threading.Lock()  # GPU 경합 방지: 추론 직렬화


class PredictDirRequest(BaseModel):
    image_dir: str


def _run_inference(image_dir: Path) -> dict:
    """inference.sh 실행 후 predictions_*.jsonl 파싱 -> {stem: {cxsmiles,...}}."""
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    env.pop("CHEMICALOCR_PYTHON", None)  # 기본 chemicalocr-env(vllm) 사용
    # 새 run 디렉토리를 식별하기 위해 실행 전 기존 목록 스냅샷
    inf_root = MG_DIR / "data" / "hf" / "inference"
    before = set(inf_root.glob("*")) if inf_root.exists() else set()

    cmd = ["bash", str(INFERENCE_SH), str(image_dir)]
    proc = subprocess.run(cmd, cwd=str(MG_DIR), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = proc.stdout.decode(errors="ignore")[-1500:]
        raise HTTPException(500, f"inference.sh failed:\n{tail}")

    after = set(inf_root.glob("*"))
    new_runs = sorted(after - before, key=lambda p: p.stat().st_mtime)
    run_dir = new_runs[-1] if new_runs else sorted(
        after, key=lambda p: p.stat().st_mtime)[-1]
    pred = run_dir / "evaluation" / "predictions_1000.jsonl"
    results = {}
    if pred.exists():
        for line in open(pred):
            if line.strip():
                d = json.loads(line)
                results[d["id"]] = {
                    "cxsmiles": d.get("cxsmiles"),
                    "cxsmiles_opt": d.get("cxsmiles_opt"),
                    "gt_cxsmiles": d.get("gt_cxsmiles"),
                    "gt_cxsmiles_opt": d.get("gt_cxsmiles_opt"),
                }
    return {"results": results, "run_dir": str(run_dir)}


def _is_allowed_image_dir(path: Path) -> bool:
    """predict_dir 입력을 PatentAgent가 생성한 mg_input 계열로 제한."""
    if path.name != "mg_input":
        return False
    if PA_ROOT in path.parents:
        return True
    return any(root == path or root in path.parents for root in EXTRA_ALLOWED_IMAGE_ROOTS)


@app.get("/health")
def health():
    return {"status": "ok", "inference_sh": INFERENCE_SH.exists()}


@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    """업로드된 이미지들을 임시 디렉토리에 저장하고 추론."""
    with _lock:
        tmp = Path(tempfile.mkdtemp(prefix="mg_svc_"))
        try:
            for uf in files:
                # 파일명(stem)이 예측 id 가 됨
                dst = tmp / Path(uf.filename).name
                dst.write_bytes(await uf.read())
            return _run_inference(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@app.post("/predict_dir")
def predict_dir(req: PredictDirRequest):
    """서버 파일시스템 상의 이미지 디렉토리를 직접 추론 (대용량/로컬용)."""
    d = Path(req.image_dir).resolve()
    if not d.is_dir():
        raise HTTPException(400, f"not a directory: {d}")
    if not _is_allowed_image_dir(d):
        raise HTTPException(403, "image_dir must be a PatentAgent mg_input directory")
    with _lock:
        return _run_inference(d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
