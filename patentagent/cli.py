"""The single public entry point for extraction and review."""

import argparse
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

from .config import Settings
from .models import model_specs
from .schemas import RunOptions


def doctor(settings):
    checks = {}
    try:
        settings.require_llm()
        checks["llm_config"] = {"ok": True, "model": settings.llm_model}
    except ValueError as exc:
        checks["llm_config"] = {"ok": False, "error": str(exc)}
    for module in ("torch", "torchvision", "timm", "transformers", "peft",
                   "accelerate", "qwen_vl_utils", "tensorflow", "decimer_segmentation"):
        checks[module] = {"ok": importlib.util.find_spec(module) is not None}
    try:
        from .inference import LocalInference
        LocalInference(settings).check()
        checks["device"] = {"ok": True, "name": settings.device}
    except Exception as exc:
        checks["device"] = {"ok": False, "error": str(exc)}
    for name, spec in model_specs(settings).items():
        local = Path(spec["source"]).expanduser().exists()
        checks[f"model_{name}"] = {
            **spec, "revision": None if local else spec["revision"], "local": local,
            "note": "가중치 로딩·API 호출은 실제 extract 실행 시 검증합니다.",
        }
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if all(item.get("ok", True) for item in checks.values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="PatentAgent 통합 추출·검토 시스템")
    subs = parser.add_subparsers(dest="command", required=True)
    extract = subs.add_parser("extract", help="특허 PDF 추출")
    extract.add_argument("pdf", type=Path)
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("--structure-pages")
    extract.add_argument("--assay-pages")
    extract.add_argument("--assay-names", default="")
    extract.add_argument("--auto-pages", action=argparse.BooleanOptionalAction, default=True)
    extract.add_argument("--markush", action=argparse.BooleanOptionalAction, default=True)
    extract.add_argument("--device", help="예: cuda:0, cuda:1")
    extract.add_argument("--replay", type=Path, help="extraction.json으로 후처리 재생 (추론 생략)")
    serve = subs.add_parser("serve", help="작업 큐·검토 웹 서버")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    subs.add_parser("doctor", help="설정·패키지·GPU 점검")
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    if args.command == "doctor":
        return doctor(settings)
    if args.command == "serve":
        import uvicorn
        uvicorn.run("gateway:app", host=args.host, port=args.port, workers=1)
        return 0
    from .pipeline import run
    if args.device:
        settings = replace(settings, device=args.device)
    options = RunOptions(
        structure_pages=args.structure_pages,
        assay_pages=args.assay_pages,
        assay_names=[name.strip() for name in args.assay_names.split(",") if name.strip()],
        auto_pages=args.auto_pages,
        markush=args.markush,
    )
    run(args.pdf, args.out, settings, options, replay=args.replay)
    return 0
