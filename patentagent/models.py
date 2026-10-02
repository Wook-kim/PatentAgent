"""Resolve immutable Glyph weight snapshots; never import external checkouts."""

import hashlib
import re
from pathlib import Path

from .config import Settings

GLYPH_REVISION = "0bf782f863d26b041ace157668928ef07c38b972"


def model_specs(settings: Settings):
    return {
        "ocsr": {"name": "OCSRGlyph", "source": settings.ocsr_checkpoint,
                 "revision": settings.ocsr_revision},
        "markush": {"name": "MarkushGlyph", "source": settings.markush_checkpoint,
                    "revision": settings.markush_revision},
        "markush_base": {"name": "Qwen3.5-2B-Base", "source": settings.markush_base_model,
                         "revision": settings.markush_base_revision},
    }


def _local_path(source):
    path = Path(source).expanduser()
    if path.exists():
        return path.resolve()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", source) or source.startswith((".", "~")):
        raise FileNotFoundError(f"모델 가중치 경로가 없습니다: {source}")
    return None


def resolve_model(settings: Settings, role: str):
    spec = model_specs(settings)[role]
    source, revision = spec["source"], spec["revision"]
    path = _local_path(source)
    local = path is not None
    if local:
        if role == "ocsr" and path.is_dir():
            path = path / "model.pth"
    else:
        default_spec = model_specs(Settings())[role]
        if source != default_spec["source"] and revision == default_spec["revision"]:
            raise ValueError(f"{role}: 사용자 모델에는 해당 모델의 REVISION을 지정하세요.")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{role}: 재현 가능한 가중치를 위해 40자리 커밋 REVISION을 지정하세요.")
        from huggingface_hub import hf_hub_download, snapshot_download
        kwargs = {"repo_id": source, "revision": revision,
                  "cache_dir": str(settings.model_dir / "hub")}
        if role == "ocsr":
            path = Path(hf_hub_download(filename="model.pth", **kwargs))
        else:
            path = Path(snapshot_download(
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
                **kwargs))
    required = path if role == "ocsr" else path / (
        "adapter_config.json" if role == "markush" else "config.json")
    if not required.is_file():
        raise FileNotFoundError(f"{role} 가중치/설정 파일이 없습니다: {required}")
    provenance = {
        **spec, "revision": None if local else revision, "local": local,
        "resolved_path": str(path), "code_revision": GLYPH_REVISION,
        "precision": settings.inference_precision,
    }
    if local:
        # Do not attribute custom local weights to the published HF revision.
        # Hash every weight/config file for a reproducible local identity.
        paths = [path] if path.is_file() else sorted(
            p for p in path.rglob("*") if p.is_file() and not p.name.startswith("."))
        provenance["files_sha256"] = {
            p.name if path.is_file() else str(p.relative_to(path)): _sha256(p)
            for p in paths
        }
    return str(path), provenance


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
