"""Configuration contains values, never paths to external source checkouts."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path("data").resolve())
    model_dir: Path = field(default_factory=lambda: Path("models").resolve())
    device: str = "cuda:0"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = ""
    llm_api_key: str = field(default="", repr=False)
    llm_timeout: float = 180
    llm_response_format: str = "json_schema"
    dpi: int = 150
    ocsr_checkpoint: str = "EdisonScientific/OCSRGlyph"
    ocsr_revision: str = "da0d049fa56effd3a07ecb15c715efdd78d9e8a0"
    markush_checkpoint: str = "EdisonScientific/MarkushGlyph"
    markush_revision: str = "2879b0380c2687a1bbdb2312ac4a810ed4887893"
    markush_base_model: str = "Qwen/Qwen3.5-2B-Base"
    markush_base_revision: str = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
    precision: str = "auto"
    ocsr_batch_size: int = 8
    markush_batch_size: int = 4
    markush_max_new_tokens: int = 1024

    @classmethod
    def from_env(cls):
        load_dotenv(Path.cwd() / ".env")
        return cls(
            data_dir=Path(os.getenv("PATENTAGENT_DATA_DIR", "data")).expanduser().resolve(),
            model_dir=Path(os.getenv("PATENTAGENT_MODEL_DIR", "models")).expanduser().resolve(),
            device=os.getenv("PATENTAGENT_DEVICE", "cuda:0"),
            llm_base_url=os.getenv("PATENTAGENT_LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_model=os.getenv("PATENTAGENT_LLM_MODEL", ""),
            llm_api_key=os.getenv("PATENTAGENT_LLM_API_KEY", ""),
            llm_timeout=float(os.getenv("PATENTAGENT_LLM_TIMEOUT", "180")),
            llm_response_format=os.getenv("PATENTAGENT_LLM_RESPONSE_FORMAT", "json_schema"),
            dpi=int(os.getenv("PATENTAGENT_DPI", "150")),
            ocsr_checkpoint=os.getenv("PATENTAGENT_OCSR_CHECKPOINT", cls.ocsr_checkpoint),
            ocsr_revision=os.getenv("PATENTAGENT_OCSR_REVISION", cls.ocsr_revision),
            markush_checkpoint=os.getenv(
                "PATENTAGENT_MARKUSH_CHECKPOINT", cls.markush_checkpoint),
            markush_revision=os.getenv("PATENTAGENT_MARKUSH_REVISION", cls.markush_revision),
            markush_base_model=os.getenv("PATENTAGENT_MARKUSH_BASE_MODEL", cls.markush_base_model),
            markush_base_revision=os.getenv(
                "PATENTAGENT_MARKUSH_BASE_REVISION", cls.markush_base_revision),
            precision=os.getenv("PATENTAGENT_PRECISION", "auto"),
            ocsr_batch_size=int(os.getenv("PATENTAGENT_OCSR_BATCH_SIZE", "8")),
            markush_batch_size=int(os.getenv("PATENTAGENT_MARKUSH_BATCH_SIZE", "4")),
            markush_max_new_tokens=int(os.getenv("PATENTAGENT_MARKUSH_MAX_NEW_TOKENS", "1024")),
        )

    @property
    def inference_precision(self):
        if self.precision not in {"auto", "float16", "float32", "bfloat16"}:
            raise ValueError("PATENTAGENT_PRECISION: auto/float16/float32/bfloat16 중 선택하세요.")
        if self.precision == "auto":
            return "float16" if self.device.startswith("cuda") else "float32"
        return self.precision

    def require_llm(self):
        if not self.llm_model:
            raise ValueError("PATENTAGENT_LLM_MODEL을 설정하세요.")
        if not self.llm_base_url.startswith(("http://", "https://")):
            raise ValueError("PATENTAGENT_LLM_BASE_URL은 http(s) 주소여야 합니다.")
        if self.llm_response_format not in {"json_schema", "none"}:
            raise ValueError("PATENTAGENT_LLM_RESPONSE_FORMAT: json_schema/none 중 선택하세요.")
        if not 72 <= self.dpi <= 300:
            raise ValueError("PATENTAGENT_DPI는 72~300 범위여야 합니다.")
