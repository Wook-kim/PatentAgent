"""Local inference in the application environment, without engine subprocesses."""

import gc
import re

import numpy as np
from PIL import Image

from .config import Settings
from .models import resolve_model
from .schemas import StructureRegion


def release_accelerator():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class LocalInference:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.provenance = {}

    def check(self):
        import torch
        precision = self.settings.inference_precision
        if any(value <= 0 for value in (
            self.settings.ocsr_batch_size, self.settings.markush_batch_size,
            self.settings.markush_max_new_tokens,
        )):
            raise ValueError("추론 batch size와 max_new_tokens는 양수여야 합니다.")
        if self.settings.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU를 사용할 수 없습니다. NVIDIA 드라이버와 설치 환경을 확인하세요.")
            torch.empty(1, device=self.settings.device)
            with torch.cuda.device(self.settings.device):
                if precision == "bfloat16" and not torch.cuda.is_bf16_supported(
                    including_emulation=False
                ):
                    raise ValueError("이 GPU는 bfloat16을 지원하지 않습니다. float16을 사용하세요.")
        elif precision != "float32":
            raise ValueError("CPU 추론은 float32 정밀도를 사용하세요.")

    def segment(self, document, pages) -> list[StructureRegion]:
        # TensorFlow's allocator otherwise retains VRAM while PyTorch loads the
        # two recognizers. Run segmentation on CPU in this single environment.
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        from decimer_segmentation import get_expanded_masks

        regions = []
        dest = document.work_dir / "regions"
        dest.mkdir(exist_ok=True)
        for page in pages:
            page_image = document.render(page)
            with Image.open(page_image) as original:
                array = np.asarray(original.convert("RGB"))
            masks = get_expanded_masks(array)
            if masks.ndim != 3:
                raise ValueError(f"DECIMER mask 형식이 잘못되었습니다: {masks.shape}")
            for idx in range(masks.shape[2]):
                y, x = np.where(masks[:, :, idx])
                if not len(x):
                    continue
                x1, y1 = max(0, int(x.min()) - 5), max(0, int(y.min()) - 5)
                x2, y2 = min(array.shape[1], int(x.max()) + 6), min(array.shape[0], int(y.max()) + 6)
                structure_id = f"{page}_{idx}"
                image = dest / f"{structure_id}.png"
                crop = array[y1:y2, x1:x2].copy()
                crop[~masks[y1:y2, x1:x2, idx].astype(bool)] = 255
                Image.fromarray(crop).save(image)
                regions.append(StructureRegion(
                    structure_id=structure_id, page=page, bbox=(x1, y1, x2, y2),
                    image=str(image), page_image=str(page_image),
                ))
        return regions

    def recognize(self, regions):
        if not regions:
            return
        from ._vendor.glyph.ocsr.postprocess import postprocess_smiles
        from ._vendor.glyph.ocsr.predict import OCSRPredictor

        checkpoint, provenance = resolve_model(self.settings, "ocsr")
        self.provenance["ocsr"] = provenance
        precision = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}
        model = None
        try:
            model = OCSRPredictor(
                checkpoint=checkpoint, device=self.settings.device,
                precision=precision[self.settings.inference_precision])
            size = self.settings.ocsr_batch_size
            for start in range(0, len(regions), size):
                batch = regions[start:start + size]
                # Preserve decoder text before upstream cleanup/canonicalization.
                predictions = model.predict_batch(
                    [r.image for r in batch], batch_size=size, postprocess=None)
                for region, raw in zip(batch, predictions, strict=True):
                    region.ocsr_model = "OCSRGlyph"
                    region.model_provenance["ocsr"] = provenance
                    region.ocsr_raw = raw
                    region.smiles = postprocess_smiles(raw) or raw or None
                    if not region.smiles:
                        region.errors.append("OCSRGlyph returned no structure")
        finally:
            del model
            release_accelerator()

    def recognize_markush(self, regions):
        if not regions:
            return
        from ._vendor.glyph.markush.inference import MarkushPredictor

        checkpoint, provenance = resolve_model(self.settings, "markush")
        base_model, base_provenance = resolve_model(self.settings, "markush_base")
        self.provenance.update(markush=provenance, markush_base=base_provenance)
        model = None
        try:
            model = MarkushPredictor(
                checkpoint=checkpoint, base_model=base_model,
                device=self.settings.device, dtype=self.settings.inference_precision,
                max_new_tokens=self.settings.markush_max_new_tokens)
            size = self.settings.markush_batch_size
            for start in range(0, len(regions), size):
                batch = regions[start:start + size]
                predictions = model.predict_many([r.image for r in batch], batch_size=size)
                for region, prediction in zip(batch, predictions, strict=True):
                    region.markush_model = "MarkushGlyph"
                    region.model_provenance.update(
                        markush=provenance, markush_base=base_provenance)
                    apply_markush_prediction(region, prediction)
        finally:
            del model
            release_accelerator()


# MarkushGlyph emits ``[\X]`` tokens that are not part of the cxsmiles_opt format, so
# opt_to_standard_cxsmiles() rejects them (GLYPH_PHASE0.md section 7.1). Each replacement
# keeps exactly one atom, so the atom indices in the trailing Sg:/m: sections stay valid.
_ESCAPED_TOKEN_RE = re.compile(r"\[\\([^\]]+)\]")
_CONCRETE_SUPERATOM = {"CH3": "C", "CH2": "C", "CH": "C", "OH": "O",
                       "NH": "N", "NH2": "N", "SH": "S"}


def repair_markush_opt(cxsmiles_opt):
    """Rewrite non-standard ``[\\X]`` tokens; return ``(repaired_opt, tokens)``.

    ``[\\*]`` is an attachment point (its index matched ``molViewConnectionPoint``
    in 8/8 Phase 0 cases), concrete superatoms become their atom, and any other
    label is kept as an ``<r>`` variable so no label is lost.
    """
    tokens = []

    def sub(match):
        label = match.group(1).strip()
        tokens.append(match.group(0))
        if label == "*":
            return "<r>_AP</r>"
        return _CONCRETE_SUPERATOM.get(label, f"<r>{label}</r>")

    return _ESCAPED_TOKEN_RE.sub(sub, cxsmiles_opt), tokens


def _try_repair(prediction):
    from ._vendor.glyph.markush.conversion import opt_to_standard_cxsmiles
    from .postprocess import _rdkit_canonical

    repaired, tokens = repair_markush_opt(prediction.cxsmiles_opt)
    if not tokens:
        return None, tokens
    try:
        converted = opt_to_standard_cxsmiles(repaired)
    except Exception:
        return None, tokens
    return (converted if _rdkit_canonical(converted)[0] is True else None), tokens


def apply_markush_prediction(region, prediction):
    """Retain evidence and report malformed output.

    Only the documented non-standard token rewrite is attempted; a repaired
    structure keeps the raw output and an error entry so it is always reviewed.
    """
    from ._vendor.glyph.markush.common import parse_stable
    from .postprocess import _rdkit_canonical

    region.markush_raw = prediction.raw
    region.markush_stable_raw = prediction.stable
    region.cxsmiles_opt = prediction.cxsmiles_opt or None
    region.cxsmiles = prediction.cxsmiles or None
    region.substituents = parse_stable(prediction.stable)
    region.markush_conversion_status = "ok"
    if "<cxsmi>" in prediction.raw and "</cxsmi>" not in prediction.raw:
        region.errors.append("MarkushGlyph output truncated: incomplete cxsmi element")
    if "<markush>" in prediction.raw and "</markush>" not in prediction.raw:
        region.errors.append("MarkushGlyph output truncated: incomplete markush element")
    if not prediction.cxsmiles_opt:
        region.markush_conversion_status = "empty"
        region.errors.append("MarkushGlyph returned no structure")
    elif prediction.conversion_error or not prediction.cxsmiles:
        error = prediction.conversion_error or "empty conversion"
        repaired, tokens = _try_repair(prediction)
        if repaired:
            region.cxsmiles = repaired
            region.markush_conversion_status = "repaired"
            region.errors.append(
                "MarkushGlyph non-standard tokens repaired, review required: "
                + ", ".join(sorted(set(tokens))) + f" (original error: {error})")
        else:
            region.markush_conversion_status = "failed"
            region.errors.append("MarkushGlyph CXSMILES conversion: " + error)
    elif _rdkit_canonical(prediction.cxsmiles)[0] is not True:
        region.markush_conversion_status = "invalid"
        region.errors.append("MarkushGlyph converted CXSMILES does not parse")
    if prediction.stable and any(
        not label or not alternatives for label, alternatives in region.substituents.items()
    ):
        region.errors.append("MarkushGlyph substituent table contains incomplete rows")
