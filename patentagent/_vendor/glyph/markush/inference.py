# Derived from EdisonScientific/glyph, Apache-2.0.
# Pinned upstream: 0bf782f863d26b041ace157668928ef07c38b972. Modified for internal inference only.
# See LICENSE.txt and SOURCES.json for provenance and modifications.
"""Public image inference API for Markush transcription."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from patentagent._vendor.glyph.markush.conversion import opt_to_standard_cxsmiles
from patentagent._vendor.glyph.markush import common

DEFAULT_MARKUSH_PROMPT = (
    "Extract the Markush structure from this image.\n"
    "Target format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; "
    "the <cxsmi> value must use canonicalized cxsmiles_opt."
)
OCSR_NO_EXTENSION_PROMPT = (
    "Extract the Markush structure from this image.\n"
    "Target format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; "
    "the <cxsmi> value must use canonicalized cxsmiles_opt.\n"
    "This image is an ordinary molecule with no Markush extension annotations; "
    "return only the base molecule in <cxsmi> and leave <stable> empty."
)
# Default model sources. These resolve from the Hugging Face Hub at the repo's
# current ``main`` (no revision is pinned here), so headline reproduction assumes
# the currently-published weights rather than a byte-exact revision.
DEFAULT_MARKUSH_ADAPTER = "EdisonScientific/MarkushGlyph"
DEFAULT_MARKUSH_BASE_MODEL = "Qwen/Qwen3.5-2B-Base"

# Default number of sampled candidates for majority-vote decoding.  The sampling
# recipe itself (temperature/top-p/seed) is owned by ``eval.fast_eval`` and
# resolved there so MV@8 reproduces the eval headline (see ``predict``).
DEFAULT_VOTE_K = 8








def _resolve_adapter(checkpoint):
    if not checkpoint or not (Path(checkpoint) / 'adapter_config.json').is_file():
        raise ValueError('Expected a local MarkushGlyph LoRA snapshot')
    return str(checkpoint)


def _resolve_base_model(base_model: str | Path | None) -> str:
    """Resolve the base model id, honoring ``MARKUSH_BASE_MODEL`` then the default."""

    if base_model:
        return str(base_model)
    return os.environ.get("MARKUSH_BASE_MODEL") or DEFAULT_MARKUSH_BASE_MODEL


@dataclass(slots=True)
class MarkushPrediction:
    """Normalized result returned by :class:`MarkushPredictor`."""

    image: str
    raw: str
    markush_xml: str
    cxsmiles_opt: str
    cxsmiles: str
    stable: str
    vote: dict[str, Any] = field(default_factory=dict)
    conversion_error: str | None = None

    @classmethod
    def from_raw(
        cls,
        image: str | Path,
        raw: str,
        *,
        vote: dict[str, Any] | None = None,
    ) -> MarkushPrediction:
        """Normalize raw model output into the public result shape."""

        cleaned = common.clean_prediction(raw)
        cxsmiles_opt, stable = common.extract_cxsmiles_and_stable(cleaned, require_wrapper=False)
        if cxsmiles_opt:
            markush_xml = (
                f"<markush><cxsmi>{cxsmiles_opt}</cxsmi><stable>{stable}</stable></markush>"
            )
        else:
            cxsmiles_opt = common.prediction_to_cxsmiles_opt(cleaned)
            stable = ""
            markush_xml = (
                f"<markush><cxsmi>{cxsmiles_opt}</cxsmi><stable></stable></markush>"
                if cxsmiles_opt
                else ""
            )
        converted, conversion_error = "", None
        if cxsmiles_opt:
            try:
                converted = opt_to_standard_cxsmiles(cxsmiles_opt)
            except Exception as exc:
                conversion_error = f"{type(exc).__name__}: {exc}"
        return cls(
            image=str(image),
            raw=raw,
            markush_xml=markush_xml,
            cxsmiles_opt=cxsmiles_opt,
            cxsmiles=converted,
            conversion_error=conversion_error,
            stable=stable,
            vote=dict(vote) if vote else {},
        )

    def asdict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


def _generate_greedy_batch(
    model: Any,
    processor: Any,
    images: list[str | Path],
    *,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> list[str]:
    """Generate one greedy completion per image with a single model call."""

    import torch
    from qwen_vl_utils import process_vision_info

    from patentagent._vendor.glyph.markush.loader import (
        ENDOFTEXT_TOKEN_ID,
        IM_END_TOKEN_ID,
        _build_messages,
        _processor_image_patch_size,
        _processor_image_pixel_kwargs,
    )
    from patentagent._vendor.glyph.markush.padding import temporary_tokenizer_padding_side

    image_patch_size = _processor_image_patch_size(processor)
    image_pixel_kwargs = _processor_image_pixel_kwargs(processor)
    message_batches = [
        _build_messages(
            str(image),
            prompt,
            image_pixel_kwargs=image_pixel_kwargs,
        )
        for image in images
    ]

    text_prompts: list[str] = []
    for messages in message_batches:
        try:
            text_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                text_prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                text_prompt = processor.apply_chat_template(messages, tokenize=False)
        text_prompts.append(text_prompt)

    image_inputs, _ = process_vision_info(
        message_batches,
        image_patch_size=image_patch_size,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    with temporary_tokenizer_padding_side(processor, "left"):
        inputs = processor(
            text=text_prompts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        prompt_len = inputs["input_ids"].shape[1]
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(
            tokenizer, "eos_token_id", None
        )

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
                eos_token_id=[ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID],
                use_cache=True,
            )

        generated = output_ids[:, prompt_len:]
        raw_predictions = [
            tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in generated
        ]

    if len(raw_predictions) != len(images):
        raise RuntimeError(
            f"Expected {len(images)} generated sequence(s), got {len(raw_predictions)}"
        )
    return raw_predictions


class MarkushPredictor:
    """Lazy single-image and batched predictor for a Markush VLM checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        base_model: str | Path | None = None,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        merge_adapter: bool = True,
        max_new_tokens: int = 1024,
        prompt: str = DEFAULT_MARKUSH_PROMPT,
    ) -> None:
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.base_model = _resolve_base_model(base_model)
        self.device = device
        self.dtype = dtype
        self.merge_adapter = merge_adapter
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self._model: Any | None = None
        self._processor: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether the model and processor are already resident in memory."""

        return self._model is not None and self._processor is not None

    def load(self) -> MarkushPredictor:
        """Load the checkpoint and processor if needed.

        Resolves the adapter to a LOCAL directory first — downloading the
        default released adapter when ``checkpoint`` is omitted, or a Hugging
        Face repo id — so ``load_model_and_processor`` always sees an on-disk
        ``adapter_config.json`` and loads LoRA weights on top of the base model.
        """

        if self.is_loaded:
            return self

        resolved_adapter = _resolve_adapter(self.checkpoint)
        self.checkpoint = resolved_adapter

        from patentagent._vendor.glyph.markush.loader import load_model_and_processor

        self._model, self._processor = load_model_and_processor(
            self.base_model,
            resolved_adapter,
            self.device,
            dtype=self.dtype,
            merge_adapter=self.merge_adapter,
        )
        return self


    def predict_many(
        self,
        images: list[str | Path],
        *,
        batch_size: int = 8,
        prompt: str | None = None,
        max_new_tokens: int | None = None,
    ) -> list[MarkushPrediction]:
        """Greedily predict images in order with one model call per batch.

        Inputs are left-padded so variable prompt and image-token lengths retain
        decoder-only generation semantics. ``prompt`` overrides the predictor
        default for every image in this call.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not images:
            return []

        self.load()
        resolved_prompt = prompt or self.prompt
        resolved_max_new_tokens = max_new_tokens or self.max_new_tokens
        predictions: list[MarkushPrediction] = []
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            raw_predictions = _generate_greedy_batch(
                self._model,
                self._processor,
                batch_images,
                prompt=resolved_prompt,
                max_new_tokens=resolved_max_new_tokens,
                device=self.device,
            )
            predictions.extend(
                MarkushPrediction.from_raw(image, raw)
                for image, raw in zip(batch_images, raw_predictions, strict=True)
            )
        return predictions


__all__ = [
    "DEFAULT_MARKUSH_ADAPTER",
    "DEFAULT_MARKUSH_BASE_MODEL",
    "DEFAULT_MARKUSH_PROMPT",
    "OCSR_NO_EXTENSION_PROMPT",
    "MarkushPrediction",
    "MarkushPredictor",
]
