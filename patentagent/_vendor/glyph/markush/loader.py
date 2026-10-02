# Derived from EdisonScientific/glyph, Apache-2.0.
# Pinned upstream: 0bf782f863d26b041ace157668928ef07c38b972. Modified for internal inference only.
# See LICENSE.txt and SOURCES.json for provenance and modifications.
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import torch
LOGGER = logging.getLogger(__name__)

ENDOFTEXT_TOKEN_ID = 248044

IM_END_TOKEN_ID = 248046

def _processor_image_patch_size(processor: Any) -> int:
    """Return the processor image patch size used by qwen_vl_utils."""

    image_processor = getattr(processor, "image_processor", None)
    patch_size = getattr(image_processor, "patch_size", None)
    if patch_size is None:
        return 14
    try:
        return int(patch_size)
    except (TypeError, ValueError):
        LOGGER.warning("Unexpected processor patch_size=%r; falling back to 14", patch_size)
        return 14

def _size_value(size: Any, key: str) -> Any:
    if size is None:
        return None
    if isinstance(size, dict):
        return size.get(key)
    return getattr(size, key, None)

def _int_pixel_bound(value: Any) -> int | None:
    if value is None:
        return None
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return None
    return value_int if value_int > 0 else None

def _processor_image_pixel_kwargs(processor: Any) -> dict[str, int]:
    """Return min/max pixel kwargs for qwen_vl_utils image loading."""

    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    min_pixels = _int_pixel_bound(getattr(image_processor, "min_pixels", None))
    max_pixels = _int_pixel_bound(getattr(image_processor, "max_pixels", None))
    if min_pixels is None:
        min_pixels = _int_pixel_bound(_size_value(size, "shortest_edge"))
    if max_pixels is None:
        max_pixels = _int_pixel_bound(_size_value(size, "longest_edge"))

    kwargs: dict[str, int] = {}
    if min_pixels is not None:
        kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        kwargs["max_pixels"] = max_pixels
    return kwargs

def load_model_and_processor(
    base_model: str,
    checkpoint: str | None,
    device: str,
    *,
    dtype: str = "bfloat16",
    merge_adapter: bool = True,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = 1_048_576,
) -> tuple[Any, Any]:
    """Load a full checkpoint or the Qwen-VL base model plus optional LoRA adapter.

    Security: ``base_model``/``checkpoint`` are loaded via
    ``transformers.from_pretrained(..., trust_remote_code=False)`` and may execute
    Python code shipped in the referenced repository (and unpickle torch weights).
    Only pass checkpoints/models from sources you trust.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }
    torch_dtype = torch_dtype_map.get(dtype, "auto")

    # --- Processor ---
    processor_path = checkpoint if checkpoint else base_model
    processor_kwargs: dict[str, Any] = {}
    if image_min_pixels is not None:
        processor_kwargs["min_pixels"] = image_min_pixels
    if image_max_pixels is not None:
        processor_kwargs["max_pixels"] = image_max_pixels
    try:
        processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=False,
            **processor_kwargs,
        )
    except Exception:
        LOGGER.info(
            "Could not load processor from %s; falling back to %s", processor_path, base_model
        )
        processor = AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=False,
            **processor_kwargs,
        )
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        if image_min_pixels is not None and hasattr(image_processor, "min_pixels"):
            image_processor.min_pixels = image_min_pixels
        if image_max_pixels is not None and hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = image_max_pixels
    if processor_kwargs:
        LOGGER.info("Configured processor image pixels: %s", processor_kwargs)

    # Chat-template fix: copy from tokenizer if processor lacks one
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(processor, "chat_template", None) is None and getattr(
        tokenizer, "chat_template", None
    ):
        processor.chat_template = tokenizer.chat_template

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    checkpoint_is_adapter = bool(checkpoint) and (Path(checkpoint) / "adapter_config.json").exists()
    model_path = base_model
    if checkpoint and not checkpoint_is_adapter:
        model_path = checkpoint

    # --- Model ---
    LOGGER.info("Loading model from %s", model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    if checkpoint and checkpoint_is_adapter:
        from peft import PeftModel

        LOGGER.info("Loading LoRA adapter from %s", checkpoint)
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
        if merge_adapter:
            LOGGER.info("Merging adapter weights")
            model = model.merge_and_unload()
    elif checkpoint:
        LOGGER.info(
            "Checkpoint %s: no local adapter_config.json found; loaded directly "
            "without a PEFT merge",
            checkpoint,
        )

    model.eval()
    model.to(device)
    return model, processor

def _build_messages(
    image_path: str,
    prompt: str,
    *,
    image_pixel_kwargs: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Build the Qwen-VL chat message dict for one sample."""
    image_content: dict[str, Any] = {"type": "image", "image": image_path}
    if image_pixel_kwargs:
        image_content.update(image_pixel_kwargs)
    return [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": prompt},
            ],
        },
    ]
