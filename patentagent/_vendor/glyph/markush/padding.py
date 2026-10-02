# Derived from EdisonScientific/glyph, Apache-2.0.
# Pinned upstream: 0bf782f863d26b041ace157668928ef07c38b972. Modified for internal inference only.
# See LICENSE.txt and SOURCES.json for provenance and modifications.
from contextlib import contextmanager
from typing import Any

@contextmanager
def temporary_tokenizer_padding_side(processor: Any, padding_side: str):
    """Temporarily set decoder generation padding without mutating training setup."""

    tokenizer = getattr(processor, "tokenizer", processor)
    missing = object()
    original = getattr(tokenizer, "padding_side", missing)
    if original is missing:
        yield
        return

    tokenizer.padding_side = padding_side
    try:
        yield
    finally:
        tokenizer.padding_side = original
