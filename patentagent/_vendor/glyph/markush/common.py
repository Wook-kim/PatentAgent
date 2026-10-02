# Derived from EdisonScientific/glyph, Apache-2.0.
# Pinned upstream: 0bf782f863d26b041ace157668928ef07c38b972. Modified for internal inference only.
# See LICENSE.txt and SOURCES.json for provenance and modifications.
from __future__ import annotations
import re

_XML_FLAGS = re.DOTALL | re.IGNORECASE

RE_MARKUSH = re.compile(
    r"<markush>\s*<cxsmi>(.*?)</cxsmi>\s*(?:<stable>(.*?)</stable>)?\s*</markush>",
    _XML_FLAGS,
)

RE_CXSMI = re.compile(r"<cxsmi>(.*?)</cxsmi>", _XML_FLAGS)

RE_STABLE = re.compile(r"<stable>(.*?)</stable>", _XML_FLAGS)

RE_THINKING = re.compile(r"<think>.*?</think>\s*", _XML_FLAGS)

_CHAT_TEMPLATE_TOKENS: tuple[str, ...] = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|im_start|>",
)

def extract_cxsmiles_and_stable(
    text: str | None, *, require_wrapper: bool = True
) -> tuple[str, str]:
    """Extract ``(cxsmi_body, stable_body)`` from a Markush prediction string.

    Parameters
    ----------
    text:
        The raw prediction or annotation string.  ``None`` and non-string
        inputs return ``("", "")``.
    require_wrapper:
        If ``True`` (default for ``metrics_string`` semantics), only
        match content inside a complete ``<markush>...</markush>``
        envelope.  Inputs containing ``<cxsmi>`` *without* the wrapper
        return ``("", "")`` so downstream lenient-mode parsing can
        decide what to do with them.
        If ``False`` (``metrics_mcs`` / ``predict`` / ``reward``
        semantics), ``<cxsmi>`` and ``<stable>`` are matched
        independently anywhere in the text.

    Returns
    -------
    Tuple of ``(cxsmi_body, stable_body)``; either entry may be the
    empty string.
    """
    if not isinstance(text, str) or not text:
        return "", ""

    if require_wrapper:
        m = RE_MARKUSH.search(text)
        if m:
            return m.group(1).strip(), (m.group(2) or "").strip()
        return "", ""

    cxsmi_match = RE_CXSMI.search(text)
    stable_match = RE_STABLE.search(text)
    return (
        cxsmi_match.group(1).strip() if cxsmi_match else "",
        stable_match.group(1).strip() if stable_match else "",
    )

def parse_stable(stable_raw: str | None, *, permissive: bool = False) -> dict[str, list[str]]:
    """Parse a ``<stable>`` payload into ``{label: [alternatives]}``.

    Format
    ------
    ``R0:hydroxyl<n>carboxyl<ns>R1:methyl``

    * ``<ns>`` separates rows (one row per R label).
    * ``<n>`` separates alternatives within a row.
    * ``:`` separates the label from its alternatives.

    Parameters
    ----------
    stable_raw:
        Body of the ``<stable>`` tag (already extracted), or ``None``.
    permissive:
        If ``False`` (``metrics_string`` semantics): only ``:`` is a
        valid label separator.  Rows without a colon are kept as a
        labelled empty list.
        If ``True`` (``metrics_mcs`` semantics): ``=`` is also accepted
        as a label separator for mildly malformed model outputs; if a
        row contains both, the one that appears first wins.
    """
    result: dict[str, list[str]] = {}
    if not stable_raw:
        return result

    for entry in stable_raw.split("<ns>"):
        entry = entry.strip()
        if not entry:
            continue

        colon = entry.find(":")
        if permissive:
            equals = entry.find("=")
            if colon == -1 and equals == -1:
                sep = -1
            elif colon == -1:
                sep = equals
            elif equals == -1:
                sep = colon
            else:
                sep = min(colon, equals)
        else:
            sep = colon

        if sep == -1:
            # No recognized separator — keep the label with no alternatives
            # rather than silently dropping it, so false-positive rows
            # are visible to downstream metrics.
            result[entry] = []
            continue

        label = entry[:sep].strip()
        alternatives = [part.strip() for part in entry[sep + 1 :].split("<n>") if part.strip()]
        if permissive:
            # metrics_mcs's behavior: only store a row when a non-empty
            # label is present, otherwise drop.
            if label:
                result[label] = alternatives
        else:
            # metrics_string's behavior: always store, even when label
            # is empty (the previous code path used ``result[label]`` directly).
            result[label] = alternatives

    return result

def clean_prediction(text: str | None) -> str:
    """Strip ``<think>`` blocks and chat-template artifacts from raw model output.

    Steps
    -----
    1. Remove every ``<think>...</think>`` reasoning block (case-insensitive,
       spanning newlines).
    2. Truncate the string at the first occurrence of any chat-template
       control token (``<|im_end|>``, ``<|endoftext|>``, ``<|im_start|>``).
       Everything after the first such token (and the token itself) is
       discarded.
    3. Trim surrounding whitespace.

    The function is idempotent: ``clean_prediction(clean_prediction(x)) ==
    clean_prediction(x)``.
    """
    if not text:
        return ""

    text = RE_THINKING.sub("", text)
    for tok in _CHAT_TEMPLATE_TOKENS:
        text = text.split(tok)[0]
    return text.strip()

def prediction_to_cxsmiles_opt(text: str | None) -> str:
    """Convert a raw model prediction into a bare ``cxsmiles_opt`` string.

    Composition of :func:`clean_prediction` and the ``<cxsmi>`` extractor,
    with a graceful fallback to the cleaned text when no ``<cxsmi>`` tag
    is present (some checkpoints emit only the SMILES core without the
    XML envelope).

    Returns ``""`` on empty input.
    """
    cleaned = clean_prediction(text)
    if not cleaned:
        return ""
    m = RE_CXSMI.search(cleaned)
    if m:
        return m.group(1).strip()
    return cleaned
