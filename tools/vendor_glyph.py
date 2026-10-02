"""Rebuild the pinned, inference-only Glyph snapshot (maintenance, not runtime).

Usage: python tools/vendor_glyph.py [--source /path/to/pinned/glyph]
"""

import argparse
import ast
import hashlib
import io
import json
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REVISION = "0bf782f863d26b041ace157668928ef07c38b972"
REPOSITORY = "https://github.com/EdisonScientific/glyph"
DEST = Path(__file__).resolve().parents[1] / "patentagent/_vendor/glyph"
HEADER = (
    "# Derived from EdisonScientific/glyph, Apache-2.0.\n"
    f"# Pinned upstream: {REVISION}. Modified for internal inference only.\n"
    "# See LICENSE.txt and SOURCES.json for provenance and modifications.\n"
)


def selected(text, names):
    lines = text.splitlines()
    nodes = []
    for node in ast.parse(text).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            name = getattr(node.targets[0], "id", None)
        if isinstance(node, ast.AnnAssign):
            name = getattr(node.target, "id", None)
        if name in names:
            start = min([node.lineno] + [
                item.lineno for item in getattr(node, "decorator_list", [])])
            nodes.append("\n".join(lines[start - 1:node.end_lineno]))
    return "\n\n".join(nodes) + "\n"


def build(source):
    records = []

    def emit(upstream, target, transform=None):
        data = (source / upstream).read_bytes()
        output = transform(data.decode()) if transform else data.decode()
        output = output.replace("glyph.", "patentagent._vendor.glyph.")
        if target.endswith(".py"):
            output = HEADER + output
        path = DEST / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output)
        records.append({
            "upstream": upstream, "path": target,
            "upstream_sha256": hashlib.sha256(data).hexdigest(),
            "vendored_sha256": hashlib.sha256(output.encode()).hexdigest(),
        })

    for name in ("config.py", "baseline_config.py", "model.py",
                 "smiles_tokenizer.py", "postprocess.py", "vocab/vocab_chars.json"):
        emit("glyph/ocsr/" + name, "ocsr/" + name)
    emit("glyph/ocsr/eval.py", "ocsr/eval.py", lambda text:
         "from rdkit import Chem, RDLogger\n\n" + selected(text, {"canonical_smiles"}))

    def ocsr_predict(text):
        # None exposes literal decoder output; True/False retain upstream behavior.
        text = text.replace("else:\n                predictions.extend(canonical_smiles",
                            "elif postprocess is False:\n                predictions.extend(canonical_smiles")
        text = text.replace("        return predictions\n", (
            "            else:\n                predictions.extend(decoded)\n"
            "        return predictions\n"))
        return text

    emit("glyph/ocsr/predict.py", "ocsr/predict.py", ocsr_predict)
    common_names = {
        "_XML_FLAGS", "RE_MARKUSH", "RE_CXSMI", "RE_STABLE", "RE_THINKING",
        "_CHAT_TEMPLATE_TOKENS", "extract_cxsmiles_and_stable", "parse_stable",
        "clean_prediction", "prediction_to_cxsmiles_opt",
    }
    emit("glyph/markush/eval/common.py", "markush/common.py", lambda text:
         "from __future__ import annotations\nimport re\n\n" + selected(text, common_names))
    def conversion(text):
        result = (
            "from __future__ import annotations\nimport re\nfrom rdkit import Chem\n\n"
            + selected(text, {"CX_SECTION_STARTS", "_RGROUP_TAG_RE", "split_cxsmiles",
                              "split_extension_sections", "opt_to_standard_cxsmiles"}))
        # Standard CXSMILES occasionally appears instead of opt. Do not discard
        # its existing atom labels when there are no optimized labels to convert.
        return result.replace(
            "    marker_base = 9000",
            '    if any(s.startswith("$") for s in split_extension_sections(extension)):\n'
            '        if _RGROUP_TAG_RE.search(core) or "[Ar]" in core:\n'
            '            raise ValueError("Mixed standard and optimized atom labels")\n'
            '        return cxsmiles_opt.strip()\n'
            "    marker_base = 9000")

    emit("glyph/markush/data/pipeline.py", "markush/conversion.py", conversion)
    loader_names = {
        "ENDOFTEXT_TOKEN_ID", "IM_END_TOKEN_ID", "_processor_image_patch_size",
        "_size_value", "_int_pixel_bound", "_processor_image_pixel_kwargs",
        "load_model_and_processor", "_build_messages",
    }

    def loader(text):
        result = (
            "from __future__ import annotations\n"
            "import logging\nfrom pathlib import Path\nfrom typing import Any\n"
            "import torch\nLOGGER = logging.getLogger(__name__)\n\n"
            + selected(text, loader_names)
        )
        # All model assets are resolved/pinned before loading by PatentAgent.
        result = result.replace("trust_remote_code=True", "trust_remote_code=False")
        result = result.replace(
            "low_cpu_mem_usage=True,", 'low_cpu_mem_usage=True,\n        attn_implementation="eager",')
        return result

    emit("glyph/markush/eval/checkpoint.py", "markush/loader.py", loader)
    emit("glyph/markush/eval/fast_eval.py", "markush/padding.py", lambda text:
         "from contextlib import contextmanager\nfrom typing import Any\n\n"
         + selected(text, {"temporary_tokenizer_padding_side"}))

    def markush_inference(text):
        tree = ast.parse(text)
        lines = text.splitlines()
        cls = next(n for n in tree.body if getattr(n, "name", "") == "MarkushPredictor")
        predict = next(n for n in cls.body if getattr(n, "name", "") == "predict")
        del lines[predict.lineno - 1:predict.end_lineno]
        text = "\n".join(lines) + "\n"
        text = text.replace("from glyph.markush.data.pipeline", "from glyph.markush.conversion")
        text = text.replace("from glyph.markush.eval import common", "from glyph.markush import common")
        text = text.replace("glyph.markush.eval.checkpoint", "glyph.markush.loader")
        text = text.replace("glyph.markush.eval.fast_eval", "glyph.markush.padding")
        text = text.replace("    return [common.clean_prediction(raw) for raw in raw_predictions]",
                            "    return raw_predictions")
        text = text.replace("            raw=cleaned,", "            raw=raw,")
        text = text.replace(
            '    vote: dict[str, Any] = field(default_factory=dict)',
            '    vote: dict[str, Any] = field(default_factory=dict)\n'
            '    conversion_error: str | None = None')
        text = text.replace(
            "        return cls(\n",
            '        converted, conversion_error = "", None\n'
            '        if cxsmiles_opt:\n'
            '            try:\n'
            '                converted = opt_to_standard_cxsmiles(cxsmiles_opt)\n'
            '            except Exception as exc:\n'
            '                conversion_error = f"{type(exc).__name__}: {exc}"\n'
            '        return cls(\n')
        text = text.replace("            cxsmiles=_standard_cxsmiles(cxsmiles_opt),",
                            "            cxsmiles=converted,\n"
                            "            conversion_error=conversion_error,")
        # Remove unused silent conversion and remote resolver helpers.
        tree = ast.parse(text)
        lines = text.splitlines()
        for node in reversed(tree.body):
            if getattr(node, "name", "") in {
                "_standard_cxsmiles", "_looks_like_hf_repo_id", "_download_adapter",
            }:
                del lines[node.lineno - 1:node.end_lineno]
        text = "\n".join(lines) + "\n"
        start = text.index("def _resolve_adapter(")
        end = text.index("\ndef _resolve_base_model", start)
        text = text[:start] + (
            "def _resolve_adapter(checkpoint):\n"
            "    if not checkpoint or not (Path(checkpoint) / 'adapter_config.json').is_file():\n"
            "        raise ValueError('Expected a local MarkushGlyph LoRA snapshot')\n"
            "    return str(checkpoint)\n\n"
        ) + text[end:]
        return text

    emit("glyph/markush/inference.py", "markush/inference.py", markush_inference)
    emit("LICENSE", "LICENSE.txt")
    for folder in ("", "ocsr", "ocsr/vocab", "markush"):
        path = DEST / folder / "__init__.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(HEADER)
    (DEST / "SOURCES.json").write_text(json.dumps({
        "repository": REPOSITORY, "revision": REVISION, "files": records,
        "modifications": [
            "Internal namespace and packaged vocabulary.",
            "Inference-only slices; no training, data pipelines, scoring or external projects.",
            "Expose literal OCSR decoder output with postprocess=None.",
            "Markush greedy batches only; preserve decoded raw text and conversion errors.",
            "Preserve existing standard CXSMILES labels; reject mixed label formats.",
            "Local adapter required; app resolves pinned model snapshots before loading.",
            "Stock Transformers, no remote code; eager attention for TITAN RTX.",
        ],
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.source:
        build(args.source)
        return
    url = f"https://codeload.github.com/EdisonScientific/glyph/tar.gz/{REVISION}"
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = response.read()
    with tempfile.TemporaryDirectory() as temporary:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(temporary, filter="data")
        build(Path(temporary) / f"glyph-{REVISION}")


if __name__ == "__main__":
    main()
