"""CPU/runtime contracts, independent of downloaded trained weights."""

from types import SimpleNamespace

import pytest
from PIL import Image

from patentagent._vendor.glyph.markush.inference import MarkushPrediction
from patentagent.config import Settings
from patentagent.inference import LocalInference, apply_markush_prediction
from patentagent.models import resolve_model
from patentagent.pipeline import assemble
from patentagent.schemas import StructureRegion


def region(number=0):
    return StructureRegion(
        structure_id=f"1_{number}", page=1, bbox=(0, 0, 20, 20),
        image=f"/unused/{number}.png", page_image="/unused/page.png",
        compound_id=str(number), identity_evidence=str(number))


def test_markush_preserves_raw_labels_and_substituent_table():
    raw = (
        "<think>draft</think><markush><cxsmi><r>R1</r>C(<r>R2</r>)O</cxsmi>"
        "<stable>R1:H<n>Me<ns>R2:F<n>Cl</stable></markush><|im_end|>"
    )
    prediction = MarkushPrediction.from_raw("test.png", raw)
    result = region()
    apply_markush_prediction(result, prediction)
    assert result.markush_raw == raw
    assert result.cxsmiles == "*C(*)O |$R1;;R2;$|"
    assert result.substituents == {"R1": ["H", "Me"], "R2": ["F", "Cl"]}
    assert result.markush_conversion_status == "ok"
    assert not result.errors
    assert assemble([result], [], {})[0]["markush_metadata"]["substituents"] == result.substituents


@pytest.mark.parametrize("value", [r"[\*]CC", r"C[\CH3]", "not-a-structure"])
def test_conversion_failure_is_reviewable_and_never_repaired(value):
    raw = f"<markush><cxsmi>{value}</cxsmi><stable></stable></markush>"
    prediction = MarkushPrediction.from_raw("test.png", raw)
    result = region()
    result.smiles = "CC"
    apply_markush_prediction(result, prediction)
    assert result.markush_raw == raw
    assert result.cxsmiles_opt == value
    assert result.cxsmiles is None
    assert result.markush_conversion_status == "failed"
    assert "ValueError" in prediction.conversion_error
    row = assemble([result], [], {})[0]
    assert row["confidence"] == "low"
    assert row["extraction_errors"]


def test_empty_and_truncated_outputs_are_flagged():
    empty = region()
    apply_markush_prediction(empty, MarkushPrediction.from_raw("a", ""))
    assert empty.markush_conversion_status == "empty"
    truncated = region()
    apply_markush_prediction(truncated, MarkushPrediction.from_raw(
        "a", "<markush><cxsmi>CC</cxsmi><stable>R1:H"))
    assert any("truncated" in error for error in truncated.errors)


def test_standard_cxsmiles_labels_survive_and_mixed_format_is_flagged():
    prediction = MarkushPrediction.from_raw("a", "*CC |$R1;;$|")
    assert prediction.cxsmiles == "*CC |$R1;;$|"
    mixed = MarkushPrediction.from_raw("a", "<r>R1</r>CC |$R2;;$|")
    assert "Mixed" in mixed.conversion_error


def test_native_inference_batch_mapping_precision_provenance_and_cleanup(monkeypatch):
    pytest.importorskip("torch")
    import patentagent._vendor.glyph.markush.inference as markush
    import patentagent._vendor.glyph.ocsr.predict as ocsr
    import patentagent.inference as native

    calls = []
    monkeypatch.setattr(native, "resolve_model", lambda settings, role:
                        (f"/weights/{role}", {"name": role, "revision": "pinned"}))
    monkeypatch.setattr(native, "release_accelerator", lambda: calls.append("release"))

    class OCSRDouble:
        def __init__(self, **kwargs):
            assert kwargs["device"] == "cuda:1"
            assert kwargs["precision"] == "fp16"

        def predict_batch(self, images, **kwargs):
            calls.append(("ocsr", images))
            assert kwargs["postprocess"] is None
            return ["CCO.[H]"] * len(images)

    class MarkushDouble:
        def __init__(self, **kwargs):
            assert kwargs["dtype"] == "float16"
            assert kwargs["base_model"] == "/weights/markush_base"

        def predict_many(self, images, **kwargs):
            calls.append(("markush", images))
            return [MarkushPrediction.from_raw(image, "CCO") for image in images]

    monkeypatch.setattr(ocsr, "OCSRPredictor", OCSRDouble)
    monkeypatch.setattr(markush, "MarkushPredictor", MarkushDouble)
    engine = LocalInference(Settings(device="cuda:1", ocsr_batch_size=2, markush_batch_size=2))
    regions = [region(i) for i in range(3)]
    engine.recognize(regions)
    engine.recognize_markush(regions)
    assert [len(c[1]) for c in calls if isinstance(c, tuple)] == [2, 1, 2, 1]
    assert calls[2] == calls[-1] == "release"
    assert regions[0].ocsr_raw == "CCO.[H]"
    assert regions[0].smiles == "CCO"
    assert all(r.ocsr_model == "OCSRGlyph" and r.markush_model == "MarkushGlyph" for r in regions)
    assert set(engine.provenance) == {"ocsr", "markush", "markush_base"}

    class WrongCount(OCSRDouble):
        def predict_batch(self, *args, **kwargs):
            return []

    monkeypatch.setattr(ocsr, "OCSRPredictor", WrongCount)
    with pytest.raises(ValueError):
        engine.recognize(regions)
    assert calls[-1] == "release"


def test_pinned_downloads_local_paths_and_custom_revision(tmp_path, monkeypatch):
    import huggingface_hub

    weights = tmp_path / "model.pth"
    weights.write_bytes(b"test weight identity")
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs)
        return str(weights)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    settings = Settings(model_dir=tmp_path / "cache")
    _, provenance = resolve_model(settings, "ocsr")
    assert downloads[0]["revision"] == settings.ocsr_revision
    assert downloads[0]["cache_dir"] == str(tmp_path / "cache/hub")
    assert provenance["local"] is False
    _, local = resolve_model(Settings(ocsr_checkpoint=str(weights)), "ocsr")
    assert local["revision"] is None and local["files_sha256"]["model.pth"]
    with pytest.raises(FileNotFoundError):
        resolve_model(Settings(ocsr_checkpoint=str(tmp_path / "missing.pth")), "ocsr")
    with pytest.raises(ValueError, match="REVISION"):
        resolve_model(Settings(ocsr_checkpoint="custom/repo"), "ocsr")


def test_markush_loader_uses_local_base_adapter_and_stock_transformers(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import peft
    import transformers

    from patentagent._vendor.glyph.markush.loader import load_model_and_processor

    (tmp_path / "adapter_config.json").write_text("{}")
    calls = []
    tokenizer = SimpleNamespace(pad_token_id=None, eos_token="eos",
                                chat_template="template", padding_side="right")
    processor = SimpleNamespace(tokenizer=tokenizer, chat_template=None,
                                image_processor=SimpleNamespace(max_pixels=None))
    model = SimpleNamespace(eval=lambda: None, to=lambda device: calls.append(device))

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return model

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **k: processor)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", load)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained",
                        lambda m, path, **kw: SimpleNamespace(merge_and_unload=lambda: m))
    actual, proc = load_model_and_processor("/local/base", str(tmp_path), "cpu", dtype="float32")
    assert actual is model and proc is processor
    assert calls[0][0] == "/local/base"
    assert calls[0][1]["attn_implementation"] == "eager"
    assert calls[0][1]["trust_remote_code"] is False
    assert tokenizer.padding_side == "left"
    assert processor.chat_template == "template"


def test_ocsr_real_architecture_checkpoint_and_cpu_generation(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from patentagent._vendor.glyph.ocsr.predict import (
        OCSRPredictor,
        _build_model,
        _build_recipe,
    )

    torch.set_num_threads(1)
    recipe = {
        "encoder_name": "swin_tiny_patch4_window7_224", "input_size": 224,
        "embed_dim": 32, "dec_num_layers": 1, "dec_attn_heads": 4,
        "dec_ff_dim": 64, "max_target_len": 4, "vocab_size": 101,
    }
    model = _build_model(_build_recipe(recipe))
    checkpoint = tmp_path / "tiny.pth"
    torch.save({"model": model.state_dict(), "recipe": recipe}, checkpoint)
    predictor = OCSRPredictor(checkpoint, device="cpu", precision="fp32")
    images = [Image.new("RGB", (80, 60), "white"), Image.new("RGB", (60, 80), "white")]
    results = predictor.predict_batch(images, batch_size=2, postprocess=None)
    assert len(results) == 2
    assert all(isinstance(value, str) for value in results)
    assert predictor.tokenizer.vocab_size == 101
