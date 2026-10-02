import hashlib
import json

import pandas as pd
import pymupdf
import pytest
from PIL import Image

from patentagent.config import Settings
from patentagent.documents import parse_pages
from patentagent.pipeline import assemble, run
from patentagent.postprocess import _looks_markush, _normalize_value
from patentagent.schemas import (
    AssayObservation,
    AssayPage,
    Identity,
    PageClassification,
    RunOptions,
    StructureRegion,
)


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "patent.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((40, 40), "Compound 007")
        doc.new_page().insert_text((40, 40), "IC50 (uM): Compound 007 < 0.5")
        doc.save(path)
    return path


class RecordedInference:
    """Deterministic contract double; never used by the application."""
    def check(self):
        pass

    def segment(self, document, pages):
        assert pages == [1]
        page = document.render(1)
        image = document.work_dir / "regions" / "1_0.png"
        image.parent.mkdir(exist_ok=True)
        with Image.open(page) as full:
            full.crop((20, 20, 200, 200)).save(image)
        return [StructureRegion(
            structure_id="1_0", page=1, bbox=(20, 20, 200, 200),
            image=str(image), page_image=str(page),
        )]

    def recognize(self, regions):
        regions[0].smiles = "CCO"
        regions[0].ocsr_model = "OCSRGlyph"
        regions[0].ocsr_raw = "OCC"

    def recognize_markush(self, regions):
        regions[0].cxsmiles = "OCC"
        regions[0].markush_model = "MarkushGlyph"
        regions[0].markush_raw = "<markush><cxsmi>OCC</cxsmi><stable></stable></markush>"
        regions[0].markush_conversion_status = "ok"


class RecordedVision:
    def __init__(self):
        self.calls = []

    def extract(self, prompt, schema, images):
        self.calls.append(schema)
        assert images[0].is_file()
        if schema is PageClassification:
            return schema(has_structures=False, has_activity="page 2." in prompt)
        if schema is Identity:
            return schema(compound_id="007", evidence_text="Compound 007")
        return AssayPage(observations=[
            AssayObservation(compound_id="007", assay_name="IC50", value_raw="< 0.5",
                             unit="µM", evidence_text="IC50 (uM): Compound 007 < 0.5"),
            AssayObservation(compound_id="007", assay_name="IC50", value_raw="0.8",
                             unit="µM", condition="second run", evidence_text="007 0.8"),
            AssayObservation(compound_id=None, assay_name="Ki", value_raw="ND",
                             evidence_text="ID illegible: ND"),
        ])


def test_native_pipeline_exports_review_and_replay(pdf, tmp_path):
    settings = Settings(device="cpu", llm_model="test-model")
    out = tmp_path / "result"
    vision = RecordedVision()
    rows = run(pdf, out, settings, RunOptions(structure_pages="1"),
               vision=vision, inference=RecordedInference())
    assert len(rows) == 2
    assert vision.calls.count(PageClassification) == 2
    assert rows[0]["agreement"] == "match"
    assert rows[0]["smiles_ocsr"] == "CCO"
    assert "smiles_molscribe" not in rows[0]
    assert rows[0]["activities"]["IC50"]["value_nM"] == 500
    assert rows[0]["activities"]["IC50 #2"]["value_nM"] == 800
    assert rows[1]["identity_status"] == "activity_only_unresolved"
    frame = pd.read_parquet(out / "merged_integration.parquet")
    assert len(frame) == 3
    assert frame["compound_id"].iloc[0] == "007"
    assert frame["ocsr_raw"].iloc[0] == "OCC"
    assert frame["markush_model"].iloc[0] == "MarkushGlyph"
    assert json.loads((out / "progress.json").read_text())[-1]["status"] == "done"
    assert json.loads((out / "run_manifest.json").read_text())["mode"] == "extract"
    import review_app
    review = review_app.ReviewServer(out)
    index = next(i for i, row in enumerate(review.rows) if row["structure_id"] == "1_0")
    page = review_app._item_page_html(review, index)
    assert "OCSRGlyph" in page and "MarkushGlyph" in page and "MolScribe" not in page
    assert "&lt;markush&gt;" in page  # Evidence is escaped, not interpreted as HTML.
    assert review_app._seg_image_bytes(review, index).startswith(b"\x89PNG")
    assert review_app._highlight_image_bytes(review, index).startswith(b"\x89PNG")
    review.save_item(index, {"verdict": "accept", "corrected_smiles": "CCN"})
    assert review.export()[index]["final_smiles"] == "CCN"
    assert json.loads((out / "merged_integration.json").read_text())[0]["canonical_smiles"] == "CCO"
    replayed = run(pdf, tmp_path / "replay", settings, RunOptions(), replay=out / "extraction.json")
    assert replayed == rows
    with pytest.raises(FileExistsError):
        run(pdf, out, settings, RunOptions())


def test_failure_has_no_completion_marker(pdf, tmp_path):
    class FailingVision:
        def extract(self, *args):
            raise ValueError("invalid response")
    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="invalid response"):
        run(pdf, out, Settings(llm_model="test"), RunOptions(),
            vision=FailingVision(), inference=RecordedInference())
    assert not (out / "merged_integration.json").exists()
    assert json.loads((out / "progress.json").read_text())[0]["status"] == "error"


def test_replay_requires_matching_pdf(pdf, tmp_path):
    recording = tmp_path / "recording.json"
    recording.write_text(json.dumps({"input_sha256": hashlib.sha256(b"other").hexdigest()}))
    with pytest.raises(ValueError, match="SHA256"):
        run(pdf, tmp_path / "out", Settings(), RunOptions(), replay=recording)


def test_ambiguous_identifiers_are_not_attached():
    regions = [StructureRegion(
        structure_id=str(i), page=1, bbox=(0, 0, 10, 10), image="unused",
        page_image="unused", compound_id="7", smiles="CCO", identity_evidence="7",
    ) for i in range(2)]
    obs = {"compound_id": "7", "assay_name": "IC50", "value_raw": "2", "page": 2, "unit": "nM"}
    rows = assemble(regions, [obs], {})
    assert len(rows) == 3
    assert not rows[0]["activities"] and not rows[1]["activities"]
    assert obs["status"] == "ambiguous"
    assert rows[2]["activities"]["IC50"]["link_method"] == "unresolved"


@pytest.mark.parametrize("raw,unit,expected", [
    ("≤0.4", "µM", 400), ("2 × 10^-3 M", None, 2e6), (">=5 nM", None, 5),
    ("1-10", "nM", None), ("5 mg/kg", "nM", None), ("ND", "nM", None),
])
def test_units_and_unsupported_values(raw, unit, expected):
    assert _normalize_value(raw, metadata={"unit_hint": unit})["value_nM"] == expected


@pytest.mark.parametrize("spec", ["0", "3", "2-1", "1,", "abc", "-1"])
def test_bad_page_ranges(spec):
    with pytest.raises(ValueError):
        parse_pages(spec, 2)


def test_wildcard_and_secret_repr():
    assert _looks_markush("*CC")
    assert not _looks_markush("CCO")
    assert "secret-value" not in repr(Settings(llm_api_key="secret-value"))
