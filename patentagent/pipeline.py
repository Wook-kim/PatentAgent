"""One application pipeline: PDF -> evidence -> recognition -> review artifacts."""

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import postprocess as pp
from .config import Settings
from .documents import PatentDocument, parse_pages
from .llm import VisionClient
from .schemas import (
    AssayPage,
    Identity,
    PageClassification,
    RunOptions,
    StructureRegion,
)

STEPS = [
    ("autopages", "페이지 분석"), ("structures", "구조 검출·인식"),
    ("activity", "화합물 ID·활성값 추출"), ("markush", "Markush 인식"),
    ("merge", "연결·검증·내보내기"),
]


def write_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def assemble(regions: list[StructureRegion], observations: list[dict], definitions: dict):
    rows = []
    for region in regions:
        mm = pp._attach_variable_definitions(
            pp._extract_markush_metadata(region.cxsmiles, region.cxsmiles_opt), definitions)
        mm["substituents"] = region.substituents
        mm["substituent_source"] = "markush_model" if region.substituents else "unavailable"
        row = {
            "compound_id": region.compound_id,
            "compound_id_raw": None,
            "page": region.page,
            "seg_key": region.structure_id,
            "structure_id": region.structure_id,
            "structure_image": region.image,
            "page_image": region.page_image,
            "bbox_pixel": list(region.bbox),
            "highlight_image": str(Path(region.image).with_name(
                f"{region.structure_id}_highlight.png")),
            "identity_evidence": region.identity_evidence,
            "identity_source": "page_vision",
            "smiles_ocsr": region.smiles,
            "ocsr_model": region.ocsr_model,
            "markush_model": region.markush_model,
            "ocsr_raw": region.ocsr_raw,
            "markush_raw": region.markush_raw,
            "markush_stable_raw": region.markush_stable_raw,
            "markush_conversion_status": region.markush_conversion_status,
            "model_provenance": region.model_provenance,
            "cxsmiles_markush": region.cxsmiles,
            "cxsmiles_opt": region.cxsmiles_opt,
            "markush_metadata": mm,
            "markush_groups": mm.get("r_groups", []),
            "substituents": region.substituents,
            "is_markush": bool(mm.get("is_markush") or pp._looks_markush(region.smiles)),
            "activities": {},
            "extraction_errors": region.errors,
        }
        rows.append(row)
    pp.enrich_with_rdkit(rows)
    lookup = defaultdict(set)
    for index, row in enumerate(rows):
        for key in pp._compound_id_keys(row["compound_id"]):
            lookup[key].add(index)
        if not row["compound_id"] or not row["identity_evidence"] or row["extraction_errors"]:
            row["confidence"] = "low"

    unresolved = {}
    for number, obs in enumerate(observations):
        source_id = obs.get("compound_id")
        candidates = set()
        for key in pp._compound_id_keys(source_id):
            candidates.update(lookup[key])
        assay = obs["assay_name"]
        metadata = pp._merge_assay_metadata(
            pp._infer_assay_metadata(assay),
            {key: obs.get(key) for key in (
                "unit", "target", "cell_line", "organism", "condition", "evidence_text")},
        )
        metadata["source_pages"] = [obs["page"]]
        record = {
            "value_raw": obs["value_raw"],
            **pp._normalize_value(obs["value_raw"], assay, metadata),
            "metadata": metadata,
            "matched_activity_id": source_id,
            "link_method": "unique_normalized_id" if len(candidates) == 1 else "unresolved",
            "link_confidence": 0.85 if len(candidates) == 1 else 0.0,
        }
        if len(candidates) == 1:
            row = rows[next(iter(candidates))]
            obs["status"] = "linked"
            obs["matched_structure_id"] = row["structure_id"]
        else:
            # Preserve ambiguities and measurements lacking a readable ID.
            identity_key = source_id or f"unidentified-observation-{number}"
            if identity_key not in unresolved:
                row = {
                    "compound_id": source_id,
                    "compound_id_raw": None,
                    "page": obs["page"], "seg_key": None,
                    "structure_id": None, "smiles_ocsr": None,
                    "cxsmiles_markush": None, "canonical_smiles": None,
                    "is_markush": False, "markush_metadata": {},
                    "activities": {}, "confidence": "low", "agreement": "activity_only",
                    "identity_status": "activity_only_unresolved",
                }
                unresolved[identity_key] = row
                rows.append(row)
            row = unresolved[identity_key]
            obs["status"] = "ambiguous" if candidates else "unlinked"
        key = assay
        suffix = 2
        while key in row["activities"]:
            key = f"{assay} #{suffix}"
            suffix += 1
        row["activities"][key] = record
    return rows


def run(pdf: Path, out_dir: Path, settings: Settings, options: RunOptions,
        *, vision=None, inference=None, replay: Path | None = None):
    pdf, out_dir = Path(pdf).resolve(), Path(out_dir).resolve()
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    # Refuse accidental result replacement; a rerun needs a new job directory.
    if (out_dir / "merged_integration.json").exists():
        raise FileExistsError(f"기존 결과가 있습니다. 새 출력 디렉토리를 사용하세요: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = [{"key": key, "label": label, "status": "pending"} for key, label in STEPS]

    def stage(key, state="running"):
        for item in progress:
            if item["key"] == key:
                item["status"] = state
        write_json(out_dir / "progress.json", progress)
        print(f"[{key}] {state}", flush=True)

    try:
        with PatentDocument(pdf, out_dir, settings.dpi) as doc:
            stage("autopages")
            if replay:
                recorded = json.loads(Path(replay).read_text())
                expected = hashlib.sha256(pdf.read_bytes()).hexdigest()
                if recorded.get("input_sha256") != expected:
                    raise ValueError("재생 데이터와 입력 PDF의 SHA256이 다릅니다.")
                regions = [StructureRegion.model_validate(r) for r in recorded["regions"]]
                provenance = recorded.get("models", {})
                observations = recorded["observations"]
                definitions = recorded.get("definitions", {})
                stage("autopages", "done")
                for key in ("structures", "activity", "markush"):
                    stage(key, "skipped")
            else:
                settings.require_llm()
                vision = vision or VisionClient(settings)
                if inference is None:
                    from .inference import LocalInference
                    inference = LocalInference(settings)
                inference.check()
                structure_pages = parse_pages(options.structure_pages, doc.page_count)
                assay_pages = parse_pages(options.assay_pages, doc.page_count)
                if options.auto_pages and (
                    not options.structure_pages or not options.assay_pages
                ):
                    for page in range(1, doc.page_count + 1):
                        result = vision.extract(
                            "Classify this patent page. has_structures: contains actual "
                            "chemical structure drawings. has_activity: contains measured "
                            "compound bioactivity values or categorical assay results. "
                            f"PDF page {page}. Text layer:\n{doc.text(page)[:16000]}",
                            PageClassification, [doc.render(page)])
                        if not options.structure_pages and result.has_structures:
                            structure_pages.append(page)
                        if not options.assay_pages and result.has_activity:
                            assay_pages.append(page)
                if not structure_pages and not assay_pages:
                    raise ValueError("구조·활성값 페이지를 찾지 못했습니다. 페이지를 직접 지정하세요.")
                write_json(out_dir / "pages.json", {
                    "structure_pages": structure_pages, "assay_pages": assay_pages})
                stage("autopages", "done")
                stage("structures")
                regions = inference.segment(doc, structure_pages) if structure_pages else []
                if regions:
                    inference.recognize(regions)
                stage("structures", "done")
                stage("activity")
                for region in regions:
                    highlighted = doc.highlight(region.page, region.bbox, region.structure_id)
                    identity = vision.extract(
                        "Read the compound identifier belonging ONLY to the chemical "
                        "structure inside the red rectangle. Use surrounding captions, "
                        "row labels and example headings. Return null if not explicit or "
                        "ambiguous. Do not substitute atom/R-group labels or invent an ID. "
                        "evidence_text must quote the relevant label/caption.",
                        Identity, [highlighted])
                    region.compound_id = identity.compound_id
                    region.identity_evidence = identity.evidence_text
                observations = []
                for page in assay_pages:
                    result = vision.extract(
                        "Transcribe every measured bioactivity observation on this page. "
                        "Keep one record per compound AND assay column/condition. "
                        "Preserve inequalities, categorical grades, units and the grade "
                        "legend verbatim. evidence_text must quote the supporting row "
                        "and header. Do not interpret synthesis yield as bioactivity. "
                        f"Requested assays (empty means all): {options.assay_names}.\n"
                        f"Current page {page} text:\n{doc.text(page)[:20000]}\n"
                        "Previous page text for continued table headings only:\n"
                        + (doc.text(page - 1)[-6000:] if page > 1 else ""),
                        AssayPage, [doc.render(page)])
                    observations.extend({**obs.model_dump(), "page": page}
                                        for obs in result.observations)
                stage("activity", "done")
                stage("markush")
                if regions and options.markush:
                    inference.recognize_markush(regions)
                    stage("markush", "done")
                else:
                    stage("markush", "skipped")
                definitions = pp._parse_markush_definition_text(
                    "\n".join(doc.text(page) for page in range(1, doc.page_count + 1)))
                provenance = getattr(inference, "provenance", {})
                write_json(out_dir / "extraction.json", {
                    "schema_version": 2,
                    "input_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                    "regions": [r.model_dump() for r in regions],
                    "observations": observations, "definitions": definitions,
                    "models": provenance,
                })
            stage("merge")
            rows = assemble(regions, observations, definitions)
            pp.write_csv(rows, out_dir / "merged_integration.csv")
            import pandas as pd
            pd.read_csv(
                out_dir / "merged_integration.csv",
                dtype={key: "string" for key in (
                    "compound_id", "compound_id_raw", "seg_key", "matched_activity_id",
                    "value_raw", "assay_name",
                )},
                keep_default_na=False, na_values=[""],
            ).to_parquet(
                out_dir / "merged_integration.parquet", index=False)
            write_json(out_dir / "assay_observations.json", observations)
            write_json(out_dir / "run_manifest.json", {
                "schema_version": 2,
                "mode": "replay" if replay else "extract",
                "input_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "structure_count": len(regions),
                "observation_count": len(observations),
                "n_compounds": len(rows),
                "llm_model": settings.llm_model,
                "models": provenance,
            })
            # This is the completion marker used by existing review clients.
            write_json(out_dir / "merged_integration.json", rows)
            stage("merge", "done")
            print(f"=== 병합 완료: {len(rows)}개 항목 ===", flush=True)
            return rows
    except Exception:
        for item in progress:
            if item["status"] == "running":
                item["status"] = "error"
        write_json(out_dir / "progress.json", progress)
        raise
