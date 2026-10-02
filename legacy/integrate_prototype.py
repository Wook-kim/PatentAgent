#!/usr/bin/env python3
"""
PatentAgent 통합 프로토타입 (Phase 1 spike)
============================================

특허 PDF 한 건에 대해 두 엔진을 오케스트레이션:

  1) BioChemInsight  : PDF -> 구조 세그먼트 검출 + SMILES(MolScribe) + 화합물 ID + (옵션) 활성값
  2) MarkushGrapher  : BioChemInsight가 떼어낸 구조 세그먼트 이미지 -> Markush CXSMILES + 치환기

두 엔진은 서로 다른 venv를 쓰므로 각각 서브프로세스로 호출하고,
결과를 화합물 ID 기준으로 병합하여 통합 테이블(merged CSV/JSON)을 만든다.

이 스크립트는 "두 엔진을 한 PDF에 대해 분기 호출해 결과를 합칠 수 있다"는 것을
실증하기 위한 최소 프로토타입이다. (시스템 python3 로 실행 — 무거운 의존성 없음)

사용:
    python3 integrate_prototype.py <pdf> --structure-pages "242-243" [--assay-pages ..] [--assay-names ..]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BCI_DIR = ROOT / "BioChemInsight"
MG_DIR = ROOT / "MarkushGrapher"

OCIE_DIR = ROOT / "OpenChemIE"
BCI_PY = BCI_DIR / ".venv" / "bin" / "python"
OCIE_PY = OCIE_DIR / ".venv" / "bin" / "python"
COREF_HELPER = ROOT / "molcoref_helper.py"
UV_BIN = "/home/wkim/.local/bin"

# ---------------------------------------------------------------------------
# Stage A — BioChemInsight: 구조(+선택적 활성) 추출
# ---------------------------------------------------------------------------
def run_biocheminsight(pdf, structure_pages, assay_pages, assay_names, out_dir,
                       engine="molscribe", gpu="1"):
    """BioChemInsight 파이프라인을 서브프로세스로 호출. structures.csv 경로 반환."""
    pdf_abs = Path(pdf).resolve()
    cmd = [
        str(BCI_PY), "pipeline.py", str(pdf_abs),
        "--structure-pages", structure_pages,
        "--engine", engine,
        "--output", str(out_dir),
    ]
    if assay_pages:
        cmd += ["--assay-pages", assay_pages]
    if assay_names:
        cmd += ["--assay-names", assay_names]

    # CUDA_VISIBLE_DEVICES 로 GPU 고정: BCI 내부(DECIMER/MolScribe)는 device 0 을
    # 쓰므로, 지정 GPU 를 '보이는 0번'으로 만들어 GPU 0 경합/OOM 을 피한다.
    env = dict(os.environ, AWS_REGION="us-east-1", AWS_DEFAULT_REGION="us-east-1",
               PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=str(gpu),
               TF_FORCE_GPU_ALLOW_GROWTH="true",
               TF_CUDNN_USE_AUTOTUNE="0")
    print(f"[BioChemInsight] 실행: {' '.join(cmd)}", flush=True)
    # 로그를 별도 파일로도 보존 (진짜 오류 추적용)
    bci_log = Path(out_dir).resolve().parent / "bci_run.log"
    with open(bci_log, "w") as lf:
        subprocess.run(cmd, cwd=str(BCI_DIR), env=env, check=True,
                       stdout=lf, stderr=subprocess.STDOUT)
    print(f"[BioChemInsight] 로그: {bci_log}", flush=True)

    structures_csv = Path(out_dir) / "structures.csv"
    if not structures_csv.is_absolute():
        structures_csv = BCI_DIR / structures_csv
    return structures_csv


def find_segment_images(out_dir):
    """BioChemInsight가 떼어낸 개별 구조 세그먼트 이미지 목록 (page/segment id 포함)."""
    base = Path(out_dir).resolve()
    segs = []
    for p in sorted(base.glob("structures_group_*/segment/segment_*.png")):
        # 파일명 형태: segment_<page>_<idx>.png
        stem = p.stem  # segment_242_0
        parts = stem.split("_")
        page, idx = parts[1], parts[2]
        segs.append({"path": p, "page": page, "idx": idx, "seg_key": f"{page}_{idx}"})
    return segs


def run_molcoref(bci_out, gpu="1"):
    """OpenChemIE MolCoref 로 페이지 이미지에서 구조↔라벨 coref 추출.

    BioChemInsight 의 structure_images/page_*.png 를 입력으로 OpenChemIE/.venv 의
    molcoref_helper.py 를 subprocess 호출. 반환: 헬퍼 JSON(list) 또는 [].
    OpenChemIE 미설치 시 빈 리스트.
    """
    if not OCIE_PY.exists():
        print("[MolCoref] OpenChemIE/.venv 없음 — coref 생략")
        return []
    base = Path(bci_out).resolve()
    pages = sorted(base.glob("structures_group_*/structure_images/page_*.png"))
    if not pages:
        print("[MolCoref] 페이지 이미지 없음 — coref 생략")
        return []

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MPLCONFIGDIR="/tmp",
               TF_FORCE_GPU_ALLOW_GROWTH="true",
               TF_CUDNN_USE_AUTOTUNE="0")
    cmd = [str(OCIE_PY), str(COREF_HELPER)] + [str(p) for p in pages]
    print(f"[MolCoref] {len(pages)}개 페이지 coref 추출 중...", flush=True)
    try:
        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=1800)
        if proc.returncode != 0:
            print(f"[MolCoref] 실패(무시): {proc.stderr.decode(errors='ignore')[-300:]}")
            return []
        return json.loads(proc.stdout.decode())
    except Exception as exc:
        print(f"[MolCoref] 예외(무시): {exc}")
        return []


def attach_coref(rows, coref_results):
    """coref 결과(라벨↔SMILES)를 rows 에 보강.

    coref pair 의 SMILES 를 RDKit canonical 로 매칭해 해당 구조 row 에 coref_label 추가.
    라벨이 BioChemInsight 의 compound_id 와 일치하면 coref_id_agree=True.
    """
    pairs = []
    for fig in coref_results or []:
        for p in fig.get("pairs", []):
            if p.get("label") and p.get("smiles"):
                _, can = _rdkit_canonical(p["smiles"], ignore_stereo=True)
                pairs.append((can, str(p["label"]).strip()))

    n_attached = 0
    for r in rows:
        r.setdefault("coref_label", None)
        r.setdefault("coref_id_agree", None)
        _, can = _rdkit_canonical(r.get("smiles_molscribe"), ignore_stereo=True)
        if not can:
            continue
        for cc, label in pairs:
            if cc and cc == can:
                r["coref_label"] = label
                r["coref_id_agree"] = (label == str(r.get("compound_id")).strip())
                n_attached += 1
                break
    return n_attached


def attach_activity_by_coref(rows, activity_map, assay_pages=None, assay_metadata_map=None):
    """MolCoref 라벨을 추가 후보로 사용해 누락된 활성값 연결을 보강."""
    activity_lookup = {
        assay_name: _build_activity_lookup(amap)
        for assay_name, amap in activity_map.items()
    }
    assay_meta = _build_assay_metadata_map(activity_map, assay_pages, assay_metadata_map)
    n_attached = 0
    for r in rows:
        label = r.get("coref_label")
        if not label:
            continue
        acts = r.setdefault("activities", {})
        for assay_name, amap in activity_lookup.items():
            if assay_name in acts:
                continue
            hit = _lookup_activity_value(amap, label)
            if not hit:
                continue
            raw, matched_id, _method = hit
            acts[assay_name] = _make_activity_record(
                raw, assay_name, matched_id, "coref_label", assay_meta)
            n_attached += 1
    return n_attached


def attach_unlinked_activity_candidates(rows, unlinked_activity_map, assay_pages=None,
                                        assay_metadata_map=None):
    """Unrestricted assay table rows를 구조 row와 보수적으로 후처리 매칭.

    exact/normalized ID 또는 Entry 번호가 일치할 때만 자동으로 activities에 붙인다.
    그 외 관측값은 별도 observations로 남겨 검수/후속 Markush enumeration에 사용한다.
    """
    assay_meta = _build_assay_metadata_map(
        unlinked_activity_map, assay_pages, assay_metadata_map)
    observations = []
    n_attached = 0
    for assay_name, amap in (unlinked_activity_map or {}).items():
        meta = assay_meta.get(assay_name, {})
        norm_meta = _metadata_with_grade_legend(meta, assay_meta.values())
        for source_id, raw in (amap or {}).items():
            match = _match_unlinked_observation(rows, source_id)
            obs = {
                "assay_name": assay_name,
                "source_id": str(source_id).strip(),
                "value_raw": raw,
                **_normalize_value(raw, assay_name, norm_meta),
                "metadata": meta,
                "matched_compound_id": None,
                "matched_seg_key": None,
                "match_method": match.get("method"),
                "link_confidence": match.get("confidence", 0.0),
                "status": "unlinked",
            }
            row = match.get("row")
            if row is not None and match.get("confidence", 0.0) >= 0.7:
                acts = row.setdefault("activities", {})
                if assay_name not in acts:
                    acts[assay_name] = _make_activity_record(
                        raw,
                        assay_name,
                        str(source_id).strip(),
                        match.get("method"),
                        assay_meta,
                    )
                    n_attached += 1
                obs.update({
                    "matched_compound_id": row.get("compound_id"),
                    "matched_seg_key": row.get("seg_key"),
                    "status": "linked",
                })
            observations.append(obs)
    return observations, n_attached


def append_activity_only_rows(rows, observations):
    """구조와 연결되지 않은 assay row를 activity-only compound row로 승격.

    이는 구조-활성 연결이 확정됐다는 의미가 아니라, SAR table의 Entry 단위
    활성값을 최종 merged 산출물에서 잃지 않기 위한 unresolved identity row다.
    """
    existing_activity_ids = set()
    for row in rows:
        for activity in (row.get("activities") or {}).values():
            matched = activity.get("matched_activity_id")
            if matched:
                existing_activity_ids.add(str(matched).strip())

    grouped = {}
    for obs in observations or []:
        if obs.get("status") != "unlinked":
            continue
        source_id = str(obs.get("source_id") or "").strip()
        if not source_id or source_id in existing_activity_ids:
            continue
        grouped.setdefault(source_id, []).append(obs)

    new_rows = []
    for source_id, obs_list in sorted(grouped.items(), key=lambda kv: _entry_sort_key(kv[0])):
        activities = {}
        for obs in obs_list:
            assay_name = obs.get("assay_name")
            if not assay_name:
                continue
            activities[assay_name] = {
                "value_raw": obs.get("value_raw"),
                "operator": obs.get("operator"),
                "value_num": obs.get("value_num"),
                "unit": obs.get("unit"),
                "value_nM": obs.get("value_nM"),
                "range_lower_nM": obs.get("range_lower_nM"),
                "range_upper_nM": obs.get("range_upper_nM"),
                "matched_activity_id": source_id,
                "link_method": "assay_table_unresolved_identity",
                "link_confidence": 0.3,
                "metadata": obs.get("metadata") or {},
            }
        if not activities:
            continue
        new_rows.append({
            "compound_id": _format_activity_only_compound_id(source_id),
            "compound_id_raw": source_id,
            "page": None,
            "seg_key": None,
            "smiles_molscribe": None,
            "cxsmiles_markush": None,
            "cxsmiles_opt": None,
            "markush_metadata": {},
            "markush_groups": [],
            "substituents": [],
            "is_markush": False,
            "activities": activities,
            "smiles_valid": None,
            "cxsmiles_valid": None,
            "canonical_smiles": None,
            "agreement": "activity_only",
            "confidence": "low",
            "identity_status": "activity_only_unresolved",
            "identity_source": "assay_table",
        })
    rows.extend(new_rows)
    return len(new_rows)


def attach_entry_names(rows, entry_name_map):
    """Entry ID 기반 row에 Table 2 compound name을 붙인다."""
    if not entry_name_map:
        return 0
    n = 0
    for row in rows:
        source_id = str(row.get("compound_id_raw") or "").strip()
        if not source_id:
            continue
        name = entry_name_map.get(source_id)
        if not name:
            continue
        row["compound_name"] = name
        row["identity_source"] = _append_source(row.get("identity_source"), "entry_name_table")
        row.setdefault("identity_status", "activity_only_unresolved")
        n += 1
    return n


def load_entry_name_map(pdf, pages=None):
    """특허 SAR table의 Entry Name 열을 추출."""
    text = _pdftotext(pdf, pages, raw=True)
    if not text:
        return {}
    entry_names = _parse_entry_name_table(text)
    if entry_names:
        print(f"[Activity] Entry Name {len(entry_names)}개 추출")
    else:
        print("[Activity] Entry Name 추출 결과 없음")
    return entry_names


def _parse_entry_name_table(text):
    import re
    table = str(text or "")
    marker = re.search(r"\bTABLE\s+2\b", table, re.I)
    if marker:
        table = table[marker.end():]
    stop = re.search(r"\bWhat\s+is\s+claimed\b", table, re.I)
    if stop:
        table = table[:stop.start()]

    entries = {}
    for m in re.finditer(r"(?ms)^\s*(\d{1,2})\s+(.+?)(?=^\s*\d{1,2}\s+\S|\Z)", table):
        entry_id = m.group(1)
        if int(entry_id) > 200:
            continue
        block = m.group(2)
        name = _clean_entry_name_block(block)
        if name:
            entries[entry_id] = name
    return entries


def _clean_entry_name_block(block):
    import re
    lines = []
    for line in str(block or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"\d{1,2}", line):
            continue
        if re.match(r"^(Entry|Name|Structure|c-Met|KDR|fit-?3)\b", line, re.I):
            continue
        line = re.sub(
            r"\s+(?:A\.?|B\.?|C\.?)(?:\s+(?:A\.?|B\.?|C\.?)){0,2}\.?\s*$",
            "",
            line,
        ).strip()
        if line:
            lines.append(line)
    if not lines:
        return None

    out = lines[0]
    for line in lines[1:]:
        if out.endswith("-") or out.endswith("["):
            out += line
        else:
            out += " " + line
    out = re.sub(r"\s+", " ", out).strip(" ;,")
    if not _looks_like_compound_name(out):
        return None
    return out or None


def _looks_like_compound_name(name):
    import re
    s = str(name or "")
    if len(s) < 12:
        return False
    if not re.search(r"[A-Za-z]", s):
        return False
    return bool(re.search(
        r"phenyl|pyr|purin|amide|carbon|fluoro|morpholin|cyclopropane|acetamide",
        s,
        re.I,
    ))


def _append_source(existing, source):
    if not existing:
        return source
    parts = [p for p in str(existing).split(",") if p]
    if source not in parts:
        parts.append(source)
    return ",".join(parts)


def _entry_sort_key(source_id):
    import re
    s = str(source_id)
    m = re.search(r"(\d+)", s)
    if m:
        return (0, int(m.group(1)), s)
    return (1, 0, s)


def _format_activity_only_compound_id(source_id):
    import re
    s = str(source_id).strip()
    if re.fullmatch(r"\d+[A-Za-z]?", s):
        return f"Entry {s}"
    if re.match(r"^(entry|example|compound)\b", s, re.I):
        return s
    return f"Activity {s}"


def extract_claim_structure_candidates(pdf, pages, out_dir, gpu="1", dpi=200,
                                       min_confidence=0.75):
    """Claim table의 Entry row crop을 MolScribe로 직접 읽어 구조 후보를 생성."""
    page_nums = _parse_pages_string(pages)
    if not page_nums:
        return {}

    crop_dir = Path(out_dir) / "claim_structures"
    crop_dir.mkdir(parents=True, exist_ok=True)

    crops = []
    for page in page_nums:
        image = _render_pdf_page(pdf, page, crop_dir, dpi=dpi)
        if image:
            crops.extend(_crop_claim_structure_rows(pdf, page, image, crop_dir))
    if not crops:
        print("[ClaimStructure] claim 구조 crop 없음")
        return {}

    raw_results = _run_molscribe_on_claim_crops(crops, gpu=gpu)
    candidates = {}
    for crop in crops:
        result = raw_results.get(str(crop["image"]))
        if not result:
            continue
        smiles = result.get("smiles")
        confidence = result.get("confidence")
        valid, canonical = _rdkit_canonical(smiles)
        if not smiles or valid is False:
            continue
        if confidence is not None and confidence < min_confidence:
            continue
        candidates[crop["source_id"]] = {
            "source_id": crop["source_id"],
            "page": crop["page"],
            "smiles": smiles,
            "canonical_smiles": canonical,
            "confidence": confidence,
            "image": str(crop["image"]),
            "bbox_pdf": crop.get("bbox_pdf"),
            "bbox_pixel": crop.get("bbox_pixel"),
            "method": "claim_table_bbox_molscribe",
        }
    print(f"[ClaimStructure] 후보 {len(candidates)}개 생성 "
          f"(crop {len(crops)}개, min_confidence={min_confidence})")
    return candidates


def _render_pdf_page(pdf, page, out_dir, dpi=200):
    prefix = Path(out_dir) / f"page_{page}"
    image = prefix.with_suffix(".png")
    cmd = [
        "pdftoppm", "-singlefile", "-f", str(page), "-l", str(page),
        "-png", "-r", str(dpi), str(Path(pdf).resolve()), str(prefix),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    except Exception as exc:
        print(f"[ClaimStructure] page {page} 렌더링 실패: {exc}")
        return None
    return image if image.exists() else None


def _crop_claim_structure_rows(pdf, page, image_path, out_dir):
    try:
        from PIL import Image
    except Exception as exc:
        print(f"[ClaimStructure] PIL 없음 — crop 생략: {exc}")
        return []

    page_info, words = _pdftotext_bbox_words(pdf, page)
    if not page_info or not words:
        return []

    structure_header = next((w for w in words if w["text"].lower() == "structure"), None)
    if not structure_header:
        return []
    table_header_y = structure_header["yMin"]

    claim_stop_y = None
    for w in words:
        if re.match(r"^\d+\.$", w["text"]) and w["xMin"] < page_info["width"] * 0.2:
            if w["yMin"] > table_header_y + 20:
                claim_stop_y = w["yMin"]
                break

    entry_words = []
    for w in words:
        if not re.fullmatch(r"\d{1,4}", w["text"]):
            continue
        if w["xMin"] > page_info["width"] * 0.2:
            continue
        if w["yMin"] <= table_header_y:
            continue
        if claim_stop_y and w["yMin"] >= claim_stop_y:
            continue
        entry_words.append(w)
    entry_words = sorted(entry_words, key=lambda w: (w["yMin"], w["xMin"]))
    if not entry_words:
        return []

    img = Image.open(image_path)
    img_w, img_h = img.size
    scale_x = img_w / page_info["width"]
    scale_y = img_h / page_info["height"]
    crops = []

    for idx, entry in enumerate(entry_words):
        next_y = entry_words[idx + 1]["yMin"] if idx + 1 < len(entry_words) else claim_stop_y
        if not next_y:
            same_table_y = [
                w["yMax"] for w in words
                if w["yMin"] > entry["yMin"] and w["xMin"] > page_info["width"] * 0.32
            ]
            next_y = max(same_table_y, default=entry["yMin"] + 120) + 14

        row_words = [w for w in words if entry["yMin"] - 2 <= w["yMin"] < next_y]
        name_right = max(
            [w["xMax"] for w in row_words if w["xMin"] < structure_header["xMin"]],
            default=page_info["width"] * 0.34,
        )
        structure_words = [
            w for w in row_words
            if w["xMin"] >= name_right - 2 or w["xMin"] >= page_info["width"] * 0.35
        ]
        x0_pt = max(page_info["width"] * 0.34,
                    min([w["xMin"] for w in structure_words], default=name_right) - 12)
        x1_pt = min(page_info["width"] * 0.80,
                    max([w["xMax"] for w in structure_words],
                        default=page_info["width"] * 0.68) + 90)
        y0_pt = max(0, entry["yMin"] - 8)
        structure_y_max = max([w["yMax"] for w in structure_words], default=next_y)
        y1_pt = min(page_info["height"], next_y - 4, structure_y_max + 5)
        if y1_pt - y0_pt < 70:
            y1_pt = min(page_info["height"], next_y - 4)
        box = (
            max(0, int(x0_pt * scale_x)),
            max(0, int(y0_pt * scale_y)),
            min(img_w, int(x1_pt * scale_x)),
            min(img_h, int(y1_pt * scale_y)),
        )
        if box[2] - box[0] < 80 or box[3] - box[1] < 80:
            continue

        crop_path = Path(out_dir) / f"page_{page}_entry_{entry['text']}.png"
        img.crop(box).save(crop_path)
        crops.append({
            "source_id": f"Entry {entry['text']}",
            "page": page,
            "image": crop_path,
            "bbox_pdf": [round(x0_pt, 2), round(y0_pt, 2), round(x1_pt, 2), round(y1_pt, 2)],
            "bbox_pixel": list(box),
        })
    return crops


def _pdftotext_bbox_words(pdf, page):
    cmd = ["pdftotext", "-bbox", "-f", str(page), "-l", str(page),
           str(Path(pdf).resolve()), "-"]
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    except Exception as exc:
        print(f"[ClaimStructure] bbox 추출 실패(page {page}): {exc}")
        return None, []

    text = proc.stdout.decode("utf-8", errors="ignore")
    page_match = re.search(r"<page\s+width=\"([0-9.]+)\"\s+height=\"([0-9.]+)\"", text)
    if not page_match:
        return None, []
    page_info = {"width": float(page_match.group(1)), "height": float(page_match.group(2))}
    word_re = re.compile(
        r"<word\s+xMin=\"([0-9.]+)\"\s+yMin=\"([0-9.]+)\"\s+"
        r"xMax=\"([0-9.]+)\"\s+yMax=\"([0-9.]+)\">(.*?)</word>"
    )
    import html
    words = []
    for m in word_re.finditer(text):
        words.append({
            "xMin": float(m.group(1)),
            "yMin": float(m.group(2)),
            "xMax": float(m.group(3)),
            "yMax": float(m.group(4)),
            "text": html.unescape(m.group(5)).strip(),
        })
    return page_info, words


def _run_molscribe_on_claim_crops(crops, gpu="1"):
    crop_paths = [str(c["image"]) for c in crops]
    code = r'''
import json
import sys
import torch
from huggingface_hub import hf_hub_download
from molscribe import MolScribe

ckpt = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth", local_dir="BioChemInsight/models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MolScribe(ckpt, device=device)
out = {}
for image in sys.argv[1:]:
    try:
        pred = model.predict_image_file(image, return_atoms_bonds=True, return_confidence=True)
    except TypeError:
        pred = model.predict_image_file(image)
    out[image] = {
        "smiles": pred.get("smiles") if isinstance(pred, dict) else None,
        "confidence": pred.get("confidence") if isinstance(pred, dict) else None,
    }
print(json.dumps(out))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MPLCONFIGDIR="/tmp",
               TF_FORCE_GPU_ALLOW_GROWTH="true",
               TF_CUDNN_USE_AUTOTUNE="0")
    try:
        proc = subprocess.run([str(BCI_PY), "-c", code] + crop_paths,
                              cwd=str(ROOT), env=env, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=1800)
        stdout = proc.stdout.decode("utf-8", errors="ignore")
        json_start = stdout.find("{")
        if json_start < 0:
            raise ValueError("MolScribe stdout에 JSON 객체가 없습니다")
        return json.loads(stdout[json_start:])
    except Exception as exc:
        stderr = getattr(exc, "stderr", b"")
        tail = stderr.decode(errors="ignore")[-500:] if stderr else ""
        print(f"[ClaimStructure] MolScribe 실행 실패: {exc} {tail}")
        return {}


def attach_claim_structures(rows, candidates):
    """claim table에서 얻은 구조 후보를 Entry row에 연결."""
    if not candidates:
        return 0
    lookup = {}
    for source_id, candidate in candidates.items():
        for key in _compound_id_keys(source_id):
            lookup[key] = candidate

    n = 0
    for row in rows:
        candidate = None
        for source in (row.get("compound_id_raw"), row.get("compound_id")):
            for key in _compound_id_keys(source):
                candidate = lookup.get(key)
                if candidate:
                    break
            if candidate:
                break
        if not candidate:
            continue
        row["smiles_molscribe"] = candidate.get("smiles")
        row["page"] = row.get("page") or candidate.get("page")
        row["structure_source"] = candidate.get("method")
        row["structure_confidence"] = candidate.get("confidence")
        row["structure_image"] = candidate.get("image")
        row["identity_source"] = _append_source(row.get("identity_source"), "claim_structure_table")
        if row.get("identity_status") == "activity_only_unresolved":
            row["identity_status"] = "structure_candidate_from_claim"
        n += 1
    return n


def _build_assay_metadata_map(activity_map, assay_pages=None, assay_metadata_map=None):
    """activity map의 assay 이름별 downstream metadata를 한 번만 구성."""
    return {
        assay_name: _merge_assay_metadata(
            _infer_assay_metadata(assay_name, assay_pages),
            (assay_metadata_map or {}).get(assay_name),
        )
        for assay_name in (activity_map or {})
    }


def _make_activity_record(raw, assay_name, matched_id, method, assay_meta):
    """activities 내부 공통 payload 생성."""
    meta = (assay_meta or {}).get(assay_name, {})
    norm_meta = _metadata_with_grade_legend(meta, (assay_meta or {}).values())
    return {
        "value_raw": raw,
        **_normalize_value(raw, assay_name, norm_meta),
        "matched_activity_id": matched_id,
        "link_method": method,
        "link_confidence": _activity_link_confidence(method),
        "metadata": meta,
    }


def _autodetect(pdf, gpu=None):
    """autodetect_pages.py 를 BioChemInsight/.venv(fitz 보유)로 호출.
    반환: (structure_pages_str, assay_pages_str, assay_names_str).
    assay_names_str 은 자동 추출된 어세이명을 콤마로 이은 문자열(예 'IC50,EC50').
    실패 시 (None, None, None)."""
    script = ROOT / "autodetect_pages.py"
    pdf_abs = Path(pdf).resolve()
    try:
        env = dict(os.environ, MPLCONFIGDIR="/tmp",
                   TF_FORCE_GPU_ALLOW_GROWTH="true",
                   TF_CUDNN_USE_AUTOTUNE="0")
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # 타임아웃 1800초: 스캔 특허는 autodetect 가 DECIMER 폴백을 돌려 느릴 수 있음
        proc = subprocess.run(
            [str(BCI_PY), str(script), str(pdf_abs)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800, env=env)
        data = json.loads(proc.stdout.decode())
        names = data.get("assay_names") or []
        return (data.get("structure_pages") or None,
                data.get("assay_pages") or None,
                ",".join(names) if names else None)
    except Exception as exc:
        print(f"[AutoPages] 자동 탐지 실패(무시): {exc}")
        return None, None, None


def _run_markushgrapher_http(img_dir, service_url):
    """REST 마이크로서비스(predict_dir)에 디렉토리 경로를 보내 추론."""
    import urllib.request
    url = service_url.rstrip("/") + "/predict_dir"
    payload = json.dumps({"image_dir": str(img_dir)}).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"})
    print(f"[MarkushGrapher] REST 호출: {url} (image_dir={img_dir})", flush=True)
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.loads(resp.read().decode())
    results = data.get("results", {})
    pred_file = data.get("run_dir", "(service)")
    return results, pred_file


# ---------------------------------------------------------------------------
# Stage A2 — BioChemInsight: 활성값(assay) 추출
# ---------------------------------------------------------------------------
def load_activity(out_dir, assay_names):
    """pipeline.py 가 생성한 <assay_name>_assay_data.json 들을 로드.
    반환: {assay_name: {compound_id: raw_value}}"""
    base = Path(out_dir).resolve()
    activity_map = {}
    if not assay_names:
        return activity_map
    for name in [a.strip() for a in assay_names.split(",") if a.strip()]:
        clean = name.replace(" ", "_").replace("/", "_")
        f = base / f"{clean}_assay_data.json"
        if f.exists():
            with open(f, encoding="utf-8") as fh:
                activity_map[name] = json.load(fh)
            print(f"[Activity] '{name}': {len(activity_map[name])}개 값 로드 ({f.name})")
        else:
            print(f"[Activity] 경고: {f} 없음")
    return activity_map


def load_unlinked_activity(out_dir, assay_names):
    """pipeline.py 가 생성한 <assay_name>_assay_unlinked_data.json 들을 로드.

    이 파일은 구조 추출 ID와 아직 연결되지 않은 SAR table row까지 보존한다.
    반환: {assay_name: {source_id: raw_value}}
    """
    base = Path(out_dir).resolve()
    activity_map = {}
    if not assay_names:
        return activity_map
    for name in [a.strip() for a in assay_names.split(",") if a.strip()]:
        clean = name.replace(" ", "_").replace("/", "_")
        f = base / f"{clean}_assay_unlinked_data.json"
        if f.exists():
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                activity_map[name] = data
                print(f"[Activity] '{name}': {len(data)}개 unlinked 관측값 로드 ({f.name})")
        else:
            print(f"[Activity] 참고: {f.name} 없음")
    return activity_map


def load_assay_metadata(out_dir, assay_names):
    """pipeline.py가 생성한 <assay_name>_assay_metadata.json들을 로드."""
    base = Path(out_dir).resolve()
    metadata = {}
    if not assay_names:
        return metadata
    for name in [a.strip() for a in assay_names.split(",") if a.strip()]:
        clean = name.replace(" ", "_").replace("/", "_")
        f = base / f"{clean}_assay_metadata.json"
        if not f.exists():
            f = base / "assay_metadata.json"
        if f.exists():
            try:
                with open(f, encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    metadata[name] = data
                    print(f"[Activity] '{name}' metadata 로드 ({f.name})")
            except Exception as exc:
                print(f"[Activity] metadata 경고: {f} 로드 실패: {exc}")
    return metadata


# ---------------------------------------------------------------------------
# Stage B — MarkushGrapher: 세그먼트 이미지 -> Markush CXSMILES
# ---------------------------------------------------------------------------
def run_markushgrapher(segment_images, work_dir, gpu="1", service_url=None):
    """세그먼트 이미지들을 MarkushGrapher에 통과시켜 {seg_key: cxsmiles} 반환.

    service_url 이 주어지면 REST 마이크로서비스(markush_service.py)에 HTTP 호출,
    아니면 기존 방식대로 inference.sh 를 subprocess 로 실행한다.
    """
    work_dir = Path(work_dir).resolve()
    img_dir = work_dir / "mg_input"
    img_dir.mkdir(parents=True, exist_ok=True)

    if not segment_images:
        print("[MarkushGrapher] 입력 세그먼트 없음 — Markush 추론 생략")
        return {}, "(no segment images)"

    # seg_key 를 파일명으로 사용 (MarkushGrapher 의 id 가 파일 stem 이 됨)
    for s in segment_images:
        dst = img_dir / f"{s['seg_key']}.png"
        dst.write_bytes(Path(s["path"]).read_bytes())

    # --- REST 서비스 경로 ---
    if service_url:
        return _run_markushgrapher_http(img_dir, service_url)

    # --- 기존 subprocess 경로 ---
    env = dict(os.environ)
    env["PATH"] = UV_BIN + os.pathsep + env.get("PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.pop("CHEMICALOCR_PYTHON", None)  # 기본 chemicalocr-env(vllm) 사용

    cmd = ["bash", "scripts/inference/inference.sh", str(img_dir)]
    print(f"[MarkushGrapher] 실행: {' '.join(cmd)} (GPU {gpu})", flush=True)
    subprocess.run(cmd, cwd=str(MG_DIR), env=env, check=True)

    # 최신 run 디렉토리의 predictions jsonl 찾기
    runs = sorted((MG_DIR / "data" / "hf" / "inference").glob("mg_input-*"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        runs = sorted((MG_DIR / "data" / "hf" / "inference").glob("*"),
                      key=lambda p: p.stat().st_mtime)
    pred_file = runs[-1] / "evaluation" / "predictions_1000.jsonl"
    results = {}
    if pred_file.exists():
        with open(pred_file) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                results[d["id"]] = {
                    "cxsmiles": d.get("cxsmiles"),
                    "cxsmiles_opt": d.get("cxsmiles_opt"),
                    "gt_cxsmiles": d.get("gt_cxsmiles"),
                    "gt_cxsmiles_opt": d.get("gt_cxsmiles_opt"),
                }
    return results, pred_file


def load_markush_from_merged(out_json):
    """기존 merged_integration.json에서 Markush 결과를 재사용."""
    p = Path(out_json)
    if not p.exists():
        print(f"[MarkushGrapher] 재사용할 기존 병합 JSON 없음: {p}")
        return {}, "(no previous merged_integration.json)"
    try:
        rows = json.load(open(p, encoding="utf-8"))
    except Exception as exc:
        print(f"[MarkushGrapher] 기존 병합 JSON 로드 실패: {exc}")
        return {}, p
    results = {}
    for row in rows or []:
        seg_key = row.get("seg_key")
        if not seg_key:
            continue
        results[seg_key] = {
            "cxsmiles": row.get("cxsmiles_markush"),
            "cxsmiles_opt": row.get("cxsmiles_opt"),
            "gt_cxsmiles": row.get("gt_cxsmiles"),
            "gt_cxsmiles_opt": row.get("gt_cxsmiles_opt"),
        }
    print(f"[MarkushGrapher] 기존 병합 JSON에서 {len(results)}개 Markush 결과 재사용")
    return results, p


def load_markush_definition_index(pdf, pages=None):
    """PDF 텍스트에서 Markush 변수 정의 후보를 추출.

    OCR/PDF text에서 위첨자가 손실될 수 있으므로 이 함수는 원문 label 기준
    index만 만들고, row별 variable label과의 exact/base 매칭은 별도 단계에서 한다.
    """
    text = _pdftotext(pdf, pages)
    if not text:
        return {}
    index = _parse_markush_definition_text(text)
    if index:
        n_defs = sum(len(v) for v in index.values())
        page_msg = pages or "all pages"
        print(f"[MarkushDefs] {n_defs}개 정의 후보 추출 ({page_msg})")
    else:
        print("[MarkushDefs] 정의 후보 없음")
    return index


def _pdftotext(pdf, pages=None, raw=False):
    cmd = ["pdftotext"]
    if raw:
        cmd.append("-raw")
    parsed_pages = _parse_pages_string(pages)
    if parsed_pages:
        cmd += ["-f", str(min(parsed_pages)), "-l", str(max(parsed_pages))]
    cmd += [str(Path(pdf).resolve()), "-"]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=120, check=False)
    except Exception as exc:
        print(f"[MarkushDefs] pdftotext 실행 실패: {exc}")
        return ""
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="ignore")[-300:]
        print(f"[MarkushDefs] pdftotext 실패: {err}")
        return ""
    return proc.stdout.decode(errors="ignore")


def _parse_markush_definition_text(text):
    import re
    normalized = text.replace("\u2014", "-").replace("\u2013", "-")
    normalized = re.sub(r"-\s*\n\s*", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    index = {}

    label_pat = r"(?:[JZR]\d*|Ar|R[0-9'\u2019\u201d]*)"
    verb_pat = r"(?:is|are)\s+(?:independently\s+)?(?:selected\s+from\s+)?"
    for m in re.finditer(
        rf"\b(?:each\s+)?(?P<label>{label_pat})\s+{verb_pat}(?P<definition>[^.;:]{{1,700}})",
        normalized,
        re.I,
    ):
        label = _clean_markush_label(m.group("label"))
        definition = _clean_definition_text(m.group("definition"))
        if not label or not definition:
            continue
        if _definition_is_noise(label, definition):
            continue
        key = _definition_key(label)
        entry = {
            "source_label": label,
            "definition": definition,
            "source": "pdf_text_rule",
        }
        index.setdefault(key, [])
        if entry not in index[key]:
            index[key].append(entry)
    return index


def _clean_markush_label(label):
    import re
    s = str(label or "").strip()
    s = s.replace("\u2019", "'").replace("\u201d", "")
    s = re.sub(r"[^A-Za-z0-9']+", "", s)
    if not s:
        return None
    if s.lower() == "ar":
        return "Ar"
    return s.upper() if len(s) == 1 else s


def _definition_key(label):
    import re
    s = _clean_markush_label(label) or ""
    return re.sub(r"[^A-Za-z0-9]+", "", s).lower()


def _definition_base_key(label):
    import re
    key = _definition_key(label)
    return re.sub(r"\d+$", "", key)


def _clean_definition_text(text):
    import re
    s = str(text or "").strip(" -,:;")
    s = re.sub(r"\s+", " ", s)
    return s[:700]


def _definition_is_noise(label, definition):
    d = str(definition or "").lower()
    if len(d) < 1:
        return True
    noisy = (
        "patent", "publication", "date of patent", "references cited",
        "field of classification", "classification search",
        "as defined in the specification", "application file",
        "according to formula", "wherein j",
    )
    if any(ch in d for ch in ("<", ">", "")):
        return True
    if len(d) > 500 and "optionally" not in d:
        return True
    return any(term in d for term in noisy)


def _attach_variable_definitions(markush_meta, definition_index):
    if not definition_index:
        markush_meta["variable_definitions"] = {}
        markush_meta["definition_coverage"] = 0.0
        return markush_meta

    labels = markush_meta.get("variable_labels") or []
    definitions = {}
    for label in labels:
        exact_key = _definition_key(label)
        base_key = _definition_base_key(label)
        hits = []
        for item in definition_index.get(exact_key, []):
            enriched = dict(item)
            enriched.update({"match": "exact", "confidence": 0.9})
            hits.append(enriched)
        if not hits and base_key and base_key != exact_key:
            base_hits = []
            for item in definition_index.get(base_key, []):
                enriched = dict(item)
                enriched.update({"match": "base_label", "confidence": 0.45})
                base_hits.append(enriched)
            short_hits = [h for h in base_hits if len(str(h.get("definition") or "")) <= 120]
            hits.extend(short_hits or base_hits[:2])
        if hits:
            definitions[label] = sorted(hits, key=_definition_rank_key)[:5]

    markush_meta["variable_definitions"] = definitions
    markush_meta["definition_coverage"] = (
        round(len(definitions) / len(labels), 3) if labels else 0.0
    )
    return markush_meta


def _definition_rank_key(item):
    definition = str(item.get("definition") or "")
    exact_rank = 0 if item.get("match") == "exact" else 1
    concrete_rank = 0 if len(definition) <= 80 else 1
    generic_penalty = definition.lower().count("optionally")
    return (exact_rank, concrete_rank, generic_penalty, len(definition), definition)


# ---------------------------------------------------------------------------
# Stage C — 병합
# ---------------------------------------------------------------------------
def merge(structures_csv, mg_results, activity_map, out_json, assay_pages=None,
          assay_metadata_map=None, markush_definition_index=None):
    """BioChemInsight structures.csv + MarkushGrapher cxsmiles + 활성값을
    seg_key / compound_id 로 병합."""
    import csv
    rows = []
    if not Path(structures_csv).exists():
        print(f"[Merge] 구조 CSV 없음 — 빈 구조 결과로 계속 진행: {structures_csv}")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        return rows

    activity_lookup = {
        assay_name: _build_activity_lookup(amap)
        for assay_name, amap in activity_map.items()
    }
    assay_meta = _build_assay_metadata_map(activity_map, assay_pages, assay_metadata_map)
    with open(structures_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # structures.csv 의 SEGMENT_FILE 에서 seg_key 추출
            seg_file = r.get("SEGMENT_FILE", "")
            seg_key = None
            if seg_file:
                stem = Path(seg_file).stem  # segment_242_0
                parts = stem.split("_")
                if len(parts) >= 3:
                    seg_key = f"{parts[1]}_{parts[2]}"
            mg = mg_results.get(seg_key, {}) if seg_key else {}

            cid_raw = r.get("COMPOUND_ID")
            cid = _normalize_compound_id(cid_raw)   # ID 정제(설명문장 제거)
            markush_meta = _extract_markush_metadata(mg.get("cxsmiles"), mg.get("cxsmiles_opt"))
            markush_meta = _attach_variable_definitions(
                markush_meta, markush_definition_index or {})

            # 활성값 조인: assay 이름별 {원문, 정규화}
            activities = {}
            for assay_name, amap in activity_lookup.items():
                hit = _lookup_activity_value(amap, cid, cid_raw)
                if hit:
                    raw, matched_id, method = hit
                    activities[assay_name] = _make_activity_record(
                        raw, assay_name, matched_id, method, assay_meta)

            rows.append({
                "compound_id": cid,
                "compound_id_raw": cid_raw if cid_raw != cid else None,
                "page": r.get("PAGE_NUM"),
                "seg_key": seg_key,
                "smiles_molscribe": r.get("SMILES"),
                "cxsmiles_markush": mg.get("cxsmiles"),
                "cxsmiles_opt": mg.get("cxsmiles_opt"),
                "markush_metadata": markush_meta,
                "markush_groups": markush_meta.get("r_groups", []),
                "substituents": markush_meta.get("substituents", []),
                "is_markush": markush_meta.get("is_markush", False),
                "activities": activities,
            })
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return rows


def _compound_id_keys(cid):
    """화합물 ID의 표기 차이를 흡수하기 위한 lookup key 목록."""
    if cid is None:
        return []
    import re
    s = str(cid).strip()
    if not s:
        return []
    keys = [s]
    compact = re.sub(r"\s+", "", s)
    keys.append(compact)
    lower = s.lower()
    keys.append(lower)
    keys.append(re.sub(r"\s+", "", lower))

    m = re.match(r"^(?:compound|compd\.?|example|ex\.?)\s*[-#:]*\s*(.+)$", s, re.I)
    if m:
        tail = m.group(1).strip()
        keys.extend([tail, re.sub(r"\s+", "", tail), tail.lower(), re.sub(r"\s+", "", tail.lower())])
    return list(dict.fromkeys(keys))


def _build_activity_lookup(activity_values):
    """활성값 dict를 원본 키 + 정규화 키로 조회 가능하게 변환."""
    lookup = {}
    for key, value in (activity_values or {}).items():
        for k in _compound_id_keys(key):
            lookup.setdefault(k, {"value": value, "source_id": str(key).strip()})
    return lookup


def _lookup_activity_value(activity_lookup, cid, cid_raw=None):
    for source, method in ((cid, "normalized_id"), (cid_raw, "raw_id")):
        for key in _compound_id_keys(source):
            if key in activity_lookup:
                hit = activity_lookup[key]
                return hit["value"], hit["source_id"], method
    return None


def _extract_entry_token(text):
    """Entry/Example/Compound 번호 기반 후처리 매칭용 토큰 추출."""
    if text is None:
        return None
    import re
    s = str(text).strip()
    patterns = [
        r"^(?:entry|example|compound|compd\.?)\s*[-#:]*\s*([0-9]+[A-Za-z]?)\b",
        r"^\(?([0-9]+[A-Za-z]?)\)?\b",
    ]
    for pat in patterns:
        m = re.search(pat, s, re.I)
        if m:
            return m.group(1).lower()
    return None


def _compact_text(text):
    if text is None:
        return ""
    import re
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _match_unlinked_observation(rows, source_id):
    source_keys = set(_compound_id_keys(source_id))
    source_entry = _extract_entry_token(source_id)
    source_compact = _compact_text(source_id)

    best = {"row": None, "method": None, "confidence": 0.0}
    for row in rows:
        candidates = [
            row.get("compound_id"),
            row.get("compound_id_raw"),
            row.get("coref_label"),
        ]
        row_keys = set()
        row_entries = set()
        row_compacts = []
        for cand in candidates:
            row_keys.update(_compound_id_keys(cand))
            ent = _extract_entry_token(cand)
            if ent:
                row_entries.add(ent)
            comp = _compact_text(cand)
            if comp:
                row_compacts.append(comp)

        if source_keys & row_keys:
            return {"row": row, "method": "unlinked_exact_id", "confidence": 0.85}
        if source_entry and source_entry in row_entries:
            return {"row": row, "method": "unlinked_entry_id", "confidence": 0.75}

        # Keep weak name containment as a candidate only; it is not auto-linked.
        for comp in row_compacts:
            if len(comp) >= 4 and comp in source_compact:
                if best["confidence"] < 0.55:
                    best = {"row": row, "method": "unlinked_name_contains_id", "confidence": 0.55}
    return best


def _activity_link_confidence(method):
    return {
        "coref_label": 0.95,
        "unlinked_exact_id": 0.85,
        "normalized_id": 0.85,
        "raw_id": 0.75,
        "unlinked_entry_id": 0.75,
        "unlinked_name_contains_id": 0.55,
    }.get(method, 0.5)


def _parse_pages_string(pages):
    if not pages:
        return []
    out = []
    for part in str(pages).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and not part.startswith("-"):
            try:
                a, b = [int(x.strip()) for x in part.split("-", 1)]
                out.extend(range(min(a, b), max(a, b) + 1))
            except ValueError:
                continue
        else:
            try:
                out.append(int(part))
            except ValueError:
                continue
    return sorted(set(out))


def _infer_assay_metadata(assay_name, assay_pages=None):
    """assay 이름/페이지에서 downstream에 필요한 최소 메타데이터를 구성."""
    import re
    name = str(assay_name or "").strip()
    endpoint = None
    for pat, label in [
        (r"\bpIC50\b", "pIC50"), (r"\bpEC50\b", "pEC50"),
        (r"IC50|IC₅₀", "IC50"), (r"EC50|EC₅₀", "EC50"),
        (r"\bGI50\b", "GI50"), (r"\bKi\b", "Ki"),
        (r"\bKd\b", "Kd"), (r"\bMIC\b", "MIC"),
    ]:
        if re.search(pat, name, re.I):
            endpoint = label
            break
    unit_hint = None
    m = re.search(r"\(([^)]*?M)\)", name)
    if m:
        u = re.search(r"(mM|nM|pM|fM|[uµμ]M|\bM)", m.group(1), re.I)
        if u:
            unit_hint = u.group(1).replace("uM", "µM").replace("μM", "µM")
    return {
        "assay_name": name,
        "assay_type": endpoint,
        "unit_hint": unit_hint,
        "source_pages": _parse_pages_string(assay_pages),
        "target": None,
        "cell_line": None,
        "organism": None,
        "condition": None,
        "evidence_text": None,
    }


def _merge_assay_metadata(base, extracted):
    if not isinstance(extracted, dict):
        return base
    out = dict(base)
    aliases = {"unit": "unit_hint"}
    for key, value in extracted.items():
        if value in (None, "", []):
            continue
        dst = aliases.get(key, key)
        if dst == "source_pages" and isinstance(value, list):
            out[dst] = value
        elif dst in out:
            out[dst] = value
        else:
            out[dst] = value
    return out


def _extract_markush_metadata(cxsmiles, cxsmiles_opt=None):
    """CXSMILES/CXSMILES_OPT에서 Markush 관련 구조 정보를 추출.

    MarkushGrapher 2.0의 현재 JSONL에는 별도 substituent table 필드가 없으므로,
    후속 분석이 가능한 최소 구조화 정보(R-label, wildcard, SGroup)를 만든다.
    """
    import re
    s = str(cxsmiles or "")
    opt = str(cxsmiles_opt or "")
    labels = []
    labels.extend(re.findall(r"\$([^$]*)\$", s))
    variable_labels = []
    for block in labels:
        for label in block.split(";"):
            label = label.strip()
            if label:
                variable_labels.append(label)
    variable_labels.extend(re.findall(r"<r>([^<]+)</r>", opt))
    variable_labels = sorted(set(variable_labels), key=lambda x: (len(x), x))
    r_groups = [label for label in variable_labels if re.fullmatch(r"R\d*", label)]
    atom_site_labels = [
        label for label in variable_labels
        if re.fullmatch(r"J\d*|Z|X|Y|Q|M", label)
    ]
    scaffold_group_labels = [
        label for label in variable_labels
        if label not in set(r_groups) and label not in set(atom_site_labels)
    ]
    r_groups = sorted(set(r_groups), key=lambda x: (len(x), x))
    wildcard_atoms = re.findall(r"\[\d*\*\]", s)
    has_sgroup = "Sg:" in s
    is_markush = bool(variable_labels or wildcard_atoms or has_sgroup or "<r>" in opt)
    return {
        "is_markush": is_markush,
        "variable_labels": variable_labels,
        "r_groups": r_groups,
        "atom_site_labels": atom_site_labels,
        "scaffold_group_labels": scaffold_group_labels,
        "variable_sites": len(set(wildcard_atoms)) if wildcard_atoms else len(r_groups),
        "wildcard_atoms": sorted(set(wildcard_atoms)),
        "has_sgroup": has_sgroup,
        "substituents": [],
        "substituent_source": "not_available_in_markushgrapher_jsonl",
    }


def _normalize_compound_id(cid):
    """Vision LLM 이 가끔 설명 문장과 함께 ID 를 반환하는 버그 대응.
    예: 'The red box is ... row labeled "238".\\n\\n238' -> '238'
    문장 안의 마지막 줄 또는 따옴표 안 토큰을 ID 로 추출."""
    if cid is None:
        return None
    cid = str(cid).strip()
    if "\n" in cid:
        # 마지막 비어있지 않은 줄을 ID 후보로
        last = [l.strip() for l in cid.splitlines() if l.strip()][-1]
        cid = last
    return cid


# 단위 -> nM 변환 계수
_UNIT_TO_NM = {
    "m": 1e9, "mm": 1e6, "um": 1e3,
    "nm": 1.0, "pm": 1e-3, "fm": 1e-6,
}
# 주: 'mM'.lower() == 'mm' 이므로 위 'mm' 키가 mM(밀리몰)를 처리한다.
_UNIT_CANON = {"um": "µM", "mm": "mM", "nm": "nM",
               "pm": "pM", "fm": "fM", "m": "M"}


def _normalize_value(raw, assay_name=None, metadata=None):
    """활성값 원문 -> {operator, value_num, unit, value_nM} 정규화.

    - 부등호(<,>,≤,≥) 분리
    - 단위(M/mM/µM/nM/pM) 인식 후 nM 표준값(value_nM) 산출
    - 단위가 원문에 없으면 assay_name 의 괄호 단위(예 "EC50 (µM)")로 보강
    - 별표 등급(****)·NT·ND 등 비수치는 value_num=None 으로 보존
    """
    import re
    s = str(raw).strip()
    out = {
        "operator": None,
        "value_num": None,
        "unit": None,
        "value_nM": None,
        "range_lower_nM": None,
        "range_upper_nM": None,
    }

    categorical = _normalize_categorical_value(s, metadata)
    if categorical:
        out.update(categorical)
        return out

    m = re.match(
        r"^\s*([<>≤≥]=?)?\s*([\d.]+(?:[eE][-+]?\d+)?)\s*"
        r"(mM|nM|pM|fM|[uµμ]M|M)?",
        s, re.IGNORECASE)
    if not (m and m.group(2)):
        return out

    out["operator"] = m.group(1) or "="
    try:
        out["value_num"] = float(m.group(2))
    except ValueError:
        return out

    unit_raw = m.group(3)
    if not unit_raw and assay_name:
        # 어세이명 괄호 안 단위 추출: "TR-FRET EC50 (µM)" -> µM
        am = re.search(r"\(([^)]*?M)\)", assay_name)
        if am:
            cand = re.search(r"(mM|nM|pM|fM|[uµμ]M|\bM)", am.group(1), re.IGNORECASE)
            if cand:
                unit_raw = cand.group(1)

    if unit_raw:
        key = unit_raw.lower().replace("µ", "u").replace("μ", "u")
        if key in _UNIT_TO_NM:
            out["unit"] = _UNIT_CANON.get(key, unit_raw)
            out["value_nM"] = out["value_num"] * _UNIT_TO_NM[key]
            if out["operator"] in ("<", "<="):
                out["range_upper_nM"] = out["value_nM"]
            elif out["operator"] in (">", ">="):
                out["range_lower_nM"] = out["value_nM"]
    return out


def _normalize_categorical_value(raw, metadata=None):
    """Assay legend 기반 A/B/C 등급값을 nM 범위로 변환."""
    import re
    grade = str(raw or "").strip().upper()
    if not re.fullmatch(r"[A-Z]", grade):
        return None
    if not isinstance(metadata, dict):
        return None

    evidence = " ".join(str(metadata.get(k) or "") for k in ("evidence_text", "condition"))
    if not evidence:
        return None

    m = re.search(
        rf"{re.escape(grade)}\s*=\s*(.*?)(?=\b[A-Z]\s*=|$)",
        evidence,
        re.I,
    )
    if not m:
        return None
    segment = m.group(1)
    values = [
        (float(num), unit)
        for num, unit in re.findall(r"([\d.]+)\s*(mM|nM|pM|fM|[uµμ]M|M)", segment, re.I)
    ]
    if not values:
        return None

    def to_nm(num, unit):
        key = unit.lower().replace("µ", "u").replace("μ", "u")
        return num * _UNIT_TO_NM[key], _UNIT_CANON.get(key, unit)

    lower = "greater than" in segment.lower() or ">" in segment
    upper = "less than" in segment.lower() or "<" in segment

    if lower and upper and len(values) >= 2:
        lo, unit = to_nm(values[0][0], values[0][1])
        hi, _ = to_nm(values[1][0], values[1][1])
        return {
            "operator": "range",
            "value_num": None,
            "unit": unit,
            "value_nM": None,
            "range_lower_nM": lo,
            "range_upper_nM": hi,
        }
    val, unit = to_nm(values[0][0], values[0][1])
    if upper:
        return {
            "operator": "<",
            "value_num": values[0][0],
            "unit": unit,
            "value_nM": val,
            "range_lower_nM": None,
            "range_upper_nM": val,
        }
    if lower:
        return {
            "operator": ">",
            "value_num": values[0][0],
            "unit": unit,
            "value_nM": val,
            "range_lower_nM": val,
            "range_upper_nM": None,
        }
    return None


def _has_grade_legend(metadata):
    if not isinstance(metadata, dict):
        return False
    import re
    text = " ".join(str(metadata.get(k) or "") for k in ("evidence_text", "condition"))
    return bool(re.search(r"\bA\s*=.*?\bB\s*=.*?\bC\s*=", text, re.I))


def _metadata_with_grade_legend(metadata, metadata_candidates=None):
    """같은 table의 공통 A/B/C legend가 특정 assay metadata에 빠진 경우 보강."""
    if _has_grade_legend(metadata):
        return metadata
    out = dict(metadata or {})
    for cand in metadata_candidates or []:
        if _has_grade_legend(cand):
            for key in ("evidence_text", "condition"):
                if cand.get(key):
                    out[f"legend_{key}"] = cand.get(key)
            out["evidence_text"] = cand.get("evidence_text") or cand.get("condition")
            return out
    return metadata


def _looks_markush(cxsmiles):
    """CXSMILES 에 R-group/SGroup 마커가 있으면 Markush 로 판단.

    정교화: 와일드카드 원자([*], [1*]), CXSMILES 의 $R..$ 라벨, SGroup(Sg:),
    또는 변수 위치 마커를 종합 판정.
    """
    if not cxsmiles:
        return False
    import re
    s = cxsmiles
    # [*], [1*], [R], <r>R</r> 등 와일드카드/변수 원자
    if re.search(r"\[\d*\*\]", s):
        return True
    if ("$R" in s) or ("Sg:" in s) or ("<r>" in s):
        return True
    return False


# ---------------------------------------------------------------------------
# RDKit 후처리: 유효성 검증 + 두 엔진 교차검증 신뢰도
# ---------------------------------------------------------------------------
def _rdkit_canonical(smiles, ignore_stereo=False):
    """SMILES/CXSMILES -> (valid, canonical). RDKit 미설치 시 (None, None)."""
    if not smiles:
        return None, None
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return None, None
    s = str(smiles).split("|")[0].strip()  # CXSMILES 의 좌표/그룹 꼬리 제거
    m = Chem.MolFromSmiles(s)
    if m is None:
        return False, None
    if ignore_stereo:
        Chem.RemoveStereochemistry(m)
    try:
        return True, Chem.MolToSmiles(m)
    except Exception:
        return True, None


def enrich_with_rdkit(rows):
    """각 row 에 RDKit 검증/교차검증 신뢰도 필드 추가.

    - smiles_valid / cxsmiles_valid : 파싱 가능 여부
    - canonical_smiles : MolScribe SMILES 의 canonical (입체 포함)
    - agreement : 두 엔진 결과 비교
        'match'           : 입체 포함 canonical 동일
        'match_skeleton'  : 입체 무시 시 동일 (입체만 차이)
        'mismatch'        : 골격도 다름
        'single'          : 한쪽만 존재
        'unparsable'      : 파싱 실패로 비교 불가
    - confidence : high / medium / low
    """
    for r in rows:
        smi = r.get("smiles_molscribe")
        cxs = r.get("cxsmiles_markush")

        v1, c1 = _rdkit_canonical(smi)
        v2, c2 = _rdkit_canonical(cxs)
        _, c1s = _rdkit_canonical(smi, ignore_stereo=True)
        _, c2s = _rdkit_canonical(cxs, ignore_stereo=True)

        r["smiles_valid"] = v1
        r["cxsmiles_valid"] = v2
        r["canonical_smiles"] = c1

        if not smi or not cxs:
            agreement = "single"
        elif v1 is False or v2 is False:
            agreement = "unparsable"
        elif c1 and c2 and c1 == c2:
            agreement = "match"
        elif c1s and c2s and c1s == c2s:
            agreement = "match_skeleton"
        elif c1 is None or c2 is None:
            agreement = "unparsable"
        else:
            agreement = "mismatch"
        r["agreement"] = agreement

        # 신뢰도: 두 엔진 골격 일치 + ID 정제 안 거침 => high
        id_clean = r.get("compound_id_raw") is None
        if agreement in ("match", "match_skeleton") and id_clean:
            conf = "high"
        elif agreement in ("match", "match_skeleton"):
            conf = "medium"
        elif agreement == "single" and (v1 or v2):
            conf = "medium"
        else:
            conf = "low"
        r["confidence"] = conf
    return rows


def write_csv(rows, out_csv):
    """통합 결과를 평탄화된 CSV 로 저장 (어세이는 첫 어세이 기준 컬럼화)."""
    import csv
    fields = [
        "compound_id", "compound_id_raw", "page", "seg_key",
        "identity_status", "identity_source", "compound_name",
        "structure_source", "structure_confidence", "structure_image",
        "smiles_molscribe", "canonical_smiles", "smiles_valid",
        "cxsmiles_markush", "cxsmiles_valid", "is_markush",
        "markush_groups", "markush_variable_labels",
        "markush_atom_site_labels", "markush_scaffold_group_labels",
        "markush_variable_sites", "markush_has_sgroup",
        "markush_definition_coverage", "markush_defined_labels",
        "agreement", "confidence", "coref_label", "coref_id_agree",
        "assay_name", "value_raw", "operator", "value_num", "unit", "value_nM",
        "range_lower_nM", "range_upper_nM",
        "matched_activity_id", "link_method", "link_confidence",
        "assay_type", "assay_unit_hint", "assay_source_pages",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            base = {k: r.get(k) for k in fields}
            mm = r.get("markush_metadata") or {}
            base.update({
                "markush_groups": ",".join(r.get("markush_groups") or []),
                "markush_variable_labels": ",".join(mm.get("variable_labels") or []),
                "markush_atom_site_labels": ",".join(mm.get("atom_site_labels") or []),
                "markush_scaffold_group_labels": ",".join(mm.get("scaffold_group_labels") or []),
                "markush_variable_sites": mm.get("variable_sites"),
                "markush_has_sgroup": mm.get("has_sgroup"),
                "markush_definition_coverage": mm.get("definition_coverage"),
                "markush_defined_labels": ",".join(
                    sorted((mm.get("variable_definitions") or {}).keys())),
            })
            acts = r.get("activities") or {}
            if acts:
                # 어세이별로 한 행씩 (long format)
                for an, a in acts.items():
                    row = dict(base)
                    row.update({
                        "assay_name": an,
                        "value_raw": a.get("value_raw"),
                        "operator": a.get("operator"),
                        "value_num": a.get("value_num"),
                        "unit": a.get("unit"),
                        "value_nM": a.get("value_nM"),
                        "range_lower_nM": a.get("range_lower_nM"),
                        "range_upper_nM": a.get("range_upper_nM"),
                        "matched_activity_id": a.get("matched_activity_id"),
                        "link_method": a.get("link_method"),
                        "link_confidence": a.get("link_confidence"),
                    })
                    meta = a.get("metadata") or {}
                    row.update({
                        "assay_type": meta.get("assay_type"),
                        "assay_unit_hint": meta.get("unit_hint"),
                        "assay_source_pages": ",".join(str(p) for p in meta.get("source_pages") or []),
                    })
                    w.writerow(row)
            else:
                w.writerow(base)


def write_assay_observations(observations, out_json, out_csv):
    """Unlinked/late-linked assay observations를 JSON/CSV로 저장."""
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(observations or [], f, ensure_ascii=False, indent=2)

    import csv
    fields = [
        "status", "assay_name", "source_id", "value_raw", "operator",
        "value_num", "unit", "value_nM", "range_lower_nM",
        "range_upper_nM", "matched_compound_id", "matched_seg_key",
        "match_method", "link_confidence",
        "assay_type", "assay_unit_hint", "assay_target", "assay_source_pages",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for obs in observations or []:
            meta = obs.get("metadata") or {}
            row = {k: obs.get(k) for k in fields}
            row.update({
                "assay_type": meta.get("assay_type"),
                "assay_unit_hint": meta.get("unit_hint"),
                "assay_target": meta.get("target"),
                "assay_source_pages": ",".join(str(p) for p in meta.get("source_pages") or []),
            })
            w.writerow(row)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="PatentAgent 통합 프로토타입")
    ap.add_argument("pdf")
    ap.add_argument("--structure-pages", default=None,
                    help="구조 페이지 (미지정+--auto-pages 시 자동 탐지)")
    ap.add_argument("--assay-pages", default=None)
    ap.add_argument("--assay-names", default=None)
    ap.add_argument("--auto-pages", action="store_true",
                    help="구조/활성 페이지를 PDF에서 자동 탐지 (autodetect_pages.py)")
    ap.add_argument("--engine", default="molscribe")
    ap.add_argument("--out", default="integration_out")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--skip-bci", action="store_true",
                    help="BioChemInsight 재실행 생략하고 기존 --out 산출물 사용")
    ap.add_argument("--mg-service-url", default=None,
                    help="MarkushGrapher REST 서비스 URL (예: http://localhost:8100). "
                         "지정 시 subprocess 대신 HTTP 호출.")
    ap.add_argument("--skip-markush", action="store_true",
                    help="기존 merged_integration.json의 Markush 결과를 재사용")
    ap.add_argument("--markush-def-pages", default=None,
                    help="Markush 변수 정의를 추출할 PDF 페이지 범위 "
                         "(예: '1-8,37-38'; 미지정 시 전체 텍스트 검색)")
    ap.add_argument("--no-markush-definitions", action="store_true",
                    help="PDF 텍스트 기반 Markush 변수 정의 추출 생략")
    ap.add_argument("--no-activity-only-rows", action="store_true",
                    help="구조와 아직 연결되지 않은 assay Entry row를 merged 결과에 추가하지 않음")
    ap.add_argument("--entry-name-pages", default=None,
                    help="Entry Name table을 추출할 PDF 페이지 범위 "
                         "(미지정 시 --assay-pages 사용)")
    ap.add_argument("--claim-structure-pages", default=None,
                    help="claim table의 Entry 구조 이미지를 직접 OCSR할 PDF 페이지 범위")
    ap.add_argument("--claim-structure-dpi", type=int, default=200,
                    help="claim 구조 crop용 PDF 렌더링 DPI")
    ap.add_argument("--claim-structure-min-confidence", type=float, default=0.75,
                    help="claim 구조 MolScribe 후보 최소 confidence")
    ap.add_argument("--with-coref", action="store_true",
                    help="OpenChemIE MolCoref 로 구조↔ID 연결 교차검증 (페이지 이미지 필요)")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bci_out = out_dir / "bci"

    # Stage 0 — (옵션) 자동 페이지 탐지 + 어세이명 자동 추출
    if args.auto_pages and not args.skip_bci:
        struct, assay, auto_names = _autodetect(args.pdf, gpu=args.gpu)
        if not args.structure_pages and struct:
            args.structure_pages = struct
            print(f"[AutoPages] 구조 페이지 자동 탐지: {struct}")
        if not args.assay_pages and assay:
            args.assay_pages = assay
            print(f"[AutoPages] 활성 페이지 자동 탐지: {assay}")
        # 어세이명: 사용자 미지정 시 자동 추출값 사용, 그것도 없으면 IC50 폴백
        if not args.assay_names and assay:
            if auto_names:
                args.assay_names = auto_names
                print(f"[AutoPages] 어세이명 자동 추출: {auto_names}")
            else:
                args.assay_names = "IC50"
                print(f"[AutoPages] 어세이명 추출 실패 → 기본값 'IC50' 사용")

    if not args.structure_pages and not args.skip_bci:
        ap.error("--structure-pages 가 필요합니다 (또는 --auto-pages 사용)")

    # Stage A
    if args.skip_bci:
        structures_csv = bci_out / "structures.csv"
        print(f"[BioChemInsight] 생략, 기존 산출물 사용: {structures_csv}")
    else:
        structures_csv = run_biocheminsight(
            args.pdf, args.structure_pages, args.assay_pages,
            args.assay_names, bci_out, engine=args.engine, gpu=args.gpu)
    print(f"[BioChemInsight] structures.csv = {structures_csv}")

    segs = find_segment_images(bci_out)
    print(f"[BioChemInsight] 구조 세그먼트 {len(segs)}개 검출")

    # Stage A2 — 활성값 로드 (pipeline.py 가 assay_pages/names 와 함께 실행되면 생성됨)
    activity_map = load_activity(bci_out, args.assay_names)
    unlinked_activity_map = load_unlinked_activity(bci_out, args.assay_names)
    assay_metadata_map = load_assay_metadata(bci_out, args.assay_names)

    # Stage B
    if args.skip_markush:
        mg_results, pred_file = load_markush_from_merged(out_dir / "merged_integration.json")
    else:
        mg_results, pred_file = run_markushgrapher(
            segs, out_dir, gpu=args.gpu, service_url=args.mg_service_url)
    print(f"[MarkushGrapher] {len(mg_results)}개 CXSMILES 예측, 결과: {pred_file}")

    markush_definition_index = {}
    if not args.no_markush_definitions:
        markush_definition_index = load_markush_definition_index(
            args.pdf, pages=args.markush_def_pages)

    # Stage C — 병합
    out_json = out_dir / "merged_integration.json"
    rows = merge(structures_csv, mg_results, activity_map, out_json,
                 assay_pages=args.assay_pages,
                 assay_metadata_map=assay_metadata_map,
                 markush_definition_index=markush_definition_index)

    # Stage D — RDKit 검증 + 두 엔진 교차검증 신뢰도
    rows = enrich_with_rdkit(rows)

    # Stage D2 — (옵션) OpenChemIE MolCoref 로 구조↔ID 연결 교차검증
    if args.with_coref:
        coref_results = run_molcoref(bci_out, gpu=args.gpu)
        n_cor = attach_coref(rows, coref_results)
        n_coref_act = attach_activity_by_coref(
            rows, activity_map, assay_pages=args.assay_pages,
            assay_metadata_map=assay_metadata_map)
        n_agree = sum(1 for r in rows if r.get("coref_id_agree"))
        print(f"[MolCoref] {n_cor}개 구조에 라벨 연결 "
              f"(BioChemInsight ID와 일치 {n_agree}개, 활성값 보강 {n_coref_act}개)")

    assay_observations, n_unlinked_attached = attach_unlinked_activity_candidates(
        rows, unlinked_activity_map, assay_pages=args.assay_pages,
        assay_metadata_map=assay_metadata_map)
    if assay_observations:
        n_unresolved = sum(1 for o in assay_observations if o.get("status") == "unlinked")
        print(f"[Activity] unlinked 관측값 {len(assay_observations)}개 보존 "
              f"(후처리 자동 연결 {n_unlinked_attached}개, 미연결 {n_unresolved}개)")
        if not args.no_activity_only_rows:
            n_activity_only = append_activity_only_rows(rows, assay_observations)
            if n_activity_only:
                print(f"[Activity] activity-only unresolved row {n_activity_only}개 추가")

    entry_name_pages = args.entry_name_pages or args.assay_pages
    entry_name_map = load_entry_name_map(args.pdf, pages=entry_name_pages)
    n_named = attach_entry_names(rows, entry_name_map)
    if n_named:
        print(f"[Activity] Entry Name {n_named}개 row에 연결")

    if args.claim_structure_pages:
        claim_candidates = extract_claim_structure_candidates(
            args.pdf, args.claim_structure_pages, out_dir, gpu=args.gpu,
            dpi=args.claim_structure_dpi,
            min_confidence=args.claim_structure_min_confidence)
        with open(out_dir / "claim_structure_candidates.json", "w", encoding="utf-8") as f:
            json.dump(claim_candidates, f, ensure_ascii=False, indent=2)
        n_claim_structures = attach_claim_structures(rows, claim_candidates)
        if n_claim_structures:
            rows = enrich_with_rdkit(rows)
            print(f"[ClaimStructure] {n_claim_structures}개 Entry row에 구조 후보 연결")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # Stage E — CSV / Parquet 출력 (검수/분석용 long format)
    out_csv = out_dir / "merged_integration.csv"
    write_csv(rows, out_csv)
    out_parquet = out_dir / "merged_integration.parquet"
    _write_parquet(out_csv, out_parquet)
    obs_json = out_dir / "assay_observations.json"
    obs_csv = out_dir / "assay_observations.csv"
    write_assay_observations(assay_observations, obs_json, obs_csv)

    n_markush = sum(1 for r in rows if r["is_markush"])
    n_act = sum(1 for r in rows if r["activities"])
    from collections import Counter
    conf_dist = Counter(r.get("confidence") for r in rows)
    agree_dist = Counter(r.get("agreement") for r in rows)
    print(f"\n=== 병합 완료: {len(rows)}개 화합물 "
          f"(Markush {n_markush}개, 활성값 보유 {n_act}개) ===")
    print(f"=== 신뢰도 분포: {dict(conf_dist)} ===")
    print(f"=== 교차검증 분포: {dict(agree_dist)} ===")
    print(f"=== 출력: {out_json}")
    print(f"===       {out_csv}")
    print(f"===       {obs_json}")
    for r in rows[:5]:
        print(json.dumps(r, ensure_ascii=False))


def _write_parquet(csv_path, parquet_path):
    """CSV -> Parquet (pandas+pyarrow 있을 때만). 실패해도 파이프라인은 계속."""
    try:
        import pandas as pd
        pd.read_csv(csv_path).to_parquet(parquet_path, index=False)
        print(f"=== Parquet: {parquet_path}")
    except Exception as exc:  # pragma: no cover
        print(f"[Parquet] 생략 ({exc})")


def _postprocess_only(out_dir):
    """기존 merged_integration.json 에 RDKit enrich + CSV 만 재적용 (디버그용)."""
    out_dir = Path(out_dir).resolve()
    out_json = out_dir / "merged_integration.json"
    rows = json.load(open(out_json, encoding="utf-8"))
    rows = enrich_with_rdkit(rows)
    json.dump(rows, open(out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    write_csv(rows, out_dir / "merged_integration.csv")
    _write_parquet(out_dir / "merged_integration.csv",
                   out_dir / "merged_integration.parquet")
    return rows


if __name__ == "__main__":
    main()
