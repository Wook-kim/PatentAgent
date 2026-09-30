#!/usr/bin/env python3
"""
자동 페이지 탐지 (auto page detection)
=====================================

특허/논문 PDF 에서 구조 페이지와 활성값(어세이) 페이지를 자동 탐지해,
integrate_prototype.py 의 --structure-pages / --assay-pages 수동 지정을 대체한다.

휴리스틱(빠른 텍스트 기반, DECIMER 불필요):
  - 구조 페이지: 텍스트 적음(<struct_text_max) + 이미지 임베드(>=1)
       특허의 구조 다이어그램은 보통 이미지로 들어가고 본문 텍스트가 거의 없다.
  - 활성 페이지: 어세이 키워드(IC50/EC50/Ki/inhibition/...) >= assay_kw_min
       + 텍스트 충분(>assay_text_min)  (표/측정값이 있는 페이지)

연속 페이지를 "243-267,270" 형태의 압축 문자열로 반환.

사용:
    python3 autodetect_pages.py <pdf> [--bci-python <path>]
    -> JSON: {"structure_pages":"243-267", "assay_pages":"268-273",
              "structure_list":[...], "assay_list":[...]}

integrate_prototype 에서 import 해서 detect_pages(pdf) 로도 사용 가능.
주의: PyMuPDF(fitz) 필요 -> BioChemInsight/.venv 의 python 으로 실행.
"""
import argparse
import json
import os
import re
from pathlib import Path

ASSAY_KW = re.compile(
    r"IC50|EC50|\bKi\b|\bKd\b|IC₅₀|EC₅₀|inhibition|\bactivity\b|"
    r"TR-?FRET|\bassay\b|\bpotency\b|\bGI50\b|\bMIC\b",
    re.IGNORECASE)

# 실제 어세이 '엔드포인트명'만 (컬럼명으로 쓸 만한 것). inhibition/activity 같은
# 일반 단어는 제외. (정규식, 정규화된 표시명) 순서 — 먼저 매치된 것이 우선.
ASSAY_NAME_PATTERNS = [
    (re.compile(r"TR-?FRET\s*(?:IC50|EC50)?", re.I), "TR-FRET"),
    (re.compile(r"\bpIC50\b", re.I), "pIC50"),
    (re.compile(r"\bpEC50\b", re.I), "pEC50"),
    (re.compile(r"IC50|IC₅₀", re.I), "IC50"),
    (re.compile(r"EC50|EC₅₀", re.I), "EC50"),
    (re.compile(r"\bGI50\b", re.I), "GI50"),
    (re.compile(r"\bKi\b", re.I), "Ki"),
    (re.compile(r"\bKd\b", re.I), "Kd"),
    (re.compile(r"\bMIC\b", re.I), "MIC"),
]


def extract_assay_names(pdf_path, assay_pages, max_names=3):
    """활성 페이지들에서 어세이 엔드포인트명을 빈도순으로 추출.

    assay_pages: 1-based 페이지 번호 리스트(detect_pages 의 assay 반환).
    반환: 빈도 높은 순 어세이명 리스트 (예: ['IC50', 'EC50']). 없으면 [].
    """
    import fitz
    from collections import Counter
    if not assay_pages:
        return []
    counts = Counter()
    doc = fitz.open(str(pdf_path))
    try:
        for p in assay_pages:
            if not (1 <= p <= doc.page_count):
                continue
            text = doc[p - 1].get_text()
            for rx, label in ASSAY_NAME_PATTERNS:
                n = len(rx.findall(text))
                if n:
                    counts[label] += n
    finally:
        doc.close()
    # TR-FRET 가 잡히면 그 안의 IC50/EC50 중복 카운트가 있을 수 있으나,
    # 빈도순 상위 max_names 개만 취하므로 실용상 문제 없음.
    return [name for name, _ in counts.most_common(max_names)]


def _bridge_gaps(pages, max_gap=2):
    """연속 구간 사이의 작은 갭(<=max_gap 페이지)을 메운다.

    예: [243..251, 253..267] 에서 252 한 칸 갭을 메워 243..267 로 잇는다.
    구조 영역 중간에 텍스트 많은 페이지(설명/표제)가 끼어 런이 끊기는 것을 보정.
    """
    if not pages:
        return []
    pages = sorted(set(pages))
    filled = [pages[0]]
    for p in pages[1:]:
        gap = p - filled[-1]
        if 1 < gap <= max_gap + 1:
            filled.extend(range(filled[-1] + 1, p))
        filled.append(p)
    return filled


def _split_runs(pages):
    """연속 구간들의 리스트로 분할."""
    if not pages:
        return []
    pages = sorted(set(pages))
    runs, cur = [], [pages[0]]
    for p in pages[1:]:
        if p == cur[-1] + 1:
            cur.append(p)
        else:
            runs.append(cur)
            cur = [p]
    runs.append(cur)
    return runs


def _largest_run(pages, min_len=3, weight=None):
    """가장 '주력'인 연속 구간을 반환.

    weight 가 주어지면 {page: 점수} 합이 최대인 런을 선택(동점/근접 시 키워드
    밀도 우선) — 활성 페이지처럼 길이만으론 구분 안 되는 경우에 사용.
    weight 없으면 최장 런. min_len 미만이면 전체 반환.
    """
    runs = _split_runs(pages)
    if not runs:
        return []
    if weight is not None:
        best = max(runs, key=lambda r: (sum(weight.get(p, 0) for p in r), len(r)))
    else:
        best = max(runs, key=len)
    return best if len(best) >= min_len else sorted(set(pages))


def _compress_ranges(pages):
    """[243,244,245,247] -> '243-245,247'."""
    if not pages:
        return ""
    pages = sorted(set(pages))
    out, start, prev = [], pages[0], pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = p
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(out)


def _detect_structures_decimer(pdf_path, doc, zoom=2, max_pages=120):
    """DECIMER 세그멘테이션으로 각 페이지의 구조 검출 수를 세어 구조 페이지 판정.

    스캔 PDF 전용(텍스트 휴리스틱이 안 먹힘). 페이지를 렌더링해 DECIMER 에 넣고
    검출 구조 수 >=1 인 페이지를 구조 페이지로 본다. 느리지만(페이지당 ~2초)
    스캔본에선 유일하게 신뢰할 수 있는 신호.

    max_pages: 스캔 범위 상한(과도하게 큰 PDF에서 폭주 방지). 초과분은 건너뛰고
    경고를 stderr 로 남긴다.
    반환: {page_1based: 구조수}
    """
    import sys
    import tempfile
    import cv2
    import fitz
    from decimer_segmentation import get_expanded_masks, apply_masks

    counts = {}
    tmp = os.path.join(tempfile.gettempdir(), "_autodetect_pg.png")
    n_scan = min(doc.page_count, max_pages)
    if doc.page_count > max_pages:
        print(f"[autodetect] DECIMER 폴백: 페이지 {doc.page_count}개 중 처음 "
              f"{max_pages}개만 스캔(상한). 구조가 뒤쪽에 있으면 수동 페이지 지정 필요.",
              file=sys.stderr)
    for i in range(n_scan):
        doc[i].get_pixmap(matrix=fitz.Matrix(zoom, zoom)).save(tmp)
        page = cv2.imread(tmp)
        if page is None:
            counts[i + 1] = 0
            continue
        try:
            segs, _ = apply_masks(page, get_expanded_masks(page))
            counts[i + 1] = len(segs)
        except Exception:
            counts[i + 1] = 0
    return counts


def detect_pages(pdf_path,
                 struct_text_max=400, struct_min_images=1,
                 assay_kw_min=2, assay_text_min=400,
                 largest_run_only=True):
    """PDF -> (structure_list, assay_list) 1-based 페이지 번호.

    largest_run_only=True 면 각 카테고리에서 가장 긴 연속 구간만 취해
    흩어진 오탐을 제거한다(특허의 주력 영역은 보통 한 블록).
    fitz 가 없으면 ImportError 를 그대로 올린다(호출측에서 처리).

    구조 페이지는 먼저 빠른 텍스트 휴리스틱(저텍스트+이미지)으로 찾고,
    0개면(= OCR 텍스트가 풍부한 스캔 특허 등) DECIMER 세그멘테이션으로
    구조를 직접 검출하는 폴백을 쓴다.
    """
    import fitz
    doc = fitz.open(str(pdf_path))

    # --- 활성 페이지: 어세이 키워드 기반 (텍스트레이어 사용) ---
    assay, kw_weight = [], {}
    for i in range(doc.page_count):
        text = doc[i].get_text()
        n_kw = len(ASSAY_KW.findall(text))
        kw_weight[i + 1] = n_kw
        if n_kw >= assay_kw_min and len(text.strip()) > assay_text_min:
            assay.append(i + 1)

    # --- 구조 페이지: 빠른 텍스트 휴리스틱 우선 ---
    structure = []
    for i in range(doc.page_count):
        pg = doc[i]
        tlen = len(pg.get_text().strip())
        n_img = len(pg.get_images())
        if tlen < struct_text_max and n_img >= struct_min_images and (i + 1) not in assay:
            structure.append(i + 1)

    used_decimer = False
    # 텍스트 휴리스틱 구조 후보가 빈약하면(스캔특허는 OCR텍스트가 많아 거의 안 잡힘,
    # 혹은 빈 페이지 1~2장만 오탐) DECIMER 세그멘테이션으로 직접 검출 폴백.
    if len(structure) < 3:
        try:
            counts = _detect_structures_decimer(pdf_path, doc)
            decimer_pages = [p for p, c in counts.items() if c >= 1]
            if len(decimer_pages) >= len(structure):
                structure = decimer_pages
                used_decimer = True
        except Exception as exc:
            print(f"[autodetect] DECIMER 폴백 실패(무시): {exc}")

    doc.close()

    if largest_run_only:
        # 구조: 최장 런 / 활성: 어세이 키워드 밀도 최대 런
        structure = _largest_run(_bridge_gaps(structure))
        assay = _largest_run(_bridge_gaps(assay), weight=kw_weight)
    return structure, assay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--struct-text-max", type=int, default=400)
    ap.add_argument("--assay-kw-min", type=int, default=2)
    args = ap.parse_args()

    s, a = detect_pages(args.pdf,
                        struct_text_max=args.struct_text_max,
                        assay_kw_min=args.assay_kw_min)
    assay_names = extract_assay_names(args.pdf, a)
    result = {
        "structure_pages": _compress_ranges(s),
        "assay_pages": _compress_ranges(a),
        "assay_names": assay_names,                  # 자동 추출된 어세이명 (빈도순)
        "structure_count": len(s),
        "assay_count": len(a),
        "structure_list": s,
        "assay_list": a,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
