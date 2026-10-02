#!/usr/bin/env python3
"""
MolCoref 헬퍼 (OpenChemIE 의 구조↔ID coreference 만 격리 사용)
==============================================================

OpenChemIE 의 interface.py 는 최상단에서 layoutparser 를 import 하는데,
MolCoref(구조-식별자 연결)에는 layoutparser 가 불필요하다. 이 헬퍼는
rxnscribe.MolDetect(coref=True) 를 직접 써서 그 의존을 우회한다.

용도: 페이지(또는 그림) 이미지에서 "분자 구조 ↔ 화합물 라벨(예: 12, 3a)" 짝을
찾아, BioChemInsight 의 Vision-LLM ID 인식을 교차검증/보강한다.

이 스크립트는 OpenChemIE/.venv (torch 1.13 + transformers 4.33.3) 에서 실행한다:
    OpenChemIE/.venv/bin/python molcoref_helper.py <page_image> [page_image2 ...]
    -> JSON: [{image, pairs:[{label, smiles, mol_bbox, label_bbox, score}]}]

REST 로 통합하려면 markush_service.py 와 유사하게 감싸면 된다.
"""
import argparse
import json
import sys
from pathlib import Path

_coref_model = None


def get_coref_model(device=None):
    global _coref_model
    if _coref_model is None:
        import torch
        from rxnscribe import MolDetect
        from huggingface_hub import hf_hub_download
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = hf_hub_download("Ozymandias314/MolDetectCkpt", "coref_best_hf.ckpt")
        _coref_model = MolDetect(ckpt, device=device, coref=True)
    return _coref_model


def extract_corefs(image_paths, molscribe=True, ocr=True):
    """페이지 이미지들에서 구조↔라벨 coref 추출.

    반환: [{image, pairs:[{label, smiles, mol_bbox, label_bbox}], n_mol, n_label}]
    """
    from PIL import Image
    model = get_coref_model()
    figures = [Image.open(p).convert("RGB") for p in image_paths]
    raw = model.predict_images(figures, coref=True, molscribe=molscribe, ocr=ocr)

    out = []
    for path, res in zip(image_paths, raw):
        bboxes = res.get("bboxes", [])
        pairs = []
        for mol_i, lab_i in res.get("corefs", []):
            mol = bboxes[mol_i] if mol_i < len(bboxes) else {}
            lab = bboxes[lab_i] if lab_i < len(bboxes) else {}
            # 라벨 텍스트: OCR 결과가 'text' 필드에 들어옴
            label_text = lab.get("text") or lab.get("label")
            if isinstance(label_text, list):
                label_text = " ".join(str(t) for t in label_text)
            pairs.append({
                "label": label_text,
                "smiles": mol.get("smiles"),
                "mol_bbox": mol.get("bbox"),
                "label_bbox": lab.get("bbox"),
                "score": mol.get("score"),
            })
        out.append({
            "image": str(path),
            "pairs": pairs,
            "n_mol": sum(1 for b in bboxes if b.get("smiles") is not None),
            "n_bbox": len(bboxes),
        })
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--no-molscribe", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = extract_corefs(
        args.images,
        molscribe=not args.no_molscribe,
        ocr=not args.no_ocr,
    )
    text = json.dumps(results, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"saved: {args.out}", file=sys.stderr)
    print(text)
