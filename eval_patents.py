#!/usr/bin/env python3
"""PatentAgent extraction evaluation helper.

Usage:
  python3 eval_patents.py <prediction.json> [--truth truth.json]

Truth format is a JSON list. Minimal fields:
  [{"compound_id": "236", "smiles": "...", "activities": {"IC50": "0.5 uM"},
    "is_markush": false}]

The script is intentionally lightweight so it can be used before a full
benchmark harness exists.
"""
import argparse
import json
from pathlib import Path


def _canon(smiles):
    if not smiles:
        return None
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(str(smiles).split("|")[0].strip())
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def _id_keys(value):
    if value is None:
        return []
    import re
    s = str(value).strip()
    if not s:
        return []
    keys = [s, s.lower(), re.sub(r"\s+", "", s.lower())]
    m = re.match(r"^(?:compound|compd\.?|example|ex\.?)\s*[-#:]*\s*(.+)$", s, re.I)
    if m:
        tail = m.group(1).strip()
        keys.extend([tail, tail.lower(), re.sub(r"\s+", "", tail.lower())])
    return list(dict.fromkeys(keys))


def _index_by_id(rows):
    out = {}
    for row in rows:
        for key in _id_keys(row.get("compound_id")):
            out.setdefault(key, row)
    return out


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(rows):
    n = len(rows)
    valid = sum(1 for r in rows if r.get("smiles_valid") is True or _canon(r.get("smiles_molscribe")))
    with_activity = sum(1 for r in rows if r.get("activities"))
    markush = sum(1 for r in rows if r.get("is_markush"))
    confidence = {}
    for r in rows:
        confidence[r.get("confidence") or "unknown"] = confidence.get(r.get("confidence") or "unknown", 0) + 1
    return {
        "n_predictions": n,
        "valid_smiles_rate": valid / n if n else 0,
        "with_activity": with_activity,
        "markush_detected": markush,
        "confidence": confidence,
    }


def compare(pred_rows, truth_rows):
    pred_idx = _index_by_id(pred_rows)
    matched = 0
    smiles_match = 0
    activity_match = 0
    markush_match = 0
    missing = []

    for truth in truth_rows:
        pred = None
        for key in _id_keys(truth.get("compound_id")):
            pred = pred_idx.get(key)
            if pred:
                break
        if not pred:
            missing.append(truth.get("compound_id"))
            continue
        matched += 1

        t_smi = _canon(truth.get("smiles") or truth.get("canonical_smiles"))
        p_smi = _canon(pred.get("canonical_smiles") or pred.get("smiles_molscribe"))
        if t_smi and p_smi and t_smi == p_smi:
            smiles_match += 1

        if "is_markush" in truth and bool(truth.get("is_markush")) == bool(pred.get("is_markush")):
            markush_match += 1

        t_acts = truth.get("activities") or {}
        p_acts = pred.get("activities") or {}
        if t_acts:
            ok = True
            for assay, expected in t_acts.items():
                got = p_acts.get(assay) or {}
                if str(got.get("value_raw")) != str(expected):
                    ok = False
                    break
            if ok:
                activity_match += 1

    denom = len(truth_rows) or 1
    return {
        "truth_count": len(truth_rows),
        "matched_by_id": matched,
        "id_recall": matched / denom,
        "smiles_accuracy_on_truth": smiles_match / denom,
        "activity_exact_accuracy_on_truth": activity_match / denom,
        "markush_accuracy_on_truth": markush_match / denom,
        "missing_compound_ids": missing[:50],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prediction_json")
    ap.add_argument("--truth", default=None)
    args = ap.parse_args()

    pred = _load(args.prediction_json)
    report = {"summary": summarize(pred)}
    if args.truth:
        report["comparison"] = compare(pred, _load(args.truth))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
