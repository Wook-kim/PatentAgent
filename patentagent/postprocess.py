"""Application-owned normalization and legacy export compatibility.

Extracted from PatentAgent integrate_prototype.py; no external engine calls.
"""
import json

_UNIT_TO_NM = {"m": 1e9, "mm": 1e6, "um": 1e3, "nm": 1.0, "pm": 1e-3, "fm": 1e-6}
_UNIT_CANON = {"um": "µM", "mm": "mM", "nm": "nM", "pm": "pM", "fm": "fM", "m": "M"}



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

    구조에서 R-label, wildcard, SGroup를 추출한다.
    모델의 별도 치환기 표는 pipeline.assemble에서 추가한다.
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
    wildcard_atoms = re.findall(r"\[\d*\*\]|\*", s.split("|", 1)[0])
    has_sgroup = "Sg:" in s
    is_markush = bool(variable_labels or wildcard_atoms or has_sgroup or "<r>" in opt)
    return {
        "is_markush": is_markush,
        "variable_labels": variable_labels,
        "r_groups": r_groups,
        "atom_site_labels": atom_site_labels,
        "scaffold_group_labels": scaffold_group_labels,
        "variable_sites": len(wildcard_atoms) if wildcard_atoms else len(r_groups),
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
        last = [line.strip() for line in cid.splitlines() if line.strip()][-1]
        cid = last
    return cid


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

    # Reject ranges or trailing prose instead of silently retaining their prefix.
    s = s.replace("≤", "<=").replace("≥", ">=").replace("−", "-")
    s = re.sub(r"\s*[×x]\s*10\s*\^?\s*([+-]?\d+)", r"e\1", s)
    m = re.fullmatch(
        r"\s*([<>]=?)?\s*(\d+(?:\.\d*)?|\.\d+)([eE][-+]?\d+)?\s*"
        r"(mM|nM|pM|fM|[uµμ]M|M)?\s*",
        s)
    if not (m and m.group(2)):
        return out

    out["operator"] = m.group(1) or "="
    try:
        out["value_num"] = float(m.group(2) + (m.group(3) or ""))
    except ValueError:
        return out

    unit_raw = m.group(4) or (metadata or {}).get("unit_hint")
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
    s = cxsmiles
    # [*], [1*], [R], <r>R</r> 등 와일드카드/변수 원자
    if "*" in s.split("|", 1)[0]:
        return True
    if ("$R" in s) or ("Sg:" in s) or ("<r>" in s):
        return True
    return False


def _rdkit_canonical(smiles, ignore_stereo=False):
    """SMILES/CXSMILES -> (valid, canonical). RDKit 미설치 시 (None, None)."""
    if not smiles:
        return None, None
    try:
        from rdkit import Chem, RDLogger
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


# 라벨이 구체적 원자단을 가리키면 실제 원소로 해소한다 (Phase 0 canon_skeleton 과 동일).
# MarkushGlyph 는 물결선을 _AP·CH3, CH2 사슬을 CH2 상위원자로 쓰고 OCSR 은 같은 자리를
# 탄소로 그리므로, 해소하지 않으면 표기 관습 차이가 mismatch 로 잡힌다.
_CONCRETE_LABEL = {
    "_AP": 6,
    "C": 6, "CH": 6, "CH2": 6, "CH3": 6,
    "N": 7, "NH": 7, "NH2": 7,
    "O": 8, "OH": 8,
    "S": 16, "SH": 16,
}


def _rdkit_skeleton(smiles):
    """상위원자 라벨·부착점까지 해소한 입체 무시 골격 canonical. 실패 시 None.

    가변 라벨(R1, Q, ...)은 isotope/atom map 을 지운 ``*`` 로 남는다. 라벨 ``H`` 는
    명시적 수소이므로 제거한다.
    """
    if not smiles:
        return None
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return None
    import re
    s = str(smiles).split("|")[0].strip()
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    labels = {}
    block = re.search(r"\|\$([^$]*)\$", str(smiles))
    if block:
        for i, label in enumerate(block.group(1).split(";")):
            if label:
                labels[i] = label.split("?")[0].strip()   # "CH2?0-3" -> "CH2"
    rw = Chem.RWMol(m)
    drop = []
    for a in rw.GetAtoms():
        if a.GetAtomicNum() != 0:
            continue
        label = labels.get(a.GetIdx(), "")
        z = _CONCRETE_LABEL.get(label)
        if z:
            a.SetAtomicNum(z)
            a.SetNoImplicit(False)
        elif label == "H":
            drop.append(a.GetIdx())
        a.SetIsotope(0)
        a.SetAtomMapNum(0)
    for i in sorted(drop, reverse=True):
        rw.RemoveAtom(i)
    m = rw.GetMol()
    Chem.RemoveStereochemistry(m)
    try:
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def enrich_with_rdkit(rows):
    """각 row 에 RDKit 검증/교차검증 신뢰도 필드 추가.

    - smiles_valid / cxsmiles_valid : 파싱 가능 여부
    - canonical_smiles : OCSR SMILES 의 canonical (입체 포함)
    - agreement : 두 엔진 결과 비교
        'match'           : 입체 포함 canonical 동일
        'match_skeleton'  : 입체 무시 시 동일 (입체만 차이)
        'match_normalized': 상위원자 라벨/부착점(_AP) 해소 후 골격 동일 (표기 관습 차이)
        'mismatch'        : 골격도 다름
        'single'          : 한쪽만 존재
        'unparsable'      : 파싱 실패로 비교 불가
    - confidence : high / medium / low
    """
    for r in rows:
        smi = r.get("smiles_ocsr", r.get("smiles_molscribe"))
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
        elif (k1 := _rdkit_skeleton(smi)) and k1 == _rdkit_skeleton(cxs):
            agreement = "match_normalized"
        elif c1 is None or c2 is None:
            agreement = "unparsable"
        else:
            agreement = "mismatch"
        r["agreement"] = agreement

        # 신뢰도: 두 엔진 골격 일치 + ID 정제 안 거침 => high
        id_clean = r.get("compound_id_raw") is None
        matched = agreement in ("match", "match_skeleton", "match_normalized")
        if matched and id_clean:
            conf = "high"
        elif matched:
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
        "smiles_ocsr", "ocsr_model", "markush_model", "canonical_smiles", "smiles_valid",
        "cxsmiles_markush", "cxsmiles_valid", "is_markush",
        "ocsr_raw", "cxsmiles_opt", "markush_raw", "markush_stable_raw",
        "markush_conversion_status", "extraction_errors", "model_provenance",
        "substituents",
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
            base["smiles_ocsr"] = r.get("smiles_ocsr", r.get("smiles_molscribe"))
            for key in ("extraction_errors", "model_provenance", "substituents"):
                base[key] = json.dumps(r.get(key), ensure_ascii=False)
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


def _write_parquet(csv_path, parquet_path):
    """CSV -> Parquet (pandas+pyarrow 있을 때만). 실패해도 파이프라인은 계속."""
    try:
        import pandas as pd
        pd.read_csv(csv_path).to_parquet(parquet_path, index=False)
        print(f"=== Parquet: {parquet_path}")
    except Exception as exc:  # pragma: no cover
        print(f"[Parquet] 생략 ({exc})")
