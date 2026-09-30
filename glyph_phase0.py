#!/usr/bin/env python3
"""Glyph Phase 0 스파이크 — 기존 산출물과 헤드투헤드 비교.

이미 추출해 둔 세그먼트 이미지(jobs/<id>/mg_input/*.png)에 OCSRGlyph / MarkushGlyph 를
돌려서, 같은 이미지에 대한 현행 엔진(MolScribe / MarkushGrapher-2) 결과와 비교한다.
정답셋이 없으므로 "정답 대비 정확도"가 아니라 아래 대리지표로 판단한다:

  - RDKit 파싱 성공률
  - 와일드카드(*) 원자 비율          → OCSR 실패의 대리지표
  - _looks_markush 판정 수           → Markush 과다판정 문제
  - 엔진 쌍별 일치율(canonical/skeleton) → 교차검증 강도
  - 현행 mismatch 건에서 신규 쌍의 일치율 → 교체 효과의 핵심 지표

실행 (glyph venv):
    glyph/.venv/bin/python glyph_phase0.py --job jobs/f80eccfc8f24
    glyph/.venv/bin/python glyph_phase0.py --job jobs/f80eccfc8f24 --report-only
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# 판정 로직은 프로덕션과 동일한 것을 쓴다 (integrate_prototype 은 stdlib 만 import)
from integrate_prototype import _looks_markush, _rdkit_canonical  # noqa: E402


# ---------------------------------------------------------------------------
# 추론
# ---------------------------------------------------------------------------
def run_glyph(seg_dir, out_jsonl, device="cuda:0", dtype="float16",
              ocsr_batch=8, markush_batch=4, limit=None):
    images = sorted(seg_dir.glob("*.png"))
    if limit:
        images = images[:limit]
    if not images:
        raise SystemExit(f"세그먼트 이미지 없음: {seg_dir}")
    print(f"[Glyph] 대상 이미지 {len(images)}개 ({seg_dir})", flush=True)

    paths = [str(p) for p in images]
    results = {p.stem: {"seg_key": p.stem, "image": str(p)} for p in images}

    # --- OCSRGlyph -------------------------------------------------------
    from glyph.ocsr.predict import OCSRPredictor

    t0 = time.time()
    print("[OCSRGlyph] 모델 로딩...", flush=True)
    ocsr = OCSRPredictor(device=device.split(":")[0])  # precision=None → CUDA 에서 fp16
    print(f"[OCSRGlyph] 로딩 완료 {time.time() - t0:.1f}s, 추론 시작", flush=True)
    t0 = time.time()
    smiles_list = ocsr.predict_batch(paths, batch_size=ocsr_batch)
    ocsr_sec = time.time() - t0
    for p, s in zip(paths, smiles_list):
        results[Path(p).stem]["smiles_ocsrglyph"] = s
    print(f"[OCSRGlyph] {len(paths)}장 {ocsr_sec:.1f}s "
          f"({ocsr_sec / len(paths):.2f}s/장)", flush=True)

    del ocsr
    import torch
    torch.cuda.empty_cache()

    # --- MarkushGlyph ----------------------------------------------------
    from glyph.markush.inference import MarkushPredictor

    t0 = time.time()
    print(f"[MarkushGlyph] 모델 로딩 (dtype={dtype})...", flush=True)
    mg = MarkushPredictor(device=device, dtype=dtype)
    mg.load()
    print(f"[MarkushGlyph] 로딩 완료 {time.time() - t0:.1f}s, 추론 시작", flush=True)
    t0 = time.time()
    preds = mg.predict_many(paths, batch_size=markush_batch)
    mk_sec = time.time() - t0
    for p, pred in zip(paths, preds):
        r = results[Path(p).stem]
        r["cxsmiles_glyph"] = pred.cxsmiles
        r["cxsmiles_opt_glyph"] = pred.cxsmiles_opt
    print(f"[MarkushGlyph] {len(paths)}장 {mk_sec:.1f}s "
          f"({mk_sec / len(paths):.2f}s/장)", flush=True)

    rows = [results[k] for k in sorted(results)]
    out_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                         encoding="utf-8")
    print(f"[Glyph] 예측 저장: {out_jsonl}", flush=True)
    return {"ocsr_sec": ocsr_sec, "markush_sec": mk_sec, "n_images": len(paths)}


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def _smiles_part(s):
    return str(s).split("|")[0].strip() if s else ""


def _wildcard_stats(s):
    """(원자수, 와일드카드 원자수). RDKit 파싱 실패 시 문자열 기반 근사."""
    s = _smiles_part(s)
    if not s:
        return 0, 0
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        m = Chem.MolFromSmiles(s)
        if m is not None:
            n = m.GetNumAtoms()
            w = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 0)
            return n, w
    except ImportError:
        pass
    return s.count("*"), s.count("*")


def canon_unlabeled(s, ignore_stereo=True):
    """R-라벨을 지운 canonical SMILES.

    엔진마다 가변 위치 표기가 다르다: MolScribe 는 맨 ``*``, OCSRGlyph 는
    ``[8*]``(isotope 라벨), MarkushGrapher 는 ``[1*]`` + ``$R8$`` 블록.
    라벨 문자열이 다르면 RDKit canonical 도 달라지므로, 같은 골격인데도
    불일치로 잡힌다. dummy 원자의 isotope/atom-map 을 0 으로 밀어
    "라벨 무시, 골격만" 비교가 가능하게 한다.
    """
    s = _smiles_part(s)
    if not s:
        return None
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return None
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetIsotope(0)
            a.SetAtomMapNum(0)
    if ignore_stereo:
        Chem.RemoveStereochemistry(m)
    try:
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def canon_ap(cx, ignore_stereo=True):
    """부착점(``_AP``) 표기까지 정규화한 canonical SMILES.

    MarkushGrapher-2 는 그림의 물결선(부착점)을 dummy 원자 + ``$_AP$`` 라벨로
    표기하는데, MolScribe/OCSRGlyph/MarkushGlyph 는 같은 물결선을 메틸(CH3)로
    읽는다. 표기 차이 때문에 골격이 동일해도 불일치로 잡히므로, ``_AP`` dummy 를
    탄소로 되돌려 "골격이 같은가"만 비교한다.

    ※ 이 정규화 없이 측정하면 MolScribe↔MG2 일치율이 18.2% 로 나오지만
      정규화 후에는 81.8% 다 (90 세그먼트 기준). 프로덕션의 mismatch 75건 중
      59건이 이 표기 차이에서 온 위양성이다.
    """
    s = _smiles_part(cx)
    if not s:
        return None
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return None
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    ap = set()
    lb = re.search(r"\|\$([^$]*)\$", str(cx or ""))
    if lb:
        ap = {i for i, l in enumerate(lb.group(1).split(";")) if l == "_AP"}
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 0:
            if a.GetIdx() in ap:
                a.SetAtomicNum(6)
                a.SetNoImplicit(False)
            a.SetIsotope(0)
            a.SetAtomMapNum(0)
    if ignore_stereo:
        Chem.RemoveStereochemistry(m)
    try:
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


# 라벨이 "구체적으로 결정된 원자단"을 가리키는 경우 → 실제 원소.
# 엔진마다 같은 그림을 상위원자 라벨로 쓰거나 원자로 그리므로, 골격 비교에서는
# 반드시 해소해야 한다. 가변 라벨(R13/Q/J1/E/Z/0-3/C1-6alkyl …)은 `*` 로 남긴다.
_CONCRETE_LABEL = {
    "_AP": 6,                                    # 부착점 ↔ 메틸 표기 차이
    "C": 6, "CH": 6, "CH2": 6, "CH3": 6,
    "N": 7, "NH": 7, "NH2": 7,
    "O": 8, "OH": 8,
    "S": 16, "SH": 16,
}


def canon_skeleton(cx, ignore_stereo=True):
    """상위원자 라벨까지 해소한 골격 canonical SMILES (**최종 주 지표**).

    `canon_ap()` 는 MG2 의 ``_AP`` 만 풀어 준다. 그런데 MarkushGlyph 는 반대로
    CH2 사슬을 ``$CH2$`` 상위원자 라벨로 쓰고(90장에서 30회), MolScribe/OCSRGlyph
    는 같은 자리를 그냥 탄소로 그린다. 즉 두 엔진의 표기 관습이 **서로 반대
    방향으로** 다르다:

        MG2          : 물결선 → ``_AP`` dummy   / CH2 → 탄소
        MarkushGlyph : 물결선 → 메틸·``[\\*]``  / CH2 → ``CH2`` dummy

    어느 쪽도 오류가 아니므로, "골격이 같은가"를 재려면 양쪽 관습을 모두 해소해야
    한다. 여기서는 라벨이 구체적 원자단이면 해당 원소로 되돌리고, 라벨 ``H`` 는
    명시적 수소이므로 원자에서 제거한다. 가변 라벨만 ``*`` 로 남는다.

    ※ 이 정규화를 빠뜨리면 MarkushGlyph 가 표기 관습 때문에 부당하게 벌점을
      받는다 — `canon_ap()` 도입 전에 MG2 가 받았던 것과 똑같은 위양성이다.
    """
    s = _smiles_part(cx)
    if not s:
        return None
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return None
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None

    labels = {}
    lb = re.search(r"\|\$([^$]*)\$", str(cx or ""))
    if lb:
        for i, l in enumerate(lb.group(1).split(";")):
            if l:
                labels[i] = l.split("?")[0].strip()   # "CH2?0-3" → "CH2"

    rw = Chem.RWMol(m)
    drop = []
    for a in rw.GetAtoms():
        if a.GetAtomicNum() != 0:
            continue
        lab = labels.get(a.GetIdx(), "")
        z = _CONCRETE_LABEL.get(lab)
        if z:
            a.SetAtomicNum(z)
            a.SetNoImplicit(False)
        elif lab == "H":
            drop.append(a.GetIdx())                  # 명시적 수소 = 골격 원자 아님
        a.SetIsotope(0)
        a.SetAtomMapNum(0)
    for i in sorted(drop, reverse=True):
        rw.RemoveAtom(i)

    m = rw.GetMol()
    if ignore_stereo:
        Chem.RemoveStereochemistry(m)
    try:
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def norm_markush_opt(s):
    """MarkushGlyph 의 ``<r>LABEL</r>`` opt 형식을 RDKit 이 읽을 수 있게 정규화.

    MarkushGlyph 는 이형 라벨(``<r>CH2</r>``, ``<r>E</r>``, ``<r>0-3</r>`` 등
    superatom/가변원자)이 섞이면 plain CXSMILES 직렬화가 빈 문자열로 나온다
    (모델 출력 자체는 opt 형식에 온전히 있다). 라벨을 와일드카드로 치환해
    골격 비교가 가능하게 만든다.
    """
    if not s:
        return ""
    s = str(s).split("|")[0]
    s = re.sub(r"<r>[^<]*</r>", "*", s)          # 변수 라벨 → 와일드카드
    s = s.replace("[\\*]", "*")
    s = re.sub(r"\[\\([A-Za-z0-9]+)\]", r"[\1]", s)  # [\CH3] → [CH3]
    return s


# MarkushGlyph 가 내보내는 규격 외 토큰. Glyph 코드베이스 어디에도 정의가 없어
# 공식 변환기 opt_to_standard_cxsmiles() 가 ValueError 를 던지고 빈 문자열이 된다.
_ESCAPED_TOKEN_RE = re.compile(r"\[\\([^\]]+)\]")

# 구체적으로 결정된 상위원자 라벨 → 실제 원자. 원자 1개를 유지해야
# 뒤따르는 Sg:/m: 의 원자 인덱스가 깨지지 않는다.
_CONCRETE_SUPERATOM = {"CH3": "C", "CH2": "C", "CH": "C", "OH": "O",
                       "NH": "N", "NH2": "N", "SH": "S"}


def repair_opt(opt):
    """규격 외 ``[\\X]`` 토큰을 정식 cxsmiles_opt 로 복원 (라벨 보존).

    모델에게 가장 유리한 해석(charitable reading)을 적용해 MarkushGlyph 품질의
    **상한**을 재는 것이 목적이다:

      ``[\\*]``   → ``<r>_AP</r>``      (부착점 — 근거는 아래)
      ``[\\CH3]`` → 탄소 ``C``           (구체적으로 결정된 메틸)
      그 외        → ``<r>X</r>``        (가변 라벨로 보존 — 정보 손실 없음)

    ``[\\*]`` 가 부착점이라는 근거: 이 토큰이 있는 17행 중 opt 꼬리에
    ``atomProp:N.molViewConnectionPoint`` 가 함께 붙은 8행에서, N 이 ``[\\*]`` 의
    원자 인덱스와 **8/8 일치**한다. 즉 MG2 의 ``_AP`` 와 같은 것을 가리킨다.

    ※ 이건 "직렬화 버그 수정"이 아니다. ``[\\X]`` 는 Glyph 의 cxsmiles_opt
      규격에 존재하지 않는 토큰이므로 모델 출력 자체가 형식을 벗어난 것이고,
      여기서 하는 일은 모델이 의도했을 바를 추측해 주는 것이다.
    """
    if not opt:
        return ""

    def sub(m):
        lab = m.group(1).strip()
        if lab == "*":
            return "<r>_AP</r>"
        if lab in _CONCRETE_SUPERATOM:
            return _CONCRETE_SUPERATOM[lab]
        return f"<r>{lab}</r>"

    return _ESCAPED_TOKEN_RE.sub(sub, str(opt))


def cxsmiles_from_opt_repaired(opt):
    """복원한 opt 를 Glyph 공식 변환기로 표준 CXSMILES(``|$...$|``) 로 바꾼다.

    라벨을 ``*`` 로 뭉개는 norm_markush_opt() 와 달리 ``$...$`` 라벨 블록과
    Sg:/m: 섹션이 온전히 남는다. 실패 시 빈 문자열.
    """
    fixed = repair_opt(opt)
    if not fixed:
        return ""
    try:
        from glyph.markush.data.pipeline import opt_to_standard_cxsmiles
    except ImportError:
        return norm_markush_opt(opt)
    try:
        return opt_to_standard_cxsmiles(fixed)
    except Exception:
        return norm_markush_opt(opt)


def effective_markush_glyph(row):
    """plain CXSMILES 가 파싱되면 그것을, 아니면 복원 변환본을 쓴다."""
    raw = row.get("cxsmiles_markushglyph")
    valid, _ = _rdkit_canonical(raw)
    if valid:
        return raw
    repaired = cxsmiles_from_opt_repaired(row.get("cxsmiles_opt_glyph"))
    if _rdkit_canonical(repaired)[0]:
        return repaired
    return norm_markush_opt(row.get("cxsmiles_opt_glyph"))


def engine_metrics(values):
    """엔진 1개의 출력 리스트 → 대리지표 dict."""
    n = len(values)
    empty = sum(1 for v in values if not _smiles_part(v))
    parse_ok = 0
    wildcard_rows = 0
    frac_sum = 0.0
    frac_n = 0
    markush_rows = 0
    for v in values:
        valid, _ = _rdkit_canonical(v)
        if valid:
            parse_ok += 1
        atoms, wild = _wildcard_stats(v)
        if wild:
            wildcard_rows += 1
        if atoms:
            frac_sum += wild / atoms
            frac_n += 1
        if _looks_markush(v):
            markush_rows += 1
    return {
        "n": n,
        "empty": empty,
        "parse_ok": parse_ok,
        "parse_ok_pct": round(100.0 * parse_ok / n, 1) if n else 0.0,
        "rows_with_wildcard": wildcard_rows,
        "rows_with_wildcard_pct": round(100.0 * wildcard_rows / n, 1) if n else 0.0,
        "mean_wildcard_atom_frac": round(frac_sum / frac_n, 3) if frac_n else None,
        "looks_markush": markush_rows,
        "looks_markush_pct": round(100.0 * markush_rows / n, 1) if n else 0.0,
    }


def pair_agreement(a_vals, b_vals):
    """두 엔진 출력의 일치율.

    3단계로 느슨해진다:
      match          — canonical 완전일치 (입체 포함, R-라벨 포함)
      match_skeleton — 입체 무시
      match_unlabeled— 입체 + R-라벨(isotope) 무시
      match_ap       — 위 + 부착점(_AP) 표기 무시   ← 현행 쌍(MolScribe↔MG2) 진단용
      match_super    — 위 + 모든 상위원자 라벨 해소  ← **엔진 간 공정 비교 (주 지표)**

    `match_ap` 는 MG2 의 관습(물결선→``_AP``)만 풀어 주므로 MarkushGlyph 의 반대
    관습(CH2 사슬→``$CH2$`` 라벨)은 그대로 벌점이 된다. 엔진끼리 비교할 때는
    반드시 `match_super`(= `canon_skeleton`)를 봐야 한다 — 이 지표로 바꾸면
    MarkushGlyph 의 단독반대율이 43.2% → 17.1% 로 떨어진다.
    """
    match = skeleton = unlabeled = ap = sup = comparable = 0
    for a, b in zip(a_vals, b_vals):
        va, ca = _rdkit_canonical(a)
        vb, cb = _rdkit_canonical(b)
        if not (va and vb and ca and cb):
            continue
        comparable += 1
        ua, ub = canon_unlabeled(a), canon_unlabeled(b)
        if ua and ub and ua == ub:
            unlabeled += 1
        pa, pb = canon_ap(a), canon_ap(b)
        if pa and pb and pa == pb:
            ap += 1
        ka, kb = canon_skeleton(a), canon_skeleton(b)
        if ka and kb and ka == kb:
            sup += 1
        if ca == cb:
            match += 1
            skeleton += 1
            continue
        _, sa = _rdkit_canonical(a, ignore_stereo=True)
        _, sb = _rdkit_canonical(b, ignore_stereo=True)
        if sa and sb and sa == sb:
            skeleton += 1
    pct = lambda x: round(100.0 * x / comparable, 1) if comparable else 0.0
    return {"comparable": comparable, "match": match, "match_pct": pct(match),
            "match_skeleton": skeleton, "match_skeleton_pct": pct(skeleton),
            "match_unlabeled": unlabeled, "match_unlabeled_pct": pct(unlabeled),
            "match_ap": ap, "match_ap_pct": pct(ap),
            "match_super": sup, "match_super_pct": pct(sup)}


def three_way_vote(rows, keys):
    """세 엔진 중 2개 이상이 골격(입체/R라벨/상위원자 라벨 무시) 일치하는 건수."""
    consensus = unanimous = none_agree = 0
    for r in rows:
        canons = [canon_skeleton(r.get(k)) for k in keys]
        avail = [c for c in canons if c]
        if len(avail) < 2:
            continue
        counts = {}
        for c in avail:
            counts[c] = counts.get(c, 0) + 1
        top = max(counts.values())
        if top >= 2:
            consensus += 1
            if top == len(avail) and len(avail) >= 3:
                unanimous += 1
        else:
            none_agree += 1
    return {"engines": list(keys), "consensus_ge2": consensus,
            "unanimous_3": unanimous, "no_agreement": none_agree}


def report(job_dir, preds_jsonl, out_json):
    merged = json.loads((job_dir / "merged_integration.json").read_text(encoding="utf-8"))
    by_seg = {r.get("seg_key"): r for r in merged if r.get("seg_key")}
    preds = [json.loads(l) for l in preds_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]

    rows = []
    for p in preds:
        base = by_seg.get(p["seg_key"], {})
        rows.append({
            "seg_key": p["seg_key"],
            "compound_id": base.get("compound_id"),
            # 현행 엔진
            "smiles_molscribe": base.get("smiles_molscribe"),
            "cxsmiles_markushgrapher": base.get("cxsmiles_markush"),
            "agreement_current": base.get("agreement"),
            "confidence_current": base.get("confidence"),
            "is_markush_current": base.get("is_markush"),
            # 신규 엔진
            "smiles_ocsrglyph": p.get("smiles_ocsrglyph"),
            "cxsmiles_markushglyph": p.get("cxsmiles_glyph"),
            "cxsmiles_opt_glyph": p.get("cxsmiles_opt_glyph"),
        })
    for r in rows:
        r["cxsmiles_markushglyph_eff"] = effective_markush_glyph(r)

    ENGINES = {
        "MolScribe (현행 OCSR)": "smiles_molscribe",
        "OCSRGlyph (신규 OCSR)": "smiles_ocsrglyph",
        "MarkushGrapher-2 (현행 Markush)": "cxsmiles_markushgrapher",
        "MarkushGlyph (plain CXSMILES)": "cxsmiles_markushglyph",
        "MarkushGlyph (opt 정규화)": "cxsmiles_markushglyph_eff",
    }
    metrics = {name: engine_metrics([r.get(k) for r in rows]) for name, k in ENGINES.items()}

    PAIRS = {
        "현행 쌍: MolScribe ↔ MarkushGrapher-2": ("smiles_molscribe", "cxsmiles_markushgrapher"),
        "신규 쌍: OCSRGlyph ↔ MarkushGlyph": ("smiles_ocsrglyph", "cxsmiles_markushglyph_eff"),
        "OCSR 교체 영향: MolScribe ↔ OCSRGlyph": ("smiles_molscribe", "smiles_ocsrglyph"),
        "Markush 교체 영향: MarkushGrapher-2 ↔ MarkushGlyph": ("cxsmiles_markushgrapher", "cxsmiles_markushglyph_eff"),
        "혼합 쌍: MolScribe ↔ MarkushGlyph": ("smiles_molscribe", "cxsmiles_markushglyph_eff"),
        "혼합 쌍: OCSRGlyph ↔ MarkushGrapher-2": ("smiles_ocsrglyph", "cxsmiles_markushgrapher"),
    }
    pairs = {name: pair_agreement([r.get(a) for r in rows], [r.get(b) for r in rows])
             for name, (a, b) in PAIRS.items()}

    # 현행 mismatch 건만 골라 신규 쌍이 구제하는지 확인
    mism = [r for r in rows if r.get("agreement_current") == "mismatch"]
    mismatch_subset = {
        "n": len(mism),
        # 현행 쌍 그대로 — 부착점 정규화만 적용하면 몇 건이 구제되는가 (모델 교체 없이)
        "현행 쌍(MolScribe↔MG2) 일치": pair_agreement(
            [r.get("smiles_molscribe") for r in mism],
            [r.get("cxsmiles_markushgrapher") for r in mism]),
        "신규 쌍(OCSRGlyph↔MarkushGlyph) 일치": pair_agreement(
            [r.get("smiles_ocsrglyph") for r in mism],
            [r.get("cxsmiles_markushglyph_eff") for r in mism]),
        "현행 MolScribe ↔ 신규 OCSRGlyph 일치": pair_agreement(
            [r.get("smiles_molscribe") for r in mism],
            [r.get("smiles_ocsrglyph") for r in mism]),
        "혼합 MolScribe ↔ MarkushGlyph 일치": pair_agreement(
            [r.get("smiles_molscribe") for r in mism],
            [r.get("cxsmiles_markushglyph_eff") for r in mism]),
    }

    # Markush 여부로 쪼개서 본다 — OCSRGlyph 는 순수 OCSR 모델이라
    # 가변 위치를 일반 원자로 정규화해 버리는 경향이 있다.
    split = {}
    for label, subset in (("현행 is_markush=True", [r for r in rows if r.get("is_markush_current")]),
                          ("현행 is_markush=False", [r for r in rows if not r.get("is_markush_current")])):
        split[label] = {
            "n": len(subset),
            "MolScribe ↔ OCSRGlyph": pair_agreement(
                [r.get("smiles_molscribe") for r in subset],
                [r.get("smiles_ocsrglyph") for r in subset]),
            "MarkushGrapher-2 ↔ MarkushGlyph": pair_agreement(
                [r.get("cxsmiles_markushgrapher") for r in subset],
                [r.get("cxsmiles_markushglyph_eff") for r in subset]),
            "MolScribe ↔ MarkushGlyph": pair_agreement(
                [r.get("smiles_molscribe") for r in subset],
                [r.get("cxsmiles_markushglyph_eff") for r in subset]),
        }

    votes = {
        "3-way (MolScribe/OCSRGlyph/MarkushGlyph)": three_way_vote(
            rows, ["smiles_molscribe", "smiles_ocsrglyph", "cxsmiles_markushglyph_eff"]),
        "4-way (전부)": three_way_vote(
            rows, ["smiles_molscribe", "smiles_ocsrglyph",
                   "cxsmiles_markushgrapher", "cxsmiles_markushglyph_eff"]),
    }

    out = {"job": str(job_dir), "n_rows": len(rows), "engine_metrics": metrics,
           "pair_agreement": pairs, "current_mismatch_subset": mismatch_subset,
           "markush_split": split, "vote": votes, "rows": rows}
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_report(out)
    print(f"\n상세: {out_json}")
    return out


def _print_report(out):
    print("\n" + "=" * 78)
    print(f"Glyph Phase 0 비교 — {out['n_rows']}개 세그먼트 ({out['job']})")
    print("=" * 78)

    print("\n[1] 엔진별 출력 품질 (정답셋 없음 → 대리지표)")
    hdr = f"{'엔진':<34}{'파싱OK':>9}{'와일드카드행':>13}{'*원자비율':>11}{'Markush판정':>12}"
    print(hdr)
    print("-" * 78)
    for name, m in out["engine_metrics"].items():
        print(f"{name:<34}{m['parse_ok_pct']:>8.1f}%{m['rows_with_wildcard_pct']:>12.1f}%"
              f"{(m['mean_wildcard_atom_frac'] if m['mean_wildcard_atom_frac'] is not None else 0):>11.3f}"
              f"{m['looks_markush']:>7}/{m['n']:<4}")

    print("\n[2] 엔진 쌍별 일치율 (교차검증 강도)")
    print("    ※ 골격=입체+R라벨 무시 / 부착점=거기서 _AP 무시 / 상위원자=모든 라벨 해소")
    print("       엔진끼리 비교할 때는 상위원자 열만 보면 된다 (표기 관습 차이 제거됨)")
    for name, p in out["pair_agreement"].items():
        print(f"  {name}")
        print(f"      비교가능 {p['comparable']:>3}  |  완전 {p['match']:>3} ({p['match_pct']:.1f}%)"
              f"  |  골격 {p['match_unlabeled']:>3} ({p['match_unlabeled_pct']:.1f}%)"
              f"  |  부착점 {p['match_ap']:>3} ({p['match_ap_pct']:.1f}%)"
              f"  |  상위원자 {p['match_super']:>3} ({p['match_super_pct']:.1f}%)")

    ms = out["current_mismatch_subset"]
    print(f"\n[3] 현행 mismatch {ms['n']}건 재판정")
    print("    ※ 프로덕션 수정은 부착점 정규화(canon_ap)만으로 충분하다 — 현행 쌍에서는")
    print("       상위원자 해소까지 가면 오히려 미세하게 낮아진다 (81.8% → 80.7%)")
    for name, p in ms.items():
        if name == "n":
            continue
        print(f"  {name}: 부착점 {p['match_ap']}/{p['comparable']} ({p['match_ap_pct']:.1f}%)"
              f"   상위원자 {p['match_super']} ({p['match_super_pct']:.1f}%)"
              f"   [정규화 전 {p['match_unlabeled']}]")

    print("\n[4] Markush 여부별 분리 (상위원자 해소 기준)")
    for label, blk in out["markush_split"].items():
        print(f"  {label} (n={blk['n']})")
        for name, p in blk.items():
            if name == "n":
                continue
            print(f"      {name:<34} {p['match_super']:>3}/{p['comparable']:<3} "
                  f"({p['match_super_pct']:.1f}%)")

    print("\n[5] 다수결 투표 가능성")
    for name, v in out["vote"].items():
        print(f"  {name}: 2표이상 합의 {v['consensus_ge2']}, 만장일치 {v['unanimous_3']}, "
              f"합의실패 {v['no_agreement']}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="jobs/f80eccfc8f24")
    ap.add_argument("--seg-dir", default=None, help="기본: <job>/mg_input")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", help="TITAN RTX(sm75) 는 bfloat16 미지원")
    ap.add_argument("--ocsr-batch", type=int, default=8)
    ap.add_argument("--markush-batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    job_dir = (ROOT / args.job).resolve()
    seg_dir = Path(args.seg_dir).resolve() if args.seg_dir else job_dir / "mg_input"
    preds_jsonl = job_dir / "glyph_phase0_preds.jsonl"
    out_json = job_dir / "glyph_phase0_report.json"

    if not args.report_only:
        timing = run_glyph(seg_dir, preds_jsonl, device=args.device, dtype=args.dtype,
                          ocsr_batch=args.ocsr_batch, markush_batch=args.markush_batch,
                          limit=args.limit)
        print(f"[Timing] {timing}", flush=True)

    report(job_dir, preds_jsonl, out_json)


if __name__ == "__main__":
    main()
