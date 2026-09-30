#!/usr/bin/env python3
"""
PatentAgent 검수 워크플로 (경량 review UI)
==========================================

통합 파이프라인 결과(merged_integration.json)를 사람이 검수하는 워크플로.
MARCUS(자체 추출용 무거운 Vue3+OCSR 플랫폼) 전체를 재현하는 대신,
우리 통합 스키마(구조+Markush+활성값+신뢰도)에 특화된 경량 검수 도구를 제공한다.

두 가지 모드:
  1) 정적 HTML 생성:  python3 review_app.py build <out_dir>
       -> <out_dir>/review.html (구조 이미지 + 데이터 + 신뢰도, 저신뢰 강조)
  2) 검수 서버:        python3 review_app.py serve <out_dir> [--port 8200]
       -> 웹에서 Ketcher 구조 편집기로 직접 수정 + 항목별 승인/수정/메모 저장
          (review_state.json). 구체 화합물은 그래픽 편집, Markush는 읽기 전용.

설계 의도:
- 저신뢰(low/medium) · 두 엔진 불일치(mismatch) · ID 정제된 항목을 우선 검수 대상으로 부각
- 구조 세그먼트 이미지(특허 원본)를 Ketcher 옆에 나란히 두어 사람이 눈으로 대조·수정
- 원본 추출값(canonical_smiles)은 보존하고, 사람이 고친 구조는 corrected_smiles 로 별도 기록
  -> 추출 vs 사람수정 비교·감사 가능
- 검수 결과(verdict/corrected_smiles/note)를 review_state.json 에 저장 -> 재현·감사 가능
"""
import argparse
import base64
import html
import json
from datetime import datetime, timezone
from pathlib import Path

# Ketcher standalone 정적 빌드 위치 (review_static/ketcher/, 완전 오프라인 자기완결)
KETCHER_DIR = Path(__file__).resolve().parent / "review_static" / "ketcher"


def _canon(smiles):
    """RDKit canonical SMILES (실패/빈값 시 None). 구조적 동일성 비교용.
    Ketcher 는 로드한 구조를 자체 정규화 형태로 재출력하므로, '사람이 실제로
    구조를 바꿨는지'는 문자열이 아니라 canonical 로 비교해야 정확하다."""
    if not smiles or not str(smiles).strip():
        return None
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        m = Chem.MolFromSmiles(str(smiles).strip())
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None


def _img_data_uri(path):
    """이미지 파일 -> base64 data URI (HTML 자체 포함용). 실패 시 None."""
    try:
        p = Path(path)
        if not p.exists():
            return None
        b = p.read_bytes()
        return "data:image/png;base64," + base64.b64encode(b).decode()
    except Exception:
        return None


def _find_segment_image(bci_out, seg_key):
    """seg_key(page_idx) -> 세그먼트(잘린 구조) 이미지 경로."""
    if not seg_key:
        return None
    base = Path(bci_out)
    for p in base.glob(f"structures_group_*/segment/segment_{seg_key}.png"):
        return p
    return None


def _find_box_json(bci_out, seg_key):
    """seg_key(page_idx) -> 세그먼트 박스좌표 JSON 경로 (highlight_<seg_key>.json)."""
    if not seg_key:
        return None
    base = Path(bci_out)
    for p in base.glob(f"structures_group_*/segment/highlight_{seg_key}.json"):
        return p
    return None


def _find_page_image(bci_out, seg_key):
    """seg_key(page_idx) -> 그 구조가 있는 페이지 원본 이미지(structure_images/page_<page>.png).
    seg_key 는 '<page>_<idx>' 형태이며 page 는 1-based 페이지 번호."""
    if not seg_key:
        return None
    page = str(seg_key).split("_")[0]
    base = Path(bci_out)
    for p in base.glob(f"structures_group_*/structure_images/page_{page}.png"):
        return p
    return None


def _render_highlight(bci_out, seg_key):
    """페이지 원본 + 추출 구조 박스를 직접 합성한 PNG bytes 반환 (없으면 None).

    BioChemInsight 의 highlight_*.png 는 좌표축 혼동(DECIMER bbox 가 [y1,x1,y2,x2]
    인데 [x1,y1,x2,y2]로 저장) + matplotlib dpi300/tight 리스케일로 박스 위치가
    어긋남. 여기서는 page 원본 좌표계에 box 를 올바른 순서로 직접 그려 정확히 맞춘다.
    """
    page_path = _find_page_image(bci_out, seg_key)
    box_path = _find_box_json(bci_out, seg_key)
    if not (page_path and page_path.exists()):
        return None
    try:
        from PIL import Image, ImageDraw
        import io
        im = Image.open(page_path).convert("RGB")
        if box_path and box_path.exists():
            raw = json.load(open(box_path)).get("box")
            # DECIMER bbox 순서는 [y1, x1, y2, x2] (page_1.png 와 동일 좌표계에서 검증됨)
            if raw and len(raw) == 4:
                y1, x1, y2, x2 = (int(v) for v in raw)
                W, H = im.size
                x1, x2 = max(0, min(x1, W - 1)), max(0, min(x2, W - 1))
                y1, y2 = max(0, min(y1, H - 1)), max(0, min(y2, H - 1))
                draw = ImageDraw.Draw(im)
                # 선명한 빨간 박스(굵기는 페이지 크기에 비례)
                lw = max(3, W // 300)
                draw.rectangle([x1, y1, x2, y2], outline=(220, 30, 30), width=lw)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _review_score(r):
    """검수 우선순위 점수(높을수록 먼저 봐야 함): 저신뢰/불일치/ID정제/무효 항목."""
    score = 0
    if r.get("confidence") == "low":
        score += 100
    elif r.get("confidence") == "medium":
        score += 50
    if r.get("agreement") == "mismatch":
        score += 40
    if r.get("compound_id_raw"):
        score += 20
    if r.get("smiles_valid") is False or r.get("cxsmiles_valid") is False:
        score += 30
    return score


def _sorted_rows(rows):
    """검수 우선순위로 정렬(저신뢰/불일치 우선)."""
    return sorted(rows, key=lambda r: -_review_score(r))


def _compound_label(row, idx=None):
    cid = row.get("compound_id")
    if cid not in (None, ""):
        return str(cid)
    raw = row.get("compound_id_raw")
    if raw not in (None, ""):
        return str(raw)
    seg = row.get("seg_key")
    if seg:
        return f"Segment {seg}"
    if idx is not None:
        return f"Row {idx + 1}"
    return "Unassigned"


def build_html(out_dir):
    out_dir = Path(out_dir).resolve()
    rows = json.load(open(out_dir / "merged_integration.json", encoding="utf-8"))
    bci_out = out_dir / "bci"

    rows_sorted = _sorted_rows(rows)

    conf_color = {"high": "#2e7d32", "medium": "#f57f17", "low": "#c62828"}
    cards = []
    for i, r in enumerate(rows_sorted):
        seg = _find_segment_image(bci_out, r.get("seg_key"))
        uri = _img_data_uri(seg) if seg else None
        img_html = (f'<img src="{uri}" style="max-width:220px;max-height:200px;'
                    f'border:1px solid #ccc"/>' if uri
                    else '<div style="color:#999">(이미지 없음)</div>')

        conf = r.get("confidence") or "?"
        color = conf_color.get(conf, "#666")

        acts = r.get("activities") or {}
        act_rows = ""
        for an, a in acts.items():
            nM = a.get("value_nM")
            nM_s = f"{nM:.3g} nM" if isinstance(nM, (int, float)) else "—"
            act_rows += (f"<tr><td>{html.escape(an)}</td>"
                         f"<td>{html.escape(str(a.get('value_raw')))}</td>"
                         f"<td>{html.escape(str(a.get('operator') or ''))}"
                         f"{html.escape(str(a.get('value_num') if a.get('value_num') is not None else ''))} "
                         f"{html.escape(str(a.get('unit') or ''))}</td>"
                         f"<td>{nM_s}</td></tr>")
        if not act_rows:
            act_rows = '<tr><td colspan="4" style="color:#999">활성값 없음</td></tr>'

        warn = []
        if r.get("compound_id_raw"):
            warn.append(f"⚠️ ID 정제됨 (원본: {html.escape(str(r['compound_id_raw'])[:60])}…)")
        if r.get("agreement") == "mismatch":
            warn.append("⚠️ 두 엔진 구조 불일치")
        if r.get("smiles_valid") is False:
            warn.append("⚠️ MolScribe SMILES 무효")
        if r.get("cxsmiles_valid") is False:
            warn.append("⚠️ MarkushGrapher CXSMILES 무효")
        if r.get("coref_id_agree") is True:
            warn.append("✅ MolCoref 라벨 일치")
        warn_html = "<br>".join(warn)

        cards.append(f"""
        <div class="card" data-conf="{conf}" data-idx="{i}">
          <div class="head">
            <span class="cid">화합물 {html.escape(str(r.get('compound_id')))}</span>
            <span class="badge" style="background:{color}">{conf}</span>
            <span class="meta">page {html.escape(str(r.get('page')))} · {html.escape(str(r.get('agreement')))} · markush={r.get('is_markush')}</span>
          </div>
          <div class="body">
            <div class="imgcol">{img_html}</div>
            <div class="datacol">
              <div class="warn">{warn_html}</div>
              <div class="smi"><b>SMILES (MolScribe):</b><br><code>{html.escape(str(r.get('smiles_molscribe') or ''))}</code></div>
              <div class="smi"><b>CXSMILES (Markush):</b><br><code>{html.escape(str(r.get('cxsmiles_markush') or ''))}</code></div>
              <table class="acts"><tr><th>어세이</th><th>원문</th><th>정규화</th><th>nM</th></tr>{act_rows}</table>
              <div class="review">
                <label>검수:
                  <select onchange="setState({i},'verdict',this.value)">
                    <option value="">—</option>
                    <option value="approved">승인</option>
                    <option value="rejected">반려</option>
                    <option value="edited">수정필요</option>
                  </select>
                </label>
                <input placeholder="메모" style="width:300px" onchange="setState({i},'note',this.value)"/>
              </div>
            </div>
          </div>
        </div>""")

    n = len(rows)
    from collections import Counter
    cdist = Counter(r.get("confidence") for r in rows)
    summary = (f"총 {n}개 · 신뢰도 "
               f"high {cdist.get('high',0)} / medium {cdist.get('medium',0)} / low {cdist.get('low',0)}")

    html_doc = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>PatentAgent 검수</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:20px;background:#fafafa}}
 h1{{font-size:20px}} .summary{{color:#555;margin-bottom:12px}}
 .filters{{margin-bottom:16px}}
 .card{{background:#fff;border:1px solid #ddd;border-radius:8px;margin-bottom:14px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
 .head{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}
 .cid{{font-weight:bold;font-size:16px}}
 .badge{{color:#fff;padding:2px 8px;border-radius:10px;font-size:12px}}
 .meta{{color:#888;font-size:12px}}
 .body{{display:flex;gap:16px}}
 .imgcol{{flex:0 0 230px}} .datacol{{flex:1}}
 .warn{{color:#c62828;font-size:13px;margin-bottom:6px}}
 .smi{{font-size:12px;margin:4px 0;word-break:break-all}} code{{background:#f4f4f4;padding:1px 3px}}
 table.acts{{border-collapse:collapse;margin:8px 0;font-size:12px}}
 table.acts th,table.acts td{{border:1px solid #ddd;padding:3px 8px}}
 .review{{margin-top:8px}}
 .topnav{{display:flex;align-items:center;gap:10px;margin-bottom:10px}}
 .topnav h1{{margin:0;flex:1}}
 a.btn{{display:inline-block;padding:5px 12px;background:#616161;color:#fff;
        text-decoration:none;border-radius:5px;font-size:13px}}
</style></head><body>
<div class="topnav">
  <h1>PatentAgent 통합 결과 검수</h1>
  <a class="btn" href="/">홈</a>
</div>
<div class="summary">{summary}</div>
<div class="filters">
  표시:
  <label><input type="checkbox" class="cf" value="low" checked onchange="applyFilter()"> low</label>
  <label><input type="checkbox" class="cf" value="medium" checked onchange="applyFilter()"> medium</label>
  <label><input type="checkbox" class="cf" value="high" checked onchange="applyFilter()"> high</label>
  <button onclick="exportState()">검수결과 내보내기(JSON)</button>
</div>
{''.join(cards)}
<script>
 const state = {{}};
 function setState(i,k,v){{ state[i]=state[i]||{{}}; state[i][k]=v; }}
 function applyFilter(){{
   const on = [...document.querySelectorAll('.cf:checked')].map(c=>c.value);
   document.querySelectorAll('.card').forEach(c=>{{
     c.style.display = on.includes(c.dataset.conf)?'block':'none';
   }});
 }}
 function exportState(){{
   const blob=new Blob([JSON.stringify(state,null,2)],{{type:'application/json'}});
   const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
   a.download='review_state.json'; a.click();
 }}
</script>
</body></html>"""

    out_html = out_dir / "review.html"
    out_html.write_text(html_doc, encoding="utf-8")
    print(f"검수 HTML 생성: {out_html}")
    print(f"  {summary}")
    print(f"  브라우저에서 열기: file://{out_html}")
    return out_html


# ===========================================================================
# serve 모드: Ketcher 구조 편집기로 직접 수정하는 검수 서버
# ===========================================================================
#
# 표준 라이브러리(http.server)만 사용 — 무거운 의존성 없이 system python3로 기동.
# 라우트:
#   GET  /                      검수 목록 (저신뢰 우선)
#   GET  /item/{i}              항목 상세 (Ketcher iframe + 원본 이미지 대조)
#   GET  /segment/{i}.png       원본 구조 세그먼트 이미지
#   GET  /ketcher/...           Ketcher standalone 정적 자산 (오프라인)
#   GET  /api/rows              전체 행 + 검수상태 JSON
#   POST /api/item/{i}          검수 저장 {verdict, corrected_smiles, note}
#   GET  /export                검수 반영 최종본 (corrected_smiles 우선)
#
# 보관 정책: 원본 추출값(canonical_smiles 등)은 불변. 사람이 고친 구조는
#            review_state.json 에 corrected_smiles + 타임스탬프로 별도 기록.

def _now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class ReviewServer:
    """검수 세션 상태(merged rows + review_state.json)를 들고 있는 핸들러 백엔드."""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir).resolve()
        self.bci_out = self.out_dir / "bci"
        self.merged_path = self.out_dir / "merged_integration.json"
        self.state_path = self.out_dir / "review_state.json"
        self.rows = _sorted_rows(json.load(open(self.merged_path, encoding="utf-8")))
        self.state = {}
        if self.state_path.exists():
            try:
                self.state = json.load(open(self.state_path, encoding="utf-8"))
            except Exception:
                self.state = {}

    # --- 상태 영속화 -------------------------------------------------------
    def save_item(self, idx, payload):
        """항목 검수 결과 저장.

        payload: {verdict, note, current_smiles, baseline_smiles}
          - current_smiles: 저장 시점 Ketcher 의 SMILES (사람이 편집했을 수 있음)
          - baseline_smiles: 로드 직후 Ketcher 가 재출력한 SMILES (편집 전 기준선)
        Ketcher 는 로드만 해도 SMILES 를 정규화해 재출력하므로, current 와
        baseline 을 RDKit canonical 로 비교해 '실제로 구조가 바뀐 경우'에만
        corrected_smiles 로 기록한다(문자열 단순비교의 오판 방지).
        (구버전 호환: payload 에 corrected_smiles 가 직접 오면 그대로 사용)
        """
        rec = self.state.get(str(idx), {})
        if "verdict" in payload:
            rec["verdict"] = payload["verdict"]
        if "note" in payload:
            rec["note"] = payload["note"]

        if "corrected_smiles" in payload:           # 구버전/직접 지정 경로
            rec["corrected_smiles"] = payload["corrected_smiles"] or None
        elif "current_smiles" in payload:           # 신 경로: canonical 비교로 판정
            cur = (payload.get("current_smiles") or "").strip()
            base = (payload.get("baseline_smiles") or "").strip()
            cur_c, base_c = _canon(cur), _canon(base)
            if cur and cur_c and cur_c != base_c:
                # 구조가 실제로 바뀜 → 사람 수정본 채택 (canonical 로 정규화 저장)
                rec["corrected_smiles"] = cur_c
            else:
                rec["corrected_smiles"] = None      # 미수정 → 추출 원본 유지
        rec["updated_at"] = _now_iso()
        self.state[str(idx)] = rec
        # 원본은 절대 건드리지 않고 state 파일에만 기록 (감사 가능)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)
        return rec

    def export(self):
        """검수 반영 최종본: corrected_smiles 있으면 우선, 없으면 원본 canonical."""
        out = []
        for i, r in enumerate(self.rows):
            rec = self.state.get(str(i), {})
            corrected = (rec.get("corrected_smiles") or "").strip()
            merged = dict(r)
            merged["review_verdict"] = rec.get("verdict")
            merged["review_note"] = rec.get("note")
            merged["review_updated_at"] = rec.get("updated_at")
            merged["corrected_smiles"] = corrected or None
            # 최종 채택 구조: 사람 수정본 > 추출 canonical
            merged["final_smiles"] = corrected or r.get("canonical_smiles")
            merged["final_source"] = "human" if corrected else "extracted"
            out.append(merged)
        return out


def _seg_image_bytes(srv, idx):
    """행 idx의 원본 세그먼트(잘린 구조) 이미지 bytes (없으면 None)."""
    try:
        r = srv.rows[idx]
    except (IndexError, ValueError):
        return None
    seg = _find_segment_image(srv.bci_out, r.get("seg_key"))
    if seg and seg.exists():
        return seg.read_bytes()
    structure_image = r.get("structure_image")
    if structure_image:
        p = Path(structure_image)
        if p.exists():
            return p.read_bytes()
    return None


def _highlight_image_bytes(srv, idx):
    """행 idx의 하이라이트 이미지 bytes — 페이지 전체+구조 박스 (없으면 None).
    BioChemInsight 의 깨진 highlight 대신 page 원본 + box 를 직접 합성(_render_highlight)."""
    try:
        r = srv.rows[idx]
    except (IndexError, ValueError):
        return None
    highlighted = _render_highlight(srv.bci_out, r.get("seg_key"))
    if highlighted:
        return highlighted
    return _render_claim_highlight(srv, r)


def _claim_candidate_for_row(srv, row):
    path = srv.out_dir / "claim_structure_candidates.json"
    if not path.exists():
        return None
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    keys = [
        str(row.get("compound_id") or "").strip(),
        f"Entry {str(row.get('compound_id_raw') or '').strip()}",
        str(row.get("compound_id_raw") or "").strip(),
    ]
    for key in keys:
        if key and key in data:
            return data[key]
    return None


def _render_claim_highlight(srv, row):
    cand = _claim_candidate_for_row(srv, row)
    if not cand:
        return None
    image_path = Path(cand.get("image") or "")
    page_path = image_path.with_name(f"page_{cand.get('page')}.png")
    bbox = cand.get("bbox_pixel")
    if not page_path.exists() or not bbox or len(bbox) != 4:
        return image_path.read_bytes() if image_path.exists() else None
    try:
        from PIL import Image, ImageDraw
        import io
        im = Image.open(page_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        x1, y1, x2, y2 = (int(v) for v in bbox)
        lw = max(3, im.size[0] // 320)
        draw.rectangle([x1, y1, x2, y2], outline=(220, 30, 30), width=lw)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return image_path.read_bytes() if image_path.exists() else None


def _list_page_html(srv, base="", home_url="/"):
    """검수 목록 페이지. base: URL 프리픽스(게이트웨이 마운트용, 예 /jobs/<id>/review)."""
    conf_color = {"high": "#2e7d32", "medium": "#f57f17", "low": "#c62828"}
    verdict_label = {"approved": "✅승인", "rejected": "❌반려",
                     "edited": "✎수정", None: "", "": ""}
    rows_html = []
    for i, r in enumerate(srv.rows):
        rec = srv.state.get(str(i), {})
        conf = r.get("confidence") or "?"
        color = conf_color.get(conf, "#666")
        is_mk = r.get("is_markush")
        flags = []
        if r.get("agreement") == "mismatch":
            flags.append("불일치")
        if r.get("compound_id_raw"):
            flags.append("ID정제")
        if is_mk:
            flags.append("Markush")
        vd = rec.get("verdict")
        vd_html = verdict_label.get(vd, html.escape(str(vd or "")))
        corrected_mark = " ✎" if (rec.get("corrected_smiles") or "").strip() else ""
        rows_html.append(f"""
        <tr class="row" data-conf="{conf}" onclick="location.href='{base}/item/{i}'">
          <td>{i+1}</td>
          <td><b>{html.escape(_compound_label(r, i))}</b></td>
          <td><span class="badge" style="background:{color}">{conf}</span></td>
          <td>{html.escape(str(r.get('agreement') or ''))}</td>
          <td>{html.escape(' · '.join(flags))}</td>
          <td>p{html.escape(str(r.get('page') or ''))}</td>
          <td class="vd">{vd_html}{corrected_mark}</td>
        </tr>""")

    from collections import Counter
    n = len(srv.rows)
    cdist = Counter(r.get("confidence") for r in srv.rows)
    n_reviewed = sum(1 for v in srv.state.values() if v.get("verdict"))
    summary = (f"총 {n}개 · 신뢰도 high {cdist.get('high',0)} / "
               f"medium {cdist.get('medium',0)} / low {cdist.get('low',0)} · "
               f"검수완료 {n_reviewed}/{n}")

    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>PatentAgent 검수</title>
<style>
 :root{{--bg:#f6f7f9;--panel:#fff;--line:#d9dee7;--text:#1f2933;--muted:#667085;--blue:#155eef}}
 *{{box-sizing:border-box}}
 body{{font-family:system-ui,sans-serif;margin:0;background:var(--bg);color:var(--text)}}
 .page{{max-width:1240px;margin:0 auto;padding:18px 20px 28px}}
 h1{{font-size:19px;line-height:1.25}}
 .summary{{color:var(--muted);margin-bottom:14px;font-size:13px}}
 .toolbar{{display:flex;gap:10px;align-items:center;justify-content:space-between;
           margin-bottom:12px;flex-wrap:wrap}}
 .filters{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;font-size:13px}}
 .search{{min-width:260px;flex:1;max-width:420px;padding:8px 10px;border:1px solid var(--line);
          border-radius:6px;background:#fff;font-size:13px}}
 .tablewrap{{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:auto}}
 table{{border-collapse:separate;border-spacing:0;width:100%;min-width:780px;background:#fff}}
 th,td{{border-bottom:1px solid #e6e9ef;padding:8px 10px;font-size:13px;text-align:left;white-space:nowrap}}
 th{{position:sticky;top:0;background:#f8fafc;color:#475467;font-weight:650;z-index:1}}
 tr.row{{cursor:pointer}} tr.row:hover{{background:#eef6ff}}
 tr.row:focus-within{{outline:2px solid var(--blue);outline-offset:-2px}}
 .badge{{color:#fff;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:650}}
 .vd{{font-weight:bold}}
 .topnav{{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:10px;
          padding:12px 0 14px;background:linear-gradient(var(--bg),var(--bg))}}
 .topnav h1{{margin:0;flex:1}}
 a.btn{{display:inline-block;padding:5px 12px;background:#1565c0;color:#fff;
        text-decoration:none;border-radius:5px;font-size:13px}}
 a.btn.secondary{{background:#616161}}
 @media (max-width:720px){{
   .page{{padding:12px}}
   .topnav{{align-items:flex-start;flex-wrap:wrap}}
   .topnav h1{{flex-basis:100%}}
   .search{{max-width:none;width:100%;min-width:0}}
 }}
</style></head><body>
<div class="page">
<div class="topnav">
  <h1>PatentAgent 통합 결과 검수</h1>
  <a class="btn secondary" href="{html.escape(home_url, quote=True)}">홈</a>
  <a class="btn" href="{base}/export" target="_blank">최종본 export</a>
</div>
<div class="summary">{summary}</div>
<div class="toolbar">
  <input class="search" id="q" placeholder="화합물, 플래그, 페이지 검색" oninput="ff()">
  <div class="filters">
    표시:
    <label><input type="checkbox" class="cf" value="low" checked onchange="ff()"> low</label>
    <label><input type="checkbox" class="cf" value="medium" checked onchange="ff()"> medium</label>
    <label><input type="checkbox" class="cf" value="high" checked onchange="ff()"> high</label>
  </div>
</div>
<div class="tablewrap">
<table>
 <tr><th>#</th><th>화합물</th><th>신뢰도</th><th>교차검증</th><th>플래그</th><th>페이지</th><th>검수</th></tr>
 {''.join(rows_html)}
</table>
</div>
</div>
<script>
 function ff(){{
   const on=[...document.querySelectorAll('.cf:checked')].map(c=>c.value);
   const q=(document.getElementById('q').value||'').trim().toLowerCase();
   document.querySelectorAll('tr.row').forEach(r=>{{
     const okConf = on.includes(r.dataset.conf);
     const okText = !q || r.textContent.toLowerCase().includes(q);
     r.style.display = (okConf && okText)?'table-row':'none';
   }});
 }}
</script>
</body></html>"""


def _item_page_html(srv, idx, base="", home_url="/"):
    """항목 상세 + Ketcher 편집기."""
    r = srv.rows[idx]
    rec = srv.state.get(str(idx), {})
    is_mk = bool(r.get("is_markush"))
    n = len(srv.rows)
    compound_label = _compound_label(r, idx)

    # 편집기 초기 로드 구조: 사람 수정본 있으면 그것, 없으면 추출 canonical
    init_smiles = (rec.get("corrected_smiles") or "").strip() or (r.get("canonical_smiles") or "")

    # 활성값 표
    acts = r.get("activities") or {}
    act_rows = ""
    for an, a in acts.items():
        nM = a.get("value_nM")
        nM_s = f"{nM:.3g} nM" if isinstance(nM, (int, float)) else "—"
        meta = a.get("metadata") or {}
        link = a.get("link_method") or ""
        conf = a.get("link_confidence")
        conf_s = f"{conf:.2g}" if isinstance(conf, (int, float)) else ""
        assay_desc = an
        if meta.get("assay_type"):
            assay_desc += f" ({meta.get('assay_type')})"
        act_rows += (f"<tr><td>{html.escape(an)}</td>"
                     f"<td>{html.escape(str(a.get('value_raw')))}</td>"
                     f"<td>{nM_s}</td>"
                     f"<td>{html.escape(link)} {html.escape(conf_s)}</td>"
                     f"<td>{html.escape(assay_desc)}</td></tr>")
    if not act_rows:
        act_rows = '<tr><td colspan="5" style="color:#999">활성값 없음</td></tr>'

    warn = []
    if r.get("compound_id_raw"):
        warn.append(f"⚠️ ID 정제됨 (원본: {html.escape(str(r['compound_id_raw'])[:80])})")
    if r.get("agreement") == "mismatch":
        warn.append("⚠️ 두 엔진 구조 불일치 — 어느 쪽이 맞는지 확인 후 수정")
    if r.get("smiles_valid") is False:
        warn.append("⚠️ MolScribe SMILES 무효")
    if r.get("cxsmiles_valid") is False:
        warn.append("⚠️ MarkushGrapher CXSMILES 무효")
    warn_html = "<br>".join(warn) or "특이사항 없음"
    groups = ", ".join(r.get("markush_groups") or [])
    markush_html = ""
    if r.get("is_markush"):
        mm = r.get("markush_metadata") or {}
        markush_html = (
            f'<div class="smi"><b>Markush groups:</b> '
            f'{html.escape(groups or "detected")} · variable sites '
            f'{html.escape(str(mm.get("variable_sites") or ""))}</div>'
        )

    prev_link = f'<a href="{base}/item/{idx-1}">← 이전</a>' if idx > 0 else "<span></span>"
    next_link = f'<a href="{base}/item/{idx+1}">다음 →</a>' if idx < n - 1 else "<span></span>"

    cur_verdict = rec.get("verdict") or ""
    cur_note = html.escape(rec.get("note") or "", quote=True)

    # Markush 안내 배너 / 편집기 잠금
    if is_mk:
        editor_banner = ('<div class="mkbanner">⚠️ Markush(일반식) 구조 — '
                         'R-group 그래픽 편집은 지원하지 않습니다. 아래 CXSMILES와 '
                         '원본 이미지로 검토하고, 필요 시 메모에 수정사항을 기록하세요. '
                         '(편집기는 참고용 읽기 전용)</div>')
    else:
        editor_banner = ""

    init_js = json.dumps(init_smiles)
    is_mk_js = "true" if is_mk else "false"

    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>화합물 {html.escape(compound_label)} 검수</title>
<style>
 :root{{--bg:#f6f7f9;--panel:#fff;--line:#d9dee7;--text:#1f2933;--muted:#667085;--blue:#155eef;--green:#2e7d32}}
 *{{box-sizing:border-box}}
 body{{font-family:system-ui,sans-serif;margin:0;background:var(--bg);color:var(--text);overflow-x:hidden}}
 .topbar{{display:flex;align-items:center;gap:12px;padding:10px 16px;background:#fff;
          border-bottom:1px solid var(--line);position:sticky;top:0;z-index:20;min-height:52px}}
 .topbar a{{color:var(--blue);text-decoration:none;font-size:13px;font-weight:650}}
 .topbar a.navbtn{{padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:#fff;color:#344054}}
 .topbar a.navbtn:hover{{background:#f2f4f7}}
 .cid{{font-weight:750;font-size:16px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
 .meta{{color:var(--muted);font-size:12px;white-space:nowrap}}
 .wrap{{display:grid;grid-template-columns:minmax(250px,300px) minmax(360px,1fr) minmax(380px,520px);
        gap:12px;padding:12px 14px;align-items:start;max-width:1800px;margin:0 auto}}
 .leftcol,.midcol,.rightcol{{min-width:0}}
 .panel{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:12px}}
 .panel h3{{margin:0 0 8px;font-size:12px;color:#475467;font-weight:750;display:flex;align-items:center;gap:8px}}
 img.seg{{width:100%;max-height:260px;object-fit:contain;border:1px solid #ccd2dc;background:#fff;border-radius:6px}}
 .pdfpanel{{position:sticky;top:64px}}
 .pdfbox{{height:calc(100vh - 142px);min-height:420px;overflow:auto;border:1px solid #ccd2dc;
          border-radius:6px;background:#fff;text-align:center}}
 img.hl{{max-width:100%;height:auto;display:block;margin:0 auto;cursor:zoom-in}}
 .hint{{font-size:11px;color:var(--muted);margin-top:6px}}
 .smi{{font-size:11px;word-break:break-all;margin:6px 0;line-height:1.35}}
 code{{background:#f2f4f7;padding:1px 3px;border-radius:4px}}
 .warn{{color:#b42318;font-size:12px;margin-bottom:8px;line-height:1.45}}
 .activity-scroll{{max-height:220px;overflow:auto;border:1px solid #eaecf0;border-radius:6px}}
 table.acts{{border-collapse:separate;border-spacing:0;width:100%;font-size:12px}}
 table.acts th,table.acts td{{border-bottom:1px solid #eaecf0;padding:6px 7px;text-align:left;vertical-align:top}}
 table.acts th{{position:sticky;top:0;background:#f8fafc;color:#475467;font-weight:650}}
 #kframe{{width:100%;height:calc(100vh - 210px);min-height:460px;border:1px solid #ccd2dc;border-radius:6px;background:#fff}}
 .mkbanner{{background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;padding:8px 10px;
            border-radius:6px;font-size:12px;margin-bottom:8px;line-height:1.4}}
 .savebar{{position:sticky;bottom:0;display:grid;grid-template-columns:auto minmax(180px,1fr) auto auto auto;
           gap:8px;align-items:center;margin-top:8px;padding:8px 0 0;background:#fff;border-top:1px solid #eaecf0}}
 .savebar label{{font-size:12px;color:#475467;display:flex;align-items:center;gap:6px}}
 .savebar select,.savebar input{{font-size:13px;padding:7px;border:1px solid #ccd2dc;border-radius:6px;background:#fff}}
 .savebar input.note{{width:100%;min-width:0}}
 button.save{{padding:8px 14px;background:var(--green);color:#fff;border:0;border-radius:6px;
              font-size:13px;font-weight:650;cursor:pointer}}
 button.reload{{padding:8px 12px;background:#667085;color:#fff;border:0;border-radius:6px;cursor:pointer;font-size:13px}}
 #status{{font-size:12px;color:var(--muted);min-width:110px}}
 @media (max-width:1320px){{
   .wrap{{grid-template-columns:minmax(250px,300px) minmax(420px,1fr);}}
   .rightcol{{grid-column:1 / -1}}
   #kframe{{height:560px}}
 }}
 @media (max-width:860px){{
   .topbar{{flex-wrap:wrap;gap:8px}}
   .cid{{flex-basis:100%;order:2}}
   .meta{{white-space:normal}}
   .wrap{{grid-template-columns:1fr;padding:10px}}
   .pdfpanel{{position:static}}
   .pdfbox{{height:70vh}}
   .savebar{{grid-template-columns:1fr;position:static}}
 }}
</style></head><body>
<div class="topbar">
  <a class="navbtn" href="{html.escape(home_url, quote=True)}">홈</a>
  <a class="navbtn" href="{base}/">목록</a>
  <span class="cid">화합물 {html.escape(compound_label)}</span>
  <span class="meta">{idx+1} / {n} · page {html.escape(str(r.get('page')))} · {html.escape(str(r.get('agreement') or ''))} · markush={is_mk}</span>
  <span style="margin-left:auto"></span>
  {prev_link} {next_link}
</div>
<div class="wrap">
  <div class="leftcol">
    <div class="panel">
      <h3>잘린 구조</h3>
      <img class="seg" id="imgSeg" src="{base}/segment/{idx}.png"
           onerror="this.outerHTML='<div style=color:#999>(이미지 없음)</div>'"/>
    </div>
    <div class="panel">
      <h3>검증 정보</h3>
      <div class="warn">{warn_html}</div>
      <div class="smi"><b>MolScribe:</b><br><code>{html.escape(str(r.get('smiles_molscribe') or ''))}</code></div>
      <div class="smi"><b>Markush CXSMILES:</b><br><code>{html.escape(str(r.get('cxsmiles_markush') or ''))}</code></div>
      {markush_html}
    </div>
    <div class="panel">
      <h3>활성값</h3>
      <div class="activity-scroll">
        <table class="acts"><tr><th>어세이</th><th>원문</th><th>nM</th><th>연결</th><th>메타</th></tr>{act_rows}</table>
      </div>
    </div>
  </div>
  <div class="midcol">
    <div class="panel pdfpanel">
      <h3>PDF 영역 — 이 구조가 추출된 원본 위치(빨간 박스)</h3>
      <div class="pdfbox">
        <img class="hl" id="imgHl" src="{base}/highlight/{idx}.png"
             title="클릭하면 새 탭에서 원본 크기로 보기"
             onclick="window.open(this.src,'_blank')"
             onerror="this.parentElement.innerHTML='<div style=color:#999;padding:20px>(PDF 영역 미리보기 없음)</div>'"/>
      </div>
      <div class="hint">스크롤로 페이지 전체 확인 · 클릭하면 새 탭에서 원본 크기로 확대</div>
    </div>
  </div>
  <div class="rightcol">
    <div class="panel">
      <h3>구조 편집 (Ketcher)</h3>
      {editor_banner}
      <iframe id="kframe" src="{base}/ketcher/index.html"></iframe>
      <div class="savebar">
        <label>검수:
          <select id="verdict">
            <option value="">—</option>
            <option value="approved">✅ 승인</option>
            <option value="edited">✎ 수정함</option>
            <option value="rejected">❌ 반려</option>
          </select>
        </label>
        <input class="note" id="note" placeholder="메모 (Markush 수정사항 등)"/>
        <button class="reload" onclick="reloadOriginal()" title="추출 원본 구조 다시 로드">↺ 원본</button>
        <button class="save" onclick="saveItem()">저장</button>
        <span id="status"></span>
      </div>
    </div>
  </div>
</div>
<script>
 const IDX = {idx};
 const BASE = {json.dumps(base)};
 const INIT_SMILES = {init_js};
 const IS_MARKUSH = {is_mk_js};
 const kframe = document.getElementById('kframe');
 let ketcherReady = false;
 // 로드 직후 Ketcher 가 재출력한 SMILES(편집 전 기준선). 저장 시 현재값과 비교해
 // '사람이 실제로 구조를 바꿨는지' 판정(canonical 비교는 서버에서).
 let baselineSmiles = '';

 // 초기 검수상태 반영
 document.getElementById('verdict').value = {json.dumps(cur_verdict)};
 document.getElementById('note').value = {json.dumps(rec.get('note') or '')};

 function setMol(smiles){{
   kframe.contentWindow.postMessage({{type:'ketcher:set', smiles: smiles||''}}, '*');
 }}
 function reloadOriginal(){{ baselineSmiles=''; setMol(INIT_SMILES); }}

 // Ketcher iframe과의 통신
 window.addEventListener('message', (ev)=>{{
   const d = ev.data||{{}};
   if(d.type==='ketcher:ready'){{
     ketcherReady = true;
     setMol(INIT_SMILES);
     // 로드가 반영될 시간을 준 뒤 기준선 캡처 (Ketcher 정규화 출력)
     setTimeout(captureBaseline, 1200);
   }} else if(d.type==='ketcher:smiles'){{
     if(d.reqId==='baseline'){{ baselineSmiles=(d.smiles||'').trim(); return; }}
     if(window._pendingSave){{ window._pendingSave(d); window._pendingSave=null; }}
   }}
 }});

 function captureBaseline(){{
   if(ketcherReady) kframe.contentWindow.postMessage({{type:'ketcher:get', reqId:'baseline'}}, '*');
 }}

 function getKetcherSmiles(){{
   return new Promise((resolve)=>{{
     if(!ketcherReady){{ resolve({{smiles:'', error:'not-ready'}}); return; }}
     window._pendingSave = resolve;
     kframe.contentWindow.postMessage({{type:'ketcher:get', reqId:1}}, '*');
     setTimeout(()=>{{ if(window._pendingSave){{ window._pendingSave({{smiles:'',error:'timeout'}}); window._pendingSave=null; }} }}, 4000);
   }});
 }}

 async function saveItem(){{
   const st = document.getElementById('status');
   st.textContent = '저장 중...';
   const verdict = document.getElementById('verdict').value;
   const note = document.getElementById('note').value;
   const body = {{verdict, note}};
   // Markush는 그래픽 편집 미지원 -> 구조 판정 생략(메모/판정만 저장)
   if(!IS_MARKUSH){{
     const res = await getKetcherSmiles();
     body.current_smiles = (res.smiles||'').trim();
     body.baseline_smiles = baselineSmiles;   // 비어있으면 서버가 추출원본 유지로 처리
   }}
   try{{
     const resp = await fetch(BASE+'/api/item/'+IDX, {{
       method:'POST', headers:{{'Content-Type':'application/json'}},
       body: JSON.stringify(body)
     }});
     const j = await resp.json();
     const edited = j.corrected_smiles ? ' · 사람 수정본 기록' : ' · 추출 원본 유지(수정 없음)';
     st.textContent = '저장됨 ('+(j.updated_at||'')+')'+edited;
   }}catch(e){{ st.textContent='저장 실패: '+e; }}
 }}
</script>
</body></html>"""


def serve(out_dir, port=8200, host="0.0.0.0"):
    import http.server
    import socketserver
    from urllib.parse import urlparse

    srv = ReviewServer(out_dir)
    if not KETCHER_DIR.exists():
        print(f"⚠️  Ketcher 정적 빌드가 없습니다: {KETCHER_DIR}")
        print("    (구조 편집기 없이 목록/검수만 동작합니다)")

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # 조용히

        def _send(self, code, body, ctype="text/html; charset=utf-8", headers=None):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_json(self, code, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

        def _serve_ketcher(self, rel):
            # /ketcher/<rel> -> KETCHER_DIR/<rel>  (디렉토리 탈출 방지)
            rel = rel.lstrip("/") or "index.html"
            target = (KETCHER_DIR / rel).resolve()
            if KETCHER_DIR.resolve() not in target.parents and target != KETCHER_DIR.resolve():
                self._send(403, "forbidden"); return
            if not target.exists() or target.is_dir():
                self._send(404, "not found"); return
            ext = target.suffix.lower()
            ctype = {".html": "text/html; charset=utf-8", ".js": "text/javascript",
                     ".mjs": "text/javascript", ".css": "text/css", ".wasm": "application/wasm",
                     ".json": "application/json", ".map": "application/json",
                     ".png": "image/png", ".svg": "image/svg+xml",
                     ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf"}.get(ext, "application/octet-stream")
            self._send(200, target.read_bytes(), ctype)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/":
                    self._send(200, _list_page_html(srv))
                elif path.startswith("/item/"):
                    idx = int(path.split("/")[2])
                    if 0 <= idx < len(srv.rows):
                        self._send(200, _item_page_html(srv, idx))
                    else:
                        self._send(404, "out of range")
                elif path.startswith("/segment/"):
                    idx = int(path.split("/")[2].split(".")[0])
                    b = _seg_image_bytes(srv, idx)
                    if b:
                        self._send(200, b, "image/png")
                    else:
                        self._send(404, "no image")
                elif path.startswith("/highlight/"):
                    idx = int(path.split("/")[2].split(".")[0])
                    b = _highlight_image_bytes(srv, idx)
                    if b:
                        self._send(200, b, "image/png")
                    else:
                        self._send(404, "no image")
                elif path.startswith("/ketcher/") or path == "/ketcher":
                    self._serve_ketcher(path[len("/ketcher"):])
                elif path == "/api/rows":
                    self._send_json(200, {"rows": srv.rows, "state": srv.state})
                elif path == "/export":
                    self._send_json(200, srv.export())
                else:
                    self._send(404, "not found")
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        def do_HEAD(self):
            self.do_GET()

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                if path.startswith("/api/item/"):
                    idx = int(path.split("/")[3])
                    length = int(self.headers.get("Content-Length", 0))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    rec = srv.save_item(idx, payload)
                    self._send_json(200, rec)
                else:
                    self._send(404, "not found")
            except Exception as e:
                self._send_json(500, {"error": str(e)})

    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    httpd = Server((host, port), Handler)
    print(f"검수 서버 기동: http://localhost:{port}/  (out_dir={srv.out_dir})")
    print(f"  항목 {len(srv.rows)}개 · 검수상태 저장: {srv.state_path}")
    print(f"  Ketcher: {'OK' if KETCHER_DIR.exists() else '없음(편집 비활성)'}")
    print("  Ctrl-C 로 종료")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n종료")
        httpd.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "serve"],
                    help="build: 정적 HTML 생성 · serve: Ketcher 검수 서버")
    ap.add_argument("out_dir", help="통합 결과 디렉토리 (merged_integration.json 포함)")
    ap.add_argument("--port", type=int, default=8200, help="serve 포트 (기본 8200)")
    ap.add_argument("--host", default="0.0.0.0", help="serve 바인드 호스트")
    args = ap.parse_args()
    if args.mode == "build":
        build_html(args.out_dir)
    elif args.mode == "serve":
        serve(args.out_dir, port=args.port, host=args.host)
