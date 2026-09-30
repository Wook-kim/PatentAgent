#!/usr/bin/env python3
"""기존 엔진(MolScribe/MarkushGrapher-2) vs Glyph(OCSRGlyph/MarkushGlyph) 육안 비교 뷰어.

`glyph_phase0.py` 가 저장한 `glyph_phase0_report.json` 을 읽어, 세그먼트별로
원본 이미지 + 4개 엔진의 구조 그림(RDKit 렌더) + 문자열을 나란히 놓은
단일 HTML 을 만든다. 사람이 "어느 엔진이 맞았나"를 직접 판정할 수 있게 하는 것이 목적.

핵심: CXSMILES 의 ``|$R8;;_AP;...$|`` 라벨 블록을 RDKit 이 그대로 파싱하므로,
MarkushGrapher-2 가 부착점을 ``_AP`` 로 표기하고 다른 엔진은 메틸로 읽는 차이가
그림에서 바로 보인다.

실행:
    glyph/.venv/bin/python glyph_compare_view.py --job jobs/f80eccfc8f24
    glyph/.venv/bin/python glyph_compare_view.py --job jobs/f80eccfc8f24 --serve --port 8210
"""

import argparse
import base64
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from glyph_phase0 import (canon_ap, canon_skeleton, canon_unlabeled,  # noqa: E402
                          _smiles_part)

ENGINES = [
    ("MolScribe",    "smiles_molscribe",            "현행 OCSR"),
    ("OCSRGlyph",    "smiles_ocsrglyph",            "신규 OCSR"),
    ("MarkushGrapher-2", "cxsmiles_markushgrapher",  "현행 Markush"),
    ("MarkushGlyph", "cxsmiles_markushglyph_eff",   "신규 Markush"),
]


# ---------------------------------------------------------------------------
# 구조 렌더링
# ---------------------------------------------------------------------------
def _mol_from_any(s):
    """SMILES / CXSMILES / MarkushGlyph opt 형식 → RDKit Mol (라벨 보존).

    RDKit 은 CXSMILES 의 ``$...$`` 라벨 블록을 atomLabel prop 으로 읽어 주므로
    ``R8`` / ``Q`` / ``_AP`` 가 그림에 그대로 나온다. MarkushGlyph 의 opt 형식
    (``<r>Q</r>``, ``[\\CH3]``)은 atom-map 을 경유해 같은 라벨을 붙인다.
    """
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    s = str(s or "").strip()
    if not s:
        return None

    # 1) opt 형식이면 라벨을 atom-map 으로 옮겨 심는다
    if "<r>" in s or "[\\" in s:
        labels = {}
        counter = [0]

        def sub(m):
            counter[0] += 1
            labels[counter[0]] = m.group(1)
            return f"[*:{counter[0]}]"

        body = s.split("|")[0]
        body = body.replace("[\\*]", "*")
        body = re.sub(r"<r>([^<]*)</r>", sub, body)
        body = re.sub(r"\[\\([A-Za-z0-9]+)\]", sub, body)
        m = Chem.MolFromSmiles(body)
        if m is None:
            return None
        for a in m.GetAtoms():
            k = a.GetAtomMapNum()
            if k and k in labels:
                a.SetProp("atomLabel", labels[k])
                a.SetAtomMapNum(0)
        return m

    # 2) CXSMILES 전체를 먼저 시도 (라벨 블록 파싱) → 실패 시 SMILES 부분만
    m = Chem.MolFromSmiles(s)
    if m is None:
        m = Chem.MolFromSmiles(_smiles_part(s))
    return m


def render_svg(s, size=(300, 210)):
    """구조 문자열 → inline SVG. 실패 시 None."""
    try:
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
        m = _mol_from_any(s)
        if m is None:
            return None
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(m)
        d = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        o = d.drawOptions()
        o.addStereoAnnotation = False
        o.clearBackground = False
        rdMolDraw2D.PrepareAndDrawMolecule(d, m)
        d.FinishDrawing()
        svg = d.GetDrawingText()
        return svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")
    except Exception:
        return None


def _img_data_uri(path):
    try:
        p = Path(path)
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() \
            if p.exists() else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 행별 판정
# ---------------------------------------------------------------------------
def classify(row):
    """행 하나에 대해 엔진별 합의 클러스터와 배지를 계산."""
    keys = [k for _, k, _ in ENGINES]
    ap = {k: canon_skeleton(row.get(k)) for k in keys}
    un = {k: canon_unlabeled(row.get(k)) for k in keys}

    avail = [k for k in keys if ap[k]]
    counts = {}
    for k in avail:
        counts[ap[k]] = counts.get(ap[k], 0) + 1
    top_n = max(counts.values()) if counts else 0
    majority = [c for c, n in counts.items() if n == top_n and top_n >= 2]
    maj = majority[0] if len(majority) == 1 else None

    cur_mismatch = row.get("agreement_current") == "mismatch"
    # rescued 판정은 프로덕션이 실제로 쓸 지표(canon_ap)로 재야 한다.
    # 합의 클러스터(ap)는 엔진 간 공정비교용 canon_skeleton 기준이라 별개.
    ms, mg2 = canon_ap(row.get("smiles_molscribe")), canon_ap(row.get("cxsmiles_markushgrapher"))
    pair_ap_ok = bool(ms and mg2 and ms == mg2)
    pair_raw_ok = bool(un["smiles_molscribe"] and un["cxsmiles_markushgrapher"]
                       and un["smiles_molscribe"] == un["cxsmiles_markushgrapher"])

    tags = []
    if cur_mismatch and pair_ap_ok:
        tags.append("rescued")          # 부착점 표기 차이로 인한 위양성
    if cur_mismatch and not pair_ap_ok:
        tags.append("realmismatch")     # 정규화해도 다름 = 진짜 불일치
    if len(avail) == 4 and top_n == 4:
        tags.append("unanimous")
    if not row.get("cxsmiles_markushglyph"):
        tags.append("mgserialfail")     # MarkushGlyph plain 직렬화 실패
    if maj and ap["cxsmiles_markushgrapher"] and ap["cxsmiles_markushgrapher"] != maj:
        tags.append("mg2_dissent")
    if maj and ap["cxsmiles_markushglyph_eff"] and ap["cxsmiles_markushglyph_eff"] != maj:
        tags.append("mkglyph_dissent")
    if top_n < 2:
        tags.append("nocons")

    status = {}
    for k in keys:
        if not ap[k]:
            status[k] = "bad"           # 파싱 불가/빈값
        elif maj and ap[k] == maj:
            status[k] = "ok"            # 다수 합의와 일치
        elif maj:
            status[k] = "diff"          # 단독/소수 의견
        else:
            status[k] = "none"          # 합의 자체가 없음
    return {"ap": ap, "status": status, "tags": tags,
            "pair_ap_ok": pair_ap_ok, "pair_raw_ok": pair_raw_ok}


CSS = """
:root{--bg:#0f1216;--card:#171c23;--line:#2a323d;--fg:#e6edf3;--dim:#8b98a5;
      --ok:#2ea043;--diff:#d29922;--bad:#f85149;--acc:#58a6ff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans KR",sans-serif}
header{position:sticky;top:0;z-index:10;background:#0f1216ee;backdrop-filter:blur(8px);
       border-bottom:1px solid var(--line);padding:14px 20px}
h1{margin:0 0 4px;font-size:17px}
.sub{color:var(--dim);font-size:12.5px}
.filters{margin-top:10px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
button{background:var(--card);color:var(--fg);border:1px solid var(--line);
       border-radius:6px;padding:5px 11px;cursor:pointer;font-size:12.5px}
button:hover{border-color:var(--acc)}
button.on{background:var(--acc);border-color:var(--acc);color:#0b0f14;font-weight:600}
input[type=search]{background:var(--card);border:1px solid var(--line);color:var(--fg);
       border-radius:6px;padding:5px 10px;font-size:12.5px;min-width:190px}
main{padding:16px 20px 60px}
.card{background:var(--card);border:1px solid var(--line);border-radius:9px;
      margin-bottom:14px;overflow:hidden}
.chead{display:flex;flex-wrap:wrap;gap:8px;align-items:center;
       padding:9px 13px;border-bottom:1px solid var(--line);background:#1b212a}
.seg{font-weight:700;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.badge{font-size:11px;padding:2px 7px;border-radius:11px;border:1px solid var(--line);color:var(--dim)}
.b-rescued{background:#1a3a24;border-color:#2ea04355;color:#7ee2a0}
.b-realmismatch{background:#3d1d1d;border-color:#f8514955;color:#ffa198}
.b-unanimous{background:#16304d;border-color:#58a6ff55;color:#9dcbff}
.b-mgserialfail{background:#3d3213;border-color:#d2992255;color:#e8c766}
.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:0}
.cell{padding:10px;border-right:1px solid var(--line);min-width:0}
.cell:last-child{border-right:0}
.cell h4{margin:0 0 2px;font-size:12px;letter-spacing:.02em}
.cell .role{color:var(--dim);font-size:10.5px;margin-bottom:6px}
.pane{background:#fff;border-radius:6px;display:flex;align-items:center;
      justify-content:center;min-height:150px;padding:4px}
.pane img{max-width:100%;height:auto;display:block}
.pane svg{max-width:100%;height:auto}
.pane.empty{background:#20262f;color:var(--dim);font-size:11.5px;min-height:150px}
.smi{margin-top:7px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
     font-size:10.5px;line-height:1.45;color:#c9d5e1;word-break:break-all;
     max-height:82px;overflow:auto}
.st{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px;vertical-align:1px}
.st-ok{background:var(--ok)}.st-diff{background:var(--diff)}
.st-bad{background:var(--bad)}.st-none{background:var(--dim)}
.orig h4{color:var(--acc)}
.legend{color:var(--dim);font-size:11.5px;margin-top:8px}
.hidden{display:none}
#count{color:var(--dim);font-size:12.5px;margin-left:6px}
@media(max-width:1250px){.grid{grid-template-columns:repeat(3,minmax(0,1fr))}}
"""

JS = """
const cards=[...document.querySelectorAll('.card')];
let flt='all', q='';
function apply(){
  let n=0;
  for(const c of cards){
    const tags=c.dataset.tags.split(' ');
    const okF = flt==='all' || tags.includes(flt);
    const okQ = !q || c.dataset.hay.includes(q);
    const show = okF && okQ;
    c.classList.toggle('hidden', !show);
    if(show) n++;
  }
  document.getElementById('count').textContent = n+'건 표시';
}
document.querySelectorAll('.filters button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.filters button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); flt=b.dataset.f; apply();
});
document.getElementById('q').oninput=e=>{q=e.target.value.toLowerCase();apply();};
apply();
"""

BADGE_LABEL = {
    "rescued": "부착점 위양성 (현행 mismatch → 실제 일치)",
    "realmismatch": "진짜 불일치",
    "unanimous": "4엔진 만장일치",
    "mgserialfail": "MarkushGlyph 직렬화 실패",
    "mg2_dissent": "MG2 단독반대",
    "mkglyph_dissent": "MarkushGlyph 단독반대",
    "nocons": "합의 없음",
}


def build(job_dir, out_html):
    rep = json.loads((job_dir / "glyph_phase0_report.json").read_text(encoding="utf-8"))
    rows = rep["rows"]

    def sort_key(r):
        p, _, i = str(r["seg_key"]).partition("_")
        return (int(p) if p.isdigit() else 0, int(i) if i.isdigit() else 0)
    rows = sorted(rows, key=sort_key)

    tally = {k: 0 for k in BADGE_LABEL}
    cards = []
    for r in rows:
        c = classify(r)
        for t in c["tags"]:
            tally[t] = tally.get(t, 0) + 1

        img = _img_data_uri(job_dir / "mg_input" / f"{r['seg_key']}.png")
        cells = [
            '<div class="cell orig"><h4>원본 세그먼트</h4>'
            f'<div class="role">page {html.escape(str(r["seg_key"]).split("_")[0])}</div>'
            + (f'<div class="pane"><img src="{img}" alt="segment"></div>'
               if img else '<div class="pane empty">이미지 없음</div>')
            + '</div>'
        ]
        for name, key, role in ENGINES:
            val = r.get(key)
            svg = render_svg(val)
            st = c["status"][key]
            cells.append(
                f'<div class="cell"><h4><span class="st st-{st}"></span>{html.escape(name)}</h4>'
                f'<div class="role">{html.escape(role)}</div>'
                + (f'<div class="pane">{svg}</div>' if svg else
                   '<div class="pane empty">' +
                   ("엔진이 빈 출력" if not _smiles_part(val) else "RDKit 렌더 불가")
                   + '</div>')
                + f'<div class="smi">{html.escape(str(val or "(없음)"))}</div></div>'
            )

        badges = [f'<span class="badge b-{t}">{html.escape(BADGE_LABEL[t])}</span>'
                  for t in c["tags"] if t in BADGE_LABEL]
        cid = r.get("compound_id") or ""
        meta = [
            f'현행판정 <b>{html.escape(str(r.get("agreement_current") or "-"))}</b>',
            f'신뢰도 <b>{html.escape(str(r.get("confidence_current") or "-"))}</b>',
            f'is_markush <b>{"Y" if r.get("is_markush_current") else "N"}</b>',
        ]
        hay = " ".join(str(r.get(k) or "") for _, k, _ in ENGINES).lower() \
            + " " + str(r["seg_key"]).lower() + " " + str(cid).lower()

        cards.append(
            f'<div class="card" data-tags="{" ".join(c["tags"])}" '
            f'data-hay="{html.escape(hay, quote=True)}">'
            f'<div class="chead"><span class="seg">{html.escape(str(r["seg_key"]))}</span>'
            + (f'<span class="badge">ID {html.escape(str(cid))}</span>' if cid else "")
            + '<span class="badge">' + '</span><span class="badge">'.join(meta) + '</span>'
            + "".join(badges)
            + f'</div><div class="grid">{"".join(cells)}</div></div>'
        )

    n = len(rows)
    fbtns = [("all", f"전체 {n}")] + [
        (k, f"{BADGE_LABEL[k]} {tally.get(k, 0)}")
        for k in ("rescued", "realmismatch", "unanimous", "mgserialfail",
                  "mg2_dissent", "mkglyph_dissent", "nocons")
        if tally.get(k)
    ]
    filters = "".join(
        '<button data-f="{}"{}>{}</button>'.format(
            k, ' class="on"' if k == "all" else "", html.escape(lab))
        for k, lab in fbtns)

    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Glyph 비교 — {html.escape(job_dir.name)}</title><style>{CSS}</style></head><body>
<header>
  <h1>기존 엔진 vs Glyph 구조 인식 비교</h1>
  <div class="sub">{html.escape(str(job_dir))} · 세그먼트 {n}개 ·
    원본 그림과 4개 엔진 결과를 나란히 보고 <b>어느 엔진이 맞았는지 직접 판정</b>하세요.</div>
  <div class="filters">{filters}
    <input type="search" id="q" placeholder="seg_key / SMILES 검색"><span id="count"></span></div>
  <div class="legend">
    <span class="st st-ok"></span>다수합의와 일치 ·
    <span class="st st-diff"></span>단독/소수의견 ·
    <span class="st st-bad"></span>파싱불가·빈출력 ·
    <span class="st st-none"></span>합의없음 &nbsp;|&nbsp;
    합의는 <b>상위원자 라벨(_AP/CH2/OH…)·R라벨·입체를 모두 해소한 골격</b> 기준.
    구조그림의 <code>R8</code>/<code>Q</code>/<code>_AP</code>
    라벨은 CXSMILES 라벨블록을 그대로 렌더한 것입니다.
  </div>
</header>
<main>{"".join(cards)}</main>
<script>{JS}</script></body></html>"""

    out_html.write_text(doc, encoding="utf-8")
    print(f"생성: {out_html}  ({out_html.stat().st_size / 1e6:.1f} MB, {n}건)")
    print("배지 집계: " + ", ".join(f"{BADGE_LABEL[k]}={v}" for k, v in tally.items() if v))
    return out_html


def write_csv(job_dir, out_csv):
    """엑셀로 훑어볼 수 있는 4엔진 나란히 CSV (판정 배지 포함)."""
    import csv
    rep = json.loads((job_dir / "glyph_phase0_report.json").read_text(encoding="utf-8"))
    fields = ["seg_key", "compound_id", "page", "agreement_current", "confidence_current",
              "is_markush_current", "verdict", "tags",
              "smiles_molscribe", "smiles_ocsrglyph",
              "cxsmiles_markushgrapher", "cxsmiles_markushglyph_eff",
              "cxsmiles_markushglyph_plain", "cxsmiles_opt_glyph"]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rep["rows"], key=lambda x: str(x["seg_key"])):
            c = classify(r)
            if "rescued" in c["tags"]:
                verdict = "부착점위양성"
            elif "realmismatch" in c["tags"]:
                verdict = "진짜불일치"
            elif "unanimous" in c["tags"]:
                verdict = "4엔진만장일치"
            else:
                verdict = "부분합의"
            w.writerow({**r,
                        "page": str(r["seg_key"]).split("_")[0],
                        "verdict": verdict,
                        "tags": " ".join(c["tags"]),
                        "cxsmiles_markushglyph_plain": r.get("cxsmiles_markushglyph")})
    print(f"생성: {out_csv}")
    return out_csv


def serve(html_path, port=8210, host="0.0.0.0"):
    """단일 HTML 을 그대로 서빙 (stdlib 만 사용)."""
    import http.server
    import socketserver
    data = html_path.read_bytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    print(f"비교 뷰어: http://localhost:{port}/   (Ctrl-C 종료)")
    with S((host, port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n종료")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="jobs/f80eccfc8f24")
    ap.add_argument("--out", default=None, help="기본: <job>/glyph_compare.html")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8210)
    args = ap.parse_args()

    job_dir = (ROOT / args.job).resolve()
    out_html = Path(args.out).resolve() if args.out else job_dir / "glyph_compare.html"
    build(job_dir, out_html)
    write_csv(job_dir, job_dir / "glyph_compare.csv")
    if args.serve:
        serve(out_html, args.port)


if __name__ == "__main__":
    main()
