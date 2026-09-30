#!/usr/bin/env python3
"""
PatentAgent 웹 서비스 게이트웨이
================================

사용자가 서버 로컬 examples/ 의 PDF를 선택하면 시작되는 단일 진입점.
(네트워크 정책상 브라우저 직접 업로드 불가 → examples/ 에 미리 둔 PDF에서 선택)

흐름:
  1) 사용자가 브라우저에서 examples/ PDF 선택 (+ 추출 옵션)
  2) 게이트웨이가 작업(job)을 큐에 넣고 job_id 즉시 반환
  3) 단일 워커가 integrate_prototype.py 를 순차 실행 (GPU 경합 방지: 동시성 1)
  4) 진행 상태/로그를 폴링으로 확인
  5) 완료되면 Ketcher 검수 UI 로 연결 (review_app 렌더 함수 재사용)

설계 원칙:
  - 게이트웨이 자체는 GPU/무거운 의존성 없음 (system python3 + fastapi/uvicorn).
    무거운 추출은 integrate_prototype.py 서브프로세스 + 기존 마이크로서비스
    (LiteLLM 4000, PaddleOCR 8010, markush_service 8100) 에 위임.
  - 159개 화합물 기준 추출에 30분+ 걸리므로 요청-응답 동기 처리 불가 → 작업 큐.
  - GPU(TITAN RTX) 경합 방지를 위해 워커는 동시성 1 (한 번에 한 작업).

실행:
  python3 gateway.py --port 8000
  (전제: LiteLLM 4000 / PaddleOCR 8010 / markush_service 8100 가동)
"""
import argparse
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, RedirectResponse
import uvicorn

# review_app 의 렌더링/상태 로직 재사용 (검수 UI 를 다시 구현하지 않음)
import review_app

ROOT = Path(__file__).resolve().parent
JOBS_DIR = ROOT / "jobs"            # 작업별 작업공간 (= integrate_prototype --out)
JOBS_DIR.mkdir(exist_ok=True)
EXAMPLES_DIR = ROOT / "examples"    # 서버 로컬 예제 PDF 디렉토리 (업로드 대신 선택)
EXAMPLES_DIR.mkdir(exist_ok=True)
INTEGRATE = ROOT / "integrate_prototype.py"


def _list_examples():
    """examples 디렉토리의 PDF 목록 (이름, 크기)."""
    out = []
    for p in sorted(EXAMPLES_DIR.glob("*.pdf")):
        out.append({"name": p.name, "size": p.stat().st_size})
    return out


MG_SERVICE_URL = os.environ.get("MG_SERVICE_URL", "http://localhost:8100")
_UA = "Mozilla/5.0 (PatentAgent)"
_PAT_NUM_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{5,}$")  # US10000000B2, EP1234567A1 ...
MAX_PDF_BYTES = int(os.environ.get("PATENTAGENT_MAX_PDF_BYTES", str(100 * 1024 * 1024)))
MAX_HTML_BYTES = int(os.environ.get("PATENTAGENT_MAX_HTML_BYTES", str(5 * 1024 * 1024)))


def _form_bool(value):
    """HTML checkbox/form 값을 명시 bool 로 변환."""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _validate_external_http_url(url):
    """임의 URL 입력의 SSRF 위험을 줄이기 위한 최소 검증."""
    u = urlparse(url)
    if u.scheme not in {"http", "https"}:
        raise ValueError("http(s) URL만 지원합니다.")
    if not u.hostname:
        raise ValueError("URL 호스트를 확인할 수 없습니다.")
    if u.username or u.password:
        raise ValueError("인증정보가 포함된 URL은 지원하지 않습니다.")

    try:
        infos = socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"URL 호스트를 해석할 수 없습니다: {u.hostname}") from exc
    for info in infos:
        addr = ip_address(info[4][0])
        if (
            addr.is_private or addr.is_loopback or addr.is_link_local
            or addr.is_multicast or addr.is_unspecified or addr.is_reserved
        ):
            raise ValueError("내부망/로컬 주소로 보이는 URL은 지원하지 않습니다.")


def _http_get(url, timeout=60, binary=False, max_bytes=None, validate=False):
    import urllib.request
    if validate:
        _validate_external_http_url(url)
    max_bytes = max_bytes if max_bytes is not None else (MAX_PDF_BYTES if binary else MAX_HTML_BYTES)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    if validate:
        class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                _validate_external_http_url(newurl)
                return super().redirect_request(req, fp, code, msg, headers, newurl)
        opener = urllib.request.build_opener(_SafeRedirectHandler)
        response = opener.open(req, timeout=timeout)
    else:
        response = urllib.request.urlopen(req, timeout=timeout)
    with response as r:
        clen = r.headers.get("Content-Length")
        if clen and int(clen) > max_bytes:
            raise ValueError(f"다운로드 파일이 너무 큽니다({int(clen)} bytes > {max_bytes} bytes).")
        chunks = []
        total = 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"다운로드 파일이 너무 큽니다(>{max_bytes} bytes).")
            chunks.append(chunk)
        data = b"".join(chunks)
    return data if binary else data.decode("utf-8", "replace")


def resolve_patent_pdf(patent_input):
    """특허 입력(번호/Google Patents 링크/PDF 직접 URL)을 PDF bytes + 권장 파일명으로 해석.

    지원 형태:
      - 특허번호:        US10000000B2, EP1234567A1
      - Google Patents:  https://patents.google.com/patent/US10000000B2/en
      - PDF 직접 URL:    https://.../foo.pdf
    반환: (pdf_bytes, filename). 실패 시 ValueError.
    """
    s = (patent_input or "").strip()
    if not s:
        raise ValueError("특허 입력이 비어 있습니다.")

    # 1) PDF 직접 URL
    if s.lower().startswith("http") and s.lower().split("?")[0].endswith(".pdf"):
        data = _http_get(s, binary=True, max_bytes=MAX_PDF_BYTES, validate=True)
        name = os.path.basename(s.split("?")[0]) or "patent.pdf"
        return data, name

    # 2) 특허번호 추출 (Google Patents 링크면 경로에서, 아니면 입력 자체)
    patent_no = None
    m = re.search(r"patents\.google\.com/patent/([A-Z0-9]+)", s, re.I)
    if m:
        patent_no = m.group(1).upper()
    elif _PAT_NUM_RE.match(s.upper().replace(" ", "")):
        patent_no = s.upper().replace(" ", "")
    if not patent_no:
        raise ValueError(f"특허번호/링크를 해석할 수 없습니다: {patent_input}")

    # 3) Google Patents 페이지에서 PDF 링크 추출
    page_url = f"https://patents.google.com/patent/{patent_no}/en"
    html_txt = _http_get(page_url, timeout=40, max_bytes=MAX_HTML_BYTES, validate=True)
    pm = re.search(r'https://patentimages\.storage\.googleapis\.com/[^"\s]+\.pdf', html_txt)
    if pm:
        data = _http_get(pm.group(0), timeout=120, binary=True, max_bytes=MAX_PDF_BYTES, validate=True)
        return data, f"{patent_no}.pdf"

    # 4) 페이지에 없으면 XHR query API 로 pdf 필드 + 패밀리국가 확인
    family = _google_patents_lookup(patent_no)
    if family.get("pdf"):
        data = _http_get(family["pdf"], timeout=120, binary=True, max_bytes=MAX_PDF_BYTES, validate=True)
        return data, f"{patent_no}.pdf"

    # PDF 없음 — 패밀리가 존재하는 국가를 안내에 포함해 ValueError
    suggest = ""
    countries = [c for c in family.get("countries", []) if c not in ("WO",)]
    if countries:
        suggest = (f" 이 특허는 {', '.join(countries)} 패밀리가 있습니다 — "
                   f"해당 국가 공보(예: {countries[0]}…)는 보통 PDF 가 있으니 그 번호로 시도해 보세요.")
    raise ValueError(
        f"{patent_no} 는 Google Patents 에 PDF 가 없습니다(최신 PCT/WO 공보는 미확보가 흔함)."
        f"{suggest} 또는 PDF 직접 URL 을 넣거나, PDF 를 서버 examples/ 에 두고 '예제 PDF 선택' 탭을 쓰세요.")


def _google_patents_lookup(patent_no):
    """Google Patents XHR query API 로 pdf 링크와 패밀리 국가코드를 조회.
    반환: {'pdf': <url 또는 ''>, 'countries': [국가코드...]}. 실패 시 빈 dict."""
    import urllib.parse
    try:
        q = urllib.parse.quote(f"q={patent_no}")
        d = _http_get(f"https://patents.google.com/xhr/query?url={q}&exp=", timeout=30,
                      max_bytes=MAX_HTML_BYTES, validate=True)
        j = json.loads(d)
        res = j["results"]["cluster"][0]["result"][0]["patent"]
        pdf = res.get("pdf") or ""
        if pdf and not pdf.startswith("http"):
            pdf = "https://patentimages.storage.googleapis.com/" + pdf
        countries = []
        try:
            for cs in res["family_metadata"]["aggregated"]["country_status"]:
                cc = cs.get("country_code")
                if cc:
                    countries.append(cc)
        except (KeyError, TypeError):
            pass
        return {"pdf": pdf, "countries": countries}
    except Exception:
        return {}


def _now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# 파이프라인 단계 정의: (key, 표시명, 완료판정 로그마커들[하나라도 있으면 완료])
PIPELINE_STEPS = [
    ("autopages", "페이지·어세이 자동 탐지",
     ["[BioChemInsight] 실행", "[BioChemInsight] structures.csv", "--structure-pages"]),
    ("structures", "구조 추출 (BioChemInsight)",
     ["[BioChemInsight] 구조 세그먼트"]),
    ("activity", "활성값 추출",
     ["[Activity]"]),
    ("markush", "Markush 구조 인식 (MarkushGrapher)",
     ["개 CXSMILES 예측"]),
    ("coref", "구조↔ID 교차검증 (MolCoref)",
     ["[MolCoref]"]),
    ("merge", "병합·검증·결과 생성",
     ["=== 병합 완료"]),
]


def _compute_steps(job_id, meta):
    """run.log 마커로 파이프라인 단계별 상태 계산.
    반환: [{key, name, state}] — state ∈ done|active|pending. (active=현재 진행중)"""
    log_path = JOBS_DIR / job_id / "run.log"
    log = ""
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8", errors="replace")

    with_coref = bool((meta.get("options") or {}).get("with_coref"))
    steps = [s for s in PIPELINE_STEPS if s[0] != "coref" or with_coref]

    status = meta.get("status")
    done_flags = [any(mk in log for mk in markers) for (_k, _n, markers) in steps]

    # 작업이 done 이면 모든 단계 완료로 간주(로그 마커 누락 대비)
    if status == "done":
        done_flags = [True] * len(steps)

    result = []
    first_pending_marked = False
    terminal = status in ("done", "error", "cancelled", "interrupted")
    for (key, name, _mk), is_done in zip(steps, done_flags):
        if is_done:
            state = "done"
        elif not first_pending_marked and not terminal:
            # 첫 번째 미완료 단계 = 현재 진행 중 (작업이 살아있을 때만)
            state = "active"
            first_pending_marked = True
        else:
            state = "pending"
        result.append({"key": key, "name": name, "state": state})
    return result


def _job_proc_pids(job_id):
    """job_id 가 cmdline 에 들어간 추출 프로세스 PID 목록 (orphan 탐지/종료용)."""
    pids = []
    try:
        out = subprocess.run(["pgrep", "-f", job_id], stdout=subprocess.PIPE,
                             timeout=5).stdout.decode()
        for line in out.split():
            try:
                pid = int(line)
                if pid != os.getpid():
                    pids.append(pid)
            except ValueError:
                pass
    except Exception:
        pass
    return pids


def _error_html(title, message):
    """사용자에게 보이는 친절한 에러 페이지 (특허 PDF 실패 등)."""
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>{review_app.html.escape(title)}</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:680px;margin:60px auto;padding:0 16px;color:#222}}
 .box{{background:#fff;border:1px solid #ffcdd2;border-left:5px solid #c62828;border-radius:8px;padding:22px}}
 h1{{font-size:19px;color:#c62828;margin:0 0 10px}}
 p{{line-height:1.7;color:#444}}
 a{{display:inline-block;margin-top:18px;color:#1565c0}}
</style></head><body>
<div class="box">
  <h1>⚠️ {review_app.html.escape(title)}</h1>
  <p>{review_app.html.escape(message)}</p>
  <a href="/">← 홈으로 돌아가기</a>
</div></body></html>"""


def _build_opts(auto_pages, structure_pages, assay_pages, assay_names, engine, with_coref):
    """업로드/특허 핸들러 공통 옵션 dict."""
    return {
        "auto_pages": _form_bool(auto_pages),
        "structure_pages": (structure_pages or "").strip(),
        "assay_pages": (assay_pages or "").strip(),
        "assay_names": (assay_names or "").strip(),
        "engine": (engine or "").strip() or "molscribe",
        "with_coref": _form_bool(with_coref),
        "gpu": "1",
    }


# ===========================================================================
# 작업 큐 + 단일 워커 (GPU 직렬화)
# ===========================================================================
class JobManager:
    def __init__(self):
        self.q = queue.Queue()
        self.jobs = {}                      # job_id -> 메타(dict)
        self.procs = {}                     # job_id -> Popen (실행 중인 것만)
        self.cancelled = set()              # 취소 요청된 job_id
        self.lock = threading.Lock()
        self._load_existing()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _meta_path(self, job_id):
        return JOBS_DIR / job_id / "job.json"

    def _save_meta(self, job_id):
        d = JOBS_DIR / job_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(
            json.dumps(self.jobs[job_id], ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_existing(self):
        """재기동 시 기존 job 메타 로드.

        'running' 이던 작업은: 산출물이 이미 있으면 done, 추출 프로세스가
        아직 살아있으면 'running'(orphan 재입양), 둘 다 아니면 interrupted.
        'queued' 이던 작업은 메모리 큐가 사라졌으므로 새 워커 큐에 재등록.
        """
        requeue = []
        for d in sorted(JOBS_DIR.glob("*/job.json")):
            try:
                meta = json.loads(d.read_text(encoding="utf-8"))
                job_id = meta["job_id"]
                if meta.get("status") == "running":
                    merged = JOBS_DIR / job_id / "merged_integration.json"
                    if merged.exists():
                        meta["status"] = "done"
                        try:
                            meta["n_compounds"] = len(json.loads(merged.read_text("utf-8")))
                        except Exception:
                            pass
                    elif _job_proc_pids(job_id):
                        meta["orphaned"] = True   # 핸들 없이 살아있는 추출 (이전 게이트웨이가 띄움)
                    else:
                        meta["status"] = "interrupted"
                        meta["error"] = "서버 재기동으로 중단됨"
                        meta["finished_at"] = meta.get("finished_at") or _now()
                elif meta.get("status") == "queued":
                    requeue.append(job_id)
                self.jobs[job_id] = meta
            except Exception:
                pass
        for job_id in requeue:
            self.q.put(job_id)

    def create(self, src_pdf, filename, opts):
        """src_pdf: 예제 PDF 경로(Path). 작업 디렉토리로 복사."""
        job_id = uuid.uuid4().hex[:12]
        d = JOBS_DIR / job_id
        d.mkdir(parents=True, exist_ok=True)
        pdf_path = d / "input.pdf"
        pdf_path.write_bytes(Path(src_pdf).read_bytes())
        meta = {
            "job_id": job_id,
            "filename": filename,
            "status": "queued",            # queued -> running -> done | error | interrupted
            "options": opts,
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "n_compounds": None,
        }
        with self.lock:
            self.jobs[job_id] = meta
            self._save_meta(job_id)
        self.q.put(job_id)
        return job_id

    def get(self, job_id):
        meta = self.jobs.get(job_id)
        # orphan(재입양된 추출): 우리 워커가 관리 안 하므로 상태를 직접 점검
        if meta and meta.get("orphaned"):
            merged = JOBS_DIR / job_id / "merged_integration.json"
            if merged.exists():
                try:
                    meta["n_compounds"] = len(json.loads(merged.read_text("utf-8")))
                except Exception:
                    pass
                meta["status"] = "done"
                meta.pop("orphaned", None)
                meta["finished_at"] = meta.get("finished_at") or _now()
                self._save_meta(job_id)
            elif not _job_proc_pids(job_id):
                meta["status"] = "error"
                meta["error"] = "추출 프로세스가 종료되었으나 결과가 없습니다(중단 추정)."
                meta.pop("orphaned", None)
                meta["finished_at"] = _now()
                self._save_meta(job_id)
        return meta

    def list(self):
        # 작업 디렉토리가 사라진 항목은 제외 (디스크와 메모리 불일치 방지)
        live = [m for m in self.jobs.values() if (JOBS_DIR / m["job_id"]).is_dir()]
        return sorted(live, key=lambda m: m["created_at"], reverse=True)

    def _run(self):
        """단일 워커 루프: 큐에서 하나씩 꺼내 integrate_prototype 실행."""
        while True:
            job_id = self.q.get()
            meta = self.jobs.get(job_id)
            if not meta or meta["status"] != "queued":
                continue
            d = JOBS_DIR / job_id
            log_path = d / "run.log"
            meta["status"] = "running"
            meta["started_at"] = _now()
            self._save_meta(job_id)
            # 큐 대기 중 취소된 경우 실행하지 않음
            if job_id in self.cancelled:
                self.cancelled.discard(job_id)
                meta["status"] = "cancelled"
                meta["error"] = "대기 중 취소됨"
                meta["finished_at"] = _now()
                self._save_meta(job_id)
                continue
            try:
                cmd = self._build_cmd(d, meta["options"])
                with open(log_path, "w") as logf:
                    logf.write(f"$ {' '.join(cmd)}\n\n")
                    logf.flush()
                    env = dict(os.environ)
                    env["PYTHONUNBUFFERED"] = "1"
                    env.setdefault("AWS_REGION", "us-east-1")
                    env.setdefault("AWS_DEFAULT_REGION", "us-east-1")
                    # 새 프로세스 그룹으로 실행 → 자식(pipeline.py 등)까지 한 번에 종료 가능
                    proc = subprocess.Popen(
                        cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
                        env=env, start_new_session=True)
                    with self.lock:
                        self.procs[job_id] = proc
                    returncode = proc.wait()
                with self.lock:
                    self.procs.pop(job_id, None)
                merged = d / "merged_integration.json"
                if job_id in self.cancelled:
                    self.cancelled.discard(job_id)
                    meta["status"] = "cancelled"
                    meta["error"] = "사용자가 작업을 중단함"
                elif returncode == 0 and merged.exists():
                    rows = json.loads(merged.read_text(encoding="utf-8"))
                    meta["n_compounds"] = len(rows)
                    meta["status"] = "done"
                else:
                    meta["status"] = "error"
                    meta["error"] = f"추출 실패 (exit={returncode}). run.log 확인."
            except Exception as e:
                meta["status"] = "error"
                meta["error"] = str(e)
            finally:
                with self.lock:
                    self.procs.pop(job_id, None)
                meta["finished_at"] = _now()
                self._save_meta(job_id)

    def cancel(self, job_id):
        """작업 취소. 실행 중이면 프로세스 그룹 종료, 대기 중이면 큐에서 스킵 표시."""
        meta = self.jobs.get(job_id)
        if not meta:
            return None
        if meta["status"] not in ("queued", "running"):
            return meta  # 이미 끝남 — 변화 없음
        self.cancelled.add(job_id)

        # orphan(이전 게이트웨이가 띄운 추출): Popen 핸들 없음 → PID 로 종료
        if meta.get("orphaned"):
            for pid in _job_proc_pids(job_id):
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.kill(pid, sig)
                    except (ProcessLookupError, PermissionError):
                        pass
            meta["status"] = "cancelled"
            meta["error"] = "사용자가 작업을 중단함"
            meta.pop("orphaned", None)
            meta["finished_at"] = _now()
            self._save_meta(job_id)
            return meta

        with self.lock:
            proc = self.procs.get(job_id)
        if proc and proc.poll() is None:
            # 프로세스 그룹 전체에 SIGTERM → 잠시 후 SIGKILL
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        else:
            # 대기 중(queued)이면 워커가 집어들 때 스킵; 상태는 즉시 반영
            meta["status"] = "cancelled"
            meta["error"] = "대기 중 취소됨"
            meta["finished_at"] = _now()
            self._save_meta(job_id)
        return meta

    def _build_cmd(self, job_dir, opts):
        cmd = [sys.executable, str(INTEGRATE), str(job_dir / "input.pdf"),
               "--out", str(job_dir),
               "--engine", opts.get("engine") or "molscribe",
               "--gpu", str(opts.get("gpu") or "1"),
               "--mg-service-url", MG_SERVICE_URL]
        if opts.get("auto_pages"):
            cmd.append("--auto-pages")
        else:
            if opts.get("structure_pages"):
                cmd += ["--structure-pages", opts["structure_pages"]]
            if opts.get("assay_pages"):
                cmd += ["--assay-pages", opts["assay_pages"]]
        if opts.get("assay_names"):
            cmd += ["--assay-names", opts["assay_names"]]
        if opts.get("with_coref"):
            cmd.append("--with-coref")
        return cmd


JM = JobManager()


# ===========================================================================
# 검수: review_app 의 ReviewServer 를 job 디렉토리에 붙여 렌더 함수 재사용
# ===========================================================================
def _review_server(job_id):
    d = JOBS_DIR / job_id
    if not (d / "merged_integration.json").exists():
        return None
    return review_app.ReviewServer(d)


# ===========================================================================
# FastAPI 앱
# ===========================================================================
app = FastAPI(title="PatentAgent Gateway", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
def index():
    return _index_html(JM.list(), _list_examples())


@app.post("/jobs")
def create_job(
    example: str = Form(...),
    auto_pages: str = Form(""),
    structure_pages: str = Form(""),
    assay_pages: str = Form(""),
    assay_names: str = Form(""),
    engine: str = Form("molscribe"),
    with_coref: str = Form(""),
):
    # 네트워크 정책상 직접 업로드 불가 → 서버 로컬 examples/ 에서 PDF 선택.
    # 디렉토리 탈출 방지: 파일명만 허용하고 examples 안에 실재하는지 확인.
    name = os.path.basename(example or "")
    src = (EXAMPLES_DIR / name).resolve()
    if EXAMPLES_DIR.resolve() not in src.parents or src.suffix.lower() != ".pdf" or not src.exists():
        raise HTTPException(400, f"예제 PDF를 찾을 수 없습니다: {example}")
    opts = _build_opts(auto_pages, structure_pages, assay_pages, assay_names, engine, with_coref)
    job_id = JM.create(src, name, opts)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.post("/jobs_from_patent")
def create_job_from_patent(
    patent: str = Form(...),
    auto_pages: str = Form(""),
    structure_pages: str = Form(""),
    assay_pages: str = Form(""),
    assay_names: str = Form(""),
    engine: str = Form("molscribe"),
    with_coref: str = Form(""),
):
    # 특허번호/Google Patents 링크/PDF URL → PDF 다운로드 → examples/ 저장 → job 생성
    try:
        pdf_bytes, fname = resolve_patent_pdf(patent)
    except Exception as e:
        return HTMLResponse(_error_html("특허 PDF 가져오기 실패", str(e)), status_code=400)
    if not pdf_bytes or not pdf_bytes[:5].startswith(b"%PDF"):
        return HTMLResponse(_error_html("특허 PDF 가져오기 실패",
                                        "받은 데이터가 PDF 가 아닙니다."), status_code=400)
    # examples/ 에 보존 (다음에 드롭다운에서도 재사용 가능)
    dest = EXAMPLES_DIR / os.path.basename(fname)
    dest.write_bytes(pdf_bytes)
    opts = _build_opts(auto_pages, structure_pages, assay_pages, assay_names, engine, with_coref)
    job_id = JM.create(dest, dest.name, opts)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str):
    meta = JM.get(job_id)
    if not meta:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return _job_html(meta)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    meta = JM.cancel(job_id)
    if not meta:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    meta = JM.get(job_id)
    if not meta:
        raise HTTPException(404, "no such job")
    out = dict(meta)
    out["steps"] = _compute_steps(job_id, meta)
    return out


@app.get("/jobs/{job_id}/log", response_class=Response)
def job_log(job_id: str):
    log = JOBS_DIR / job_id / "run.log"
    if not log.exists():
        return Response("(로그 없음)", media_type="text/plain; charset=utf-8")
    # 마지막 ~200줄만
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = "\n".join(lines[-200:])
    return Response(tail, media_type="text/plain; charset=utf-8")


# --- 검수 라우트 (job 별로 review_app 렌더 재사용, base 프리픽스) ----------
@app.get("/jobs/{job_id}/review", response_class=HTMLResponse)
@app.get("/jobs/{job_id}/review/", response_class=HTMLResponse)
def review_list(job_id: str):
    srv = _review_server(job_id)
    if not srv:
        if not (JOBS_DIR / job_id).is_dir():
            msg = ("이 작업의 데이터가 서버에서 삭제되었습니다. "
                   "홈에서 다시 추출을 시작해 주세요.")
        else:
            msg = "아직 추출이 완료되지 않았거나 결과 파일이 없습니다."
        return HTMLResponse(_error_html("검수 결과 없음", msg), status_code=404)
    return review_app._list_page_html(srv, base=f"/jobs/{job_id}/review", home_url="/")


@app.get("/jobs/{job_id}/review/item/{idx}", response_class=HTMLResponse)
def review_item(job_id: str, idx: int):
    srv = _review_server(job_id)
    if not srv or not (0 <= idx < len(srv.rows)):
        raise HTTPException(404, "항목 없음")
    return review_app._item_page_html(srv, idx, base=f"/jobs/{job_id}/review", home_url="/")


@app.get("/jobs/{job_id}/review/segment/{name}")
def review_segment(job_id: str, name: str):
    srv = _review_server(job_id)
    if not srv:
        raise HTTPException(404, "no job")
    try:
        idx = int(name.split(".")[0])
    except ValueError:
        raise HTTPException(400, "bad index")
    b = review_app._seg_image_bytes(srv, idx)
    if not b:
        raise HTTPException(404, "no image")
    return Response(b, media_type="image/png")


@app.get("/jobs/{job_id}/review/highlight/{name}")
def review_highlight(job_id: str, name: str):
    """페이지 전체 + 구조 박스 미리보기 (추출 영역)."""
    srv = _review_server(job_id)
    if not srv:
        raise HTTPException(404, "no job")
    try:
        idx = int(name.split(".")[0])
    except ValueError:
        raise HTTPException(400, "bad index")
    b = review_app._highlight_image_bytes(srv, idx)
    if not b:
        raise HTTPException(404, "no image")
    return Response(b, media_type="image/png")


@app.post("/jobs/{job_id}/review/api/item/{idx}")
async def review_save(job_id: str, idx: int, payload: dict):
    srv = _review_server(job_id)
    if not srv:
        raise HTTPException(404, "no job")
    rec = srv.save_item(idx, payload)
    return rec


@app.get("/jobs/{job_id}/review/export")
def review_export(job_id: str):
    srv = _review_server(job_id)
    if not srv:
        raise HTTPException(404, "no job")
    return JSONResponse(srv.export())


# --- Ketcher 정적 자산 (job 별 경로에서 동일 빌드 서빙) --------------------
_KETCHER_CTYPE = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript",
    ".mjs": "text/javascript", ".css": "text/css", ".wasm": "application/wasm",
    ".json": "application/json", ".map": "application/json",
    ".png": "image/png", ".svg": "image/svg+xml",
    ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf",
}


@app.get("/jobs/{job_id}/review/ketcher/{path:path}")
def review_ketcher(job_id: str, path: str):
    base = review_app.KETCHER_DIR.resolve()
    rel = path or "index.html"
    target = (base / rel).resolve()
    if base not in target.parents and target != base:
        raise HTTPException(403, "forbidden")
    if not target.exists() or target.is_dir():
        raise HTTPException(404, "not found")
    ctype = _KETCHER_CTYPE.get(target.suffix.lower(), "application/octet-stream")
    return Response(target.read_bytes(), media_type=ctype)


# ===========================================================================
# HTML 페이지 (업로드 / 작업상태)
# ===========================================================================
def _fmt_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _index_html(jobs, examples):
    rows = []
    badge = {"queued": "#777", "running": "#1565c0", "done": "#2e7d32",
             "error": "#c62828", "interrupted": "#e65100", "cancelled": "#9c27b0"}
    label = {"queued": "대기", "running": "추출중", "done": "완료",
             "error": "실패", "interrupted": "중단됨", "cancelled": "취소됨"}
    for m in jobs:
        st = m["status"]
        nc = m.get("n_compounds")
        link = (f'<a href="/jobs/{m["job_id"]}/review">검수</a>'
                if st == "done" else
                f'<a href="/jobs/{m["job_id"]}">상태</a>')
        rows.append(f"""<tr>
          <td><code>{m['job_id']}</code></td>
          <td>{review_app.html.escape(m['filename'])}</td>
          <td><span class="b" style="background:{badge.get(st,'#777')}">{label.get(st,st)}</span></td>
          <td>{nc if nc is not None else '—'}</td>
          <td style="font-size:12px;color:#888">{m['created_at']}</td>
          <td>{link}</td>
        </tr>""")
    job_table = ("<table><tr><th>작업ID</th><th>파일</th><th>상태</th><th>화합물</th>"
                 "<th>생성</th><th></th></tr>" + "".join(rows) + "</table>"
                 if rows else '<p style="color:#888">아직 생성된 작업이 없습니다.</p>')

    # examples 디렉토리 PDF 선택 옵션
    if examples:
        opts_html = "".join(
            f'<option value="{review_app.html.escape(e["name"])}">'
            f'{review_app.html.escape(e["name"])} ({_fmt_size(e["size"])})</option>'
            for e in examples)
        example_selector = (f'<select name="example" id="example" required>'
                            f'<option value="" disabled selected>— 예제 PDF 선택 —</option>'
                            f'{opts_html}</select>')
        submit_disabled = ""
    else:
        example_selector = ('<div style="color:#c62828">examples/ 디렉토리에 PDF가 없습니다. '
                            f'서버의 <code>{EXAMPLES_DIR}</code> 에 PDF를 넣어주세요.</div>')
        submit_disabled = "disabled"

    # 두 입력방식(예제선택/특허링크)이 공유하는 옵션 블록 (suffix 로 id 구분)
    def opts_block(sfx):
        return f"""
    <label><input type="checkbox" name="auto_pages" id="auto{sfx}" checked onchange="toggleOpts('{sfx}')">
      페이지 자동 탐지 (구조/활성 페이지를 자동으로 찾음)</label>
    <div class="opts" id="manualOpts{sfx}">
      <label>구조 페이지 <span class="hint">(예: 242-243)</span></label>
      <input type="text" name="structure_pages" placeholder="242-243">
      <label>활성 페이지 <span class="hint">(예: 269-272)</span></label>
      <input type="text" name="assay_pages" placeholder="269-272">
    </div>
    <label>어세이명 <span class="hint">(비워두면 PDF에서 자동 추출 — IC50/EC50/Ki/TR-FRET 등)</span></label>
    <input type="text" name="assay_names" placeholder="자동 추출 (필요 시 직접 지정)">
    <label><input type="checkbox" name="with_coref"> MolCoref 교차검증 (구조↔ID 연결, 느림)</label>"""

    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>PatentAgent — 특허 화합물 추출</title>
<style>
 :root{{--bg:#f6f7f9;--panel:#fff;--line:#d9dee7;--text:#1f2933;--muted:#667085;--blue:#155eef}}
 *{{box-sizing:border-box}}
 body{{font-family:system-ui,sans-serif;margin:0;background:var(--bg);color:var(--text)}}
 .page{{max-width:1180px;margin:0 auto;padding:24px 20px 36px}}
 .head{{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:18px}}
 h1{{font-size:22px;line-height:1.2;margin:0}} h2{{font-size:15px;margin:0 0 12px;color:#344054}}
 .subtitle{{font-size:13px;color:var(--muted);margin-top:6px}}
 .layout{{display:grid;grid-template-columns:minmax(360px,460px) minmax(0,1fr);gap:16px;align-items:start}}
 .card,.jobs{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:18px;box-shadow:0 1px 2px rgba(16,24,40,.04)}}
 label{{display:block;margin:10px 0 4px;font-size:13px;font-weight:650;color:#344054}}
 input[type=text]{{width:100%;padding:9px 10px;border:1px solid #ccd2dc;border-radius:6px;font-size:14px;background:#fff}}
 select{{width:100%;padding:9px 10px;border:1px solid #ccd2dc;border-radius:6px;font-size:14px;background:#fff}}
 .opts{{display:none;border-left:3px solid #d0d5dd;padding-left:14px;margin-top:8px}}
 .opts.show{{display:block}}
 button{{margin-top:16px;width:100%;padding:10px 18px;background:var(--blue);color:#fff;border:0;border-radius:6px;font-size:15px;font-weight:650;cursor:pointer}}
 button:disabled{{background:#aaa;cursor:not-allowed}}
 .hint{{font-size:12px;color:var(--muted);margin-top:2px}}
 .tabs{{display:flex;gap:6px;margin-bottom:14px;border-bottom:1px solid var(--line)}}
 .tab{{padding:9px 14px;cursor:pointer;border:1px solid transparent;border-bottom:none;
       border-radius:6px 6px 0 0;font-size:13px;color:#667085}}
 .tab.active{{background:#fff;border-color:var(--line);color:var(--blue);font-weight:700;margin-bottom:-1px}}
 .pane{{display:none}} .pane.active{{display:block}}
 .tablewrap{{overflow:auto;border:1px solid var(--line);border-radius:8px;background:#fff}}
 table{{border-collapse:separate;border-spacing:0;width:100%;min-width:680px;background:#fff}}
 th,td{{border-bottom:1px solid #eaecf0;padding:8px 10px;font-size:13px;text-align:left;vertical-align:middle}}
 th{{background:#f8fafc;color:#475467;font-weight:650}} .b{{color:#fff;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:650}}
 td a{{color:var(--blue);font-weight:650;text-decoration:none}}
 code{{background:#f0f0f0;padding:1px 4px;border-radius:3px;font-size:12px}}
 @media (max-width:900px){{
   .page{{padding:16px 12px}}
   .head{{display:block}}
   .layout{{grid-template-columns:1fr}}
 }}
</style></head><body>
<div class="page">
<div class="head">
  <div>
    <h1>PatentAgent</h1>
    <div class="subtitle">특허 PDF에서 화합물 구조, Markush 정보, assay 활성값을 추출합니다.</div>
  </div>
</div>
<div class="layout">
<div class="card">
  <div class="tabs">
    <div class="tab active" id="tabEx" onclick="showTab('Ex')">예제 PDF</div>
    <div class="tab" id="tabPat" onclick="showTab('Pat')">특허 링크/번호</div>
  </div>

  <div class="pane active" id="paneEx">
    <form action="/jobs" method="post">
      <label>특허 PDF 선택 <span class="hint">(서버의 examples/ 디렉토리)</span></label>
      {example_selector}
      {opts_block("Ex")}
      <button type="submit" {submit_disabled}>추출 시작 ▶</button>
    </form>
  </div>

  <div class="pane" id="panePat">
    <form action="/jobs_from_patent" method="post" onsubmit="return onPatSubmit()">
      <label>특허번호 또는 링크</label>
      <input type="text" name="patent" id="patentInput" required
             placeholder="US10000000B2  또는  https://patents.google.com/patent/US10000000B2/en  또는  PDF URL">
      <div class="hint">특허번호 · Google Patents 링크 · PDF 직접 URL 모두 지원. PDF를 받아 examples/ 에 저장 후 처리합니다.</div>
      {opts_block("Pat")}
      <button type="submit" id="patBtn">PDF 가져와 추출 ▶</button>
    </form>
  </div>
</div>

<div class="jobs">
  <h2>작업 목록</h2>
  <div class="tablewrap">{job_table}</div>
</div>
</div>
</div>

<script>
 function toggleOpts(sfx){{
   document.getElementById('manualOpts'+sfx).classList.toggle('show',
     !document.getElementById('auto'+sfx).checked);
 }}
 toggleOpts('Ex'); toggleOpts('Pat');
 function showTab(t){{
   for(const x of ['Ex','Pat']){{
     document.getElementById('tab'+x).classList.toggle('active', x===t);
     document.getElementById('pane'+x).classList.toggle('active', x===t);
   }}
 }}
 function onPatSubmit(){{
   const b=document.getElementById('patBtn');
   b.disabled=true; b.textContent='PDF 다운로드 중… (잠시 기다려주세요)';
   return true;
 }}
</script>
</body></html>"""


def _job_html(meta):
    st = meta["status"]
    job_id = meta["job_id"]
    done = st == "done"
    poll = st in ("queued", "running")
    cancellable = st in ("queued", "running")
    badge = {"queued": "#777", "running": "#1565c0", "done": "#2e7d32",
             "error": "#c62828", "interrupted": "#e65100", "cancelled": "#9c27b0"}.get(st, "#777")
    label = {"queued": "대기 중", "running": "추출 진행 중…", "done": "완료",
             "error": "실패", "interrupted": "중단됨", "cancelled": "취소됨"}.get(st, st)
    opts = meta.get("options", {})
    opts_str = ("자동 페이지 탐지" if opts.get("auto_pages")
                else f"구조 {opts.get('structure_pages') or '—'} / 활성 {opts.get('assay_pages') or '—'}")
    err = (f'<p style="color:#c62828">⚠️ {review_app.html.escape(str(meta.get("error")))}</p>'
           if meta.get("error") else "")
    action = (f'<a class="btn" href="/jobs/{job_id}/review">→ 검수 UI 열기 ({meta.get("n_compounds")}개 화합물)</a>'
              if done else "")
    cancel_btn = (f'''<form action="/jobs/{job_id}/cancel" method="post" style="display:inline"
        onsubmit="return confirm('이 작업을 중단하시겠습니까? 진행 중인 추출이 종료됩니다.')">
        <button class="cancel" type="submit">■ 작업 중단</button></form>'''
        if cancellable else "")

    # 초기 단계 체크리스트 (서버사이드 렌더 → 폴링으로 갱신)
    steps = _compute_steps(job_id, meta)
    steps_html = _steps_html(steps)

    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>작업 {job_id}</title>
<style>
 :root{{--bg:#f6f7f9;--panel:#fff;--line:#d9dee7;--text:#1f2933;--muted:#667085;--blue:#155eef;--green:#2e7d32}}
 *{{box-sizing:border-box}}
 body{{font-family:system-ui,sans-serif;margin:0;background:var(--bg);color:var(--text)}}
 .page{{max-width:960px;margin:0 auto;padding:24px 20px 36px}}
 .toplink{{display:inline-flex;align-items:center;margin-bottom:14px;color:var(--blue);text-decoration:none;font-size:13px;font-weight:650}}
 a{{color:var(--blue)}} .b{{color:#fff;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:650}}
 .card{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:18px;margin-bottom:14px;box-shadow:0 1px 2px rgba(16,24,40,.04)}}
 pre{{background:#111827;color:#e5e7eb;padding:12px;border-radius:6px;overflow:auto;max-height:340px;font-size:12px;line-height:1.45}}
 a.btn{{display:inline-block;padding:9px 16px;background:var(--green);color:#fff;text-decoration:none;border-radius:6px;font-size:14px;font-weight:650}}
 button.cancel{{padding:8px 14px;background:#b42318;color:#fff;border:0;border-radius:6px;font-size:13px;cursor:pointer}}
 .meta{{font-size:13px;color:var(--muted);line-height:1.55}}
 ul.steps{{list-style:none;padding:0;margin:4px 0}}
 ul.steps li{{display:flex;align-items:center;gap:11px;padding:9px 4px;font-size:15px;
              border-bottom:1px solid #f0f0f0}}
 ul.steps li:last-child{{border-bottom:none}}
 .ic{{width:22px;height:22px;flex:0 0 22px;display:flex;align-items:center;justify-content:center}}
 li.done .tx{{color:#1b5e20;font-weight:600}}
 li.done .ic{{color:#2e7d32;font-size:18px}}
 li.active .tx{{color:#0d47a1;font-weight:600}}
 li.pending .tx{{color:#bbb}}
 li.pending .ic{{color:#ddd}}
 .spin{{width:17px;height:17px;border:3px solid #bbdefb;border-top-color:#1565c0;
        border-radius:50%;animation:sp .8s linear infinite}}
 @keyframes sp{{to{{transform:rotate(360deg)}}}}
 details{{margin-top:6px}} summary{{cursor:pointer;color:#1565c0;font-size:13px}}
</style></head><body>
<div class="page">
<a class="toplink" href="/">홈</a>
<div class="card">
  <h1 style="font-size:20px">작업 <code>{job_id}</code> <span class="b" id="badge" style="background:{badge}">{label}</span></h1>
  <p class="meta">파일: {review_app.html.escape(meta['filename'])}<br>
     옵션: {opts_str} · 어세이: {review_app.html.escape(opts.get('assay_names') or '자동 추출')}<br>
     생성: {meta['created_at']}{' · 완료: ' + meta['finished_at'] if meta.get('finished_at') else ''}</p>
  <div id="err">{err}</div>
  <div id="action">{action} {cancel_btn}</div>
</div>
<div class="card">
  <h2 style="font-size:15px">진행 상황</h2>
  <ul class="steps" id="steps">{steps_html}</ul>
  <details>
    <summary>상세 실행 로그 보기</summary>
    <pre id="log">로딩 중…</pre>
  </details>
</div>
</div>
<script>
 const JOB = "{job_id}";
 let polling = {str(poll).lower()};
 function iconFor(state){{
   if(state==='done') return '<span class="ic">✓</span>';
   if(state==='active') return '<span class="ic"><span class="spin"></span></span>';
   return '<span class="ic">○</span>';
 }}
 function renderSteps(steps){{
   document.getElementById('steps').innerHTML = steps.map(s =>
     `<li class="${{s.state}}">${{iconFor(s.state)}}<span class="tx">${{s.name}}</span></li>`
   ).join('');
 }}
 async function refresh(){{
   try{{
     const meta = await (await fetch(`/api/jobs/${{JOB}}`)).json();
     if(meta.steps) renderSteps(meta.steps);
     fetch(`/jobs/${{JOB}}/log`).then(r=>r.text()).then(t=>{{
       const el=document.getElementById('log'); if(el) el.textContent=t||'(로그 없음)';
     }});
     if(polling && meta.status!=='queued' && meta.status!=='running'){{
       location.reload();  // 완료/실패/취소 시 페이지 갱신 (검수 버튼·뱃지 반영)
       return;
     }}
   }}catch(e){{}}
   if(polling) setTimeout(refresh, 3000);
 }}
 refresh();
</script>
</body></html>"""


def _steps_html(steps):
    """단계 리스트 → 초기 <li> HTML (JS renderSteps 와 동일 마크업)."""
    def icon(state):
        if state == "done":
            return '<span class="ic">✓</span>'
        if state == "active":
            return '<span class="ic"><span class="spin"></span></span>'
        return '<span class="ic">○</span>'
    return "".join(
        f'<li class="{s["state"]}">{icon(s["state"])}'
        f'<span class="tx">{review_app.html.escape(s["name"])}</span></li>'
        for s in steps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PatentAgent 웹 게이트웨이")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    print(f"PatentAgent 게이트웨이: http://localhost:{args.port}/")
    print(f"  작업 디렉토리: {JOBS_DIR}")
    print(f"  MarkushGrapher 서비스: {MG_SERVICE_URL}")
    print("  전제: LiteLLM 4000 / PaddleOCR 8010 / markush_service 8100 가동")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
