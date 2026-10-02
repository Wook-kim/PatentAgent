# PatentAgent

특허 PDF에서 화합물 구조·Markush 구조·활성값을 추출하고 Ketcher로 검토하는 시스템입니다.
현재 런타임은 **한 Python 패키지·한 설치 환경**으로 구성됩니다.
Linux NVIDIA GPU 서버를 대상으로 하며, LLM은 설정한 OpenAI 호환 멀티모달 API에 직접 연결합니다.

## 실행 구조

```text
patentagent serve → 작업 큐 → patentagent extract
                                │
                 PDF 렌더링·페이지 분석 (PyMuPDF + LLM)
                                │
                 구조 영역 검출 (DECIMER, CPU)
                                │
                 구조 인식 (내장 OCSRGlyph, GPU)
                                │
                 화합물 ID·활성표 전사 (설정한 LLM API)
                                │
                 Markush 인식 (내장 MarkushGlyph + Qwen3.5-2B, GPU)
                                │
                 ID 연결·RDKit 검증·JSON/CSV/Parquet
                                │
                 검토 화면·내장 Ketcher·수정본 내보내기
```

웹 서버는 GPU 경합을 피하도록 작업을 하나씩 실행합니다. 작업별 프로세스는 같은 패키지와
Python 환경을 사용하며, 중단과 메모리 반환을 위한 격리 단위입니다.
GPU 모델은 순차적으로 로드·해제합니다. Glyph 추론 코드도 패키지 안에 포함되어 있으며
별도 프로젝트 체크아웃·가상환경·추론 서버가 필요하지 않습니다.
Glyph는 잘린 구조 이미지를 인식하므로 PDF의 구조 영역 검출은 DECIMER가 담당합니다.

일반 라이브러리(PyTorch, Transformers, RDKit 등), 모델 가중치, LLM API는 필요합니다.
이 변경은 모델을 새로 학습하거나 기존 코드의 라이선스를 없애는 작업이 아닙니다.

## 설치 및 설정

Python **3.11**, `uv`, NVIDIA 드라이버가 있는 Linux 서버에서:

```bash
uv sync --locked --python 3.11 --extra inference
cp .env.example .env
```

`.env`의 다음 값을 설정하세요.

```dotenv
PATENTAGENT_LLM_BASE_URL=https://your-endpoint/v1
PATENTAGENT_LLM_MODEL=your-vision-model
PATENTAGENT_LLM_API_KEY=your-key
PATENTAGENT_DEVICE=cuda:1
```

API는 `/chat/completions`의 `image_url` 입력과 JSON 응답을 지원해야 합니다.
인증 없는 자체 서버는 키를 비워둘 수 있습니다. 키는 실행 명령이나 결과 manifest에 기록하지 않습니다.
GPU 번호는 실제 장치에 맞춰 지정하세요. GPU가 하나라면 `cuda:0`을 사용합니다.
`.env`는 **명령을 실행한 디렉토리**에서 읽으며 환경변수가 우선합니다.

Linux의 PyTorch 2.7.1은 CUDA 12.6 휠로 고정되어 있으며 **glibc 2.28 이상**이 필요합니다.
기존 드라이버 560 환경에서 사용할 구성입니다. TITAN RTX에서는 기본 `auto` 정밀도가
`float16`을 선택합니다. `bfloat16`은 지원하는 장치에서만 명시적으로 사용할 수 있습니다.
실제 서버에서 아래 점검과 추출을 실행해 확인해야 합니다.

```bash
uv run --locked --extra inference patentagent doctor
uv run --locked --extra inference patentagent serve --host 0.0.0.0 --port 8000
```

브라우저에서 서버의 8000번 포트로 접속합니다. `examples/`의 PDF를 선택하거나 특허 URL을
입력할 수 있습니다. 작업·로그·검토 결과는 기본 `data/jobs/`에 저장됩니다.
`doctor`는 설정·설치·장치 접근을 점검합니다. API 호출 성공이나 모델 정확도를 검증하는 명령은 아닙니다.

CLI에서 직접 실행:

```bash
uv run --locked --extra inference patentagent extract examples/patent.pdf \
  --out data/run-001 \
  --structure-pages 242-243 --assay-pages 269-272 --no-auto-pages
```

페이지 번호는 PDF의 **1부터 시작하는 실제 페이지 순서**입니다.
페이지를 생략하면 전체 PDF를 LLM으로 분류하므로 페이지 수에 비례해 API 호출이 발생합니다.
`--auto-pages` 상태에서도 명시한 페이지 범위는 유지합니다.
일반 구조만 필요하면 `--no-markush`를 사용할 수 있습니다.
완료된 출력 디렉토리는 덮어쓰지 않으므로 재실행 시 새 `--out`을 지정합니다.

## 가중치와 포함된 코드

첫 추론 시 아래 가중치를 내려받습니다. 대형 가중치는 저장소에 포함하지 않습니다.

| 역할 | 기본 가중치 | 저장 위치 |
|---|---|---|
| 구조 검출 | DECIMER `mask_rcnn_molecule.h5` | DECIMER 설치 디렉토리 (라이브러리 기본 동작) |
| 구조 인식 | `EdisonScientific/OCSRGlyph`, `model.pth` | `models/hub` |
| Markush 어댑터 | `EdisonScientific/MarkushGlyph` | `models/hub` |
| Markush 기반 모델 | `Qwen/Qwen3.5-2B-Base` | `models/hub` |

기본 가중치 리비전은 `.env.example`과 `Settings`에서 고정합니다.
다른 Hugging Face 모델을 지정할 때에는 그 모델에 해당하는 40자리 커밋 리비전도 지정합니다.
로컬 가중치는 `PATENTAGENT_OCSR_CHECKPOINT`에 `.pth` 파일 또는 이를 포함한 폴더,
`PATENTAGENT_MARKUSH_CHECKPOINT`에 LoRA 어댑터 폴더,
`PATENTAGENT_MARKUSH_BASE_MODEL`에 Qwen 모델 폴더를 지정합니다.
로컬 모델에는 공개 리비전을 붙이지 않고 실제 파일의 SHA256을 기록합니다.

추론 코드는 `patentagent/_vendor/glyph/`에 포함했습니다.
원본 저장소·커밋·파일 체크섬·수정 내역은 해당 폴더의 `SOURCES.json`에 있고,
Apache-2.0 라이선스도 함께 배포합니다.
유지보수용 `tools/vendor_glyph.py`는 스냅샷을 재생성하며 설치·실행 중에는 호출하지 않습니다.
학습·평가 도구를 제외해 `datasets`와 별도 Markush scorer가 필요하지 않습니다.
ChemicalOCR, MolScribe, MarkushGrapher 및 전용 Transformers 포크는 현재 런타임에서 제거했습니다.

기본 배치는 OCSR 8장, Markush 4장입니다. 메모리가 부족하면
`PATENTAGENT_OCSR_BATCH_SIZE`, `PATENTAGENT_MARKUSH_BATCH_SIZE`를 낮추세요.
Markush는 greedy 생성이며 최대 토큰 수는 `PATENTAGENT_MARKUSH_MAX_NEW_TOKENS`로 설정합니다.
출력이 잘리면 원문과 오류를 남겨 검토 대상으로 표시합니다.

## 결과와 검토

- `extraction.json`: 페이지/영역/ID 근거/활성값 원문과 입력 PDF 해시
- `merged_integration.json`: 구조별 연결 결과. 모든 내보내기가 성공한 뒤 생성하는 완료 표시
- `merged_integration.csv`, `.parquet`: 측정값별 행으로 펼친 결과
- `assay_observations.json`: 연결되지 않은 측정값을 포함한 전체 전사 결과
- `pages/`, `regions/`: 원본 페이지·구조 이미지·강조 표시
- `progress.json`, `run_manifest.json`: 단계 상태와 실행 정보
- `review_state.json`: 검토 화면에서 저장한 수정·판정. 추출 원본과 별도로 저장

화합물 ID가 하나의 구조에 대응할 때만 자동 연결합니다. 같은 ID가 여러 구조에 붙어 있거나
ID를 읽을 수 없으면 측정값을 미연결 항목으로 보존합니다. 반복 측정값도 덮어쓰지 않습니다.
모델 간 구조 일치는 정확도를 보증하지 않으며, 근거 이미지와 함께 검토해야 합니다.

새 결과 스키마는 버전 2이며 일반 구조 필드는 `smiles_ocsr`입니다.
`ocsr_model`, `markush_model`, `model_provenance`에 실제 모델과 리비전을 기록합니다.
일반 구조의 후처리 전 텍스트(`ocsr_raw`), Markush 원문(`markush_raw`),
치환기 표 원문(`markush_stable_raw`), `cxsmiles_opt`, 변환 상태와 오류도 보존합니다.
잘못된 화학 토큰을 추정해서 자동 복원하지 않으며, 실패 항목은 낮은 신뢰도로 검토에 남깁니다.
이 정보는 JSON과 CSV/Parquet 모두에 포함되고 검토 화면의 “모델 출력 원문”에서 확인할 수 있습니다.
기존 `smiles_molscribe` 결과는 검토 화면에서 계속 읽을 수 있습니다.

기존 추출 기록의 후처리만 다시 확인하려면:

```bash
uv run --locked patentagent extract examples/patent.pdf \
  --out data/replay-001 --replay data/run-001/extraction.json
```

`--replay`는 API·추론을 실행하지 않습니다. 동일 PDF 해시를 확인하고 기존 이미지 경로를
참조하므로 원래 실행 디렉토리를 유지해야 합니다. 결과 manifest에 `mode: replay`를 기록합니다.

## 이전 구성에서의 변경

| 이전 | 현재 |
|---|---|
| BioChemInsight 외부 pipeline | 내부 PDF/영역/활성값 모듈 |
| Bedrock·LiteLLM 설정 | OpenAI 호환 API URL·모델·키 |
| MolScribe | 내장 OCSRGlyph |
| ChemicalOCR + MarkushGrapher 서비스 또는 shell 실행 | 내장 MarkushGlyph |
| OpenChemIE MolCoref 교차검증 | 페이지 이미지에서 ID 근거 전사 + 유일한 ID 연결 |
| 별도 Ketcher 배포 경로 | Python 패키지에 정적 빌드 포함 |

이전 실행 스크립트는 `legacy/`에 보관했습니다. 루트 `integrate_prototype.py`는 새 CLI로
연결하는 호환 진입점입니다. `--engine`, `--gpu`, `--mg-service-url`, `--with-coref` 등
예전 엔진 전용 옵션은 지원하지 않습니다.
이전의 실험적 화합물명 기반 보정·청구항 전용 후처리·MolCoref와 기능 동등성을 보장하지 않습니다.
구조 페이지를 지정하거나 자동 탐지하면 청구항의 구조 그림도 같은 검출 경로로 처리합니다.
기존 `jobs/` 폴더를 계속 쓰려면 `PATENTAGENT_DATA_DIR=.`로 설정하세요.
과거 설계·실측 결과는 `AGENT_DESIGN.md`, `STATUS.md`에 남아 있습니다.

## 개발 검증

```bash
uv sync --locked --extra inference --group dev
uv run --locked --extra inference pytest -q
uv run --locked ruff check --config pyproject.toml patentagent tests tools/vendor_glyph.py
```

테스트는 합성 PDF와 기록된 응답으로 추출→연결→저장→검토 경로, API 요청 계약, 오류 처리,
ID 중복, 단위 정규화를 검증합니다. Glyph의 원문 보존·치환기 파싱·변환 실패·배치 순서·
모델 출처와 작은 무작위 OCSR 모델의 체크포인트 로딩·CPU 생성 연산도 검증합니다.
2026-10-02 전환 검증에서 테스트 33개와 정적 검사가 통과했습니다.
공개 Qwen 입력 처리기와 작은 무작위 Qwen3.5 모델로 LoRA 저장·로딩·병합 및
이미지 2장 배치 생성도 CPU에서 확인했습니다. 이는 실제 학습 가중치의 정확도 검증은 아닙니다.
**Linux NVIDIA GPU에서 실제 가중치를 로딩한 전체 PDF 추론·정확도·메모리 사용량은 아직 미검증입니다.**
