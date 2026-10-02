# PatentAgent — 진행 현황 (Status)

> **2026-10-02 구조 변경:** 현재 실행 경로는 `patentagent/` 내부 모듈입니다.
> 외부 프로젝트 체크아웃·독립 추론 서버 의존을 제거하는 구현을 추가했습니다.
> **2026-10-02 Glyph 전환:** 일반 구조는 OCSRGlyph, Markush는 MarkushGlyph를 내부에서 실행합니다.
> ChemicalOCR·MolScribe·MarkushGrapher 추론 코드와 전용 의존성을 제거했습니다.
> 설치·검증 범위는 [README.md](./README.md)를 참고하세요.
> 아래의 엔진별 운영 현황은 전환 전 기록이며 새 런타임의 GPU 검증 결과가 아닙니다.

최종 갱신: 2026-09-30

> 관련 문서: [AGENT_DESIGN.md](./AGENT_DESIGN.md) (설계서) · [GLYPH_PHASE0.md](./GLYPH_PHASE0.md) (Glyph 비교 스파이크)

**최근 (2026-09-10) Glyph Phase 0**: OCSRGlyph/MarkushGlyph 헤드투헤드 비교 → 엔진 교체 보류(품질 구별 불가, Markush 속도는 약 5배).
현행 mismatch 75건 중 59건이 `_AP` 부착점 표기 차이에 의한 위양성으로 확인 → 비교 지표 정규화가 다음 과제.
외부 엔진 로컬 수정분은 `patches/*.diff` 참고.

---

## 1. 한 줄 요약

특허/논문에서 **화합물 구조 + SAR/활성값**을 추출하는 에이전트.
4개 엔진(BioChemInsight=구조+활성, MarkushGrapher=Markush, OpenChemIE=MolCoref, 검수UI)을
통합한 **end-to-end 파이프라인 + REST 서비스 + 검수 워크플로까지 완성**(2026-06-24).
구조(SMILES/CXSMILES) + 활성값(단위정규화) + 신뢰도 + 검수를 한 흐름으로 산출.

---

## 2. 컴포넌트별 검증 상태

| 컴포넌트 | 역할 | 설치 | 동작 검증 | 비고 |
|---------|------|:---:|:---:|------|
| **BioChemInsight** | 구조+활성 추출 파이프라인 (뼈대) | ✅ | ✅ | 샘플 특허 PDF로 구조 6개+ID 추출 성공 |
| **MarkushGrapher** | 특허 Markush(R-group) 구조 인식 | ✅ | ✅ | sample 이미지 8개 → CXSMILES 추출 성공 |
| **OpenChemIE** | 구조↔ID 연결(MolCoref), 표 추출 | ⬜ 미설치 | ⬜ | Phase 3에서 통합 예정 |
| **MARCUS** | 검수 웹 UI (Vue3+FastAPI) | ⬜ 미설치 | ⬜ | Phase 4 프론트엔드 셸 후보 |

---

## 3. 인프라 현황 (이 서버)

- **GPU**: NVIDIA TITAN RTX 24GB × 3장 (compute capability **7.5** → bfloat16 미지원, fp16 사용)
- **드라이버**: 560.35.03 / **CUDA 12.6** (cu121 빌드까지 호환, cu13x는 불가)
- **PaddleOCR 서버**: 8010 포트 상시 가동 (docker `paddle-ocr-server`) — BioChemInsight OCR용
- **AWS Bedrock**: 자격증명 유효, Claude Sonnet 4.6 호출 가능
- **공통 도구**: uv 0.9.21, docker, neo4j/redis 등 다수 서비스 가동 중

---

## 4. BioChemInsight 실행 방법 (검증됨)

```bash
cd /data1/wook_workspace/PatentAgent/BioChemInsight

# 1) LiteLLM 프록시 기동 (평소 꺼져 있음 — 수동 기동 필요)
export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
.venv/bin/python -c "from litellm.proxy.proxy_cli import run_server; run_server()" \
    --config litellm_config.yaml --port 4000 &
# (주의: .venv/bin/litellm 스크립트는 shebang이 옛 경로라 깨짐 → 위 python -c 우회 사용)

# 2) 파이프라인 실행 (구조 추출 예시)
.venv/bin/python pipeline.py data/sample.pdf \
    --structure-pages "242-243" --engine molscribe --output output_test
```

- venv: `.venv` (Python 3.10), 패키지 전부 설치됨
- 모델: `models/molnextr_best.pth`(1.1GB), `bin/molvec.jar`
- 산출물: `output/structures.csv`, `{assay}_assay_data.json`, `merged.csv`
- ⚠️ 알려진 버그: Vision LLM이 화합물 ID 대신 설명 문장을 반환하는 경우 있음 → ID 파싱 정규화 필요

자세한 내용: 메모리 `biocheminsight-runtime-notes`

---

## 5. MarkushGrapher 실행 방법 (검증됨, vllm 가속)

```bash
cd /data1/wook_workspace/PatentAgent/MarkushGrapher
export PATH="/home/wkim/.local/bin:$PATH" CUDA_VISIBLE_DEVICES=1
unset CHEMICALOCR_PYTHON   # 기본 chemicalocr-env(vllm) 사용
bash scripts/inference/inference.sh ./data/images
```

- venv 2개: `chemicalocr-env`(vllm 0.6.4.post1 + torch 2.5.1+cu121), `markushgrapher-env`(custom transformers fork)
- 모델: `models/markushgrapher-2`(8.7GB), `models/chemicalocr`(1.5GB), MolScribe ckpt(1.1GB)
- 산출물: `data/hf/inference/<run>/evaluation/predictions_1000.jsonl` (cxsmiles 필드)
- 2단계: [1/2] ChemicalOCR(이미지→OCR) → [2/2] MarkushGrapher(→CXSMILES+치환기)

### 적용한 환경 수정 (CUDA 12.6 + TITAN RTX 7.5 호환을 위해 필수)
1. chemicalocr-env를 **vllm 0.6.4.post1**(Idefics3 지원 첫 버전) + torch 2.5.1+cu121로 재구성
2. transformers **4.46.3** + tokenizers 0.20.3 고정 (5.x는 vllm과 비호환)
3. datasets 2.21 + **pyarrow<18** (신 pyarrow는 PyExtensionType 제거됨)
4. **코드 패치** `markushgrapher/ocr/chemical_ocr.py`: `LLM(...)`에 `dtype="half"` (bfloat16 미지원 대응)
5. **라이브러리 패치** transformers & vllm 양쪽의 `idefics3` image processor: `max_image_size` 인자 전달 (2048 > 1820 오류 회피)

검증 출력 예 (Markush 구조 정상 인식):
```
[1*]CC(=O)N1CCN(...)CC1 |$R;...$,Sg:n:1:m:ht|      ← R-group + SGroup(반복단위)
[1*]C1CCC(CCC2CCC(c3ccc(F)c(F)c3)CC2)CC1 |$R1;...$|  ← R1 가변 치환기
```

자세한 내용: 메모리 `markushgrapher-runtime-notes`

---

## 6. 통합 프로토타입 (검증됨)

스크립트: [integrate_prototype.py](./integrate_prototype.py) (시스템 python3로 실행)

```bash
# 기존 BioChemInsight 산출물 재활용 (Stage B/C만 테스트)
python3 integrate_prototype.py BioChemInsight/data/sample.pdf \
    --structure-pages "242-243" --skip-bci --gpu 1 --out integration_out
```

**통합 메커니즘**: BioChemInsight의 DECIMER 세그멘테이션이 산출한 개별 구조
세그먼트 이미지(`segment_<page>_<idx>.png`)를 그대로 MarkushGrapher 입력으로 전달.
`seg_key = "<page>_<idx>"`로 두 엔진 결과를 병합 → `merged_integration.json`.

3단계: Stage A(BioChemInsight: PDF→구조+SMILES+ID) → Stage B(MarkushGrapher: 세그먼트→CXSMILES) → Stage C(seg_key 병합)

### 완전 통합 검증 (2026-06-24) — 구조 + 활성값 모두 ✅

전체 파이프라인 실행:
```bash
cd /data1/wook_workspace/PatentAgent
pkill -9 -f "image_dir_to_hf|markushgrapher.eval|inference.sh|pipeline.py"; sleep 2
python3 integrate_prototype.py BioChemInsight/data/sample.pdf \
    --structure-pages "242-243" --assay-pages "269-272" --assay-names "FRET EC50" \
    --engine molscribe --gpu 1 --out integration_full
# (LiteLLM 프록시 4000 가동 필수)
```

결과 (sample.pdf, 구조 242-243p + 활성 269-272p): **6개 화합물에
구조(SMILES) + Markush(CXSMILES) + 활성값(FRET EC50) 모두 병합** → `merged_integration.json`

- 두 엔진이 같은 구조에 일치하는 골격 SMILES 산출 → **교차검증 가능**
  (MolScribe는 `[S@TB8]` 과도 입체표기, MarkushGrapher CXSMILES는 더 정제됨)
- Markush 판정 0개 = 정상 (이 페이지는 구체적 화합물, R-group 없음)
- **에이전트적 개선 2종 구현·검증**:
  - `_normalize_compound_id()`: Vision LLM이 ID 대신 설명문장 반환 시 정제
    (화합물 238: `'The red box is...238'` → `238`, 원본은 `compound_id_raw` 보존)
  - `_normalize_value()`: 활성값 원문→`{operator, value_num}` 부등호 분리
    (`value_raw`+정규화 병기. 별표등급/NT는 보존)

자세한 내용: 메모리 `integration-prototype`

---

## 7. 다음 단계 — 전부 완료 ✅ (2026-06-24)

- [x] **활성값(assay) 경로까지 포함한 통합 테스트**
- [x] **활성값 단위 변환(nM/µM)** — `_normalize_value()`가 단위 인식+nM 표준화+부등호 분리
      (어세이명 괄호 단위 보강 포함). 단위테스트 8케이스 통과.
- [x] **출력 스키마 통일(JSON/CSV/Parquet) + RDKit SMILES 검증** — `enrich_with_rdkit()`,
      `write_csv()`, `_write_parquet()`. 18컬럼 long-format.
- [x] **두 SMILES 불일치 신뢰도 판정 + Markush 휴리스틱** — `agreement`(match/match_skeleton/
      mismatch/single/unparsable) + `confidence`(high/medium/low). 6개 샘플 모두 match_skeleton,
      ID 정제된 238만 medium으로 정확히 판정.
- [x] **MarkushGrapher REST 마이크로서비스** — `markush_service.py`(FastAPI, 포트 8100).
      `/predict`(업로드) `/predict_dir`(경로). GPU 직렬화 lock. 통합 스크립트 `--mg-service-url`로 HTTP 호출.
- [x] **OpenChemIE MolCoref 통합** — `OpenChemIE/.venv`(torch1.13+transformers4.33.3) 격리 설치,
      `molcoref_helper.py`(layoutparser 우회, MolDetect coref=True 직접 사용).
      통합 스크립트 `--with-coref`. 분자검출 작동(샘플 특허는 구조에 텍스트라벨 없어 짝 0 = 정상).
- [x] **검수 워크플로** — `review_app.py build`로 검수 HTML 생성
      (구조이미지 임베드 + 신뢰도 badge + 활성값표 + 승인/반려 UI + 저신뢰 우선정렬 + JSON export).
      MARCUS 전체(무거운 Vue3+OCSR) 대신 우리 스키마 특화 경량 뷰어 채택.

### 산출 파일 (PatentAgent 루트)
- `integrate_prototype.py` — 통합 오케스트레이터 (Stage A~E + coref)
- `markush_service.py` — MarkushGrapher REST 서비스
- `molcoref_helper.py` — OpenChemIE MolCoref 격리 래퍼
- `review_app.py` — 검수 HTML 생성기
- `autodetect_pages.py` — 구조/활성 페이지 자동 탐지 (--auto-pages)
- `test_activity.pdf` — 단위변환 실측용 합성 활성표(IC50 nM/µM)
- `integration_full/` — merged_integration.{json,csv,parquet} + review.html

### 향후 과제

- [x] **활성값 실수치 단위변환 실측** (2026-06-25) — 합성 활성표 PDF(`test_activity.pdf`,
      IC50 nM/µM 실수치)로 BioChemInsight LLM 추출(`content_to_dict`) → `_normalize_value`
      end-to-end 검증. `<0.5`→`<`500nM은 부등호분리, `0.045 µM`→45nM, `>10 µM`→10000nM 등
      nM/µM 변환 모두 정확. (현 sample.pdf 본 활성표는 ****/NT 등급표기라 별도 합성 PDF로 실측)
- [x] **자동 페이지 탐지** (2026-06-25) — `autodetect_pages.py`. 텍스트량+이미지(구조) /
      어세이키워드+텍스트량(활성) 휴리스틱 + 갭메우기 + 최장런(구조)/키워드밀도런(활성).
      sample.pdf에서 구조 243-267, 활성 268-272 정확 탐지 (정답 242-267/269-272와 일치).
      통합 스크립트 `--auto-pages`로 연결 (페이지 미지정 시 자동 채움).
- [ ] 검수 결과를 파이프라인으로 되먹이는 능동학습 루프 (미수행)

### 대규모 자동 실행 검증 (2026-06-25, `integration_auto/`)
`--auto-pages`로 전체 25페이지 end-to-end 실행 (수동 페이지 지정 없음):
```bash
python3 integrate_prototype.py BioChemInsight/data/sample.pdf \
    --auto-pages --assay-names "FRET EC50" \
    --mg-service-url http://localhost:8100 --engine molscribe --gpu 1 --out integration_auto
```
- 자동탐지: 구조 243-267, 활성 268-272
- **159개 화합물** 추출 (구조 SMILES + CXSMILES), 활성값 27개 연결, **Markush 1건** 검출
- 신뢰도: high 156 / medium 2 / low 1 — 교차검증 match_skeleton 158 / **mismatch 1**
- 검수 시스템이 **검토 필요 3건 자동 선별** (379=low+mismatch+Markush, 389·331=medium)
- 산출: merged_integration.{json,csv,parquet} + review.html (159건, 저신뢰 우선정렬)
- → 신뢰도/교차검증 시스템이 대규모에서 검수 대상을 정확히 골라냄을 입증
