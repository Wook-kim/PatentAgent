# PatentAgent — 화합물 구조 + SAR/활성값 추출 에이전트 설계서

> 논문·특허 문서를 입력하면 화합물 구조(SMILES/CXSMILES)와 SAR/활성값을
> 자동으로 추출·연결·검증하여 구조화된 데이터셋으로 산출하는 에이전트.

작성일: 2026-06-24

---

## 1. 목표와 범위

### 목표
- **입력**: 논문 또는 특허 PDF (이미지 스캔본 포함)
- **출력**: 화합물별 `{ID, SMILES/CXSMILES, 치환기 정의, 활성값(IC50/EC50/Ki...), 어세이 메타데이터, 출처(page/figure/table)}` 통합 테이블 (CSV/JSON/Parquet)

### 핵심 차별점
1. **특허 Markush(일반식, R-group) 지원** — MarkushGrapher 통합 (경쟁 도구 대부분 미지원)
2. **구조 ↔ 화합물 ID ↔ 활성값 자동 연결** — coreference 기반
3. **에이전트적 검증 루프** — SMILES 유효성, 활성값 정규화, 신뢰도 기반 재시도

### 비범위 (1차 버전 제외)
- 활성값으로부터 신규 활성 예측 (이건 별도 CADD MCP 활용 가능)
- 합성 경로/수율 추출 (OpenChemIE가 일부 가능하나 후순위)

---

## 2. 자산 매핑 — 무엇을 재사용하는가

| 기능 | 채택 컴포넌트 | 출처 프로젝트 | 비고 |
|------|--------------|--------------|------|
| PDF → 페이지 이미지/텍스트 | `utils/pdf_utils.py` + Docling | BioChemInsight / MARCUS | |
| 구조 위치 검출(세그멘테이션) | DECIMER Segmentation | BioChemInsight | |
| 일반 구조 → SMILES | MolScribe / MolNexTR | BioChemInsight | 엔진 선택 가능 |
| **Markush 구조 → CXSMILES + 치환기 테이블** | **MarkushGrapher 2.0** | MarkushGrapher | **특허 전용 경로** |
| 구조 ↔ 텍스트 ID 연결 | MolCoref | OpenChemIE | coreference |
| 화합물 ID 인식(보조) | Vision LLM | BioChemInsight | coref 실패 시 fallback |
| OCR(표/텍스트 → 마크다운) | PaddleOCR / DotsOCR | BioChemInsight | |
| **활성값 추출(표/텍스트 → {ID:값})** | **`activity_parser.py` (LLM)** | BioChemInsight | **이미 구현** |
| 표 구조 추출 + 측정 컬럼 태깅 | `tableextractor.py` | OpenChemIE | SAR 테이블 보강 |
| 구조 시각화/검증 UI | CDK/RDKit + Ketcher | MARCUS | 사람 검수 단계 |

**결론**: BioChemInsight를 오케스트레이션 뼈대로, MarkushGrapher(특허)·OpenChemIE(연결/표)·MARCUS(검수 UI)를 모듈로 흡수.

---

## 3. 전체 아키텍처

```
                        ┌──────────────────────────────────────┐
                        │   Orchestrator Agent (LLM tool-use)   │
                        │   - 문서 라우팅 / 단계 호출 / 검증·재시도 │
                        └──────────────────────────────────────┘
                                          │
   ┌──────────────┬───────────────┬───────┴────────┬──────────────────┐
   ▼              ▼               ▼                ▼                  ▼
[0. Ingest]  [1. Layout]    [2. Structure]   [3. Activity]      [4. Link & Verify]
 PDF 파싱     페이지/블록     구조 추출         SAR/활성 추출        병합·정규화·검증
 분류         분류(표/그림/    ├ 일반: MolScribe  ├ OCR→마크다운       ├ ID 조인
 (논문/특허)   텍스트)         └ Markush:         └ LLM 추출           ├ RDKit 검증
                              MarkushGrapher     {ID: 값}            ├ 단위 정규화
                                                                    └ coref 보정
                                          │
                                          ▼
                              [5. Output] CSV / JSON / Parquet
                              + (옵션) MARCUS 검수 UI
```

---

## 4. 단계별 상세 설계

### Stage 0 — Ingest & 문서 분류
- PDF 메타데이터/텍스트 밀도로 **논문 vs 특허** 판별 (특허: 청구항/실시예 패턴, 균일 레이아웃)
- 산출: 문서 타입, 페이지 수, 페이지별 이미지(2x) + Docling 텍스트
- **라우팅 결정**: 특허이면 Markush 경로 활성화

### Stage 1 — Layout 분석
- 페이지를 블록 단위로 분류: `구조 그림 / 표 / 본문 텍스트`
- OpenChemIE `extract_figures_from_pdf`, `extract_tables_from_pdf` 활용
- 산출: 블록별 bbox + 타입 → Stage 2/3로 분배

### Stage 2 — 구조 추출 (분기)
```
for each figure block:
    if 문서가 특허 and Markush 패턴 감지(R-group 라벨/치환기 테이블 인접):
        → MarkushGrapher: 이미지 → CXSMILES + 치환기 테이블
    else:
        → DECIMER 세그멘테이션 → MolScribe/MolNexTR → SMILES (+molblock)
```
- 산출: `{structure_id, smiles|cxsmiles, substituents?, page, bbox, image_path}`
- **검증**: RDKit 파싱 성공 여부 → 실패 시 다른 엔진으로 재시도(에이전트 루프)

### Stage 3 — SAR/활성값 추출
- 표/텍스트 블록 → OCR(PaddleOCR) → 마크다운
- BioChemInsight `content_to_dict()` (LLM)로 `{화합물ID: 활성값}` 추출
- **보강**: 어세이 메타데이터 추출 (LLM 프롬프트 확장)
  - 어세이 종류(IC50/EC50/Ki/%inhibition), 타겟/세포주/종, readout, 단위, 조건
- 산출: `{compound_id, assay_name, value_raw, value_norm, unit, metadata}`

### Stage 4 — 연결 & 검증 (에이전트 핵심)
1. **ID 연결**: 구조의 화합물 ID ↔ 활성 테이블의 행 ID 조인
   - 1차: OpenChemIE MolCoref (그림 라벨 ↔ 구조)
   - 2차: BioChemInsight Vision LLM (red-box ID 인식)
   - 3차: 문자열/Levenshtein 매칭 (`Compound 12` ≈ `12`)
2. **SMILES 검증**: RDKit canonicalize, 무효 구조 플래그
3. **활성값 정규화**: `<0.1`, `1.2×10³`, 단위(nM/μM) → 표준 단위 + 부등호 분리
4. **신뢰도 스코어링**: 각 필드에 confidence → 임계값 미만은 검수 큐로
5. **재시도 루프**: 연결 실패/저신뢰 항목은 다른 방법·다른 페이지 범위로 재추출

### Stage 5 — 출력 & 검수
- `merged.csv` / `compounds.json` / Parquet
- 컬럼: `compound_id, smiles, cxsmiles, substituents, assay, value, unit, operator, target, source_page, confidence, needs_review`
- (옵션) MARCUS 웹 UI로 저신뢰 항목 사람 검수 + Ketcher 구조 수정

---

## 5. 에이전트 설계 (Orchestrator)

### 도구(Tool) 정의 — LLM이 호출하는 함수
```
- classify_document(pdf)            → {type, page_map}
- detect_layout(pages)              → blocks[]
- extract_structure(image, mode)    → {smiles|cxsmiles, substituents}
- extract_activity(table_md, ids)   → {id: value, metadata}
- resolve_coref(figure)             → {structure↔label}
- validate_smiles(smiles)           → {valid, canonical}
- normalize_value(raw, unit)        → {value, unit, operator}
- merge_and_score(structs, acts)    → rows[] with confidence
```

### 제어 흐름
- **결정론적 골격 + LLM 판단 지점**: 파이프라인 순서는 코드로 고정,
  분기(Markush 여부)·재시도·저신뢰 처리만 LLM이 판단
- **검증 우선**: 각 추출 결과는 다음 단계 진입 전 검증 통과 필요
- 구현: LangGraph 또는 단순 상태머신 + 함수 호출 (초기엔 후자 권장)

---

## 6. 구현 로드맵

> 진행 현황은 [STATUS.md](./STATUS.md) 참조. 아래 로드맵 항목 대부분 2026-06-24~25 완료.

### Phase 1 — 베이스라인 ✅
- [x] BioChemInsight 단독 동작 확인 (2026-06-24)
- [x] 출력 스키마 통일 (JSON/CSV/Parquet, `write_csv`/`_write_parquet`)
- [x] RDKit SMILES 검증 + 활성값 정규화 모듈 (`enrich_with_rdkit`/`_normalize_value`)

### Phase 2 — 특허 Markush 통합 ✅
- [x] MarkushGrapher 추론 동작 확인 (2026-06-24, vllm 가속)
- [x] MarkushGrapher 마이크로서비스 (`markush_service.py`, REST 8100)
- [x] Markush 감지 휴리스틱 (`_looks_markush`: R-group/SGroup/와일드카드)
- [x] CXSMILES(치환기 포함)를 출력 스키마에 통합

### Phase 3 — 연결 정확도 향상 ✅(부분)
- [x] OpenChemIE MolCoref 통합 (`molcoref_helper.py`, `--with-coref`)
- [x] ID 매칭 fallback (`_normalize_compound_id` + coref 교차검증)
- [ ] 어세이 메타데이터 추출 프롬프트 확장 (향후)

### Phase 4 — 에이전트화 & 검수 ✅(부분)
- [x] 결정론적 오케스트레이터 + 신뢰도 판정 (`integrate_prototype.py`, `confidence`)
- [x] 검수 워크플로 (`review_app.py`, 저신뢰 우선 + 승인/반려 UI)
- [x] 자동 페이지 탐지 (`autodetect_pages.py`, `--auto-pages`)
- [ ] 배치 처리 + 평가셋 정확도 측정 / 능동학습 루프 (향후)

---

## 7. 주요 리스크 & 대응

| 리스크 | 영향 | 대응 |
|--------|------|------|
| GPU 메모리 부족 (모델 다수 동시 로드) | 높음 | 단계별 모델 로드/언로드, 서비스 분리 |
| 환경 충돌 (MarkushGrapher는 transformers 포크/별도 env) | 높음 | MarkushGrapher를 **별도 마이크로서비스**로 격리, REST 호출 |
| OCR 품질 → 활성값 오류 전파 | 중간 | 다중 OCR 비교, LLM 표 재구성, 신뢰도 플래그 |
| 구조 ↔ 활성 ID 불일치 | 중간 | 3단계 매칭 fallback + 사람 검수 |
| 활성값 단위/엔드포인트 비표준 | 중간 | 정규화 사전 + LLM 보조 |
| LLM API 비용/레이트리밋 | 낮음 | 청킹·캐싱, 로컬 모델 fallback |

---

## 8. 확정된 요구사항 (2026-06-24)

| 항목 | 결정 | 설계 영향 |
|------|------|----------|
| **주 타깃** | **특허 위주** | **MarkushGrapher 통합 = 최우선**. Markush 경로가 1급 기능. 청구항/실시예 파싱 강화 |
| **배포 형태** | **웹 서비스 + 검수 UI** | **MARCUS 프론트엔드(Vue3)/FastAPI를 셸로 채택**. 업로드→추출→검수→수정→export 워크플로 |
| **활성값 처리** | **표준 단위 변환까지** | 정규화 모듈 필수: 단위 통일(nM/μM), 부등호 분리, 엔드포인트 정규화. `value_raw` + `value_norm` 둘 다 보존 |

### 위 결정에 따른 우선순위 재조정

1. **특허 Markush가 1급** → 로드맵 Phase 2(MarkushGrapher)를 Phase 1과 병행/조기 착수
2. **웹 서비스가 베이스** → BioChemInsight 파이프라인을 백엔드 엔진으로, MARCUS UI를 프론트로 결합
   - BioChemInsight도 자체 React 프론트 있음 → **MARCUS UI vs BioChemInsight UI 중 택1** 또는 통합 필요 (아래 결정 필요)
3. **단위 변환 필수** → Stage 3/4에 정규화 사전 + RDKit/직접 변환 로직 추가, 검수 UI에 원문·정규화 값 병기

### 남은 소소한 결정 (진행하며 확정 가능)
- 프론트엔드: MARCUS(Vue3) 채택 vs BioChemInsight(React) 채택 vs 신규 → **MARCUS 권장** (검수/세션/Ketcher 편집기 이미 보유)
- LLM 백엔드: BioChemInsight의 Bedrock(Claude)+LiteLLM 유지 권장
```
