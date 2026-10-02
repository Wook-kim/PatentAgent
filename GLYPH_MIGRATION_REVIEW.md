# Glyph 교체 타당성 검토

검토일: 2026-10-02

> 후속 구현: 사용자의 교체 지시에 따라 내부 Glyph 엔진으로 전환했다.
> 현재 설치·실행 방법은 [README.md](./README.md)를 따른다.
> 아래는 전환 전 타당성 검토 기록이다. 최종 구현은 평가·학습 도구를 제외했으므로
> 후보 환경의 `datasets`/`pyarrow>=21`은 요구하지 않는다.

## 판단

**OCSRGlyph + MarkushGlyph로 두 구조 인식 엔진을 교체할 수 있다.**
현재의 단일 패키지 구성을 유지하면서, Glyph의 추론에 필요한 코드와 변환기를 내부에 포함하는 방식을 권장한다.
이번 작업은 소스·API·의존성·변환 동작 검토이며 실제 추론 엔진은 변경하지 않았다.

| 현재 구성 | 교체 구성 | 영향 |
|---|---|---|
| DECIMER 영역 검출 | 유지 | Glyph는 잘린 구조 이미지에서 인식하는 모델이다. PDF 내 구조 영역 검출을 대체하지 않는다. |
| MolScribe | OCSRGlyph | 이미지 → SMILES |
| ChemicalOCR + MarkushGrapher-2 | MarkushGlyph | 이미지 → CXSMILES 및 치환기 표. 별도 ChemicalOCR 단계 제거 |
| OpenAI 호환 LLM | 유지 | 페이지 분류·화합물 ID·활성표 전사 |
| 연결·내보내기·Ketcher 검토 | 유지 및 출력 매핑 수정 | 엔진명과 원문·오류 기록을 새 모델에 맞게 변경 |

Glyph는 한 저장소에 **두 모델**을 제공한다. 일반 구조와 Markush 구조를 하나의 동일 모델로 처리하는 것은 아니다.

## 확인한 공개 버전

- 소스: [`EdisonScientific/glyph@0bf782f863d26b041ace157668928ef07c38b972`](https://github.com/EdisonScientific/glyph/tree/0bf782f863d26b041ace157668928ef07c38b972)
- OCSR 가중치: `EdisonScientific/OCSRGlyph@da0d049fa56effd3a07ecb15c715efdd78d9e8a0`
- Markush LoRA: `EdisonScientific/MarkushGlyph@2879b0380c2687a1bbdb2312ac4a810ed4887893`
- Markush 기반 모델: `Qwen/Qwen3.5-2B-Base@b1485b2fa6dfa1287294f269f5fb618e03d52d7c`

코드는 Apache-2.0이며 세 모델의 Hugging Face 메타데이터도 Apache-2.0으로 표시되어 있다.
포함한 코드의 저작권·라이선스·NOTICE 및 수정 표시를 보존해야 한다.
공식 기본 다운로드는 revision을 고정하지 않으므로 통합 시 코드와 가중치를 각각 고정한다.

## 실제 API 대응

- `OCSRPredictor(checkpoint=..., device="cuda:1", precision="fp16").predict_batch(paths)`
  → 입력 순서와 같은 SMILES 목록.
- `MarkushPredictor(checkpoint=..., base_model=..., device="cuda:1", dtype="float16").predict_many(paths)`
  → `raw`, `markush_xml`, `cxsmiles_opt`, `cxsmiles`, `stable`을 가진 결과 목록.

현재 `StructureRegion.smiles`, `.cxsmiles`, `.cxsmiles_opt`에 대응시킬 수 있다.
`stable`은 문자열이므로 `substituents: dict`에 넣기 위한 별도 파싱이 필요하다.
모델 원문과 변환 실패 이유도 저장하도록 스키마를 확장하는 것이 적절하다.

변경 지점:

1. `patentagent/inference.py`: `recognize()`와 `recognize_markush()` 교체, ChemicalOCR 경로 제거.
2. `patentagent/config.py`, `.env.example`, `cli.py`: 두 Glyph 체크포인트·기반 모델·정밀도·배치 설정.
3. `schemas.py`, `pipeline.py`, `postprocess.py`, `review_app.py`: 출력·치환기 표·원문·실패·모델 출처 연결.
4. `pyproject.toml`, `uv.lock`, `_vendor/`: 추론 코드와 의존성 교체.

`smiles_molscribe` 같은 기존 결과 키를 유지할 경우 호환용 필드임을 명시하고 실제 모델 출처를 따로 기록해야 한다.
검토 화면에 Glyph 결과를 “MolScribe”로 표시하면 안 된다.

## 단일 환경 설치 가능성

현재 버전을 그대로 둔 채 Glyph를 추가 설치하는 방식은 충돌한다.

| 패키지 | 현재 PatentAgent | Glyph 선언 또는 전이 의존성 |
|---|---|---|
| PyTorch | `2.5.1` | `>=2.7` |
| Transformers | `4.46.3` | `5.6.2` |
| timm | `0.4.12` | `>=1.0` |
| huggingface-hub | `<1` | Transformers에서 `>=1.5,<2` 요구 |
| pyarrow | `<20` | Glyph의 datasets extra에서 `>=21` 요구 |

Linux x86_64 / Python 3.11 / glibc 2.28 이상 / CUDA 12.6 대상으로
`uv pip compile`을 실행해, 기존 웹·PDF·OpenAI 클라이언트와 DECIMER를 유지한 후보 환경의
**의존성 해결이 성공**하는 것을 확인했다. 설치 또는 GPU 실행 성공을 뜻하지는 않는다.

주요 해결 버전:

```text
torch             2.7.1+cu126
torchvision       0.22.1+cu126
transformers      5.6.2
timm              1.0.30
peft              0.19.1
accelerate        1.15.0
huggingface-hub    1.33.0
datasets          4.8.5
pyarrow           23.0.1
numpy             1.26.4
tensorflow        2.15.1
DECIMER-Segmentation 1.5.0
```

TITAN RTX(sm75)는 `float16`을 명시한다. MarkushPredictor 기본값과 공식 간편 스크립트의
CUDA 선택값은 `bfloat16`이므로 그대로 사용하지 않는다.
공식 저장소의 cu128 설정을 그대로 복사하기보다 기존 드라이버 560.35.03에 맞춘 cu126 후보부터 검증한다.
새 PyTorch 휠은 glibc 2.28 이상이 필요하므로 서버 OS 확인도 필요하다.

`glyph @ git+...` 실행, 별도 Glyph 가상환경, 외부 추론 서버는 이전의 “외부 프로젝트 의존 제거”
목표와 맞지 않는다. 선택한 추론 코드·어휘·변환기를 내부에 포함하고 일반 라이브러리와
가중치만 설치하는 구성이 적절하다. 평가용 외부 MG2 scorer는 운영 추론에 필요하지 않다.

## Markush 변환에서 재현한 문제

현재 공개 코드의 `MarkushPrediction.from_raw()`를 기존 CPU 환경에서 직접 실행했다.

| 모델의 `cxsmiles_opt` | 반환된 `cxsmiles` |
|---|---|
| `CCO` | `CCO` |
| `<r>R1</r>CC` | `*CC \|$R1;;$\|` |
| `[\*]CC` | 빈 문자열 |
| `C[\CH3]` | 빈 문자열 |

공식 `_standard_cxsmiles()`는 변환 예외를 잡아 빈 문자열을 반환한다.
따라서 `cxsmiles_opt`가 존재한다는 이유만으로 성공 처리하면 안 된다.
원문·표준 변환본·실패 사유를 구분하고, 실패한 항목을 검토 대상으로 남겨야 한다.
기존 `repair_opt()`의 추정 복원을 적용한다면 원문을 덮어쓰지 않고 보정 이력과 신뢰도를 별도로 기록해야 한다.

## 성능 근거의 범위

[공식 README](https://github.com/EdisonScientific/glyph/blob/0bf782f863d26b041ace157668928ef07c38b972/README.md)의
개발팀 보고값:

- OCSR USPTO canonical: OCSRGlyph 93.8%, MolScribe 88.4%.
- Markush greedy: IP5-M 58.2%, M2S 61.2%, USPTO-Markush 59.5%.
- 같은 표의 MG2 비교값: 53.2%, 56.0%, 55.0%.

우리 특허의 정확도가 같은 폭으로 개선된다는 근거는 아니다.

[기존 Phase 0 기록](./GLYPH_PHASE0.md)에는 TITAN RTX에서 90개 이미지에 대해 OCSR 0.40초/장,
Markush 2.10초/장, Markush batch 4 약 4.9GB라는 측정이 있다.
이번 작업공간에는 해당 원시 예측·보고 JSON이 없어 재산출하지 못했다.
기록의 “약 5배”는 이전 외부 엔진 구성과의 비교이며, 방금 통합한 내부 런타임과의 실측 비교가 아니다.
기존 비교는 정답셋이 없고 표기 정규화·추정 복원을 포함한다. 비유의적 차이(p=0.359)는 동등성의 증명이 아니다.

## 권고

**교체 구현을 진행할 기술적 근거는 충분하다.** 현재 목표인 코드·설치·실행 구조 단순화에도 부합한다.
현재 구성도 이미 한 환경이므로 추가 이점은 ChemicalOCR 제거와 구형 MolScribe/Markush 전용 코드 제거다.

교체 확정 시 두 모델을 차례로 로드·해제하고, 기존 결과와 같은 입력 crop으로 비교한다.
일반 화합물·입체화학·R-group·부착점·반복단위·가변 결합 위치를 포함한 사람이 확인한 표본에서
구조 정확도, CXSMILES 변환 실패율, 속도, 최대 VRAM을 함께 측정한다.
실제 서버의 전체 가중치 로딩·GPU 추론과 해당 품질 검증은 이번 확인 범위에서 실행하지 않았다.
