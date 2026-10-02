# Ketcher 임베드 빌드 소스

검수 UI(review_app.py / gateway.py)가 쓰는 Ketcher standalone 정적 빌드의 소스.
산출물은 `../patentagent/static/ketcher/` 에 배치됨.

## 재빌드
```bash
npm install            # node 20+ (ketcher 3.15.0)
npx vite build         # -> dist-embed/
cp -r dist-embed/* ../patentagent/static/ketcher/
```

## 핵심: vite.config.js
ketcher-react 의 raphael 동적 require 때문에 commonjsOptions.transformMixedEsModules
+ optimizeDeps.include 필수. 없으면 브라우저에서 "require is not defined" → 빈 화면.

## 헤드리스 검증
```bash
npm install puppeteer
JOB=<job_id> node ket_test.cjs   # 마운트/렌더 확인
JOB=<job_id> node ket_flow.cjs   # postMessage 왕복 + 저장
```
