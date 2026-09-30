const puppeteer = require('puppeteer');
(async () => {
  const browser = await puppeteer.launch({headless:'new', args:['--no-sandbox','--disable-setuid-sandbox','--disable-gpu']});
  const page = await browser.newPage();
  const logs = [];
  page.on('pageerror', e => logs.push('[pageerror] '+e.message));

  await page.goto(`http://localhost:8000/jobs/${process.env.JOB}/review/item/0`, {waitUntil:'networkidle2', timeout:60000});
  await new Promise(r=>setTimeout(r,12000)); // ketcher ready + setMol

  // 부모 페이지에서 ketcherReady 상태 + getKetcherSmiles() 직접 호출
  const result = await page.evaluate(async () => {
    // 부모 window 의 getKetcherSmiles 사용 (저장 로직과 동일 경로)
    const res = await getKetcherSmiles();
    return { ketcherReady: window.ketcherReady, init: INIT_SMILES.slice(0,40), got: res };
  }).catch(e => 'eval-error: '+e.message);
  console.log('=== postMessage 왕복 (setMol→getSmiles) ===');
  console.log(JSON.stringify(result, null, 2));

  // 실제 저장까지: verdict 설정 후 saveItem()
  const saveLog = await page.evaluate(async () => {
    document.getElementById('verdict').value = 'approved';
    await saveItem();
    return document.getElementById('status').textContent;
  }).catch(e=>'save-error: '+e.message);
  console.log('=== 저장 결과 status ===');
  console.log(saveLog);
  console.log('=== pageerror ===', logs.join('\n')||'(없음)');
  await browser.close();
})();
