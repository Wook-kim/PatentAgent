const puppeteer = require('puppeteer');
(async () => {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu']
  });
  const page = await browser.newPage();
  const logs = [];
  page.on('console', m => logs.push(`[console.${m.type()}] ${m.text()}`));
  page.on('pageerror', e => logs.push(`[pageerror] ${e.message}`));
  page.on('requestfailed', r => logs.push(`[reqfail] ${r.url()} :: ${r.failure().errorText}`));

  const url = `http://localhost:8000/jobs/${process.env.JOB}/review/item/0`;
  console.log('GET', url);
  await page.goto(url, {waitUntil: 'networkidle2', timeout: 60000});
  // Ketcher 로딩 대기
  await new Promise(r => setTimeout(r, 12000));

  // iframe 내부 상태 확인
  const frames = page.frames();
  let kinfo = 'no-kframe';
  for (const f of frames) {
    if (f.url().includes('/ketcher/')) {
      kinfo = await f.evaluate(() => ({
        hasRoot: !!document.getElementById('root'),
        rootChildren: document.getElementById('root') ? document.getElementById('root').children.length : -1,
        hasKetcher: typeof window.ketcher !== 'undefined',
        bodyText: document.body.innerText.slice(0, 200)
      })).catch(e => 'frame-eval-error: ' + e.message);
    }
  }
  console.log('=== iframe(ketcher) 상태 ===');
  console.log(JSON.stringify(kinfo, null, 2));
  console.log('=== 콘솔/에러 로그 ===');
  console.log(logs.join('\n') || '(로그 없음)');
  await browser.close();
})();
