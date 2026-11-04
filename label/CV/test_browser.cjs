const {chromium}=require('playwright');
const fs=require('fs'),path=require('path'),http=require('http'),assert=require('assert');
const root=__dirname,manifest=JSON.parse(fs.readFileSync(path.join(root,'results/manifest.json')));
const artifacts=path.join(root,'cache/browser');fs.mkdirSync(artifacts,{recursive:true});
const server=http.createServer((req,res)=>{const name=path.resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname));if(name!==root&&!name.startsWith(root+path.sep)){res.writeHead(403);res.end();return;}const file=name===root?path.join(root,'index.html'):name;fs.readFile(file,(error,body)=>{if(error){res.writeHead(404);res.end();return;}const ext=path.extname(file);res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.json':'application/json'})[ext]||'application/octet-stream');res.end(body);});});
(async()=>{
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const base=`http://127.0.0.1:${server.address().port}`;
const existing='/home/delen/.cache/ms-playwright/chromium_headless_shell-1248/chrome-linux/headless_shell';
const alternatives=[existing,'/home/delen/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell'];
const executablePath=process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE||alternatives.find(p=>fs.existsSync(p));
const browser=await chromium.launch({headless:true,executablePath,args:['--no-sandbox']});let checks=0;
try{
for(const [protocol,url] of [['file','file://'+path.join(root,'index.html')],['http',base+'/index.html']]){
 const context=await browser.newContext({viewport:{width:1440,height:1050},acceptDownloads:true});const page=await context.newPage(),errors=[];page.on('pageerror',error=>errors.push(error.message));
 await page.goto(url);await page.waitForSelector('.card');assert.equal(await page.locator('.card').count(),150);assert((await page.locator('.card').first().boundingBox()).height>=100);checks++;
 await page.locator('[data-band="0"]').click();assert.equal(await page.locator('.card').count(),15);await page.locator('[data-band="0"]').click();checks++;
 await page.locator('#search').fill('no-such-visualization');assert.equal(await page.locator('.card').count(),0);assert(await page.locator('#empty').isVisible());await page.locator('#search').fill('');checks++;
 const entry=manifest.entries.find(e=>e.count>0)||manifest.entries.find(e=>e.status!=='not_applicable')||manifest.entries[0];
 await page.locator('#search').fill(entry.filename);await page.locator(`.card[data-id="${entry.image_id}"]`).click();await page.waitForFunction(id=>window.CV_FIGURES[id]&&document.querySelector('#main-image').complete,entry.image_id);
 assert.equal(await page.locator('#detail-score').innerText(),entry.mean_human_complexity_rating.toFixed(1));assert.equal(await page.locator('#detail-count').innerText(),entry.count!==null?String(entry.count):entry.status==='not_applicable'?'N/A':entry.status==='failed'?'Failed':'—');checks++;
 for(const mode of ['original','segmented','overlay']){await page.locator(`[data-mode="${mode}"]`).click();await page.waitForFunction(()=>document.querySelector('#main-image').complete);assert.equal(await page.locator('#main-image').evaluate(i=>i.naturalWidth),entry.width);assert.equal(await page.locator('#main-image').evaluate(i=>i.naturalHeight),entry.height);}checks++;
 await page.locator('#size-mode').selectOption('native');assert(Math.abs((await page.locator('#image-stage').boundingBox()).width-entry.width)<1);await page.locator('#size-mode').selectOption('fit');checks++;
 if(await page.locator('#instance-list button').count()){await page.locator('#instance-list button').first().click();assert(await page.locator('#instance-detail').isVisible());await page.waitForTimeout(50);assert(await page.locator('#instance-detail canvas').count());checks++;}
 await page.locator('#review-quality').selectOption('mixed');await page.locator('[name=issue][value=missed]').check();await page.locator('#review-notes').fill('QA review: check small marks.');
 await page.reload();await page.waitForSelector('.card');await page.locator('#search').fill(entry.filename);await page.locator(`.card[data-id="${entry.image_id}"]`).click();assert.equal(await page.locator('#review-quality').inputValue(),'mixed');assert.equal(await page.locator('#review-notes').inputValue(),'QA review: check small marks.');checks++;
 const dl=page.waitForEvent('download');await page.locator('#export-reviews').click();const download=await dl,file=path.join(artifacts,protocol+'-reviews.json');await download.saveAs(file);const exported=JSON.parse(fs.readFileSync(file));assert.equal(exported.run_id,manifest.run_id);assert.equal(exported.reviews[entry.image_id].quality,'mixed');checks++;
 await page.evaluate(()=>localStorage.clear());await page.reload();await page.waitForSelector('.card');await page.locator('#import-reviews').setInputFiles(file);await page.waitForFunction(()=>document.querySelector('#review-total').textContent.startsWith('1 /'));checks++;
 await page.locator('#import-reviews').setInputFiles({name:'wrong-run.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({...exported,run_id:'wrong'}))});assert((await page.locator('#review-message').innerText()).includes('different run'));checks++;
 await page.locator('#search').fill('');await page.locator(`.card[data-id="${entry.image_id}"]`).click();await page.evaluate(()=>{document.querySelector('#review-message').textContent='';window.scrollTo(0,0)});await page.screenshot({path:path.join(artifacts,protocol+'-desktop.png'),fullPage:true});await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(artifacts,protocol+'-mobile.png'),fullPage:true});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));checks++;
 assert.deepEqual(errors,[]);await context.close();
}
const context=await browser.newContext(),page=await context.newPage();const fixture=JSON.parse(JSON.stringify(manifest));fixture.entries[0].status='failed';fixture.entries[0].count=null;fixture.entries[1].status='complete';fixture.entries[1].count=0;fixture.entries[2].status='not_applicable';fixture.entries[2].count=null;
await page.route('**/data.js',route=>route.fulfill({contentType:'text/javascript',body:'window.CV_DATA='+JSON.stringify(fixture)}));await page.goto(base+'/index.html');await page.waitForSelector('.card');
for(const [i,text]of [[0,'Failed'],[1,'0'],[2,'N/A']]){await page.locator(`.card[data-id="${fixture.entries[i].image_id}"]`).click();assert.equal(await page.locator('#detail-count').innerText(),text);checks++;}await context.close();
fs.writeFileSync(path.join(root,'results/browser-verification.json'),JSON.stringify({passed:true,checks,protocols:['file','http'],viewports:[1440,390],verified_at:new Date().toISOString()},null,2)+'\n');console.log(`PASS: ${checks} browser checks across file:// and HTTP, desktop and mobile.`);
}finally{await browser.close();server.close();}
})().catch(error=>{console.error(error);server.close();process.exitCode=1;});
