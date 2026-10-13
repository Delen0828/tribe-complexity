/* Browser validation: NODE_PATH=<directory containing playwright> node visualizer/test_studies.cjs [URL] */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {firefox} = require('playwright');

const names = ['kylie', 'kylie_bar', 'kylie_area', 'hayeong', 'hayeong_color',
  'hayeong_hand_drawn', 'hayeong_cartoon_image'];
const base = process.argv[2] || 'http://127.0.0.1:8000/';
const screenshots = path.resolve('tribe/outputs/study_browser');
const url = relative => new URL(relative, base).href;

async function main() {
  fs.mkdirSync(screenshots, {recursive: true});
  const browser = await firefox.launch({headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
    const errors = [];
    const checked = [];
    page.on('pageerror', error => errors.push(String(error)));
    async function loadedImages(selector) {
      await page.locator(selector).evaluateAll(images => images.forEach(image => { image.loading = 'eager'; }));
      await page.waitForFunction(selector => [...document.querySelectorAll(selector)].every(
        image => image.complete && image.naturalWidth > 0 && !image.hidden), selector);
      assert.equal(await page.locator('.map-error:visible').count(), 0);
    }
    await page.goto(url('tribe/outputs/web/'));
    for (const name of names) assert.equal(await page.locator(`a[href="study-${name}.html"]`).count(), 1);
    for (const name of names) {
      await page.goto(url(`tribe/outputs/web/study-${name}.html`));
      await page.waitForSelector('#brain-workspace:visible');
      const report = await page.evaluate(() => window.eval('report'));
      assert(report.complete && report.scope === 'study');
      assert.equal(await page.locator('#report-title').innerText(), report.title);
      assert(await page.locator('#dataset-control').isHidden());
      const layouts = await page.locator('#views option').evaluateAll(options => options.map(o => o.value));
      assert.deepEqual(layouts, ['four', 'inferior', 'all']);
      for (const attribute of report.attributes) {
        await page.selectOption('#attribute', attribute.key);
        const groups = report.groups.filter(g => g.attribute === attribute.key);
        assert.equal(await page.locator('#groups button').count(), groups.length);
        for (const [position, group] of groups.entries()) {
          const button = page.locator('#groups button').nth(position);
          assert.equal(await button.innerText(), group.label);
          await button.click();
          assert.equal(await page.locator('#members a').count(), Math.min(24, group.count));
          const sources = await page.locator('#members img').evaluateAll(images => images.map(image => image.src));
          assert.deepEqual(sources.map(source => Number(path.basename(source, '.png'))), group.indices.slice(0, 24));
          if (name.startsWith('hayeong')) {
            for (const detail of await page.locator('#members span').allTextContents()) {
              assert(!detail.toLowerCase().includes('complexity'));
            }
          }
          await loadedImages('#members img');
        }
        for (const mode of ['mean', 'contrast']) {
          await page.selectOption('#statistic', mode);
          for (const layout of layouts) {
            await page.selectOption('#views', layout);
            const count = {four: 4, inferior: 1, all: 5}[layout];
            assert.equal(await page.locator('#groups tbody tr').count(), count);
            assert.equal(await page.locator('#groups .group-map').count(), groups.length * count);
            await loadedImages('#groups .group-map');
          }
        }
        const selectedURL = page.url();
        const selectedGroup = await page.locator('#groups button[aria-pressed="true"]').getAttribute('data-value');
        await page.reload();
        await page.waitForSelector('#brain-workspace:visible');
        assert.equal(page.url(), selectedURL);
        assert.equal(await page.locator('#groups button[aria-pressed="true"]').getAttribute('data-value'), selectedGroup);
        assert.equal(await page.locator('#statistic').inputValue(), 'contrast');
      }
      await page.selectOption('#attribute', report.attributes[0].key);
      await page.selectOption('#statistic', 'mean');
      await page.selectOption('#views', 'all');
      await loadedImages('#groups .group-map');
      await page.locator('.brain-grid-scroll').evaluate(element => { element.scrollLeft = 0; });
      await page.screenshot({path: path.join(screenshots, `${name}-desktop.png`), fullPage: true});
      await page.setViewportSize({width: 390, height: 844});
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.screenshot({path: path.join(screenshots, `${name}-mobile.png`), fullPage: true});
      await page.setViewportSize({width: 1440, height: 1100});
      checked.push({page: name, stimuli: report.count, attributes: report.attributes.length,
        groups: report.groups.length, layouts});
    }
    for (const name of names) {
      await page.goto(url(`visualizer/brain.html?dataset=${name}`));
      await page.waitForSelector('#brain-workspace:visible');
      assert.equal(await page.evaluate(() => window.eval('report.dataset')), name.startsWith('kylie') ? 'kylie' : 'hayeong');
      assert((await page.locator('#members a').first().getAttribute('href')).includes('/inputs/'));
      await loadedImages('#groups .group-map');
    }
    // Shared UI changes must preserve the existing MassVis explorer.
    await page.goto(url('visualizer/brain.html?dataset=massvis_sample'));
    await page.waitForSelector('#brain-workspace:visible');
    const massvis = await page.evaluate(() => window.eval('report'));
    assert.equal(massvis.attributes.length, 31);
    for (const attribute of massvis.attributes) {
      await page.selectOption('#attribute', attribute.key);
      const groups = massvis.groups.filter(group => group.attribute === attribute.key);
      assert.equal(await page.locator('#groups button').count(), groups.length);
      for (const mode of ['mean', 'contrast']) {
        await page.selectOption('#statistic', mode);
        await loadedImages('#groups .group-map');
      }
    }
    assert.deepEqual(errors, []);
    const result = {status: 'PASS', pages: checked, massvis_attributes: 31,
      map_modes: ['mean', 'contrast'], checks: ['cortical assets and member thumbnails',
        'group labels and membership', 'no invented Hayeong scores', 'URL restoration',
        'root visualizer routes', 'mobile overflow', 'MassVis regression'], screenshots};
    fs.writeFileSync(path.join(screenshots, 'verification.json'), JSON.stringify(result, null, 2) + '\n');
    process.stdout.write(JSON.stringify(result, null, 2) + '\n');
  } finally {
    await browser.close();
  }
}
main().catch(error => { process.stderr.write(String(error.stack || error) + '\n'); process.exitCode = 1; });
