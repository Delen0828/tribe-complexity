/* Run with NODE_PATH pointing to an installed Playwright package. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {firefox} = require('playwright');
const base = process.argv[2] || 'http://127.0.0.1:8765/';
const url = path => new URL(path, base).href;
const names = ['massvis_sample', 'kylie', 'kylie_bar', 'kylie_area', 'hayeong',
  'hayeong_color', 'hayeong_hand_drawn', 'arunkumar', 'arunkumar_experiment_1', 'arunkumar_experiment_2'];

async function main() {
  const browser = await firefox.launch({headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    page.setDefaultTimeout(60000);
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    async function grid() {
      await page.waitForFunction(() => document.querySelector('#groups .cell-map') &&
        document.querySelector('#brain-status').textContent === '');
      return page.evaluate(() => window.eval('analysis'));
    }
    async function members(indices) {
      assert.equal(await page.locator('#members a').count(), Math.min(24, indices.length));
      const actual = await page.locator('#members img').evaluateAll(images => images.map(image =>
        Number(new URL(image.src).pathname.split('/').pop().replace('.png', ''))));
      assert.deepEqual(actual, indices.slice(0, 24));
    }
    async function firstMap() {
      await page.locator('.group-map').first().evaluate(image => {image.loading = 'eager';});
      await page.waitForFunction(() => {
        const image = document.querySelector('.group-map');
        return image && image.complete && image.naturalWidth > 0 && !image.hidden;
      });
    }
    await page.goto(url('visualizer/'));
    assert.equal(await page.locator('.home-grid a').count(), 6);
    assert.equal(await page.locator('nav[aria-label="Dataset explorers"] a').count(), 2);
    assert.equal(await page.locator('nav[aria-label="Simulation results"] a').count(), 4);
    await page.locator('.home-grid a').first().click();
    await page.waitForSelector('#workspace:visible');
    await page.fill('#index', '42'); await page.locator('#lookup button').click();
    assert.equal(await page.locator('#image-id').innerText(), 'Study ID 42.png');
    await page.waitForFunction(() => document.querySelector('#image').naturalWidth > 0);

    for (const name of names) {
      await page.goto(url(`visualizer/brain.html?dataset=${name}`));
      let data = await grid();
      const report = await page.evaluate(() => window.eval('report'));
      if (name === 'massvis_sample') {
        assert.deepEqual(data.axes.x.map(g => g.label), ['Low', 'Mid', 'High']);
        const counts = data.axes.x.map(g => g.indices.length);
        assert(Math.max(...counts) - Math.min(...counts) <= 1);
        const scores = data.axes.x.map(g => report.rows.filter(row => g.indices.includes(row.index)).map(row => row.score));
        assert(Math.max(...scores[0]) <= Math.min(...scores[1]));
        assert(Math.max(...scores[1]) <= Math.min(...scores[2]));
      }
      assert.equal(await page.locator('#views').inputValue(), 'inferior');
      assert.equal(await page.locator('.sample-square').count(), 0);
      assert.equal(await page.locator('#x-axis option').count(), report.attributes.length);
      assert.equal(await page.locator('#y-axis option').count(), report.attributes.length + 1);
      await members(report.rows.map(row => row.index));
      const fieldsBox = await page.locator('.field-panel').boundingBox();
      const galleryBox = await page.locator('#members-section').boundingBox();
      const tableBox = await page.locator('.brain-card').boundingBox();
      assert(galleryBox.x >= fieldsBox.x + fieldsBox.width);
      assert(tableBox.y >= Math.max(fieldsBox.y + fieldsBox.height, galleryBox.y + galleryBox.height));
      const gallery = await page.locator('#members').boundingBox();
      const firstExample = await page.locator('#members a').first().boundingBox();
      assert(Math.abs(firstExample.width * 3 + 24 - gallery.width) < 2);
      if (name.startsWith('kylie')) assert(report.attributes.some(a => a.key === 'complexity' && a.kind === 'bin'));
      if (name.startsWith('hayeong')) {
        assert.deepEqual(report.attributes.slice(0, 5).map(a => a.key),
          ['has_color', 'bar_is_hand_drawn', 'font_is_hand_drawn', 'is_comic', 'bar_is_image']);
        assert(report.rows.every(row => row.attributes.experiment === 2));
        if (name === 'hayeong') assert.equal(report.selected_count, 24);
        assert.equal(await page.locator('#dataset option[value="hayeong_cartoon_image"]').count(), 0);
      }
      if (name.startsWith('arunkumar')) {
        const experiment = name.endsWith('_1') ? 1 : name.endsWith('_2') ? 2 : null;
        assert.equal(report.count, experiment === 1 ? 500 : experiment === 2 ? 100 : 600);
        assert.equal(report.attributes.length, experiment ? 40 : 41);
        assert.equal(report.attributes[0].key, 'image_information_rating');
        assert(report.rows.every(row => row.score >= -10 && row.score <= 10 &&
          (!experiment || row.attributes.experiment === experiment)));
        for (const attribute of report.attributes) {
          const response = await page.request.get(url('api/analysis?' + new URLSearchParams({
            dataset: name, x: attribute.key, y: '', statistic: 'mean'})));
          assert(response.ok());
          assert.equal((await response.json()).cells.reduce((sum, cell) => sum + cell.count, 0), report.count);
        }
      }
      for (const attribute of report.attributes.filter(a => a.kind === 'binary' ||
        (name.startsWith('kylie') && a.key === 'complexity'))) {
        const response = await page.request.get(url('api/analysis?' + new URLSearchParams({
          dataset: name, x: attribute.key, y: '', statistic: 'mean'})));
        assert(response.ok());
        const grouped = await response.json();
        assert.equal(grouped.cells.reduce((sum, c) => sum + c.count, 0), report.count);
      }
      await firstMap();
      const second = report.attributes.at(-1).key;
      await page.selectOption('#y-axis', second);
      data = await grid();
      assert.equal(await page.locator('#groups tbody tr').count(), data.axes.y.length);
      assert.equal(await page.locator('#groups td').count(), data.axes.x.length * data.axes.y.length);
      assert.equal(data.cells.reduce((sum, c) => sum + c.count, 0), report.count);
      assert.deepEqual(await page.locator('#groups td').evaluateAll(cells => cells.map(c => Number(c.dataset.count))),
        data.cells.map(c => c.count));
      await page.locator('.axis-tick[data-axis="x"]').first().click();
      await members(data.axes.x[0].indices);
      await page.locator('.axis-tick[data-axis="y"]').last().click();
      await members(data.axes.y.at(-1).indices);
      const cell = data.cells.find(c => c.count);
      await page.locator(`.cell-map[data-x="${cell.x}"][data-y="${cell.y}"]`).click();
      await members(cell.indices);
      const chosenURL = page.url();
      await page.reload(); await grid();
      assert.equal(page.url(), chosenURL);
      await members(cell.indices);
      if (name.startsWith('hayeong')) assert(!(await page.locator('#members').innerText()).includes('Complexity'));
      await page.setViewportSize({width: 390, height: 844});
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.setViewportSize({width: 1440, height: 1000});
      process.stdout.write(`${name}: X/Y cells, ticks, examples, URL, maps and mobile PASS\n`);
    }
    await page.goto(url('visualizer/brain.html?dataset=massvis_sample&x=black_and_white&y=multi_panel'));
    await grid();
    for (const mode of ['mean', 'contrast']) {
      await page.selectOption('#statistic', mode); await grid();
      for (const view of ['inferior', 'four', 'all']) {
        await page.selectOption('#views', view); await firstMap();
        const src = await page.locator('.group-map').first().getAttribute('src');
        assert.equal(new URL(src, base).searchParams.get('views'), view);
      }
    }
    await page.locator('#swap-axes').click(); await grid();
    assert.equal(await page.locator('#x-axis').inputValue(), 'multi_panel');
    assert.equal(await page.locator('#y-axis').inputValue(), 'black_and_white');
    await page.fill('#field-search', 'complexity');
    assert.equal(await page.locator('.field:visible').count(), 1);
    await page.locator('.field:visible button').first().click(); await grid();
    assert.equal(await page.locator('#x-axis').inputValue(), 'complexity');
    // Check every published variable through the API without rendering unused combinations.
    const report = await page.evaluate(() => window.eval('report'));
    for (const attribute of report.attributes) {
      for (const axis of ['x', 'y']) {
        const params = new URLSearchParams({dataset: 'massvis_sample', x: axis === 'x' ? attribute.key : 'complexity',
          y: axis === 'y' ? attribute.key : '', statistic: 'mean'});
        const response = await page.request.get(url('api/analysis?' + params));
        assert(response.ok());
        const data = await response.json();
        if (attribute.key === 'complexity') assert.deepEqual(data.axes[axis].map(g => g.label), ['Low', 'Mid', 'High']);
        assert.equal(data.cells.reduce((sum, c) => sum + c.count, 0), report.count);
        assert.deepEqual(data.axes[axis].flatMap(g => g.indices).sort((a, b) => a - b),
          report.rows.map(row => row.index).sort((a, b) => a - b));
      }
    }
    // Scrolling loads every example in the selected bin beyond the first batch.
    await page.selectOption('#x-axis', 'charts'); await page.selectOption('#y-axis', '');
    const pagination = await grid();
    const largest = pagination.axes.x.reduce((a, b) => a.indices.length > b.indices.length ? a : b);
    await page.locator(`.axis-tick[data-axis="x"][data-x="${largest.value}"]`).click();
    if (largest.indices.length > 24) {
      for (let loaded = 24; loaded < largest.indices.length; loaded += 24) {
        await page.locator('#members').evaluate(gallery => {gallery.scrollLeft = gallery.scrollWidth;});
        await page.waitForFunction(count => document.querySelectorAll('#members a').length >= count,
          Math.min(loaded + 24, largest.indices.length));
      }
      assert.equal(await page.locator('#members a').count(), largest.indices.length);
      assert.deepEqual(await page.locator('#members img').evaluateAll(images => images.map(image =>
        Number(new URL(image.src).pathname.split('/').pop().replace('.png', '')))), largest.indices);
    }
    await page.goto(url('visualizer/brain.html?dataset=massvis_all&x=black_and_white'));
    await grid();
    const full = await page.evaluate(() => window.eval('report'));
    assert.equal(full.selected_count, 5800);
    const fullComplexity = await page.request.get(url('api/analysis?dataset=massvis_all&x=complexity'));
    assert(fullComplexity.ok());
    assert.deepEqual((await fullComplexity.json()).axes.x.map(g => g.label), ['Low', 'Mid', 'High']);
    if (!full.complete) assert((await page.locator('#sample-summary').innerText()).startsWith('Partial'));
    await firstMap();
    await page.goto(url('tribe/outputs/web/study-kylie.html')); await grid();
    assert.equal(await page.evaluate(() => window.eval('report.count')), 24);
    await page.goto(url('visualizer/brain.html?dataset=missing'));
    await grid(); // An unknown bookmark falls back to the available default.
    assert.deepEqual(errors, []);
    await page.goto(url('visualizer/brain.html?dataset=massvis_sample&x=black_and_white&y=multi_panel'));
    await grid(); await firstMap();
    fs.mkdirSync('/tmp/tribe-visualizer-qa', {recursive: true});
    await page.screenshot({path: '/tmp/tribe-visualizer-qa/analysis-desktop.png', fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    await page.screenshot({path: '/tmp/tribe-visualizer-qa/analysis-mobile.png', fullPage: true});
    await page.goto(url('visualizer/'));
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({path: '/tmp/tribe-visualizer-qa/home-mobile.png', fullPage: true});
    // Static previews return HTML for API paths; explain how to start the backend.
    await page.route('**/api/reports', route => route.fulfill({status: 404,
      contentType: 'text/html', body: '<html>File not found</html>'}));
    await page.goto(url('visualizer/brain.html'));
    await page.waitForFunction(() => document.querySelector('#brain-status').textContent.includes('Visualizer API unavailable'));
    assert((await page.locator('#brain-status').innerText()).includes('./start-visualizer.sh'));
    await page.unroute('**/api/reports');
    // Some preview servers return a JSON 404 rather than HTML for absent routes.
    await page.route('**/api/reports', route => route.fulfill({status: 404,
      contentType: 'application/json', body: JSON.stringify({success: false, error: 'Not found'})}));
    await page.goto(url('visualizer/brain.html'));
    await page.waitForFunction(() => document.querySelector('#brain-status').textContent.includes('Visualizer API unavailable'));
    await page.unroute('**/api/reports');
    // A stopped backend should be distinguishable from missing predictions.
    await page.route('**/api/reports', route => route.abort('failed'));
    await page.goto(url('visualizer/brain.html'));
    await page.waitForFunction(() => document.querySelector('#brain-status').textContent.includes('Cannot reach the visualizer server'));
    await page.unroute('**/api/reports');
    await page.route('**/api/report?*', route => route.fulfill({status: 404,
      contentType: 'application/json', body: JSON.stringify({error: 'Predictions unavailable'})}));
    await page.goto(url('visualizer/brain.html'));
    await page.waitForFunction(() => document.querySelector('#brain-status').textContent === 'Predictions unavailable');
    await page.unroute('**/api/report?*');
    assert.deepEqual(errors, []);
    process.stdout.write('PASS: 10 study routes, both Arunkumar experiments and all 39 metadata fields, experiment 2 features, 31 MassVis variables on both axes, 3 cortical layouts, 2 map modes, full-run coverage, navigation, scrolling gallery and connection diagnostics\n');
  } finally {await browser.close();}
}
main().catch(error => {process.stderr.write(error.stack + '\n'); process.exitCode = 1;});
