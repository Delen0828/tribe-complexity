/* End-to-end check against a visualizer with the precomputed MassVis report. */
const assert = require('node:assert/strict');
const {firefox} = require('playwright');

async function main() {
  const browser = await firefox.launch({headless: true});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    const base = process.argv[2] || 'http://127.0.0.1:8000/';
    await page.goto(new URL('visualizer/', base).href);
    await page.getByRole('link', {name: /Ridge interpretability/}).click();
    await page.locator('#results').waitFor();
    assert.match(await page.locator('#summary').innerText(), /Full-model OOF R²/);
    const table = page.locator('#regions tr');
    assert.ok(await table.count() > 100);
    async function scores() {
      return table.locator('td:last-child').evaluateAll(cells => cells.map(c => Number(c.textContent)));
    }
    const descending = await scores();
    assert.deepEqual(descending, [...descending].sort((a, b) => b-a));
    await page.locator('#sort').selectOption('asc');
    assert.deepEqual(await scores(), descending.reverse());
    await page.locator('#metric').selectOption('permutation');
    assert.notDeepEqual(await scores(), descending);
    for (const metric of ['ablation', 'permutation', 'haufe']) {
      await page.locator('#metric').selectOption(metric);
      await page.locator('#view').selectOption('all');
      assert.equal(await page.locator('#region-section').isVisible(), metric !== 'haufe');
      assert.equal(await page.locator('#time').isVisible(), metric === 'haufe');
      const limit = await page.locator('#high').textContent();
      for (const time of metric === 'haufe' ? ['0','1','2'] : ['0']) {
        if (metric === 'haufe') await page.locator('#time').selectOption(time);
        await page.waitForFunction(() => {
          const images = [...document.querySelectorAll('#panels img')];
          return images.length === 5 && images.every(img => img.complete && img.naturalWidth > 0);
        });
        assert.equal(await page.locator('#high').textContent(), limit);
      }
    }
    await page.route('**/ridge_interpretability/report.json', route => route.fulfill({status:404, body:''}));
    await page.reload();
    await page.getByText('python label/stats/interpret_ridge.py', {exact: true}).waitFor();
    assert.equal(await page.locator('#results').isVisible(), false);
    await page.getByRole('link', {name:'Prediction pipeline', exact:true}).click();
    await page.getByRole('link', {name:'Ridge interpretability', exact:true}).waitFor();
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
