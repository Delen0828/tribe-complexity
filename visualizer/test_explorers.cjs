/* Run with NODE_PATH pointing to an installed Playwright package. */
const assert = require('node:assert/strict');
const {firefox} = require('playwright');
const base = process.argv[2] || 'http://127.0.0.1:8766/';
const url = path => new URL(path, base).href;

async function main() {
  const browser = await firefox.launch({headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    async function imageLoaded() {
      await page.waitForFunction(() => {
        const image = document.querySelector('#image');
        return image.complete && image.naturalWidth > 0 && !image.hidden;
      });
    }
    async function counts(expected) {
      assert.deepEqual(await page.locator('.hist-bar').evaluateAll(bars =>
        bars.map(bar => Number(bar.dataset.count))), expected);
    }
    async function lookup(index) {
      await page.fill('#index', String(index));
      await page.locator('#lookup button').click();
      assert.equal(await page.locator('#index').inputValue(), String(index));
      await imageLoaded();
    }
    await page.goto(url('visualizer/'));
    assert.deepEqual(await page.locator('.home-grid a').evaluateAll(links =>
      links.map(link => link.getAttribute('href'))), [
      'data.html', 'arunkumar.html', 'brain.html?dataset=massvis_sample',
      'brain.html?dataset=kylie', 'brain.html?dataset=hayeong', 'brain.html?dataset=arunkumar']);
    const datasetCards = await page.locator('nav[aria-label="Dataset explorers"]').boundingBox();
    const simulationCards = await page.locator('nav[aria-label="Simulation results"]').boundingBox();
    assert(simulationCards.y > datasetCards.y + datasetCards.height);
    await page.locator('.home-grid a').first().click();
    await page.waitForSelector('#workspace:visible');
    await lookup(42);
    assert.equal(await page.locator('#image-id').innerText(), 'Study ID 42.png');
    assert.equal(await page.locator('.hist-bar').count(), 20);

    await page.goto(url('visualizer/arunkumar.html'));
    await page.waitForSelector('#workspace:visible');
    await imageLoaded();
    await counts([0, 0, 7, 51, 38, 52, 80, 47, 28, 16, 16, 46, 34, 53, 36, 43, 40, 13, 0, 0]);
    assert(await page.locator('.hist-bar').evaluateAll(bars => bars.length === 20 &&
      bars.every(bar => Number(bar.dataset.end) - Number(bar.dataset.start) === 1)));
    assert.equal(await page.locator('.hist-bar[role="button"]').count(), 16);
    assert.equal(await page.locator('#score').innerText(), '-1.65');
    assert.equal(await page.locator('#current-interval').innerText(), 'I3');
    assert.equal(await page.locator('#dimensions tbody tr').count(), 39);
    assert.equal(await page.locator('tr[data-key="temporal"] td').innerText(), 'Yes');
    assert.equal(await page.locator('tr[data-key="photorealistic"] td').innerText(), 'No');
    assert.equal(await page.locator('tr[data-key="sentiment"] td').innerText(), 'Neutral');
    assert.deepEqual(await page.locator('.hist-boundary').evaluateAll(lines =>
      lines.map(line => [Number(line.dataset.value), Number(line.getAttribute('x1'))])),
      [[-6, 142], [-3, 217], [3, 367], [6, 442]]);
    assert.deepEqual(await page.locator('#rating-intervals li').allTextContents(), [
      'I1 · Image-like [−10, −6]', 'I2 · (−6, −3]', 'I3 · (−3, 3)',
      'I4 · [3, 6)', 'I5 · Information-like [6, 10]']);
    await page.locator('.hist-bar').nth(2).click();
    assert.equal(await page.locator('#index').inputValue(), '459');
    await page.locator('.hist-bar').nth(12).focus();
    await page.keyboard.press('Enter');
    assert.equal(await page.locator('#index').inputValue(), '175');
    assert(await page.locator('.hist-bar').nth(12).evaluate(bar => bar === document.activeElement));
    await page.locator('.hist-bar').nth(7).focus();
    await page.keyboard.press('Space');
    assert.equal(await page.locator('#index').inputValue(), '17');
    await page.goBack();
    assert.equal(await page.locator('#index').inputValue(), '175');

    await lookup(75);
    assert.equal(await page.locator('#score').innerText(), '-7.16');
    await page.selectOption('#experiment', '2');
    await imageLoaded();
    await counts([0, 0, 3, 8, 8, 7, 11, 6, 2, 1, 3, 6, 8, 6, 7, 13, 6, 5, 0, 0]);
    assert.equal(await page.locator('#index').inputValue(), '501');
    assert.equal(await page.locator('#image-id').innerText(), 'Original stimulus 75');
    assert.equal(await page.locator('#score').innerText(), '-7.74');
    assert.equal(await page.locator('#current-interval').innerText(), 'I1');
    assert(await page.locator('#group-ratings').isVisible());
    assert((await page.locator('#rating-spread').innerText()).includes('SD 2.13'));
    assert(await page.locator('#previous').isDisabled());
    assert.equal(new URL(page.url()).searchParams.get('experiment'), '2');
    await page.goBack();
    assert.equal(await page.locator('#experiment').inputValue(), 'all');
    assert.equal(await page.locator('#index').inputValue(), '75');
    await counts([0, 0, 7, 51, 38, 52, 80, 47, 28, 16, 16, 46, 34, 53, 36, 43, 40, 13, 0, 0]);
    await page.goForward();
    assert.equal(await page.locator('#index').inputValue(), '501');
    await page.locator('#next').click();
    assert.equal(await page.locator('#index').inputValue(), '502');
    await page.locator('#previous').click();
    assert.equal(await page.locator('#index').inputValue(), '501');
    for (const [index, filename] of [[505, '5-76.jpeg'], [509, '9-89.png'], [529, '29-88.png']]) {
      await lookup(index);
      assert.equal(await page.locator('#image-title').innerText(), filename);
      const source = decodeURIComponent(await page.locator('#image').getAttribute('src'));
      assert(source.endsWith(`/Expt 2 - 100 stimuli/${filename}`));
      assert.equal(await page.locator('#original').getAttribute('href'),
        await page.locator('#image').getAttribute('src'));
    }
    await page.locator('.hist-bar').nth(17).click();
    assert.equal(await page.locator('#index').inputValue(), '600');
    assert.equal(await page.locator('#score').innerText(), '+7.28');
    assert(await page.locator('#next').isDisabled());
    await page.selectOption('#experiment', '1');
    await counts([0, 0, 4, 43, 30, 45, 69, 41, 26, 15, 13, 40, 26, 47, 29, 30, 34, 8, 0, 0]);
    assert.equal(await page.locator('#index').inputValue(), '431');
    assert.equal(await page.locator('#score').innerText(), '+7.29');
    assert(await page.locator('#group-ratings').isHidden());
    await lookup(1);
    assert(await page.locator('#previous').isDisabled());
    await lookup(500);
    assert(await page.locator('#next').isDisabled());

    await page.goto(url('visualizer/arunkumar.html?experiment=2&index=509'));
    await page.waitForSelector('#workspace:visible');
    assert.equal(await page.locator('#index').inputValue(), '509');
    await page.reload();
    await page.waitForSelector('#workspace:visible');
    assert.equal(await page.locator('#index').inputValue(), '509');
    await imageLoaded();
    for (const width of [390, 720, 900]) {
      await page.setViewportSize({width, height: 844});
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
        `Explorer overflows at width ${width}`);
    }
    await page.setViewportSize({width: 1440, height: 1000});
    await page.screenshot({path: '/tmp/tribe-arunkumar-explorer.png', fullPage: true});
    await page.goto(url('visualizer/'));
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.setViewportSize({width: 1440, height: 1000});
    await page.screenshot({path: '/tmp/tribe-portal.png', fullPage: true});
    assert.deepEqual(errors, []);
    process.stdout.write('Dataset explorers: portal, annotations, histogram, experiments, images, keyboard, URLs and mobile PASS\n');
  } finally {
    await browser.close();
  }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
