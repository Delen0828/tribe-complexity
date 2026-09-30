"""Browser smoke test. Requires Playwright Firefox and a server at --url."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000/visualizer/brain.html')
    parser.add_argument('--firefox', help='Optional installed Firefox executable')
    parser.add_argument('--screenshots', type=Path, default=Path('/tmp/massvis-browser-check'))
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        launch = dict(headless=True)
        if args.firefox:
            launch['executable_path'] = args.firefox
        browser = p.firefox.launch(**launch)
        page = browser.new_page(viewport=dict(width=1440, height=1100))
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_selector('#brain-workspace:visible')
        assert page.locator('#attribute option').count() == 5
        assert '100 predictions loaded' in page.locator('#brain-status').inner_text()
        assert page.locator('#groups button').count() == 10
        for attribute in ['complexity','charts','colors','quantitative','categorical']:
            page.select_option('#attribute',attribute)
            buttons = page.locator('#groups button')
            assert buttons.count() > 0
            for position in (0,buttons.count()-1):
                buttons.nth(position).click()
                assert buttons.nth(position).get_attribute('aria-pressed') == 'true'
                count = int(page.locator('#group-count').inner_text().split('=')[1])
                assert page.locator('#members a').count() == min(count,24)
                for statistic in ['mean','contrast']:
                    page.select_option('#statistic',statistic)
                    page.wait_for_function('''() => {
                        const img = document.getElementById('brain-map');
                        return img.complete && img.naturalWidth > 1000 && !img.hidden;
                    }''')
                    assert f'_{statistic}.png' in page.locator('#brain-map').get_attribute('src')
                    assert page.locator('#map-error').is_hidden()
        # Verify pagination with the largest chart-count group.
        page.select_option('#attribute','charts')
        buttons=page.locator('#groups button')
        counts=[int(text.split('n=')[1]) for text in buttons.all_inner_texts()]
        buttons.nth(counts.index(max(counts))).click()
        if max(counts)>24:
            page.locator('#more-members').click()
            assert page.locator('#members a').count()==min(max(counts),48)
        page.select_option('#attribute','complexity')
        page.select_option('#statistic','mean')
        page.wait_for_function("document.getElementById('brain-map').complete")
        page.screenshot(path=str(args.screenshots/'desktop.png'),full_page=True)
        # Refresh restores the selected grouping and statistic from the URL.
        page.select_option('#attribute','categorical')
        page.select_option('#statistic','contrast')
        selected=page.url
        page.reload()
        page.wait_for_selector('#brain-workspace:visible')
        assert page.locator('#attribute').input_value()=='categorical'
        assert page.locator('#statistic').input_value()=='contrast'
        assert page.url==selected
        # Unavailable full results must clear the previous report.
        page.select_option('#dataset','massvis_all')
        page.wait_for_function("document.getElementById('brain-status').textContent.includes('Whole-dataset results are unavailable')")
        assert page.locator('#brain-workspace').is_hidden()
        assert page.locator('#attribute').is_disabled()
        page.select_option('#dataset','massvis_sample')
        page.wait_for_selector('#brain-workspace:visible')
        page.set_viewport_size(dict(width=390,height=844))
        page.wait_for_function("document.getElementById('brain-map').complete")
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(args.screenshots/'mobile.png'),full_page=True)
        assert not errors, errors
        # Existing annotation viewer and its new entry point still function.
        page.goto(args.url.replace('brain.html','?index=42'))
        page.wait_for_selector('#workspace:visible')
        assert page.locator('a[href="brain.html"]').count()==1
        assert page.locator('#index').input_value()=='42'
        browser.close()
    print(json.dumps(dict(status='PASS', attributes=5, map_modes=2,
                          checks=['loaded cortical images','member counts and pagination',
                                  'URL restoration','unavailable full-data state',
                                  'mobile overflow','annotation viewer navigation'],
                          screenshots=str(args.screenshots)), indent=2))


if __name__ == '__main__':
    main()
