"""Browser smoke test. Requires Playwright Firefox and a server at --url."""
import argparse
import json
from pathlib import Path
from urllib.parse import urljoin
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
        report = page.evaluate('report')
        attributes = [a['key'] for a in report['attributes']]
        assert page.locator('#attribute option').count() == len(attributes) == 31
        assert f"{report['count']:,} predictions loaded" in page.locator('#brain-status').inner_text()
        assert page.locator('#groups button').count() == 10
        layouts = page.locator('#views option').evaluate_all('options => options.map(option => option.value)')
        assert page.locator('#views').input_value() == report.get('view_layout', 'four')
        baseline_layout = 'four' if 'four' in layouts else 'inferior'
        baseline_count = 4 if baseline_layout == 'four' else 1
        page.select_option('#views', baseline_layout)

        def loaded_maps():
            # Load off-screen columns too, so every rendered group is checked.
            page.locator('#groups .group-map').evaluate_all("images => images.forEach(image => image.loading = 'eager')")
            page.wait_for_function('''() => [...document.querySelectorAll('#groups .group-map')].every(
                image => image.complete && image.naturalWidth >= 300 * parseFloat(image.style.width) / 100 && !image.hidden)''')
            assert page.locator('#groups .map-error:visible').count() == 0

        for attribute in attributes:
            page.select_option('#attribute',attribute)
            groups = [g for g in report['groups'] if g['attribute'] == attribute]
            buttons = page.locator('#groups button')
            assert buttons.count() == len(groups) > 0
            assert page.locator('#groups tbody tr').count() == baseline_count
            assert page.locator('#groups .group-map').count() == len(groups)*baseline_count
            for position, group in enumerate(groups):
                icons = page.locator('#groups thead .sample-size-icons').nth(position)
                assert icons.locator('.sample-square').count() == group['count']
                if attribute == 'category' or attribute in ('titles', 'multi_panel', 'bar'):
                    assert buttons.nth(position).inner_text() == group['label']
            for position in sorted({0,buttons.count()-1}):
                buttons.nth(position).click()
                assert buttons.nth(position).get_attribute('aria-pressed') == 'true'
                assert page.locator('#groups button[aria-pressed="true"]').count() == 1
                count = groups[position]['count']
                assert page.locator('#members a').count() == min(count,24)
                assert f'{count} stimuli shown' in page.locator('#member-range').inner_text()
            for statistic in ['mean','contrast']:
                page.select_option('#statistic',statistic)
                for layout in layouts:
                    page.select_option('#views', layout)
                    count = {'four': 4, 'inferior': 1, 'all': 5}[layout]
                    assert page.locator('#groups tbody tr').count() == count
                    assert page.locator('#groups .group-map').count() == len(groups) * count
                    assert page.locator('#group-count').inner_text().endswith(f"{count} {'view' if count == 1 else 'views'}")
                    loaded_maps()
                    for source in page.locator('#groups .group-map').evaluate_all('images => images.map(image => image.src)'):
                        assert source.endswith((f'_{statistic}.png', f'_{statistic}_inferior.png'))
                        if layout == 'inferior':
                            assert source.endswith(f'_{statistic}_inferior.png')
                    if layout != 'four':
                        bottom = page.locator('#groups tbody tr').last
                        assert 'both hemispheres' in bottom.locator('th').inner_text()
                        assert bottom.locator('.group-map').first.evaluate("image => image.style.width === '100%' && image.style.left === '0%'")
                page.select_option('#views', baseline_layout)
        # Verify pagination with the largest chart-count group.
        page.select_option('#attribute','charts')
        buttons=page.locator('#groups button')
        counts=[g['count'] for g in report['groups'] if g['attribute'] == 'charts']
        buttons.nth(counts.index(max(counts))).click()
        if max(counts)>24:
            page.locator('#more-members').click()
            assert page.locator('#members a').count()==min(max(counts),48)
        page.select_option('#attribute','complexity')
        page.select_option('#statistic','mean')
        loaded_maps()
        page.locator('.brain-grid-scroll').evaluate('element => element.scrollLeft = 0')
        page.screenshot(path=str(args.screenshots/'desktop.png'),full_page=True)
        if 'inferior' in layouts:
            page.select_option('#views', 'inferior')
            loaded_maps()
            page.screenshot(path=str(args.screenshots/'inferior.png'),full_page=True)
        # Refresh restores a string-valued category group and the map mode.
        page.select_option('#attribute','category')
        page.locator('#groups button').last.click()
        category = page.locator('#groups button[aria-pressed="true"]').get_attribute('data-value')
        page.select_option('#statistic','contrast')
        selected=page.url
        page.reload()
        page.wait_for_selector('#brain-workspace:visible')
        assert page.locator('#attribute').input_value()=='category'
        assert page.locator('#statistic').input_value()=='contrast'
        assert page.locator('#views').input_value()==('inferior' if 'inferior' in layouts else 'four')
        assert page.locator('#groups button[aria-pressed="true"]').get_attribute('data-value')==category
        assert page.url==selected
        # A report offers only the assets it rendered; legacy reports still
        # crop their standard four-view strips when metadata is absent.
        report_url = page.evaluate('baseURL.href')
        variants = [layout for layout in layouts if layout != 'all']
        if 'four' in layouts:
            variants.append('legacy')
        for layout in variants:
            variant = dict(report)
            if layout == 'legacy':
                variant.pop('views', None)
                variant.pop('view_layout', None)
            else:
                variant['view_layout'] = layout
                variant['views'] = [view for view in report.get('views', [])
                                    if (view['key'] == 'inferior') == (layout == 'inferior')]
                if not variant['views']:
                    variant.pop('views')
            def serve_variant(route):
                route.fulfill(json=variant)
            page.route(report_url, serve_variant)
            page.reload()
            page.wait_for_selector('#brain-workspace:visible')
            expected = 'four' if layout == 'legacy' else layout
            assert page.locator('#views option').evaluate_all('options => options.map(option => option.value)') == [expected]
            assert page.locator('#views').input_value() == expected
            assert page.locator('#groups tbody tr').count() == (1 if expected == 'inferior' else 4)
            loaded_maps()
            page.unroute(report_url, serve_variant)
        page.reload()
        page.wait_for_selector('#brain-workspace:visible')
        # Simulate unavailable results even when a full/partial report exists.
        page.route('**/massvis_all/explorer.json', lambda route: route.fulfill(status=404, body='Unavailable'))
        page.select_option('#dataset','massvis_all')
        page.wait_for_function("document.getElementById('brain-status').textContent.includes('This report is unavailable')")
        assert page.locator('#brain-workspace').is_hidden()
        assert page.locator('#attribute').is_disabled()
        assert page.locator('#views').is_disabled()
        page.select_option('#dataset','massvis_sample')
        page.wait_for_selector('#brain-workspace:visible')
        page.set_viewport_size(dict(width=390,height=844))
        loaded_maps()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(args.screenshots/'mobile.png'),full_page=True)
        assert not errors, errors
        # Existing annotation viewer and its new entry point still function.
        page.goto(urljoin(args.url, './?index=42'))
        page.wait_for_selector('#workspace:visible')
        assert page.locator('a[href="brain.html"]').count()==1
        assert page.locator('#index').input_value()=='42'
        browser.close()
    print(json.dumps(dict(status='PASS', attributes=len(attributes), map_modes=2, view_layouts=layouts,
                          checks=['loaded cortical images','member counts and pagination',
                                  'URL restoration','unavailable full-data state',
                                  'mobile overflow','annotation viewer navigation'],
                          screenshots=str(args.screenshots)), indent=2))


if __name__ == '__main__':
    main()
