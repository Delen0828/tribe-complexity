"""Shared report discovery, routing, and partial-run coverage checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from dataset import ATTRIBUTES, export_report, report_rows
from web import build_site


class WebTests(unittest.TestCase):
    def test_catalog_routes_and_legacy_redirects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('massvis_sample', 'custom run'):
                folder = root / name
                folder.mkdir()
                (folder / 'explorer.json').write_text(json.dumps({'count': 4, 'complete': name != 'custom run'}))
                (folder / 'brain.js').write_text('old duplicated viewer')
            (root / 'spectrum').mkdir()
            (root / 'spectrum/spectrum.png').touch()
            (root / 'unrendered').mkdir()
            (root / 'comparison.png').touch()
            hub = build_site(root)
            catalog = json.loads((hub.parent / 'reports.json').read_text())
            self.assertEqual([r['id'] for r in catalog], ['custom run', 'massvis_sample'])
            self.assertEqual(catalog[0]['data'], '../custom%20run/explorer.json')
            self.assertIn('partial', catalog[0]['label'])
            self.assertIn('spectrum-spectrum.html', hub.read_text())
            self.assertIn('comparison.html', hub.read_text())
            self.assertIn('unrendered', hub.read_text())
            self.assertIn('data-catalog="reports.json"', (hub.parent / 'brain.html').read_text())
            self.assertIn('../web/brain.html?dataset=custom%20run', (root / 'custom run/index.html').read_text())
            self.assertFalse((root / 'massvis_sample/brain.js').exists())
            self.assertEqual(build_site(root).read_text(), hub.read_text())

    def test_partial_requires_opt_in_and_records_exact_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            output.mkdir()
            rows = [dict(index=i, attributes={key: i for key in ATTRIBUTES}) for i in (0, 1)]
            for row in rows:
                row['attributes']['category'] = 'S' if row['index'] else 'G'
            with self.assertRaisesRegex(ValueError, 'No completed'):
                report_rows(rows, output, True)
            (output / 'prediction_0.npz').touch()
            with self.assertRaisesRegex(ValueError, '0/2'):
                report_rows(rows, output)
            (output / 'prediction_0.json').touch()
            with self.assertRaisesRegex(ValueError, '1/2'):
                report_rows(rows, output)
            selected = report_rows(rows, output, True)
            self.assertEqual(selected, rows[:1])
            selection = dict(rows=rows, scope='all', seed=0, eligible_count=2, missing_indices=[])
            export_report(selected, np.array([[1., 2.]]), output, selection, {}, render=False)
            report = json.loads((output / 'explorer.json').read_text())
            self.assertFalse(report['complete'])
            self.assertEqual(report['pending_indices'], [1])
            self.assertEqual(report['selected_count'], 2)
            self.assertEqual(report['count'], 1)
            attributes = {a['key']: a for a in report['attributes']}
            self.assertEqual(set(attributes), set(ATTRIBUTES))
            self.assertEqual(attributes['chart_types']['count_labels'], ['chart type', 'chart types'])
            self.assertEqual(attributes['titles']['kind'], 'binary')
            self.assertEqual(attributes['category']['kind'], 'category')
            self.assertEqual(next(g['label'] for g in report['groups'] if g['attribute'] == 'category'), 'Government')
            with np.load(output / 'aggregate_maps.npz') as saved:
                np.testing.assert_array_equal(saved['grand_mean'], [1., 2.])
                np.testing.assert_array_equal(saved['contrasts'], np.zeros((len(ATTRIBUTES), 2)))


if __name__ == '__main__':
    unittest.main()
