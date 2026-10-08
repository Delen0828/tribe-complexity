"""Rendered layout coverage, bilateral inferior maps, and report metadata."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from dataset import ATTRIBUTES, export_report, render_groups, view_metadata


class CorticalViewTests(unittest.TestCase):
    def test_rendered_layouts_and_both_inferior_hemispheres(self):
        # Opposite signs make a missing or duplicated hemisphere detectable.
        values = np.concatenate([np.full(10242, -1.), np.full(10242, 1.)])
        for layout in ('four', 'inferior', 'all'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                group = dict(attribute='complexity', value=0, label='0–10',
                             mean=values, contrast=values * .5)
                limits = render_groups([group], output, views=layout)
                self.assertEqual(limits, dict(mean=1., contrast=.5))
                for kind in ('mean', 'contrast'):
                    assets = {group[f'{kind}_{view["image_suffix"]}']: view['panels']
                              for view in view_metadata(layout)}
                    self.assertEqual(len(assets), 2 if layout == 'all' else 1)
                    for path, panels in assets.items():
                        with Image.open(output / path) as image:
                            self.assertEqual(image.size, (330 * panels, 374))
                            if panels == 1:
                                # Inspect only the brain region, above the color bar.
                                rgb = np.asarray(image.convert('RGB'))[:270].astype(int)
                                red = rgb[:, :, 0] - rgb[:, :, 2] > 60
                                blue = rgb[:, :, 2] - rgb[:, :, 0] > 60
                                self.assertGreater(np.count_nonzero(red), 100)
                                self.assertGreater(np.count_nonzero(blue), 100)
                                # Inferior orientation puts anatomical right
                                # (positive here) on the displayed left.
                                self.assertLess(np.where(red)[1].mean(), np.where(blue)[1].mean())
                self.assertEqual('mean_image' in group, layout != 'inferior')
                self.assertEqual('mean_inferior_image' in group, layout != 'four')

    def test_report_layout_keeps_aggregates_and_selection_independent(self):
        rows = [dict(index=0, attributes={key: 0 for key in ATTRIBUTES})]
        rows[0]['attributes']['category'] = 'G'
        selection = dict(rows=rows, scope='all', seed=0, eligible_count=1, missing_indices=[])
        for layout, count in [('four', 4), ('inferior', 1), ('all', 5)]:
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'run'
                output.mkdir()
                export_report(rows, np.array([[1., 2.]]), output, selection, {}, render=False, views=layout)
                report = json.loads((output / 'explorer.json').read_text())
                self.assertEqual(report['view_layout'], layout)
                self.assertEqual(len(report['views']), count)
                self.assertEqual(report['views'], view_metadata(layout))
                self.assertTrue(report['complete'])
                with np.load(output / 'aggregate_maps.npz') as saved:
                    np.testing.assert_array_equal(saved['grand_mean'], [1., 2.])
        with self.assertRaisesRegex(ValueError, 'Unknown view layout'):
            view_metadata('invalid')


if __name__ == '__main__':
    unittest.main()
