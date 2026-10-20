"""Thesis study membership, provenance, and existing-pipeline aggregation."""
from collections import Counter
from copy import deepcopy
import hashlib
import unittest

import numpy as np

from dataset import compatible_selection
from studies import REPOSITORY, prepare_manifest, study_variables
from studies.ashcraft import CORRELATION_CLASS, correlation_class
from visualizer.analysis import aggregate_xy
from visualizer.server import Visualizer


class AshcraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = prepare_manifest('ashcraft')

    def test_authoritative_pilot4_has_actual_joint_counts_and_no_invented_ratings(self):
        rows = self.manifest['rows']
        self.assertEqual([r['index'] for r in rows], list(range(1, 73)))
        counts = Counter((r['attributes']['correlation_direction'], r['attributes']['has_outlier']) for r in rows)
        self.assertEqual(counts, {('positive', 1): 24, ('positive', 0): 12,
                                  ('negative', 1): 12, ('negative', 0): 24})
        self.assertEqual({a['key'] for a in self.manifest['attributes']},
                         {'correlation_direction', 'has_outlier', 'correlation_class'})
        for row in rows:
            self.assertNotIn('score', row)
            self.assertEqual(hashlib.sha256((REPOSITORY / row['source']).read_bytes()).hexdigest(), row['sha256'])
            self.assertEqual((REPOSITORY / row['source']).read_bytes(), (REPOSITORY / row['original_source']).read_bytes())
        z = [r['outlier_z'] for r in rows if r['attributes']['has_outlier']]
        self.assertEqual((min(z), max(z)), (3.042, 7.968))

    def test_actual_memberships_use_the_existing_equal_image_contrast(self):
        rows = self.manifest['rows']
        maps = np.arange(72 * 4).reshape(72, 4)
        grid = aggregate_xy(rows, maps, self.manifest['attributes'],
                            'correlation_direction', 'has_outlier', 'contrast')
        self.assertEqual([c['count'] for c in grid['cells']], [24, 12, 12, 24])
        np.testing.assert_allclose(sum(c['values'] * c['count'] for c in grid['cells']), 0, atol=1e-10)
        by_index = {r['index']: i for i, r in enumerate(rows)}
        for cell in grid['cells']:
            np.testing.assert_allclose(cell['values'], maps[[by_index[i] for i in cell['indices']]].mean(0) - maps.mean(0))
        self.assertIn('ashcraft', {r['id'] for r in Visualizer().catalog()})

    def test_absolute_correlation_strength_and_boundaries(self):
        for value, expected in [(0, 'low'), (.299999, 'low'), (.3, 'mid'),
                                (.699999, 'mid'), (.7, 'high'), (1, 'high')]:
            for sign in (-1, 1):
                self.assertEqual(correlation_class(sign * value), expected)
        for value in (float('nan'), float('inf'), -1.001, 1.001):
            with self.assertRaises(ValueError):
                correlation_class(value)

    def test_legacy_saved_classes_can_resume_without_changing_prediction_identity(self):
        current = dict(dataset='ashcraft', protocol='unchanged',
                       rows=deepcopy(self.manifest['rows']), attributes=self.manifest['attributes'])
        previous = deepcopy(current)
        old_labels = {'negSrngCor': 'Strong negative', 'negCor': 'Negative',
                      'posCor': 'Positive', 'posSrngCor': 'Strong positive'}
        for row in previous['rows']:
            r = row['published_r']
            row['attributes']['correlation_class'] = (
                ('pos' if r > 0 else 'neg') + ('SrngCor' if abs(r) >= .7 else 'Cor'))
        previous['attributes'][-1]['values'] = old_labels
        before = deepcopy(previous)
        rows, definitions = study_variables('ashcraft', previous['rows'], previous['attributes'])
        self.assertEqual(rows, current['rows'])
        self.assertEqual(definitions[-1], CORRELATION_CLASS)
        self.assertEqual(previous, before)
        self.assertTrue(compatible_selection(previous, current))
        for field in ('sha256', 'published_r'):
            changed = deepcopy(current)
            changed['rows'][0][field] = 'changed' if field == 'sha256' else .5
            self.assertFalse(compatible_selection(previous, changed))
        self.assertFalse(compatible_selection(previous, dict(current, protocol='changed')))

    def test_strength_groups_work_on_either_axis_with_direction_separate(self):
        rows = self.manifest['rows']
        maps = np.arange(72 * 4).reshape(72, 4)
        by_index = {r['index']: r for r in rows}
        for x, y, axis in [('correlation_class', 'correlation_direction', 'x'),
                           ('correlation_direction', 'correlation_class', 'y')]:
            grid = aggregate_xy(rows, maps, self.manifest['attributes'], x, y, 'contrast')
            self.assertEqual([g['label'] for g in grid['axes'][axis]], ['Low', 'Mid', 'High'])
            self.assertEqual(sum(c['count'] for c in grid['cells']), 72)
            np.testing.assert_allclose(sum(c['values'] * c['count'] for c in grid['cells']), 0, atol=1e-10)
            for group in grid['axes'][axis]:
                self.assertEqual({by_index[i]['attributes']['correlation_direction'] for i in group['indices']},
                                 {'positive', 'negative'})
                for index in group['indices']:
                    magnitude = abs(by_index[index]['published_r'])
                    self.assertTrue(magnitude < .3 if group['value'] == 'low' else
                                    .3 <= magnitude < .7 if group['value'] == 'mid' else magnitude >= .7)


if __name__ == '__main__':
    unittest.main()
