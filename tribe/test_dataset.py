"""Sampling, aggregation, provenance, and runtime regression tests."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from dataset import (ATTRIBUTES, BINARY_ATTRIBUTES, ROOT, aggregate_maps, compatible_selection,
                     estimate_runtime, load_predictions, load_stimuli, select_stimuli)
from preprocessing import PROTOCOL


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.missing = load_stimuli(ROOT.parent / 'label/output/labels.csv')

    def test_sample_and_all(self):
        sample = select_stimuli(self.rows, 'sample')
        self.assertEqual(len(sample), 100)
        self.assertEqual(len({r['index'] for r in sample}), 100)
        self.assertEqual([sum(r['bin'] == b for r in sample) for b in range(10)], [10]*10)
        self.assertEqual(sample, select_stimuli(self.rows, 'sample', seed=0))
        self.assertNotEqual(sample, select_stimuli(self.rows, 'sample', seed=1))
        self.assertEqual(select_stimuli(self.rows, 'all'), self.rows)
        self.assertEqual(len(self.rows), 5800)
        self.assertFalse(self.missing)
        self.assertEqual(len(ATTRIBUTES), 31)

    def test_all_published_feature_columns_are_available(self):
        with (ROOT.parent / 'label/output/labels.csv').open(newline='') as stream:
            published = list(csv.DictReader(stream))
        columns = {key for key in published[0] if key.split('.')[0] in ('text', 'color', 'data', 'design')}
        columns.update(['mean_human_complexity_rating', 'category'])
        self.assertEqual({column for _, column in ATTRIBUTES.values()}, columns)
        for row, raw in zip(self.rows, published):
            self.assertEqual(set(row['attributes']), set(ATTRIBUTES))
            for key, (_, column) in ATTRIBUTES.items():
                if key == 'complexity':
                    continue
                expected = raw[column] if key == 'category' else int(raw[column])
                self.assertEqual(row['attributes'][key], expected)

    def test_insufficient_bin(self):
        with self.assertRaisesRegex(ValueError, 'need 10'):
            select_stimuli(self.rows[:10], 'sample')
        with self.assertRaises(ValueError):
            select_stimuli(self.rows, 'sample', per_bin=0)

    def test_bins_and_invalid_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'labels.csv'
            source = self.rows[0]
            columns = ['index', 'filename', 'image_path', 'mean_human_complexity_rating'] + [v[1] for k,v in ATTRIBUTES.items() if k != 'complexity']
            def write(scores, **overrides):
                with path.open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=columns)
                    writer.writeheader()
                    for i, score in enumerate(scores):
                        row = dict.fromkeys(columns, 0)
                        row.update(index=i, filename=source['filename'], image_path=source['image_path'], mean_human_complexity_rating=score, category='G')
                        row.update(overrides)
                        writer.writerow(row)
            write([0, 9.99, 10, 89.99, 90, 100])
            rows, _ = load_stimuli(path)
            self.assertEqual([r['bin'] for r in rows], [0,0,1,8,9,9])
            for score in ['nan', -1, 101]:
                write([score])
                with self.assertRaises(ValueError):
                    load_stimuli(path)
            for column, value in [('design.chart_types_count', -1), ('data.quantitative', 1.5),
                                  ('color.color_count', 'nan'), ('text.titles', 2),
                                  ('design.multi_panel', -1), ('category', 'unknown')]:
                write([50], **{column: value})
                with self.assertRaisesRegex(ValueError, f'Invalid {column}'):
                    load_stimuli(path)

    def test_exact_group_means_and_contrasts(self):
        rows = [dict(index=i, attributes={key: int(i > 0) for key in ATTRIBUTES}) for i in range(3)]
        for row in rows:
            row['attributes']['category'] = 'S' if row['index'] else 'G'
        maps = np.array([[0, 3], [6, 9], [12, 15]], dtype=float)
        groups, grand = aggregate_maps(rows, maps)
        np.testing.assert_array_equal(grand, [6,9])
        for key in ATTRIBUTES:
            selected = [g for g in groups if g['attribute'] == key]
            self.assertEqual(selected[0]['indices'], [0])
            self.assertEqual(selected[1]['indices'], [1,2])
            np.testing.assert_array_equal(selected[1]['mean'], [9,12])
            np.testing.assert_array_equal(selected[1]['contrast'], [3,3])
            np.testing.assert_allclose(sum(g['mean']*g['count'] for g in selected)/3, grand)
            np.testing.assert_allclose(sum(g['contrast']*g['count'] for g in selected), [0,0])
            if key in BINARY_ATTRIBUTES:
                self.assertEqual([g['label'] for g in selected], ['No', 'Yes'])
            elif key == 'category':
                self.assertEqual([g['value'] for g in selected], ['G', 'S'])
                self.assertEqual([g['label'] for g in selected], ['Government', 'Science'])
        with self.assertRaises(ValueError):
            aggregate_maps(rows, [[float('nan')]]*3)

    def test_sample_and_full_feature_groups_partition_stimuli(self):
        for scope in ('sample', 'all'):
            rows = select_stimuli(self.rows, scope)
            maps = np.arange(len(rows)*2, dtype=float).reshape(len(rows), 2)
            groups, grand = aggregate_maps(rows, maps)
            self.assertEqual({g['attribute'] for g in groups}, set(ATTRIBUTES))
            for attribute in ATTRIBUTES:
                selected = [g for g in groups if g['attribute'] == attribute]
                self.assertEqual(sorted(i for g in selected for i in g['indices']), sorted(r['index'] for r in rows))
                np.testing.assert_allclose(sum(g['mean']*g['count'] for g in selected)/len(rows), grand)
                np.testing.assert_allclose(sum(g['contrast']*g['count'] for g in selected), [0, 0], atol=1e-7)

    def test_existing_selection_accepts_only_added_annotations(self):
        current = dict(scope='sample', seed=0, protocol=PROTOCOL, background_rgb=[255]*3,
                       labels_sha256='unchanged-labels', rows=copy.deepcopy(self.rows[:2]))
        for row in current['rows']:
            row['sha256'] = 'source-hash'
        previous = copy.deepcopy(current)
        for row in previous['rows']:
            row['attributes'] = {key: row['attributes'][key] for key in
                                 ('complexity', 'charts', 'colors', 'quantitative', 'categorical')}
        self.assertTrue(compatible_selection(previous, current))
        self.assertTrue(compatible_selection(current, current))
        for key, value in [('labels_sha256', 'changed'), ('seed', 1), ('protocol', 'changed'),
                           ('background_rgb', [0]*3)]:
            changed = copy.deepcopy(current)
            changed[key] = value
            self.assertFalse(compatible_selection(previous, changed))
        for key, value in [('index', -1), ('sha256', 'changed'), ('score', -1), ('bin', -1)]:
            changed = copy.deepcopy(current)
            changed['rows'][0][key] = value
            self.assertFalse(compatible_selection(previous, changed))
        changed = copy.deepcopy(current)
        changed['rows'][0]['attributes']['charts'] += 1
        self.assertFalse(compatible_selection(previous, changed))
        self.assertFalse(compatible_selection(current, previous))
        changed = copy.deepcopy(current)
        changed['rows'].reverse()
        self.assertFalse(compatible_selection(previous, changed))
        changed = copy.deepcopy(current)
        changed['precision'] = 'bf16'
        self.assertFalse(compatible_selection(previous, changed))

    def test_timing_excludes_cache_hits(self):
        receipts = [dict(encoded_windows=1, total_seconds=4, device='cuda'),
                    dict(encoded_windows=1, total_seconds=6, device='cuda'),
                    dict(encoded_windows=0, total_seconds=.1, device='cuda')]
        result = estimate_runtime(receipts, 5800, 10)
        self.assertEqual(result['measured_count'], 2)
        self.assertEqual(result['mean_seconds_per_stimulus'], 5)
        self.assertEqual(result['full_prediction_seconds'], 29010)
        self.assertEqual(estimate_runtime(receipts[-1:],5800,10)['measured_count'], 0)

    def test_prediction_provenance_and_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            row = dict(index=42, sha256='source-hash')
            prediction = path/'prediction_42.npz'
            np.savez_compressed(prediction, predictions=np.ones((3,20484)), times=[0,1,2])
            receipt = dict(identity=dict(protocol=PROTOCOL, source_sha256='source-hash',
                                        background_rgb=[255]*3, model='facebook/tribev2'),
                           prediction_sha256=hashlib.sha256(prediction.read_bytes()).hexdigest())
            (path/'prediction_42.json').write_text(json.dumps(receipt))
            maps, _ = load_predictions([row],path,[255]*3)
            self.assertEqual(maps.shape,(1,20484))
            with self.assertRaisesRegex(ValueError,'Stale or corrupted'):
                load_predictions([dict(index=42,sha256='changed')],path,[255]*3)
            prediction.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'Stale or corrupted'):
                load_predictions([row],path,[255]*3)


if __name__ == '__main__':
    unittest.main()
