"""Sampling, aggregation, provenance, and runtime regression tests."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from dataset import (ATTRIBUTES, ROOT, aggregate_maps, estimate_runtime, load_predictions,
                     load_stimuli, select_stimuli)
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
        self.assertEqual(len(ATTRIBUTES), 5)

    def test_insufficient_bin(self):
        with self.assertRaisesRegex(ValueError, 'need 10'):
            select_stimuli(self.rows[:10], 'sample')
        with self.assertRaises(ValueError):
            select_stimuli(self.rows, 'sample', per_bin=0)

    def test_bins_and_invalid_labels(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'labels.csv'
            source = self.rows[0]
            columns = ['index', 'filename', 'image_path', 'mean_human_complexity_rating'] + [v[1] for k,v in ATTRIBUTES.items() if k != 'complexity']
            def write(scores):
                with path.open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=columns)
                    writer.writeheader()
                    for i, score in enumerate(scores):
                        row = dict.fromkeys(columns, 0)
                        row.update(index=i, filename=source['filename'], image_path=source['image_path'], mean_human_complexity_rating=score)
                        writer.writerow(row)
            write([0, 9.99, 10, 89.99, 90, 100])
            rows, _ = load_stimuli(path)
            self.assertEqual([r['bin'] for r in rows], [0,0,1,8,9,9])
            for score in ['nan', -1, 101]:
                write([score])
                with self.assertRaises(ValueError):
                    load_stimuli(path)

    def test_exact_group_means_and_contrasts(self):
        rows = [dict(index=i, attributes={key: int(i > 0) for key in ATTRIBUTES}) for i in range(3)]
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
        with self.assertRaises(ValueError):
            aggregate_maps(rows, [[float('nan')]]*3)

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
