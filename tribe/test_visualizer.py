"""Dynamic X/Y aggregation, raw prediction discovery, and simulator separation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from preprocessing import PROTOCOL
from gpu_encoder import prediction_identity
from studies import REPOSITORY, HAYEONG_FEATURES
from study import run_study
import dataset
from visualizer.analysis import aggregate_xy, complexity_tertiles
from visualizer.server import Visualizer


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.attributes = [dict(key='complexity', kind='bin'), dict(key='color', kind='binary')]
        self.rows = [dict(index=i, attributes=dict(complexity=c, color=b))
                     for i, (c, b) in enumerate([(0, 0), (0, 1), (0, 1), (9, 1)])]
        self.maps = np.array([[0, 2], [6, 8], [12, 14], [30, 32]])

    def test_intersections_use_equal_stimulus_means_and_shared_scale(self):
        grid = aggregate_xy(self.rows, self.maps, self.attributes, 'complexity', 'color')
        self.assertEqual(len(grid['cells']), 20)
        cell = next(c for c in grid['cells'] if c['x'] == 0 and c['y'] == 1)
        self.assertEqual(cell['indices'], [1, 2])
        np.testing.assert_array_equal(cell['values'], [9, 11])
        self.assertEqual(grid['limit'], 32)
        self.assertIsNone(next(c for c in grid['cells'] if c['x'] == 9 and c['y'] == 0)['values'])
        self.assertEqual(grid['axes']['x'][-1]['label'], '90–100')
        self.assertEqual(grid['axes']['x'][0]['indices'], [0, 1, 2])
        self.assertEqual(grid['axes']['y'][1]['indices'], [1, 2, 3])

    def test_contrast_baseline_weights_stimuli_rather_than_cells(self):
        grid = aggregate_xy(self.rows, self.maps, self.attributes, 'complexity', 'color', 'contrast')
        cell = next(c for c in grid['cells'] if c['x'] == 0 and c['y'] == 1)
        np.testing.assert_array_equal(cell['values'], [9, 11] - self.maps.mean(axis=0))
        np.testing.assert_allclose(sum(c['values'] * c['count'] for c in grid['cells'] if c['count']), 0)
        self.assertEqual(grid['limit'], 18)

    def test_unassigned_y_same_variable_and_invalid_selection(self):
        grid = aggregate_xy(self.rows, self.maps, self.attributes, 'complexity')
        self.assertEqual(len(grid['axes']['y']), 1)
        np.testing.assert_array_equal(grid['cells'][0]['values'], [6, 8])
        same = aggregate_xy(self.rows, self.maps, self.attributes, 'color', 'color')
        self.assertEqual([c['count'] for c in same['cells']], [1, 0, 0, 3])
        for args in [('unknown', '', 'mean'), ('color', 'unknown', 'mean'), ('color', '', 'bad')]:
            with self.assertRaises(ValueError):
                aggregate_xy(self.rows, self.maps, self.attributes, *args)


class ComplexityTertileTests(unittest.TestCase):
    def setUp(self):
        self.attributes = [dict(key='complexity', label='Perceived complexity', kind='bin')]
        self.rows = [dict(index=index, score=score, attributes={'complexity': min(int(score // 10), 9)})
                     for index, score in enumerate([9, 5, 5, 10, 7, 100, 0, 80, 13])]
        self.maps = np.arange(18).reshape(9, 2)

    def test_thirds_rank_original_scores_and_aggregate_on_either_axis(self):
        rows, definitions = complexity_tertiles(self.rows, self.attributes, self.rows)
        for x, y, axis in [('complexity', '', 'x'), ('complexity', 'complexity', 'y')]:
            grid = aggregate_xy(rows, self.maps, definitions, x, y)
            self.assertEqual([g['label'] for g in grid['axes'][axis]], ['Low', 'Mid', 'High'])
            self.assertEqual([g['indices'] for g in grid['axes'][axis]], [[1, 2, 6], [0, 3, 4], [5, 7, 8]])
            self.assertEqual(sum(c['count'] for c in grid['cells']), 9)
            for cell in grid['cells']:
                if cell['count']:
                    np.testing.assert_array_equal(cell['values'], self.maps[cell['indices']].mean(axis=0))
        self.assertEqual(self.attributes[0]['kind'], 'bin')
        self.assertEqual(self.rows[0]['attributes']['complexity'], 0)

    def test_partial_predictions_keep_membership_and_all_three_groups(self):
        rows, definitions = complexity_tertiles(self.rows[:2], self.attributes, self.rows)
        self.assertEqual([r['attributes']['complexity'] for r in rows], ['mid', 'low'])
        grid = aggregate_xy(rows, self.maps[:2], definitions, 'complexity')
        self.assertEqual([g['label'] for g in grid['axes']['x']], ['Low', 'Mid', 'High'])
        self.assertEqual([c['count'] for c in grid['cells']], [1, 1, 0])

    def test_ties_are_deterministic_and_group_sizes_differ_by_at_most_one(self):
        for count in (1, 2, 6, 100):
            originals = [dict(index=index, score=50, attributes={'complexity': 5}) for index in range(count)]
            rows, definitions = complexity_tertiles(originals, self.attributes, list(reversed(originals)))
            grid = aggregate_xy(rows, np.ones((count, 2)), definitions, 'complexity')
            counts = [c['count'] for c in grid['cells']]
            self.assertLessEqual(max(counts) - min(counts), 1)
            ordered = [index for group in grid['axes']['x'] for index in group['indices']]
            self.assertEqual(ordered, list(range(count)))


class PredictionViewerTests(unittest.TestCase):
    def test_saved_massvis_selections_gain_stable_thirds_during_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'tribe/outputs/massvis_sample'
            folder.mkdir(parents=True)
            rows = [dict(index=index, score=index, filename=f'{index}.png', sha256=f'source-{index}',
                         attributes={'complexity': 0}) for index in range(9)]
            selection = dict(rows=rows, background_rgb=[255] * 3,
                             attributes=[dict(key='complexity', label='Perceived complexity', kind='bin')])
            path = folder / 'selection.json'
            path.write_text(json.dumps(selection))
            original = path.read_bytes()
            viewer = Visualizer(tmp)
            for index in (4, 0):
                prediction = folder / f'prediction_{index}.npz'
                np.savez_compressed(prediction, predictions=np.full((3, 20484), index + 1), times=[0, 1, 2])
                receipt = dict(identity=prediction_identity(PROTOCOL, rows[index]['sha256'], [255] * 3),
                               prediction_sha256=hashlib.sha256(prediction.read_bytes()).hexdigest())
                prediction.with_suffix('.json').write_text(json.dumps(receipt))
                report, _ = viewer.load('massvis_sample')
                self.assertEqual(report['attributes'][0]['kind'], 'tertile')
                middle = next(row for row in report['rows'] if row['index'] == 4)
                self.assertEqual(middle['attributes']['complexity'], 'mid')
                grid = viewer.analysis('massvis_sample', '', 'complexity', '', 'mean')
                self.assertEqual([g['label'] for g in grid['axes']['x']], ['Low', 'Mid', 'High'])
                self.assertEqual(next(c for c in grid['cells'] if c['x'] == 'mid')['indices'], [4])
            custom, _ = viewer.load('', '/tribe/outputs/massvis_sample/selection.json')
            self.assertEqual(custom['attributes'][0]['kind'], 'tertile')
            self.assertEqual(path.read_bytes(), original)

    def test_existing_study_predictions_gain_variables_without_rewriting_selection(self):
        for name, expected_keys in [('kylie', ['complexity']),
                                    ('hayeong', [a['key'] for a in HAYEONG_FEATURES])]:
            with self.subTest(dataset=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder = root / 'tribe/outputs' / name
                folder.mkdir(parents=True)
                manifest = json.loads((REPOSITORY / f'tribe/studies/manifests/{name}.json').read_text())
                originals = [manifest['rows'][0]]
                if name == 'hayeong':
                    originals.append(next(r for r in manifest['rows'] if r['attributes']['experiment'] == 2))
                rows = [dict(row, attributes={k: v for k, v in row['attributes'].items() if k not in expected_keys})
                        for row in originals]
                selection = dict(dataset=name, rows=rows, background_rgb=[255] * 3,
                                 attributes=[a for a in manifest['attributes'] if a['key'] not in expected_keys])
                path = folder / 'selection.json'
                path.write_text(json.dumps(selection))
                original = path.read_bytes()
                for row in rows:
                    prediction = folder / f'prediction_{row["index"]}.npz'
                    np.savez_compressed(prediction, predictions=np.full((3, 20484), row['index']), times=[0, 1, 2])
                    receipt = dict(identity=prediction_identity(PROTOCOL, row['sha256'], [255] * 3),
                                   prediction_sha256=hashlib.sha256(prediction.read_bytes()).hexdigest())
                    prediction.with_suffix('.json').write_text(json.dumps(receipt))
                viewer = Visualizer(root)
                metadata, maps = viewer.load(name)
                self.assertEqual((metadata['count'], metadata['selected_count'], metadata['complete']), (1, 1, True))
                self.assertEqual([r['index'] for r in metadata['rows']], [rows[-1]['index']])
                if name == 'hayeong':
                    self.assertNotIn('experiment', [a['key'] for a in metadata['attributes']])
                self.assertTrue(set(expected_keys).issubset({a['key'] for a in metadata['attributes']}))
                for key in expected_keys:
                    grid = viewer.analysis(name, '', key, '', 'mean')
                    self.assertEqual(sum(c['count'] for c in grid['cells']), 1)
                np.testing.assert_array_equal(maps, np.full((1, 20484), rows[-1]['index']))
                self.assertEqual(path.read_bytes(), original)

    def test_raw_predictions_work_without_exported_report_and_refresh_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'tribe/outputs/massvis_sample'
            (folder / 'inputs').mkdir(parents=True)
            rows = [dict(index=i, sha256=f'source-{i}', filename=f'{i}.png', attributes={'color': i}) for i in range(2)]
            selection = dict(rows=rows, background_rgb=[255] * 3, attributes=[dict(key='color', kind='binary')])
            (folder / 'selection.json').write_text(json.dumps(selection))
            def prediction(index):
                p = folder / f'prediction_{index}.npz'
                np.savez_compressed(p, predictions=np.full((3, 20484), index + 1.), times=[0, 1, 2])
                receipt = dict(identity=prediction_identity(PROTOCOL, rows[index]['sha256'], [255] * 3),
                               prediction_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                (folder / f'prediction_{index}.json').write_text(json.dumps(receipt))
            prediction(0)
            viewer = Visualizer(root)
            metadata, maps = viewer.load('massvis_sample')
            self.assertFalse(metadata['complete'])
            self.assertEqual((metadata['count'], metadata['selected_count']), (1, 2))
            np.testing.assert_array_equal(maps[0], np.ones(20484))
            self.assertNotIn('sha256', metadata['rows'][0])
            grid = viewer.analysis('massvis_sample', '', 'color', '', 'mean')
            self.assertEqual(grid['cells'][0]['indices'], [0])
            prediction(1)
            updated, maps = viewer.load('massvis_sample')
            self.assertTrue(updated['complete'])
            self.assertNotEqual(metadata['revision'], updated['revision'])
            self.assertFalse((folder / 'explorer.json').exists())
            self.assertFalse((folder / 'brain_maps').exists())
            (folder / 'prediction_1.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'corrupted'):
                viewer.load('massvis_sample')

    def test_paths_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            viewer = Visualizer(tmp)
            for dataset in ['..', '../outside', 'a/b', 'a\\b']:
                with self.assertRaises(ValueError):
                    viewer.folder(dataset)
            with self.assertRaises(FileNotFoundError):
                viewer.folder('', '/../../etc/selection.json')
            with self.assertRaises(ValueError):
                viewer.folder('', '/etc/passwd')

    def test_study_simulation_finishes_without_aggregation_or_plotting(self):
        manifest = json.loads((REPOSITORY / 'tribe/studies/manifests/kylie.json').read_text())
        with tempfile.TemporaryDirectory() as tmp, patch('study.prepare_manifest', return_value=manifest), \
                patch('study.subprocess.run') as inference, patch('study.export_report') as export:
            output = Path(tmp) / 'kylie'
            run_study('kylie', output=output, predict=True)
            inference.assert_called_once()
            export.assert_not_called()
            self.assertTrue((output / 'selection.json').exists())
            self.assertFalse((output.parent / 'kylie_bar').exists())

    def test_massvis_predict_cli_does_not_export_or_plot(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch('sys.argv', ['dataset.py', '--predict', '--per-bin', '1', '--output-dir', tmp]), \
                patch('dataset.subprocess.run') as inference, patch('dataset.export_report') as export:
            dataset.main()
            inference.assert_called_once()
            export.assert_not_called()
            self.assertTrue((Path(tmp) / 'selection.json').exists())
            self.assertFalse((Path(tmp) / 'aggregate_maps.npz').exists())


if __name__ == '__main__':
    unittest.main()
