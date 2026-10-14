"""Published workbook joins, experiment-specific sources, and prediction reuse."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from dataset import load_predictions
from gpu_encoder import prediction_identity
from preprocessing import PROTOCOL
from studies import REPOSITORY, prepare_manifest
from studies.arunkumar import FIELDS, read_workbook
from study import reuse_identical_stimuli, run_study
from visualizer.server import Visualizer


class ArunkumarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = prepare_manifest('arunkumar')

    def test_both_experiments_keep_their_ratings_and_changed_images(self):
        manifest = self.manifest
        rows = manifest['rows']
        self.assertEqual(len(rows), 600)
        self.assertEqual(len({row['index'] for row in rows}), 600)
        self.assertEqual(len({row['sha256'] for row in rows}), 503)
        self.assertEqual(manifest['unique_prediction_count'], 503)
        self.assertEqual(len(manifest['aliases']), 97)
        self.assertEqual(manifest['rating_interval_counts'], {
            '1': {'I1': 47, 'I2': 144, 'I3': 161, 'I4': 106, 'I5': 42},
            '2': {'I1': 11, 'I2': 26, 'I3': 26, 'I4': 26, 'I5': 11}})
        by_index = {row['index']: row for row in rows}
        self.assertEqual(by_index[75]['score'], -7.16)
        self.assertEqual(by_index[501]['original_stimulus_index'], 75)
        self.assertEqual(by_index[501]['score'], -7.74)
        self.assertEqual(by_index[501]['reuse_prediction_from'], 75)
        self.assertEqual(by_index[501]['ratings']['experiment_2']['group_a']['mean'], -7.48)
        changed = {row['filename'] for row in rows[500:] if 'reuse_prediction_from' not in row}
        self.assertEqual(changed, {'5-76.jpeg', '9-89.png', '29-88.png'})
        for row in rows:
            self.assertEqual(hashlib.sha256((REPOSITORY / row['source']).read_bytes()).hexdigest(), row['sha256'])
            self.assertNotIn('complexity', row['attributes'])
            if 'reuse_prediction_from' in row:
                self.assertEqual(row['sha256'], by_index[row['reuse_prediction_from']]['sha256'])
            rating = row['ratings'][f'experiment_{row["attributes"]["experiment"]}']
            self.assertEqual(row['score'], rating['mean'])
            self.assertEqual(row['attributes']['image_information_rating'], rating['interval'])

    def test_all_39_taxonomy_fields_are_available_with_original_values(self):
        manifest = self.manifest
        keys = {key for fields in FIELDS.values() for key, _ in fields}
        self.assertEqual(len(keys), 39)
        self.assertEqual({definition['key'] for definition in manifest['attributes']},
                         keys | {'image_information_rating', 'experiment'})
        by_index = {row['index']: row for row in manifest['rows']}
        values = by_index[1]['attributes']
        self.assertEqual([values[key] for key in ('chart_category', 'chart_type', 'domain', 'source', 'topic')],
                         ['Grid', 'Heatmap', 'Infographics', 'Information is Beautiful', 'Health'])
        self.assertEqual(values['temporal'], 1)
        self.assertEqual(values['photorealistic'], 0)
        self.assertEqual(by_index[500]['attributes']['human_depiction'], 0)
        self.assertEqual(by_index[500]['published_features']['Visual Elements']['Human Depiction (HD)'], 'n ')
        for row in manifest['rows'][500:]:
            original = by_index[row['original_stimulus_index']]
            self.assertEqual({key: row['attributes'][key] for key in keys},
                             {key: original['attributes'][key] for key in keys})
        for path, digest in manifest['source_documents'].items():
            self.assertEqual(hashlib.sha256((REPOSITORY / path).read_bytes()).hexdigest(), digest)
            self.assertNotIn('Demographics', path)

    def test_simulation_infers_503_images_and_preserves_600_observations(self):
        with tempfile.TemporaryDirectory() as tmp, patch('study.subprocess.run') as inference, \
                patch('study.reuse_identical_stimuli', return_value=[]), patch('study.export_report') as export:
            output = Path(tmp)
            run_study('arunkumar', output=output, predict=True)
            command = inference.call_args.args[0]
            manifest = json.loads(Path(command[command.index('--manifest') + 1]).read_text())
            self.assertEqual(len(manifest), 503)
            self.assertEqual(len(json.loads((output / 'inputs/manifest.json').read_text())), 600)
            self.assertEqual(len(json.loads((output / 'selection.json').read_text())['rows']), 600)
            export.assert_not_called()

    def test_identical_prediction_reuse_has_valid_independent_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            target = output / 'inputs' / PROTOCOL / 'ffffff'
            target.mkdir(parents=True)
            video = target / '1.mp4'
            video.write_bytes(b'validated static video')
            (target / '1.png').write_bytes(b'prepared image')
            prediction = output / 'prediction_1.npz'
            np.savez_compressed(prediction, predictions=np.ones((3, 20484)), times=[0, 1, 2])
            rows = [dict(index=1, sha256='same-source'),
                    dict(index=501, sha256='same-source', reuse_prediction_from=1)]
            receipt = dict(identity=prediction_identity(PROTOCOL, 'same-source', [255] * 3),
                           prepared=dict(index=1, video=str(video), video_sha256=hashlib.sha256(video.read_bytes()).hexdigest()),
                           prediction_sha256=hashlib.sha256(prediction.read_bytes()).hexdigest())
            (output / 'prediction_1.json').write_text(json.dumps(receipt))
            self.assertEqual(reuse_identical_stimuli(rows, output, [255] * 3, 'fp32'), [501])
            maps, receipts = load_predictions(rows, output, [255] * 3)
            np.testing.assert_array_equal(maps, np.ones((2, 20484)))
            reused = receipts[1]
            self.assertEqual(reused['prepared']['index'], 501)
            self.assertEqual(reused['encoded_windows'], 0)
            self.assertEqual(reused['reused_from_index'], 1)
            self.assertEqual((output / 'inputs/501.mp4').read_bytes(), video.read_bytes())
            self.assertTrue((output / 'events_501.csv').is_file())
            changed = [rows[0], dict(rows[1], sha256='different-source')]
            with self.assertRaisesRegex(ValueError, 'different source bytes'):
                reuse_identical_stimuli(changed, output, [255] * 3, 'fp32')

    def test_experiment_views_select_their_own_ratings_maps_and_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'tribe/outputs/arunkumar'
            output.mkdir(parents=True)
            manifest_folder = root / 'tribe/studies/manifests'
            manifest_folder.mkdir(parents=True)
            (manifest_folder / 'arunkumar.json').write_text(json.dumps(self.manifest))
            selection = dict(dataset='arunkumar', rows=self.manifest['rows'],
                             attributes=self.manifest['attributes'], background_rgb=[255] * 3)
            (output / 'selection.json').write_text(json.dumps(selection))
            for index in (1, 501, 505):
                row = self.manifest['rows'][index - 1]
                path = output / f'prediction_{index}.npz'
                np.savez_compressed(path, predictions=np.full((3, 20484), index), times=[0, 1, 2])
                receipt = dict(identity=prediction_identity(PROTOCOL, row['sha256'], [255] * 3),
                               prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                path.with_suffix('.json').write_text(json.dumps(receipt))
            viewer = Visualizer(root)
            combined, _ = viewer.load('arunkumar')
            self.assertEqual((combined['count'], combined['selected_count']), (3, 600))
            for experiment, indices, selected in [(1, [1], 500), (2, [501, 505], 100)]:
                name = f'arunkumar_experiment_{experiment}'
                report, maps = viewer.load(name)
                self.assertEqual((report['count'], report['selected_count']), (len(indices), selected))
                self.assertEqual([row['index'] for row in report['rows']], indices)
                self.assertEqual(len(report['attributes']), 40)
                np.testing.assert_array_equal(maps[:, 0], indices)
                analysis = viewer.analysis(name, '', 'image_information_rating', '', 'contrast')
                self.assertEqual(sum(cell['count'] for cell in analysis['cells']), len(indices))
                for cell in viewer.grids[analysis['id']]['cells']:
                    np.testing.assert_array_equal(cell['values'], np.zeros(20484))

    def test_sparse_shared_inline_and_cached_formula_workbook_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'table.xlsx'
            with ZipFile(path, 'w') as archive:
                archive.writestr('xl/workbook.xml', f'<workbook xmlns="{NS_URL}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Table" r:id="rId1"/></sheets></workbook>')
                archive.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
                archive.writestr('xl/sharedStrings.xml', f'<sst xmlns="{NS_URL}"><si><r><t>File</t></r><r><t> name</t></r></si></sst>')
                archive.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{NS_URL}"><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="C1" t="inlineStr"><is><t>Mean</t></is></c><c r="AA1"><f>1+2</f><v>3</v></c></row></sheetData></worksheet>')
            row = read_workbook(path, ['Table'])['Table'][0]
            self.assertEqual(row[:3], ['File name', None, 'Mean'])
            self.assertEqual(row[26], 3)
            with self.assertRaisesRegex(ValueError, 'Missing workbook sheets'):
                read_workbook(path, ['Absent'])


NS_URL = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'

if __name__ == '__main__':
    unittest.main()
