"""Study selection, matched comparison membership, provenance, and viewer routing."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
from dataset import aggregate_maps, export_report
from preprocessing import PROTOCOL
from studies import REPOSITORY, extract_archive, prepare_manifest, study_variables
from study import save_selection


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kylie = prepare_manifest('kylie')
        cls.hayeong = prepare_manifest('hayeong')

    def test_unique_authoritative_sources_and_aliases(self):
        for manifest, count in [(self.kylie, 24), (self.hayeong, 40)]:
            self.assertEqual(len(manifest['rows']), count)
            self.assertEqual(len({r['index'] for r in manifest['rows']}), count)
            self.assertEqual(len({r['source'] for r in manifest['rows']}), count)
            for row in manifest['rows']:
                source = REPOSITORY / row['source']
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), row['sha256'])
                self.assertNotIn('response_summary', row['source'])
                self.assertNotIn('extract_clutter', row['source'])
        self.assertEqual(len(self.kylie['aliases']), 6)
        self.assertEqual(len(self.hayeong['aliases']), 5)
        self.assertEqual(sum(r['rating_count'] for r in self.kylie['rows']), 759)
        self.assertTrue(all(0 <= r['score'] <= 100 for r in self.kylie['rows']))
        self.assertTrue(all('score' not in r for r in self.hayeong['rows']))

    def test_comparisons_preserve_the_matched_design(self):
        for comparison in self.kylie['comparisons']:
            rows = [r for r in self.kylie['rows'] if r['index'] in comparison['indices']]
            self.assertEqual(len(rows), 12)
            self.assertEqual({r['attributes']['chart_type'] for r in rows}, {comparison['id']})
            for design in range(6):
                self.assertEqual({r['attributes']['topic'] for r in rows if r['attributes']['design_variant'] == design},
                                 {'covid', 'penguins'})
        comparisons = {c['id']: c for c in self.hayeong['comparisons']}
        for key, count in [('color', 16), ('hand_drawn', 12), ('cartoon_image', 16)]:
            comparison = comparisons[key]
            rows = [r for r in self.hayeong['rows'] if r['index'] in comparison['indices']]
            self.assertEqual(len(rows), count)
            for definition in comparison['attributes']:
                if definition['kind'] in ('binary', 'bin'):
                    continue
                values = {r['attributes'][definition['key']] for r in rows}
                populations = [sum(r['attributes'][definition['key']] == v for r in rows) for v in values]
                self.assertEqual(len(set(populations)), 1)
            if key != 'cartoon_image':
                for condition in {r['attributes']['condition'] for r in rows}:
                    self.assertEqual({r['attributes']['topic'] for r in rows if r['attributes']['condition'] == condition},
                                     {'emission', 'ice', 'medal', 'trump'})
        self.assertIn('different topics', comparisons['cartoon_image']['description'])

    def test_perceived_complexity_uses_human_ratings_and_includes_100(self):
        definition = next(a for a in self.kylie['attributes'] if a['key'] == 'complexity')
        self.assertEqual((definition['label'], definition['kind']), ('Perceived complexity', 'bin'))
        for row in self.kylie['rows']:
            self.assertLessEqual(row['attributes']['complexity'] * 10, row['score'])
            self.assertLess(row['score'], (row['attributes']['complexity'] + 1) * 10)
        boundaries = [dict(attributes={}, score=score) for score in (0, 10, 99.9, 100)]
        updated, _ = study_variables('kylie', boundaries, [])
        self.assertEqual([row['attributes']['complexity'] for row in updated], [0, 1, 9, 9])
        self.assertTrue(all(not row['attributes'] for row in boundaries))
        for comparison in self.kylie['comparisons']:
            self.assertIn('complexity', [a['key'] for a in comparison['attributes']])

    def test_hayeong_features_distinguish_bar_font_color_and_illustration(self):
        keys = ['has_color', 'bar_is_hand_drawn', 'font_is_hand_drawn', 'is_comic', 'bar_is_image']
        self.assertEqual([a['key'] for a in self.hayeong['attributes'][:5]], keys)
        by_filename = {row['filename']: row['attributes'] for row in self.hayeong['rows']}
        expected = {
            'building_standard.png': [0, 0, 0, 0, 0],
            'scaffold_embellished.png': [1, 1, 1, 1, 0],
            'covid_embellished.png': [1, 0, 0, 0, 1],
            'emission_font_color.png': [1, 0, 1, 0, 0],
            'emission_handrawn_blackwhite.png': [0, 1, 1, 1, 0],
            'emission_handrawn_color.png': [1, 1, 1, 1, 0],
            'emission_standard_blackwhite.png': [0, 0, 0, 0, 0],
            'emission_standard_color.png': [1, 0, 0, 0, 0],
            'emission_image_color.png': [1, 0, 0, 0, 1],
            'medal_color_font.png': [1, 0, 1, 0, 0],
        }
        for filename, values in expected.items():
            with self.subTest(filename=filename):
                self.assertEqual([by_filename[filename][key] for key in keys], values)
        for comparison in self.hayeong['comparisons']:
            self.assertEqual([a['key'] for a in comparison['attributes'][:5]], keys)

    def test_custom_aggregation_and_separate_pages_without_fake_scores(self):
        manifest = self.hayeong
        rows = manifest['rows']
        maps = np.arange(len(rows) * 2).reshape(len(rows), 2)
        groups, grand = aggregate_maps(rows, maps, manifest['attributes'])
        for definition in manifest['attributes']:
            selected = [g for g in groups if g['attribute'] == definition['key']]
            np.testing.assert_allclose(sum(g['mean'] * g['count'] for g in selected) / len(rows), grand)
            np.testing.assert_allclose(sum(g['contrast'] * g['count'] for g in selected), [0, 0], atol=1e-12)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'hayeong'
            output.mkdir()
            selection = dict(rows=rows, scope='study', seed=0, protocol=PROTOCOL,
                             eligible_count=len(rows), missing_indices=[], dataset='hayeong',
                             title=manifest['title'], description=manifest['description'],
                             sampling=manifest['sampling'], attributes=manifest['attributes'])
            export_report(rows, maps, output, selection, {}, render=False)
            report = json.loads((output / 'explorer.json').read_text())
            self.assertEqual(report['sampling'], manifest['sampling'])
            self.assertEqual(report['attributes'], manifest['attributes'])
            self.assertNotIn('score', report['rows'][0])
            self.assertFalse((output.parent / 'web').exists())
            self.assertFalse((output / 'brain_maps').exists())

    def test_selection_refuses_changed_source_or_protocol(self):
        rows = [dict(row, source=str(REPOSITORY / row['source'])) for row in self.kylie['rows'][:1]]
        selection = dict(rows=rows, protocol=PROTOCOL, background_rgb=[255] * 3, attributes=[])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            save_selection(output, selection)
            save_selection(output, selection)
            additions = copy.deepcopy(selection)
            additions['attributes'] = [dict(key='complexity', label='Perceived complexity', kind='bin')]
            save_selection(output, additions)
            changed_definition = copy.deepcopy(additions)
            changed_definition['attributes'][0]['kind'] = 'count'
            with self.assertRaisesRegex(ValueError, 'Different selection'):
                save_selection(output, changed_definition)
            changed = copy.deepcopy(selection)
            changed['background_rgb'] = [0] * 3
            with self.assertRaisesRegex(ValueError, 'Different selection'):
                save_selection(output, changed)

    def test_archive_paths_and_changed_extracted_content_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'archive.zip'
            with ZipFile(path, 'w') as archive:
                archive.writestr('../escape.png', b'invalid')
            with self.assertRaisesRegex(ValueError, 'Unsafe archive path'):
                extract_archive(path, root / 'extracted')
            with ZipFile(path, 'w') as archive:
                archive.writestr('stimuli/one.png', b'original')
                archive.writestr('__MACOSX/junk', b'ignored')
            extract_archive(path, root / 'extracted')
            self.assertFalse((root / 'extracted/__MACOSX').exists())
            (root / 'extracted/stimuli/one.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'differs from archive'):
                extract_archive(path, root / 'extracted')


if __name__ == '__main__':
    unittest.main()
