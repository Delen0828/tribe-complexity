"""Integrity checks against the published source, and deterministic rebuilds."""
import csv
import json
import unittest
import statistics
from collections import defaultdict

from build import ROOT, build


class DatasetIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build()
        cls.records = json.loads((ROOT / 'label/output/labels.json').read_text())

    def test_every_published_feature_preserved(self):
        with (ROOT / 'label/source/annotations.csv').open(newline='') as handle:
            source = {int(r['image_id'][:-4]): r for r in csv.DictReader(handle)}
        self.assertEqual(len(self.records), 5800)
        for record in self.records:
            original = source[record['index']]
            self.assertEqual(record['filename'], original['filename'])
            self.assertEqual(record['mean_human_complexity_rating'], float(original['mean_human_complexity_rating']))
            for dimension in ('text', 'color', 'data', 'design'):
                for key, value in record[dimension].items():
                    self.assertEqual(value, int(original[key]))

    def test_complete_local_mapping(self):
        self.assertEqual(self.report['missing_images'], [])
        self.assertEqual(self.report['matched_images'], 5800)
        self.assertEqual(len(self.report['aliases']), 6)
        for record in self.records:
            self.assertTrue((ROOT / record['image_path']).is_file())

    def test_rebuild_is_deterministic(self):
        paths = list((ROOT / 'label/output').glob('*'))
        before = {p: p.read_bytes() for p in paths}
        build()
        self.assertEqual(before, {p: p.read_bytes() for p in paths})

    def test_human_rating_spread(self):
        ratings = defaultdict(list)
        with (ROOT / 'label/source/ratings.csv').open(newline='') as handle:
            for row in csv.DictReader(handle):
                ratings[row['image_id']].append(float(row['response']))
        for record in self.records:
            values = ratings[record['image_id']]
            self.assertEqual(record['human_rating_count'], 10)
            self.assertAlmostEqual(statistics.mean(values), record['mean_human_complexity_rating'])
            variance = sum((x - statistics.mean(values)) ** 2 for x in values) / (len(values) - 1)
            self.assertAlmostEqual(record['human_rating_variance'], variance)
            self.assertAlmostEqual(record['human_rating_sd'] ** 2, variance)

    def test_csv_matches_json(self):
        with (ROOT / 'label/output/labels.csv').open(newline='') as handle:
            flat = list(csv.DictReader(handle))
        self.assertEqual(len(flat), len(self.records))
        for row, record in zip(flat, self.records):
            self.assertEqual(int(row['index']), record['index'])
            self.assertEqual(row['image_path'], record['image_path'])
            for group in ('text', 'color', 'data', 'design'):
                for key, value in record[group].items():
                    self.assertEqual(int(row[f'{group}.{key}']), value)


if __name__ == '__main__':
    unittest.main()
