"""Verify the numerical port against the actual notebook and published metadata."""
import ast
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.distance import mahalanobis

from reproduce import NOTEBOOK, ROOT, export_pilot, generate_points, parse_filename


class ReproductionTests(unittest.TestCase):
    def test_port_matches_current_notebook_for_all_72_point_clouds(self):
        # Run the reviewed numerical part of the source notebook itself. Strip
        # plotting, file I/O, and Colab download statements using its AST.
        source = ''.join(json.loads(NOTEBOOK.read_text())['cells'][3]['source'])
        tree = ast.parse(source)
        body = []
        for node in tree.body:
            if isinstance(node, ast.For):
                numerical = []
                for statement in node.body:
                    if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                            and ast.unparse(statement.value.func) == 'plt.clf'):
                        break
                    numerical.append(statement)
                capture = ast.parse('captured.append((x.copy(), y.copy(), float(r), outlierTREND))').body
                node.body = numerical + capture
                body.append(node)
                break
            if isinstance(node, (ast.Assign, ast.If)):
                body.append(node)
            elif isinstance(node, ast.Expr) and ast.unparse(node).startswith('np.random.'):
                body.append(node)
        captured = []
        namespace = dict(np=np, os=os, mahalanobis=mahalanobis, captured=captured)
        state = np.random.get_state()
        try:
            exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                         str(NOTEBOOK), 'exec'), namespace)
        finally:
            np.random.set_state(state)
        generated = list(generate_points())
        self.assertEqual(len(captured), 72)
        for (row, points), (x, y, r, suffix) in zip(generated, captured):
            np.testing.assert_array_equal(points, np.column_stack((x, y)))
            self.assertEqual(row['exact_r'], r)
            self.assertEqual(row['has_outlier'], int(bool(suffix)))
            self.assertEqual(points.shape, (251, 2))
            self.assertEqual(np.sign(r), np.sign(row['target_r']))
            if row['has_outlier']:
                self.assertTrue(6 <= row['exact_z'] <= 10)
        self.assertEqual(sum(row['has_outlier'] for row, _ in generated), 36)

    def test_export_agrees_with_independent_published_rsa_metadata(self):
        with (ROOT / 'DataVizThesis_StimGen/RSA/DataVizStim_Parsed_Metadata.csv').open() as stream:
            reference = list(csv.DictReader(stream))
        with tempfile.TemporaryDirectory() as temporary:
            rows = export_pilot(Path(temporary))
            self.assertEqual(len(rows), len(reference))
            for row, published in zip(rows, reference):
                self.assertEqual(row['filename'], published['stim_fns'])
                self.assertEqual(row['r'], float(published['r']))
                self.assertEqual(row['has_outlier'], int(published['outlier'] == 'True'))
                self.assertEqual(row['ink_density'], published['ink_density'])
                self.assertEqual((Path(temporary) / 'stimuli' / row['filename']).read_bytes(),
                                 (ROOT / 'DataVizThesis_Script/dataVizSetPilot_4' / row['filename']).read_bytes())
            changed = Path(temporary) / 'stimuli' / rows[0]['filename']
            changed.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed export'):
                export_pilot(Path(temporary))

    def test_invalid_metadata_is_rejected(self):
        for name in ('../fake.png', 'scatter_plot1_posCor(r=-0.500).png',
                     'scatter_plot1_negCor(r=0.000).png'):
            with self.assertRaises(ValueError):
                parse_filename(name)


if __name__ == '__main__':
    unittest.main()
