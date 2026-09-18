"""Check saved input provenance and inference/comparison artifacts."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
rows = json.loads((root/'inputs/manifest.json').read_text())
predictions = []
for row in rows:
    source = root/'inputs'/row['image_id']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row['sha256']
    saved = np.load(root/f"outputs/prediction_{row['index']}.npz")
    pred = saved['predictions']
    assert pred.shape == (12, 20484)
    assert np.isfinite(pred).all()
    np.testing.assert_array_equal(saved['times'], np.arange(12))
    predictions.append(pred)
assert not np.array_equal(*predictions), 'Unexpected identical image predictions'
maps = np.load(root/'outputs/mean_maps.npz')
np.testing.assert_array_equal(maps['image_751'], predictions[0].mean(0))
np.testing.assert_array_equal(maps['image_4849'], predictions[1].mean(0))
np.testing.assert_array_equal(maps['difference'], maps['image_751']-maps['image_4849'])
metrics = json.loads((root/'outputs/comparison.json').read_text())
np.testing.assert_allclose(metrics['spatial_pearson_r'],
                           np.corrcoef(maps['image_751'], maps['image_4849'])[0,1])
for path in ['index.html', 'outputs/comparison.png', 'outputs/comparison.pdf']:
    assert (root/path).stat().st_size > 1000
print('PASS: source hashes, finite 12 × 20,484 predictions, aligned times, distinct outputs,')
print('      mean maps, signed difference, correlation, and rendered artifacts.')
