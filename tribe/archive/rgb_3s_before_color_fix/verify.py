"""Validate source provenance, prepared stimuli, t=0 maps, and report contents."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from moviepy import VideoFileClip
from pypdf import PdfReader

root = Path(__file__).resolve().parent
rows = json.loads((root/'inputs/manifest.json').read_text())
meta = json.loads((root/'outputs/run_metadata.json').read_text())
assert meta['protocol'] == 'paper_3s_t0_even_crop_v1'
assert meta['duration_seconds'] == 3 and meta['reported_time_index'] == 0
predictions = []
for row, prepared in zip(rows, meta['prepared_inputs']):
    source = root/'inputs'/row['image_id']
    assert row['index'] == prepared['index']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row['sha256']
    original = np.array(Image.open(source).convert('RGB'))
    h,w = original.shape[:2]
    expected = original[:h-h%2,:w-w%2]
    assert prepared['original_size'] == [w,h]
    assert prepared['prepared_size'] == [w-w%2,h-h%2]
    image_path = root/'inputs'/meta['protocol']/row['image_id']
    np.testing.assert_array_equal(np.array(Image.open(image_path)), expected)
    video = root/prepared['video']
    assert hashlib.sha256(video.read_bytes()).hexdigest() == prepared['video_sha256']
    with VideoFileClip(str(video)) as clip:
        assert clip.audio is None
        assert clip.duration == 3 and clip.fps == 16
        frames = 0
        for frame in clip.iter_frames():
            np.testing.assert_array_equal(frame, expected)
            frames += 1
        assert frames == 48
    saved = np.load(root/f"outputs/prediction_{row['index']}.npz")
    pred = saved['predictions']
    assert pred.shape == (3,20484) and np.isfinite(pred).all()
    np.testing.assert_array_equal(saved['times'], np.arange(3))
    predictions.append(pred)
assert not np.array_equal(*predictions)
maps = np.load(root/'outputs/selected_maps.npz')
np.testing.assert_array_equal(maps['image_751'],predictions[0][0])
np.testing.assert_array_equal(maps['image_4849'],predictions[1][0])
np.testing.assert_array_equal(maps['difference'],predictions[0][0]-predictions[1][0])
np.testing.assert_allclose(maps['demeaned_difference'],maps['difference']-maps['difference'].mean(),atol=1e-7)
assert abs(float(maps['demeaned_difference'].mean())) < 1e-7
metrics = json.loads((root/'outputs/comparison.json').read_text())
np.testing.assert_allclose(metrics['spatial_pearson_r'],np.corrcoef(maps['image_751'],maps['image_4849'])[0,1])
assert not (root/'outputs/mean_maps.npz').exists(), 'Stale temporal-mean output'
for path in ['index.html','outputs/comparison.png','outputs/directional_contrasts.png','outputs/comparison.pdf']:
    assert (root/path).stat().st_size > 1000
pdf = PdfReader(root/'outputs/comparison.pdf')
assert len(pdf.pages) == 2
text = '\n'.join(p.extract_text() for p in pdf.pages)
assert 't=0' in text and 'Whole-brain mean removed' in text
assert 'Cortical RMS' not in text and 'Spatial correlation' not in text
html = (root/'index.html').read_text()
assert 'metrics' not in html and 'magnitude over time' not in html
assert np.load(root/'archive/original_12s/outputs/prediction_751.npz')['predictions'].shape == (12,20484)
print('PASS: source hashes; even crop; every lossless video frame; no audio; 3-second clips;')
print('      finite 3 x 20,484 predictions; exact t=0 selection; raw/de-meaned contrasts;')
print('      two-page neuroimaging report without time-series plots; original run preserved.')
