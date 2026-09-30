"""Validate source provenance, prepared stimuli, t=0 maps, and report contents."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from moviepy import VideoFileClip
from pypdf import PdfReader
import imageio_ffmpeg

from preprocessing import PROTOCOL, prepare_stimulus

root = Path(__file__).resolve().parent
rows = json.loads((root/'inputs/manifest.json').read_text())
meta = json.loads((root/'outputs/run_metadata.json').read_text())
assert meta['protocol'] == PROTOCOL
assert meta['duration_seconds'] == 3 and meta['reported_time_index'] == 0
predictions = []
color_checks = {}
for row, prepared in zip(rows, meta['prepared_inputs']):
    source = root/'inputs'/row['image_id']
    assert row['index'] == prepared['index']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row['sha256']
    original = np.array(Image.open(source).convert('RGB'))
    h,w = original.shape[:2]
    background = '#%02x%02x%02x' % tuple(prepared['background_rgb'])
    expected_image, geometry = prepare_stimulus(Image.fromarray(original), background)
    expected = np.array(expected_image)
    for key, value in geometry.items():
        assert prepared[key] == value
    assert prepared['original_size'] == [w,h]
    assert prepared['prepared_size'] == [292,292]
    image_path = (root/prepared['video']).with_suffix('.png')
    np.testing.assert_array_equal(np.array(Image.open(image_path)), expected)
    video = root/prepared['video']
    assert hashlib.sha256(video.read_bytes()).hexdigest() == prepared['video_sha256']
    stream = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-i',str(video)],
                            capture_output=True,text=True).stderr
    assert 'yuv420p(tv, bt709' in stream and 'High 4:4:4' not in stream, stream
    errors = []
    with VideoFileClip(str(video)) as clip:
        assert clip.audio is None
        assert clip.duration == 3 and clip.fps == 16
        frames = 0
        first_frame = clip.get_frame(0)
        for frame in clip.iter_frames():
            np.testing.assert_array_equal(frame,first_frame)
            error = np.abs(frame.astype(float)-expected.astype(float))
            # YUV420 subsamples chroma; verify fidelity rather than claiming losslessness.
            assert error.mean() < 4 and np.percentile(error,99) < 35
            errors.append(float(error.mean()))
            frames += 1
        assert frames == 48
    color_checks[str(row['index'])] = {'ffmpeg_max_frame_mae_8bit':max(errors)}
    if sys.platform == 'darwin':
        cache = root/'cache/color-verification'
        cache.mkdir(exist_ok=True,parents=True)
        native = cache/f"native-{row['index']}.png"
        subprocess.run(['swift','-module-cache-path',str(cache/'swift'),str(root/'verify_macos.swift'),
                        str(video),str(native)],check=True,capture_output=True)
        native_pixels = np.array(Image.open(native).convert('RGB'))
        error = np.abs(native_pixels.astype(float)-expected.astype(float))
        assert error.mean() < 4 and np.percentile(error,99) < 35
        color_checks[str(row['index'])]['macos_native_mae_8bit'] = float(error.mean())
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
assert 'Original stimulus | Image 751' in text and 'Original stimulus | Image 4849' in text
assert 'Cortical RMS' not in text and 'Spatial correlation' not in text
html = (root/'index.html').read_text()
assert 'metrics' not in html and 'magnitude over time' not in html
assert np.load(root/'archive/original_12s/outputs/prediction_751.npz')['predictions'].shape == (12,20484)
(root/'outputs/color_validation.json').write_text(json.dumps(color_checks,indent=2)+'\n')
print('PASS: source hashes; centered padded stimuli; all video frames within color-error bounds;')
print('      native macOS decoding: ' + ('PASS' if sys.platform == 'darwin' else 'SKIPPED (requires macOS)'))
print('      finite 3 x 20,484 predictions; exact t=0 selection; raw/de-meaned contrasts;')
print('      two-page report with original stimuli and no time-series plots; original run preserved.')
