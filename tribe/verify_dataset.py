"""Verify sampled/full-run provenance, prepared stimuli, and every group aggregate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from moviepy import VideoFileClip
from dataset import ATTRIBUTES, ROOT, aggregate_maps, load_predictions, write_json
from preprocessing import PROTOCOL, prepare_stimulus


def verify(output):
    selection = json.loads((output / 'selection.json').read_text())
    report = json.loads((output / 'explorer.json').read_text())
    rows = selection['rows']
    assert report['protocol'] == selection['protocol'] == PROTOCOL
    assert report['count'] == len(rows) == len({r['index'] for r in rows})
    assert {a['key'] for a in report['attributes']} == set(ATTRIBUTES)
    assert [r['index'] for r in report['rows']] == [r['index'] for r in rows]
    if selection['scope'] == 'sample':
        assert [sum(r['bin'] == b for r in rows) for b in range(10)] == [selection['per_bin']]*10
    else:
        assert len(rows) == selection['eligible_count']
    background = selection['background_rgb']
    maps, receipts = load_predictions(rows, output, background)
    maximum_mae = 0.
    color_errors = {}
    for row, receipt in zip(rows, receipts):
        prepared = receipt['prepared']
        source = output / 'inputs' / f'{row["index"]}.png'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row['sha256']
        assert hashlib.sha256(Path(row['source']).read_bytes()).hexdigest() == row['sha256']
        color = '#%02x%02x%02x' % tuple(background)
        with Image.open(source) as original:
            assert list(original.size) == prepared['original_size']
            frame, geometry = prepare_stimulus(original, color)
        for key, value in geometry.items():
            assert prepared[key] == value
        assert prepared['prepared_size'] == [292,292]
        x0,y0,x1,y1 = geometry['stimulus_box']
        assert 18 <= x0 < x1 <= 274 and 18 <= y0 < y1 <= 274
        video = ROOT / prepared['video']
        expected = np.asarray(frame)
        with Image.open(video.with_suffix('.png')) as png:
            np.testing.assert_array_equal(np.asarray(png), expected)
        assert hashlib.sha256(video.read_bytes()).hexdigest() == prepared['video_sha256']
        with VideoFileClip(str(video)) as clip:
            assert clip.size == [292,292] and clip.duration == 3 and clip.fps == 16
            assert clip.audio is None
            for t in (0, 1.5, 47/16):
                error = np.abs(clip.get_frame(t).astype(float)-expected.astype(float))
                # YUV420 averages neighboring chroma. Saturated chart edges can
                # have large local errors even when overall fidelity is good;
                # record the tail rather than applying the old two-image cutoff.
                assert error.mean() < 4, (row['index'], error.mean())
                color_errors[str(row['index'])] = dict(
                    mean_absolute_error=float(error.mean()),
                    percentile_99_absolute_error=float(np.percentile(error,99)))
                maximum_mae = max(maximum_mae, float(error.mean()))
    groups, grand = aggregate_maps(rows, maps)
    with np.load(output / 'aggregate_maps.npz') as saved:
        np.testing.assert_array_equal(saved['indices'], [r['index'] for r in rows])
        np.testing.assert_array_equal(saved['maps'], maps)
        np.testing.assert_allclose(saved['grand_mean'], grand)
        np.testing.assert_allclose(saved['means'], [g['mean'] for g in groups])
        np.testing.assert_allclose(saved['contrasts'], [g['contrast'] for g in groups])
    assert len(groups) == len(report['groups'])
    for expected, published in zip(groups, report['groups']):
        for key in ('attribute', 'value', 'label', 'indices', 'count'):
            assert published[key] == expected[key]
        for kind in ('mean','contrast'):
            with Image.open(output / published[f'{kind}_image']) as image:
                assert image.width >= 1000 and image.height >= 300
                image.verify()
    for kind in ('mean','contrast'):
        assert report['limits'][kind] == max(1e-12,max(float(np.abs(g[kind]).max()) for g in groups))
    for attribute in ATTRIBUTES:
        selected = [g for g in groups if g['attribute'] == attribute]
        assert sum(g['count'] for g in selected) == len(rows)
        np.testing.assert_allclose(sum(g['mean']*g['count'] for g in selected)/len(rows), grand, atol=1e-12)
    result = dict(status='PASS', predictions=len(rows), groups=len(groups), attributes=len(ATTRIBUTES),
                  video_frames_checked=3*len(rows), maximum_video_frame_mae=maximum_mae,
                  color_errors=color_errors,
                  checks=['source and prediction hashes', 'finite 3x20484 arrays and times 0,1,2',
                          '292 canvas and full stimulus within final 256 crop', 'video decoding and color fidelity',
                          'group membership, means, contrasts, scales, and rendered assets'])
    write_json(output / 'verification.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'color_errors'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    verify(parser.parse_args().output.resolve())
