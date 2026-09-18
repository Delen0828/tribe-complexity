"""Predict average-subject cortical responses to matched static MASSVIS stimuli."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(ROOT / 'cache/huggingface')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from moviepy import ImageClip
from PIL import Image
from tribev2 import TribeModel

PROTOCOL = 'paper_3s_t0_even_crop_v1'
DURATION = 3


def cache_identical_visual_windows():
    """Memoize byte-identical clips, retaining the native mean-token features.

    No model weights or temporal sampling change. The official downstream
    extractor uses mean token aggregation, so pooling before caching is equivalent.
    """
    from neuralset.extractors.video import _HFVideoModel
    original = _HFVideoModel.predict_hidden_states
    def predict(self, images, audio=None):
        if audio is not None:
            return original(self, images, audio)
        key = hashlib.sha256(images.tobytes()).hexdigest()
        cached = getattr(self, '_static_window_cache', None)
        if cached is not None and cached[0] == key:
            return cached[1]
        states = original(self, images, audio)
        pooled = states.mean(dim=2, keepdim=True).cpu()
        self._static_window_cache = (key, pooled)
        print(f'Encoded unique visual window {key[:12]}', flush=True)
        return pooled
    _HFVideoModel.predict_hidden_states = predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(8)
    cache_identical_visual_windows()
    started = time.time()
    updates = {
        'data.num_workers': 0,
        'data.batch_size': 1,
        'data.video_feature.image.device': args.device,
        'data.video_feature.use_audio': False,
    }
    model = TribeModel.from_pretrained(
        'facebook/tribev2', cache_folder=ROOT / f'cache/features/{PROTOCOL}',
        device=args.device, config_update=updates,
    )
    print('Pretrained TRIBE loaded successfully', flush=True)
    if args.smoke_only:
        return
    assert model.data.video_feature.image.token_aggregation == 'mean'
    rows = json.loads((ROOT / 'inputs/manifest.json').read_text())
    prepared = []
    for row in rows:
        index = row['index']
        target = ROOT / 'inputs' / PROTOCOL
        target.mkdir(exist_ok=True)
        video = target / f'{index}.mp4'
        source = ROOT / f'inputs/{index}.png'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row['sha256']
        with Image.open(source) as image:
            w, h = image.size
            # The paper specifies even dimensions but not which edge to crop.
            # Remove at most one pixel from right/bottom; never scale or pad.
            crop_box = (0, 0, w-w%2, h-h%2)
            frame = image.convert('RGB').crop(crop_box)
            frame.save(target / f'{index}.png')
            pixels = np.asarray(frame)
        clip = ImageClip(pixels).with_duration(DURATION)
        clip.write_videofile(str(video), fps=16, codec='libx264rgb', audio=False,
                             ffmpeg_params=['-crf', '0', '-pix_fmt', 'rgb24'], logger=None)
        clip.close()
        prepared.append(dict(index=index, source_sha256=row['sha256'],
                             original_size=[w,h], prepared_size=list(frame.size),
                             crop_box=list(crop_box), video=str(video.relative_to(ROOT)),
                             video_sha256=hashlib.sha256(video.read_bytes()).hexdigest()))
        events = pd.DataFrame([dict(type='Video', start=0., duration=float(DURATION),
                                    filepath=str(video), timeline=f'{PROTOCOL}_{index}', subject='default')])
        events.to_csv(ROOT / f'outputs/events_{index}.csv', index=False)
        print(f'Predicting image {index}', flush=True)
        preds, segments = model.predict(events)
        times = np.array([s.start for s in segments])
        assert preds.shape == (DURATION, 20484), preds.shape
        assert np.isfinite(preds).all()
        np.testing.assert_array_equal(times, np.arange(DURATION))
        np.savez_compressed(ROOT / f'outputs/prediction_{index}.npz', predictions=preds, times=times)
        print(f'Image {index}: {preds.shape}, range {preds.min():.4f} to {preds.max():.4f}', flush=True)
    (ROOT / 'outputs/run_metadata.json').write_text(json.dumps({
        'protocol': PROTOCOL,
        'model': 'facebook/tribev2', 'device': args.device, 'duration_seconds': DURATION,
        'reported_time_index': 0, 'reported_time_seconds': 0,
        'fps': 16, 'codec': 'libx264rgb', 'lossless': True,
        'crop_policy': 'Right/bottom edge to even dimensions; no scaling or padding',
        'prepared_inputs': prepared,
        'source_documents': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            ROOT.parent/'Can_a_Neural_Encoding_Model_Replicate_an_fMRI_Visu.pdf',
            ROOT.parent/'supplemental_materials.pdf']},
        'code_commit': 'af58661791a351a448a489042a28f6c37e1c14b7',
        'weight_revisions': {p.parent.parent.name: p.name for p in
                             (ROOT/'cache/huggingface/hub').glob('models--*/snapshots/*')},
        'duplicate_window_cache': 'SHA-256 byte identity; native mean-token aggregation',
        'visual_only': True, 'torch': torch.__version__, 'elapsed_seconds': time.time()-started,
        'summary': 'First prediction only (t=0); separate 3-second silent static clips.',
        'units': 'Model output units; not percent BOLD or complexity ratings.',
    }, indent=2)+'\n')


if __name__ == '__main__':
    main()
