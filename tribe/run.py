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

from preprocessing import PROTOCOL, prepare_stimulus
DURATION = 3
ENCODE_STATS = {'unique_windows': 0}


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
        ENCODE_STATS['unique_windows'] += 1
        states = original(self, images, audio)
        pooled = states.mean(dim=2, keepdim=True).cpu()
        self._static_window_cache = (key, pooled)
        print(f'Encoded unique visual window {key[:12]}', flush=True)
        return pooled
    _HFVideoModel.predict_hidden_states = predict


def write_events(video, index, output):
    events = pd.DataFrame([dict(type='Video', start=0., duration=float(DURATION),
                               filepath=str(video), timeline=f'{PROTOCOL}_{index}', subject='default')])
    events.to_csv(output / f'events_{index}.csv', index=False)
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true',
                        help='Reuse predictions only when their receipt and file hashes validate')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'inputs/manifest.json')
    parser.add_argument('--inputs-dir', type=Path, default=ROOT / 'inputs')
    parser.add_argument('--outputs-dir', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--background', default='white',
                        help='Padding color (Pillow name or #RRGGBB); match experiment background')
    args = parser.parse_args()
    # Validate the color before loading models or writing outputs.
    _, geometry = prepare_stimulus(Image.new('RGB', (1, 1)), args.background)
    background_key = ''.join(f'{c:02x}' for c in geometry['background_rgb'])
    args.inputs_dir = args.inputs_dir.resolve()
    args.outputs_dir = args.outputs_dir.resolve()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    cache_identical_visual_windows()
    started = time.time()
    updates = {
        'data.num_workers': 0,
        'data.batch_size': 1,
        'data.video_feature.image.device': args.device,
        'data.video_feature.use_audio': False,
    }
    if not args.prepare_only:
        model = TribeModel.from_pretrained(
            'facebook/tribev2', cache_folder=ROOT / f'cache/features/{PROTOCOL}/{background_key}/intra_qp10',
            device=args.device, config_update=updates,
        )
        print('Pretrained TRIBE loaded successfully', flush=True)
        if args.smoke_only:
            return
        assert model.data.video_feature.image.token_aggregation == 'mean'
    rows = json.loads(args.manifest.read_text())
    prepared = []
    target = args.inputs_dir / PROTOCOL / background_key
    target.mkdir(parents=True, exist_ok=True)
    model_load_seconds = time.time() - started
    for row in rows:
        item_started = time.perf_counter()
        index = row['index']
        video = target / f'{index}.mp4'
        source = args.inputs_dir / f'{index}.png'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row['sha256']
        receipt_path = args.outputs_dir / f'prediction_{index}.json'
        prediction_path = args.outputs_dir / f'prediction_{index}.npz'
        identity = dict(protocol=PROTOCOL, source_sha256=row['sha256'],
                        background_rgb=geometry['background_rgb'], model='facebook/tribev2')
        if args.resume and not args.prepare_only and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['identity'] != identity:
                raise ValueError(f'Prediction identity mismatch for {index}; use a new output directory')
            if (prediction_path.is_file() and video.is_file()
                    and receipt['prediction_sha256'] == hashlib.sha256(prediction_path.read_bytes()).hexdigest()
                    and receipt['prepared']['video_sha256'] == hashlib.sha256(video.read_bytes()).hexdigest()):
                with np.load(prediction_path) as saved:
                    assert saved['predictions'].shape == (DURATION, 20484)
                    assert np.isfinite(saved['predictions']).all()
                    np.testing.assert_array_equal(saved['times'], np.arange(DURATION))
                write_events(video, index, args.outputs_dir)
                prepared.append(receipt['prepared'])
                print(f'Reusing validated prediction {index}', flush=True)
                continue
        with Image.open(source) as image:
            w, h = image.size
            frame, geometry = prepare_stimulus(image, args.background)
            frame.save(target / f'{index}.png')
            pixels = np.asarray(frame)
        clip = ImageClip(pixels).with_duration(DURATION)
        clip.write_videofile(str(video), fps=16, codec='libx264', audio=False,
                             ffmpeg_params=['-qp', '10', '-g', '1', '-preset', 'slow',
                                            '-vf', 'scale=in_range=pc:out_range=tv:out_color_matrix=bt709,setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709',
                                            '-pix_fmt', 'yuv420p', '-profile:v', 'high',
                                            '-color_range', 'tv', '-colorspace', 'bt709',
                                            '-color_primaries', 'bt709', '-color_trc', 'bt709',
                                            '-movflags', '+faststart'], logger=None)
        clip.close()
        # Keep previously shared video paths pointing to the current compatible clips.
        aliases = [args.inputs_dir / f'{index}.mp4']
        if args.inputs_dir == ROOT / 'inputs':
            aliases.append(ROOT / f'inputs/paper_3s_t0_even_crop_v1/{index}.mp4')
        for alias in aliases:
            alias.parent.mkdir(parents=True, exist_ok=True)
            alias.unlink(missing_ok=True)
            alias.symlink_to(os.path.relpath(video, alias.parent))
        prepared.append(dict(index=index, source_sha256=row['sha256'],
                             original_size=[w,h], prepared_size=list(frame.size),
                             **geometry, video=os.path.relpath(video, ROOT),
                             video_sha256=hashlib.sha256(video.read_bytes()).hexdigest()))
        if args.prepare_only:
            continue
        events = write_events(video, index, args.outputs_dir)
        print(f'Predicting image {index}', flush=True)
        inference_started = time.perf_counter()
        encodes_before = ENCODE_STATS['unique_windows']
        preds, segments = model.predict(events)
        inference_seconds = time.perf_counter() - inference_started
        times = np.array([s.start for s in segments])
        assert preds.shape == (DURATION, 20484), preds.shape
        assert np.isfinite(preds).all()
        np.testing.assert_array_equal(times, np.arange(DURATION))
        temporary = prediction_path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, predictions=preds, times=times)
        temporary.replace(prediction_path)
        receipt = dict(identity=identity, prepared=prepared[-1],
                       prediction_sha256=hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                       inference_seconds=inference_seconds,
                       total_seconds=time.perf_counter()-item_started,
                       encoded_windows=ENCODE_STATS['unique_windows']-encodes_before,
                       device=args.device)
        temporary_receipt = receipt_path.with_suffix('.tmp.json')
        temporary_receipt.write_text(json.dumps(receipt, indent=2)+'\n')
        temporary_receipt.replace(receipt_path)
        print(f'Image {index}: {preds.shape}, range {preds.min():.4f} to {preds.max():.4f}', flush=True)
    (target / 'preparation_metadata.json').write_text(json.dumps({
        'protocol': PROTOCOL, 'prepared_inputs': prepared,
    }, indent=2) + '\n')
    if args.prepare_only:
        print(f'Prepared {len(prepared)} compatible videos', flush=True)
        return
    (args.outputs_dir / 'run_metadata.json').write_text(json.dumps({
        'protocol': PROTOCOL,
        'model_load_seconds': model_load_seconds,
        'model': 'facebook/tribev2', 'device': args.device, 'duration_seconds': DURATION,
        'reported_time_index': 0, 'reported_time_seconds': 0,
        'fps': 16, 'codec': 'libx264', 'lossless': False, 'quantizer': 10, 'gop_size': 1,
        'pixel_format': 'yuv420p', 'color_space': 'bt709', 'color_range': 'tv',
        'crop_policy': 'Full image fit within centered 256x256 on 292x292 padded canvas',
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
