"""Sample or predict all annotated MASSVIS stimuli and export grouped cortical maps."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from preprocessing import PROTOCOL

ROOT = Path(__file__).resolve().parent
ATTRIBUTES = {
    'complexity': ('Perceived complexity', 'mean_human_complexity_rating'),
    'charts': ('Number of charts', 'design.chart_count'),
    'colors': ('Number of distinct colors', 'color.color_count'),
    'quantitative': ('Number of quantitative variables', 'data.quantitative'),
    'categorical': ('Number of categorical variables', 'data.categorical'),
}


def write_json(path, data):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_stimuli(labels):
    rows, missing = [], []
    with labels.open(newline='') as stream:
        for raw in csv.DictReader(stream):
            index = int(raw['index'])
            score = float(raw['mean_human_complexity_rating'])
            if not np.isfinite(score) or not 0 <= score <= 100:
                raise ValueError(f'Invalid complexity for {index}: {score}')
            source = ROOT.parent / raw['image_path']
            if not raw['image_path'] or not source.is_file():
                missing.append(index)
                continue
            values = {'complexity': min(int(score // 10), 9)}
            for key, (_, column) in ATTRIBUTES.items():
                if key != 'complexity':
                    count = float(raw[column])
                    if not np.isfinite(count) or count < 0 or not count.is_integer():
                        raise ValueError(f'Invalid {column} for {index}: {raw[column]}')
                    values[key] = int(count)
            rows.append(dict(index=index, score=score, bin=values['complexity'],
                             attributes=values, image_path=raw['image_path'],
                             source=str(source.resolve()), filename=raw['filename']))
    if len({row['index'] for row in rows}) != len(rows):
        raise ValueError('Duplicate stimulus indices')
    if not rows:
        raise ValueError('No available annotated stimuli')
    return sorted(rows, key=lambda row: row['index']), missing


def select_stimuli(rows, scope, per_bin=10, seed=0):
    if scope == 'all':
        return list(rows)
    if per_bin < 1:
        raise ValueError('--per-bin must be positive')
    rng = np.random.default_rng(seed)
    selected = []
    for bucket in range(10):
        eligible = sorted((r for r in rows if r['bin'] == bucket), key=lambda r: r['index'])
        if len(eligible) < per_bin:
            raise ValueError(f'Complexity bin {bucket*10}–{(bucket+1)*10} has {len(eligible)} images; need {per_bin}')
        positions = rng.choice(len(eligible), size=per_bin, replace=False)
        selected.extend(eligible[int(p)] for p in positions)
    return selected


def aggregate_maps(rows, maps):
    """Equal stimulus weight, t=0; counts are exact labels, not inferred from pixels."""
    maps = np.asarray(maps)
    if maps.ndim != 2 or len(rows) != len(maps) or not np.isfinite(maps).all():
        raise ValueError('Invalid map matrix')
    grand_mean = maps.mean(axis=0, dtype=np.float64)
    groups = []
    for attribute in ATTRIBUTES:
        for value in sorted({r['attributes'][attribute] for r in rows}):
            positions = [i for i, r in enumerate(rows) if r['attributes'][attribute] == value]
            mean = maps[positions].mean(axis=0, dtype=np.float64)
            label = (f'{value*10}–{(value+1)*10}' + (' (inclusive)' if value == 9 else ' (upper excluded)')
                     if attribute == 'complexity' else str(value))
            groups.append(dict(attribute=attribute, value=value, label=label,
                               indices=[rows[i]['index'] for i in positions],
                               count=len(positions), mean=mean, contrast=mean-grand_mean))
    return groups, grand_mean


def load_predictions(rows, output, background_rgb):
    maps, receipts = [], []
    for row in rows:
        index = row['index']
        path = output / f'prediction_{index}.npz'
        receipt = json.loads((output / f'prediction_{index}.json').read_text())
        expected = dict(protocol=PROTOCOL, source_sha256=row['sha256'],
                        background_rgb=background_rgb, model='facebook/tribev2')
        if receipt['identity'] != expected or hashlib.sha256(path.read_bytes()).hexdigest() != receipt['prediction_sha256']:
            raise ValueError(f'Stale or corrupted prediction for {index}')
        with np.load(path) as saved:
            values = saved['predictions']
            if values.shape != (3, 20484) or not np.isfinite(values).all():
                raise ValueError(f'Invalid prediction for {index}')
            np.testing.assert_array_equal(saved['times'], [0, 1, 2])
            maps.append(values[0].copy())
        receipts.append(receipt)
    return np.asarray(maps), receipts


def estimate_runtime(receipts, total_count, setup_seconds):
    # Cached feature extraction is much faster and cannot estimate a fresh full run.
    measured = [r for r in receipts if r['encoded_windows'] > 0]
    if not measured:
        return dict(measured_count=0, total_stimuli=total_count,
                    note='No uncached encoder timings available; no fresh-run estimate.')
    seconds = np.array([r['total_seconds'] for r in measured])
    mean = float(seconds.mean())
    return dict(measured_count=len(measured), total_stimuli=total_count,
                device=measured[0]['device'], setup_seconds=setup_seconds,
                sampled_processing_seconds=float(seconds.sum()),
                mean_seconds_per_stimulus=mean, median_seconds_per_stimulus=float(np.median(seconds)),
                full_prediction_seconds=setup_seconds + total_count * mean,
                remaining_prediction_seconds=max(0, total_count-len(receipts))*mean,
                note='Linear estimate from uncached stimulus preparation + inference + save on this device. '
                     'Excludes report rendering, downloads, and interruptions; balanced sample, not a confidence interval.')


def render_groups(groups, output):
    from compare import brain
    from nilearn import datasets
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    fs = datasets.fetch_surf_fsaverage(mesh='fsaverage5', data_dir=str(ROOT / 'cache/nilearn'))
    folder = output / 'brain_maps'
    folder.mkdir(exist_ok=True)
    limits = {kind: max(1e-12, max(float(np.abs(g[kind]).max()) for g in groups))
              for kind in ('mean', 'contrast')}
    views = [('left', 'lateral'), ('left', 'medial'), ('right', 'lateral'), ('right', 'medial')]
    for number, group in enumerate(groups, 1):
        for kind in ('mean', 'contrast'):
            fig = plt.figure(figsize=(12, 3.4), facecolor='white')
            gs = fig.add_gridspec(1, 4, wspace=0)
            limit = limits[kind]
            for column, (hemi, view) in enumerate(views):
                brain(fig, gs[0, column], fs, group[kind], hemi, view, limit)
            fig.subplots_adjust(top=.92, bottom=.28, left=0, right=1)
            cax = fig.add_axes([.34, .15, .32, .03])
            fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-limit, limit), cmap='RdBu_r'),
                         cax=cax, orientation='horizontal', label='Predicted response (model units)')
            name = f"{group['attribute']}_{group['value']}_{kind}.png"
            fig.savefig(folder / name, dpi=110, facecolor='white')
            plt.close(fig)
            group[f'{kind}_image'] = f'brain_maps/{name}'
        print(f'Rendered group {number}/{len(groups)}: {group["attribute"]} {group["label"]}', flush=True)
    return limits


def export_report(rows, maps, output, selection, timing, render=True):
    started = time.perf_counter()
    groups, grand_mean = aggregate_maps(rows, maps)
    limits = render_groups(groups, output) if render else {}
    np.savez_compressed(output / 'aggregate_maps.npz',
                        means=np.asarray([g['mean'] for g in groups]),
                        contrasts=np.asarray([g['contrast'] for g in groups]),
                        grand_mean=grand_mean, indices=[r['index'] for r in rows], maps=maps)
    public_groups = [{k: v for k, v in g.items() if k not in ('mean', 'contrast')} for g in groups]
    timing['report_seconds'] = time.perf_counter()-started
    write_json(output / 'timing.json', timing)
    write_json(output / 'explorer.json', dict(
        schema_version=1, protocol=PROTOCOL, scope=selection['scope'], seed=selection['seed'],
        count=len(rows), eligible_count=selection['eligible_count'], missing_indices=selection['missing_indices'],
        attributes=[dict(key=k, label=v[0]) for k,v in ATTRIBUTES.items()],
        rows=[{k:v for k,v in r.items() if k not in ('source', 'sha256')} for r in rows],
        groups=public_groups, limits=limits, timing=timing,
        aggregation='Equal-weight arithmetic mean across stimuli, first predicted response at t=0.',
        contrast='Group mean minus the mean of all selected stimuli (including this group).',
        sampling='Ten-point complexity bins; upper edge excluded except 100. Sampling without replacement.'
                 if selection['scope'] == 'sample' else 'All available annotated stimuli.'))
    print(f'Explorer data: {output / "explorer.json"}', flush=True)
    print(json.dumps(timing, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=['sample', 'all'], default='sample')
    parser.add_argument('--per-bin', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--labels', type=Path, default=ROOT.parent / 'label/output/labels.csv')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--background', default='white')
    parser.add_argument('--device', default='cuda')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--predict', action='store_true', help='Predict missing stimuli, then export the web report')
    action.add_argument('--sample-only', action='store_true', help='Save selection and source manifest only')
    parser.add_argument('--reuse-from', type=Path, help='Reuse validated prediction artifacts from a previous dataset run')
    args = parser.parse_args()
    from PIL import ImageColor
    background = list(ImageColor.getrgb(args.background))
    if len(background) != 3:
        parser.error('Background must be an opaque RGB color')
    output = (args.output_dir or ROOT / f'outputs/massvis_{args.scope}').resolve()
    eligible, missing = load_stimuli(args.labels)
    rows = select_stimuli(eligible, args.scope, args.per_bin, args.seed)
    for row in rows:
        row['sha256'] = hashlib.sha256(Path(row['source']).read_bytes()).hexdigest()
    selection = dict(scope=args.scope, seed=args.seed, per_bin=args.per_bin if args.scope == 'sample' else None,
                     protocol=PROTOCOL, background_rgb=background, eligible_count=len(eligible),
                     missing_indices=missing, labels_sha256=hashlib.sha256(args.labels.read_bytes()).hexdigest(), rows=rows)
    selection_path = output / 'selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        parser.error('Output has a different selection or preprocessing; choose a new --output-dir')
    (output / 'inputs').mkdir(parents=True, exist_ok=True)
    write_json(selection_path, selection)
    manifest = []
    for row in rows:
        target = output / 'inputs' / f'{row["index"]}.png'
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
            shutil.copyfile(row['source'], target)
        manifest.append(dict(index=row['index'], sha256=row['sha256']))
    write_json(output / 'inputs/manifest.json', manifest)
    print(f'Selected {len(rows)} / {len(eligible)} stimuli; bins: {dict(sorted(Counter(r["bin"] for r in rows).items()))}', flush=True)
    if missing:
        print(f'Unavailable annotated sources: {missing}', flush=True)
    if args.sample_only:
        return
    if args.reuse_from:
        reuse_predictions(rows, args.reuse_from.resolve(), output, background)
    if args.predict:
        subprocess.run([sys.executable, str(ROOT / 'run.py'), '--device', args.device,
                        '--manifest', str(output / 'inputs/manifest.json'), '--inputs-dir', str(output / 'inputs'),
                        '--outputs-dir', str(output), '--background', args.background, '--resume'], check=True)
    maps, receipts = load_predictions(rows, output, background)
    metadata = json.loads((output / 'run_metadata.json').read_text())
    timing = estimate_runtime(receipts, len(eligible), metadata['model_load_seconds'])
    export_report(rows, maps, output, selection, timing)


def reuse_predictions(rows, source, output, background):
    """Import valid sampled predictions into a full run, preserving their timing receipts."""
    for row in rows:
        index = row['index']
        receipt_path = source / f'prediction_{index}.json'
        if not receipt_path.is_file() or (output / receipt_path.name).exists():
            continue
        _, receipts = load_predictions([row], source, background)
        receipt = receipts[0]
        old_video = ROOT / receipt['prepared']['video']
        if hashlib.sha256(old_video.read_bytes()).hexdigest() != receipt['prepared']['video_sha256']:
            raise ValueError(f'Corrupted prepared video for {index}')
        target = output / 'inputs' / PROTOCOL / ''.join(f'{c:02x}' for c in background)
        target.mkdir(parents=True, exist_ok=True)
        for suffix in ('.png', '.mp4'):
            shutil.copyfile(old_video.with_suffix(suffix), target / f'{index}{suffix}')
        import os
        receipt['prepared']['video'] = os.path.relpath(target / f'{index}.mp4', ROOT)
        shutil.copyfile(source / f'prediction_{index}.npz', output / f'prediction_{index}.npz')
        write_json(output / receipt_path.name, receipt)


if __name__ == '__main__':
    main()
