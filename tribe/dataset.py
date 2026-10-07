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
from parallelism import positive_threads
from gpu_encoder import prediction_identity

ROOT = Path(__file__).resolve().parent
ATTRIBUTES = {
    'complexity': ('Perceived complexity', 'mean_human_complexity_rating'),
    'charts': ('Number of charts', 'design.chart_count'),
    'colors': ('Number of distinct colors', 'color.color_count'),
    'quantitative': ('Number of quantitative variables', 'data.quantitative'),
    'categorical': ('Number of categorical variables', 'data.categorical'),
    'chart_types': ('Number of distinct chart types', 'design.chart_types_count'),
    'no_text': ('No text', 'text.no_text'),
    'axes_labels': ('Axis labels', 'text.axes_labels'),
    'axes_text': ('Axis text', 'text.axes_text'),
    'titles': ('Titles', 'text.titles'),
    'annotations': ('Annotations', 'text.annotations'),
    'captions': ('Captions', 'text.captions'),
    'legend_text': ('Legend text', 'text.legend_text'),
    'legend_title': ('Legend titles', 'text.legend_title'),
    'text_only': ('Text only', 'text.text_only'),
    'black_and_white': ('Black and white', 'color.black_and_white'),
    'background_color': ('Non-white background', 'color.background_color'),
    'multi_panel': ('Multiple panels', 'design.multi_panel'),
    'area': ('Area chart', 'design.area'),
    'bar': ('Bar chart', 'design.bar'),
    'circle': ('Circle chart', 'design.circle'),
    'diagram': ('Diagram', 'design.diagram'),
    'distribution': ('Distribution chart', 'design.distribution'),
    'grid_or_matrix': ('Grid or matrix', 'design.grid_or_matrix'),
    'line': ('Line chart', 'design.line'),
    'map': ('Map', 'design.map'),
    'point': ('Point chart', 'design.point'),
    'table': ('Table', 'design.table'),
    'text': ('Text chart', 'design.text'),
    'trees_or_networks': ('Trees or networks', 'design.trees_or_networks'),
    'category': ('Source category', 'category'),
}
COUNT_LABELS = {
    'charts': ('chart', 'charts'),
    'colors': ('color', 'colors'),
    'quantitative': ('quantitative variable', 'quantitative variables'),
    'categorical': ('categorical variable', 'categorical variables'),
    'chart_types': ('chart type', 'chart types'),
}
BINARY_ATTRIBUTES = set(ATTRIBUTES) - set(COUNT_LABELS) - {'complexity', 'category'}
CATEGORY_LABELS = {'G': 'Government', 'I': 'Infographic', 'N': 'News', 'S': 'Science'}


def group_label(attribute, value):
    if attribute == 'complexity':
        return f'{value*10}–{(value+1)*10}' + (' (inclusive)' if value == 9 else ' (upper excluded)')
    if attribute == 'category':
        return CATEGORY_LABELS[value]
    if attribute in BINARY_ATTRIBUTES:
        return 'Yes' if value else 'No'
    return str(value)


def attribute_metadata():
    attributes = []
    for key, (label, _) in ATTRIBUTES.items():
        kind = ('bin' if key == 'complexity' else 'category' if key == 'category'
                else 'binary' if key in BINARY_ATTRIBUTES else 'count')
        item = dict(key=key, label=label, kind=kind)
        if key in COUNT_LABELS:
            item['count_labels'] = COUNT_LABELS[key]
        attributes.append(item)
    return attributes


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
                if key == 'complexity':
                    continue
                if key == 'category':
                    if raw[column] not in CATEGORY_LABELS:
                        raise ValueError(f'Invalid {column} for {index}: {raw[column]}')
                    values[key] = raw[column]
                    continue
                count = float(raw[column])
                if (not np.isfinite(count) or count < 0 or not count.is_integer()
                        or (key in BINARY_ATTRIBUTES and count not in (0, 1))):
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
            groups.append(dict(attribute=attribute, value=value, label=group_label(attribute, value),
                               indices=[rows[i]['index'] for i in positions],
                               count=len(positions), mean=mean, contrast=mean-grand_mean))
    return groups, grand_mean


def load_predictions(rows, output, background_rgb, precision='fp32'):
    maps, receipts = [], []
    for row in rows:
        index = row['index']
        path = output / f'prediction_{index}.npz'
        receipt = json.loads((output / f'prediction_{index}.json').read_text())
        expected = prediction_identity(PROTOCOL, row['sha256'], background_rgb, precision)
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


def estimate_runtime(receipts, total_count, setup_seconds, benchmark=None):
    if benchmark:
        mean = benchmark['processing_wall_seconds'] / benchmark['count']
        return dict(measured_count=benchmark['count'], total_stimuli=total_count,
                    device=benchmark['device'], thread=benchmark['thread'], setup_seconds=setup_seconds,
                    batch_size=benchmark.get('batch_size',1), precision=benchmark.get('precision','fp32'),
                    sampled_processing_seconds=benchmark['processing_wall_seconds'],
                    mean_seconds_per_stimulus=mean,
                    full_prediction_seconds=setup_seconds+total_count*mean,
                    remaining_prediction_seconds=max(0,total_count-len(receipts))*mean,
                    note='Estimate from measured uncached wall-clock throughput for this recorded execution configuration. '
                         'Excludes report rendering, downloads, and interruptions.')
    # Cached feature extraction is much faster and cannot estimate a fresh full run.
    measured = [r for r in receipts if r['encoded_windows'] > 0 and r.get('execution') not in ('prefetched', 'pooled_vjepa_v1')]
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


def export_static_site(output):
    """Refresh the shared viewer and discover all sibling reports."""
    from web import build_site
    build_site(output.parent)


def export_report(rows, maps, output, selection, timing, render=True):
    started = time.perf_counter()
    groups, grand_mean = aggregate_maps(rows, maps)
    limits = render_groups(groups, output) if render else {}
    np.savez_compressed(output / 'aggregate_maps.npz',
                        means=np.asarray([g['mean'] for g in groups]),
                        contrasts=np.asarray([g['contrast'] for g in groups]),
                        grand_mean=grand_mean, indices=[r['index'] for r in rows], maps=maps)
    completed_indices = {row['index'] for row in rows}
    public_groups = [{k: v for k, v in g.items() if k not in ('mean', 'contrast')} for g in groups]
    timing['report_seconds'] = time.perf_counter()-started
    write_json(output / 'timing.json', timing)
    write_json(output / 'explorer.json', dict(
        schema_version=1, protocol=PROTOCOL, scope=selection['scope'], seed=selection['seed'],
        precision=selection.get('precision','fp32'),
        count=len(rows), selected_count=len(selection['rows']),
        complete=len(rows) == len(selection['rows']),
        pending_indices=[r['index'] for r in selection['rows'] if r['index'] not in completed_indices],
        eligible_count=selection['eligible_count'], missing_indices=selection['missing_indices'],
        attributes=attribute_metadata(),
        rows=[{k:v for k,v in r.items() if k not in ('source', 'sha256')} for r in rows],
        groups=public_groups, limits=limits, timing=timing,
        aggregation='Equal-weight arithmetic mean across stimuli, first predicted response at t=0.',
        contrast='Group mean minus the mean of all included stimuli (including this group).',
        sampling='Partial run: only completed predictions are included.' if len(rows) != len(selection['rows']) else 'Ten-point complexity bins; upper edge excluded except 100. Sampling without replacement.'
                 if selection['scope'] == 'sample' else 'All available annotated stimuli.'))
    export_static_site(output)
    print(f'Static report: {output / "index.html"}', flush=True)
    print(f'Explorer data: {output / "explorer.json"}', flush=True)
    print(json.dumps(timing, indent=2), flush=True)


def report_rows(rows, output, allow_partial=False):
    completed = [r for r in rows if (output / f"prediction_{r['index']}.npz").is_file()
                 and (output / f"prediction_{r['index']}.json").is_file()]
    if len(completed) != len(rows) and not allow_partial:
        raise ValueError(f'{len(completed)}/{len(rows)} predictions complete. Resume inference or use --allow-partial.')
    if not completed:
        raise ValueError('No completed predictions available to report')
    return completed


def compatible_selection(previous, current):
    """Allow added grouping annotations while preserving every prediction identity."""
    if previous.keys() != current.keys():
        return False
    if any(previous[key] != current[key] for key in current if key != 'rows'):
        return False
    if len(previous['rows']) != len(current['rows']):
        return False
    for old, new in zip(previous['rows'], current['rows']):
        if {k: v for k, v in old.items() if k != 'attributes'} != {k: v for k, v in new.items() if k != 'attributes'}:
            return False
        if any(new['attributes'].get(key) != value for key, value in old['attributes'].items()):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=['sample', 'all'], default='sample')
    parser.add_argument('--per-bin', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--labels', type=Path, default=ROOT.parent / 'label/output/labels.csv')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--background', default='white')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--thread', type=positive_threads, default=8,
                        help='Concurrent CPU stimulus-preparation workers (default: 8)')
    parser.add_argument('--batch-size', type=positive_threads, default=1,
                        help='Stimuli per GPU encoder forward (default: 1)')
    parser.add_argument('--cpu-threads', type=positive_threads, default=8,
                        help='PyTorch CPU compute threads (default: 8)')
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--predict', action='store_true', help='Predict missing stimuli, then export the web report')
    action.add_argument('--sample-only', action='store_true', help='Save selection and source manifest only')
    parser.add_argument('--allow-partial', action='store_true',
                        help='Report only completed predictions, explicitly labeled as partial')
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
    if args.precision != 'fp32':
        selection['precision'] = args.precision
    selection_path = output / 'selection.json'
    if selection_path.exists() and not compatible_selection(json.loads(selection_path.read_text()), selection):
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
        reuse_predictions(rows, args.reuse_from.resolve(), output, background, args.precision)
    if args.predict:
        subprocess.run([sys.executable, str(ROOT / 'run.py'), '--device', args.device,
                        '--manifest', str(output / 'inputs/manifest.json'), '--inputs-dir', str(output / 'inputs'),
                        '--outputs-dir', str(output), '--background', args.background, '--thread', str(args.thread), '--batch-size', str(args.batch_size),
                        '--cpu-threads', str(args.cpu_threads), '--precision', args.precision, '--resume'], check=True)
    rows = report_rows(rows, output, args.allow_partial)
    maps, receipts = load_predictions(rows, output, background, args.precision)
    metadata_path = output / 'run_metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {'model_load_seconds': 0}
    timing = estimate_runtime(receipts, len(eligible), metadata['model_load_seconds'], metadata.get('benchmark'))
    if not metadata_path.exists():
        timing['note'] += ' Model setup timing is unavailable for this interrupted run.'
    export_report(rows, maps, output, selection, timing)


def reuse_predictions(rows, source, output, background, precision='fp32'):
    """Import valid sampled predictions into a full run, preserving their timing receipts."""
    for row in rows:
        index = row['index']
        receipt_path = source / f'prediction_{index}.json'
        if not receipt_path.is_file() or (output / receipt_path.name).exists():
            continue
        _, receipts = load_predictions([row], source, background, precision)
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
