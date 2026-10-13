"""Run controlled studies through the existing TRIBE engine and shared cortical viewer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from dataset import (ROOT, VIEW_LAYOUTS, compatible_selection, estimate_runtime, export_report,
                     load_predictions, report_rows, reuse_predictions, write_json)
from parallelism import positive_threads
from preprocessing import PROTOCOL
from studies import REPOSITORY, prepare_manifest


def save_selection(output, selection):
    path = output / 'selection.json'
    if path.exists() and not compatible_selection(json.loads(path.read_text()), selection):
        raise ValueError(f'Different selection or preprocessing in {output}; choose a new output directory')
    (output / 'inputs').mkdir(parents=True, exist_ok=True)
    write_json(path, selection)
    for row in selection['rows']:
        target = output / 'inputs' / f'{row["index"]}.png'
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
            shutil.copyfile(row['source'], target)
    write_json(output / 'inputs/manifest.json',
               [dict(index=row['index'], sha256=row['sha256']) for row in selection['rows']])


def run_study(dataset, output=None, predict=False, sample_only=False, prepare_only=False,
              allow_partial=False, background='white', device='cuda', thread=8,
              batch_size=1, cpu_threads=8, precision='fp32', views='all'):
    from PIL import ImageColor
    rgb = list(ImageColor.getrgb(background))
    if len(rgb) != 3:
        raise ValueError('Background must be an opaque RGB color')
    manifest = prepare_manifest(dataset)
    output = (output or ROOT / 'outputs' / dataset).resolve()
    rows = [dict(row, source=str(REPOSITORY / row['source'])) for row in manifest['rows']]
    selection = dict(scope='study', seed=0, protocol=PROTOCOL, background_rgb=rgb,
                     precision=precision, dataset=dataset, title=manifest['title'],
                     description=manifest['description'], sampling=manifest['sampling'],
                     attributes=manifest['attributes'], archive_sha256=manifest['archive_sha256'],
                     eligible_count=len(rows), missing_indices=[], rows=rows)
    save_selection(output, selection)
    print(f'{dataset}: {len(rows)} unique study stimuli selected', flush=True)
    if sample_only:
        return
    if predict or prepare_only:
        command = [sys.executable, str(ROOT / 'run.py'), '--device', device,
                   '--manifest', str(output / 'inputs/manifest.json'), '--inputs-dir', str(output / 'inputs'),
                   '--outputs-dir', str(output), '--background', background,
                   '--thread', str(thread), '--batch-size', str(batch_size),
                   '--cpu-threads', str(cpu_threads), '--precision', precision, '--resume']
        if prepare_only:
            command.append('--prepare-only')
        subprocess.run(command, check=True)
    if prepare_only:
        return
    included = report_rows(rows, output, allow_partial)
    maps, receipts = load_predictions(included, output, rgb, precision)
    metadata_path = output / 'run_metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    timing = estimate_runtime(receipts, len(rows), metadata.get('model_load_seconds', 0), metadata.get('benchmark'))
    export_report(included, maps, output, selection, timing, views=views)
    for comparison in manifest['comparisons']:
        comparison_rows = [row for row in rows if row['index'] in comparison['indices']]
        folder = output.with_name(output.name + '_' + comparison['id'])
        subset = dict(selection, rows=comparison_rows, eligible_count=len(comparison_rows),
                      title=comparison['title'], description=comparison['description'],
                      attributes=comparison['attributes'], sampling=comparison['description'])
        save_selection(folder, subset)
        reuse_predictions(comparison_rows, output, folder, rgb, precision)
        completed = report_rows(comparison_rows, folder, allow_partial)
        subset_maps, subset_receipts = load_predictions(completed, folder, rgb, precision)
        subset_timing = estimate_runtime(subset_receipts, len(comparison_rows), 0)
        subset_timing['note'] = 'Predictions reused from the parent study run; no additional inference. ' + subset_timing['note']
        export_report(completed, subset_maps, folder, subset, subset_timing, views=views)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['kylie', 'hayeong'], required=True)
    parser.add_argument('--output-dir', dest='output', type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--predict', action='store_true')
    action.add_argument('--sample-only', action='store_true')
    action.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--background', default='white')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--thread', type=positive_threads, default=8)
    parser.add_argument('--batch-size', type=positive_threads, default=1)
    parser.add_argument('--cpu-threads', type=positive_threads, default=8)
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--views', choices=VIEW_LAYOUTS, default='all')
    run_study(**vars(parser.parse_args()))
