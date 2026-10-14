"""Prepare controlled studies and generate their TRIBE prediction data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from dataset import (ROOT, compatible_selection, estimate_runtime, export_report,
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


def reuse_identical_stimuli(rows, output, background, precision, prepare_only=False):
    """Give byte-identical study copies their own validated inputs and receipts."""
    target = output / 'inputs' / PROTOCOL / ''.join(f'{value:02x}' for value in background)
    reused = []
    by_index = {row['index']: row for row in rows}
    for row in rows:
        canonical = row.get('reuse_prediction_from')
        if canonical is None:
            continue
        original = by_index[canonical]
        if original['sha256'] != row['sha256']:
            raise ValueError('Cannot reuse predictions from different source bytes')
        receipt = None
        if not prepare_only:
            _, receipts = load_predictions([original], output, background, precision)
            receipt = receipts[0]
            if hashlib.sha256((ROOT / receipt['prepared']['video']).read_bytes()).hexdigest() != receipt['prepared']['video_sha256']:
                raise ValueError(f'Corrupted prepared video for {canonical}')
        for suffix in ('.png', '.mp4'):
            shutil.copyfile(target / f'{canonical}{suffix}', target / f'{row["index"]}{suffix}')
        video = target / f'{row["index"]}.mp4'
        alias = output / 'inputs' / video.name
        alias.unlink(missing_ok=True)
        alias.symlink_to(os.path.relpath(video, alias.parent))
        if receipt is not None:
            from run import write_events
            write_events(video, row['index'], output)
            shutil.copyfile(output / f'prediction_{canonical}.npz', output / f'prediction_{row["index"]}.npz')
            receipt['prepared'] = dict(receipt['prepared'], index=row['index'], video=os.path.relpath(video, ROOT))
            receipt.update(reused_from_index=canonical, encoded_windows=0, inference_seconds=0,
                           total_seconds=0, preparation_seconds=0, brain_prediction_seconds=0,
                           encoder_batch_seconds=0)
            write_json(output / f'prediction_{row["index"]}.json', receipt)
        reused.append(row['index'])
    return reused


def run_study(dataset, output=None, predict=False, sample_only=False, prepare_only=False,
              allow_partial=False, background='white', device='cuda', thread=8,
              batch_size=1, cpu_threads=8, precision='fp32'):
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
                     attributes=manifest['attributes'], archive_sha256=manifest.get('archive_sha256'),
                     eligible_count=len(rows), missing_indices=[], rows=rows)
    if 'source_documents' in manifest:
        selection.update(source_documents=manifest['source_documents'], source_sha256=manifest['source_sha256'])
    save_selection(output, selection)
    unique_count = len({row['sha256'] for row in rows})
    print(f'{dataset}: {len(rows)} study stimuli selected ({unique_count} distinct source images)', flush=True)
    if sample_only:
        return
    if predict or prepare_only:
        prediction_manifest = output / 'inputs/manifest.json'
        if any('reuse_prediction_from' in row for row in rows):
            prediction_manifest = output / 'inputs/prediction_manifest.json'
            write_json(prediction_manifest, [dict(index=row['index'], sha256=row['sha256'])
                       for row in rows if 'reuse_prediction_from' not in row])
        command = [sys.executable, str(ROOT / 'run.py'), '--device', device,
                   '--manifest', str(prediction_manifest), '--inputs-dir', str(output / 'inputs'),
                   '--outputs-dir', str(output), '--background', background,
                   '--thread', str(thread), '--batch-size', str(batch_size),
                   '--cpu-threads', str(cpu_threads), '--precision', precision, '--resume']
        if prepare_only:
            command.append('--prepare-only')
        subprocess.run(command, check=True)
        reused = reuse_identical_stimuli(rows, output, rgb, precision, prepare_only)
        if reused and not prepare_only:
            metadata_path = output / 'run_metadata.json'
            metadata = json.loads(metadata_path.read_text())
            metadata.update(selected_count=len(rows), unique_prediction_count=len(rows) - len(reused),
                            reused_stimulus_indices=reused)
            write_json(metadata_path, metadata)
            print(f'Reused {len(reused)} validated predictions for byte-identical stimulus copies', flush=True)
    if prepare_only or predict:
        return
    included = report_rows(rows, output, allow_partial)
    maps, receipts = load_predictions(included, output, rgb, precision)
    metadata_path = output / 'run_metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    timing = estimate_runtime(receipts, unique_count, metadata.get('model_load_seconds', 0), metadata.get('benchmark'))
    if unique_count != len(rows):
        timing['selected_observations'] = len(rows)
        timing['note'] += ' Byte-identical stimulus observations reuse the same prediction.'
    export_report(included, maps, output, selection, timing)
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
        export_report(completed, subset_maps, folder, subset, subset_timing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['kylie', 'hayeong', 'arunkumar'], required=True)
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
    run_study(**vars(parser.parse_args()))
