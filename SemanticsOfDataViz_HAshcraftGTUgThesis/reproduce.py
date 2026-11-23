"""Export the authoritative pilot-4 stimuli, or regenerate the notebook's recipe.

The export preserves published PNG bytes: the saved pilot and the current
generator are different revisions. Generation is an explicit, separate action.
No MATLAB, Colab, Excel engine, browser, or new dependencies are required.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / 'DataVizThesis_Script/DataVizThesis_HAScript.m'
NOTEBOOK = ROOT / 'DataVizThesis_StimGen/DataVizStim(figure)Gen.ipynb'
WORKBOOK = ROOT / 'DataVizThesis_Script/stimSetPilot_4.xlsx'
ORIGINALS = ROOT / 'DataVizThesis_Script/dataVizSetPilot_4'
DEFAULT_OUTPUT = ROOT / 'reproduced/pilot4'
BLOCK_ORDER = 'XPQMOPMXQOPQMOX'
TASKS = {
    'P': ('Passive detection', 'Press 1 when you see the figure.', '1; accuracy is not scored'),
    'Q': ('Trend judgment', 'What is trend in the data?', '1 = positive; 2 = negative'),
    'M': ('Motion judgment', 'Which direction is the figure moving?', '1 = left; 2 = right'),
    'O': ('Outlier judgment', 'Is there an outlier in the data?', '1 = yes; 2 = no'),
}
PATTERN = re.compile(
    r'^scatter_plot(?P<index>\d+)_(?P<tag>posSrngCor|posCor|negSrngCor|negCor)'
    r'\(r=(?P<r>[+-]?\d+\.\d+)\)'
    r'(?:_out\(MD=(?P<md>\d+\.\d+),z=(?P<z>\d+\.\d+)\))?\.png$')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_filename(filename):
    match = PATTERN.fullmatch(filename)
    if not match:
        raise ValueError(f'Unrecognized stimulus filename: {filename}')
    values = match.groupdict()
    r = float(values['r'])
    tag = values['tag']
    if (r > 0) != tag.startswith('pos') or r == 0 or not -1 <= r <= 1:
        raise ValueError(f'Invalid correlation in {filename}')
    return dict(index=int(values['index']), filename=filename, r=r,
                correlation_direction='positive' if tag.startswith('pos') else 'negative',
                correlation_class=tag, has_outlier=int(values['md'] is not None),
                mahalanobis_distance=float(values['md']) if values['md'] else None,
                outlier_z=float(values['z']) if values['z'] else None,
                ink_density='unknown')


def published_rows():
    """Read the actual MATLAB stimulus list using only the Python standard library."""
    namespace = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(WORKBOOK) as archive:
        sheet = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        names = [''.join(t.text or '' for t in row.findall('.//s:t', namespace))
                 for row in sheet.findall('s:sheetData/s:row', namespace)]
    if names[0] != 'stim_fns' or len(names) != 73 or len(set(names)) != 73:
        raise ValueError('Expected the authoritative 72 unique pilot-4 filenames')
    if set(names[1:]) != {p.name for p in ORIGINALS.glob('*.png')}:
        raise ValueError('Workbook and pilot-4 PNGs disagree')
    rows = [parse_filename(name) for name in names[1:]]
    if [r['index'] for r in rows] != list(range(1, 73)):
        raise ValueError('Unexpected pilot-4 workbook ordering')
    return rows


def summarize(rows):
    """Keep task design, visual pools, and protocol parameters in separate tidy CSVs."""
    counts = Counter((r['correlation_direction'], r['has_outlier']) for r in rows)
    conditions = []
    for code, (task, prompt, response) in TASKS.items():
        for direction in ('positive', 'negative'):
            for outlier in (0, 1):
                conditions.append(dict(
                    task_code=code, task=task, correlation_direction=direction,
                    has_outlier=outlier, available_images=counts[direction, outlier],
                    trials_per_task_per_run=18, blocks_per_task_per_run=3,
                    trials_per_block=6, stimulus_seconds=3, intertrial_fixation_seconds=0.5,
                    prompt_seconds=3, prompt=prompt, response_mapping=response,
                    cell_trials_per_task='Variable; only the task marginals are balanced',
                    task_balance='9 positive / 9 negative and 9 outlier / 9 no-outlier per task',
                    simulated='Image only; task instructions and motion are omitted',
                    source='DataVizThesis_Script/DataVizThesis_HAScript.m'))
    write_csv(ROOT / 'study_conditions.csv', conditions)
    write_csv(ROOT / 'visual_conditions.csv', [dict(
        correlation_direction=direction, has_outlier=outlier, count=count,
        r_min=min(r['r'] for r in rows if (r['correlation_direction'], r['has_outlier']) == (direction, outlier)),
        r_max=max(r['r'] for r in rows if (r['correlation_direction'], r['has_outlier']) == (direction, outlier)))
        for (direction, outlier), count in sorted(counts.items())])
    parameters = [
        ('authoritative_stimuli', '72 pilot-4 PNGs referenced by stimSetPilot_4.xlsx', 'workbook and PNGs'),
        ('chart_type', 'Scatterplots only; bar/line/violin occur only in a notebook heading', 'generator code'),
        ('generator_seed_in_current_notebook', '1; not the pilot-4 image seed', 'generator cell 3'),
        ('base_points', '250 plus 1 appended point; 251 total', 'generator cell 3'),
        ('target_correlations', '72 evenly spaced values from +1 to -1; zero is absent', 'generator cell 3'),
        ('correlation_labels', 'Strong positive >=0.7; positive (0,0.7); negative (-0.7,0); strong negative <=-0.7', 'generator cell 3'),
        ('correlation_metadata_precision', 'Filename values rounded to 3 decimals; target r and final r differ', 'generator cell 3'),
        ('outliers', '36 of 72; randomly allocated, not a balanced direction-by-outlier factorial', 'generator and workbook'),
        ('current_generator_outlier_rule', 'Cauchy candidate; Mahalanobis-distance z in [6,10]; retain target sign', 'generator cell 3'),
        ('pilot4_outlier_z_range', f"{min(r['outlier_z'] for r in rows if r['has_outlier']):.3f} to {max(r['outlier_z'] for r in rows if r['has_outlier']):.3f}; differs from current generator", 'published filenames'),
        ('ink_density', 'Unknown for all 72; not an implemented factor in the supplied set', 'RSA metadata and generator'),
        ('raster', '732 x 732 RGBA; 2.44 inches square at 300 dpi; blue s=5 markers; bottom/left axes; no ticks or labels', 'PNG metadata and generator'),
        ('matlab_rng_seed', '1234; NumPy and MATLAB RNG calls are not interchangeable', 'MATLAB rng(1234)'),
        ('first_run_block_order', BLOCK_ORDER, 'MATLAB block1'),
        ('later_run_block_order', BLOCK_ORDER[::-1], 'MATLAB block2'),
        ('fixation_blocks', '3 per run; code X; 12 seconds each', 'MATLAB block loop'),
        ('task_blocks', '12 per run; 6 stimuli each; 3-second prompt per block', 'MATLAB block loop'),
        ('nominal_run_duration_seconds', '324 excluding scanner wait and frame timing', '3*12 + 12*3 + 72*(3+0.5)'),
        ('stimulus_reuse', 'Each of 72 appears once per run; each task has 18 trials', 'MATLAB stimulus/task pools'),
        ('later_run_task_assignment', 'All runs after the first use firstRunTaskRows([end,1:end-1]); repeated runs do not further rotate', 'MATLAB run loop'),
        ('image_scale', 'Fit within 80% of screen width and height; white background', 'MATLAB imgScale'),
        ('motion', 'All tasks: left/right linear travel of 0.005*screen_width in 3 seconds; no >3 consecutive identical directions', 'MATLAB motionVec and totalTravelPixels'),
        ('task_prompt_appearance', 'Times New Roman 80; black on white; prompt displayed before each block', 'MATLAB PTB setup'),
        ('simulation_protocol', 'Existing TRIBE: independent silent static 3s clips, 16fps, 292px canvas/256px crop, FP32, t=0 means', 'tribe/README.md'),
        ('interpretation', 'Descriptive model predictions; no task contrast, participant fMRI validation, or human complexity score', 'supplied resources and simulation scope'),
        ('proposal_status', 'Proposal describes 2 main tasks and future localizers; executable MATLAB implements P/Q/M/O; no localizer stimuli supplied', 'proposal versus executable script'),
    ]
    write_csv(ROOT / 'study_parameters.csv', [dict(parameter=k, value=v, evidence=e) for k, v, e in parameters])


def export_pilot(output=DEFAULT_OUTPUT):
    rows = published_rows()
    target = output / 'stimuli'
    target.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = ORIGINALS / row['filename']
        destination = target / row['filename']
        source_hash = digest(source)
        if destination.exists() and digest(destination) != source_hash:
            raise ValueError(f'Refusing to replace changed export: {destination}')
        if not destination.exists():
            shutil.copyfile(source, destination)
        row.update(sha256=source_hash, reproduction='published raster; byte-identical')
    write_csv(output / 'stimulus_metadata.csv', rows)
    provenance = dict(mode='pilot4', stimuli=72,
                      method='Lossless export of authoritative published rasters; not regenerated coordinates',
                      source_sha256={str(p.relative_to(ROOT)): digest(p)
                                     for p in (SCRIPT, NOTEBOOK, WORKBOOK, Path(__file__))},
                      protocol='Static images for the existing independent 3-second TRIBE pipeline',
                      limitation='MATLAB motion, preceding prompts, attention, and block timing are not simulated')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    summarize(rows)
    return rows


def generate_points(seed=1, z_min=6, z_max=10):
    """Port cell 3 faithfully, including its 251st-point and random-call behavior."""
    import numpy as np
    from scipy.spatial.distance import mahalanobis
    if not 0 < z_min < z_max:
        raise ValueError('Require 0 < z_min < z_max')
    # RandomState deliberately matches the legacy np.random.seed notebook API.
    rng = np.random.RandomState(seed)
    outliers = np.array([True] * 36 + [False] * 36)
    rng.shuffle(outliers)
    for index, n in enumerate(np.linspace(1, -1, 72), 1):
        data = rng.multivariate_normal([0, 0], np.eye(2), 250)
        x, y = data[:, 0], data[:, 1]
        x = (x - np.mean(x)) / np.std(x)
        y = y - np.mean(y)
        y = y - (np.dot(x, y) / np.dot(x, x)) * x
        y = y / np.std(y)
        y = np.sign(n) * x if abs(n) >= 1 else n * x + np.sqrt(1 - n**2) * y
        md = z = None
        if outliers[index - 1]:
            points = np.column_stack((x, y))
            center = np.array([np.mean(x), np.mean(y)])
            inverse = np.linalg.pinv(np.cov(points, rowvar=False))
            distances = np.array([mahalanobis(p, center, inverse) for p in points])
        for attempt in range(1_000_000):
            if outliers[index - 1]:
                cauchy = abs(rng.standard_cauchy())
                theta = rng.uniform(0, 2 * np.pi)
                point = center + [np.std(x) * cauchy * np.cos(theta),
                                  np.std(y) * cauchy * np.sin(theta)]
                md = mahalanobis(point, center, inverse)
                z = (md - np.mean(distances)) / np.std(distances, ddof=1)
            else:
                point = rng.multivariate_normal([0, 0], np.eye(2), 1)[0]
            xx, yy = np.append(x, point[0]), np.append(y, point[1])
            r = np.corrcoef(xx, yy)[0, 1]
            if np.sign(r) == np.sign(n) and (not outliers[index - 1] or z_min <= z <= z_max):
                break
        else:
            raise RuntimeError(f'No admissible point after 1,000,000 attempts: stimulus {index}')
        tag = 'posSrngCor' if r >= .7 else 'posCor' if r > 0 else 'negSrngCor' if r <= -.7 else 'negCor'
        suffix = f'_out(MD={md:.3f},z={z:.3f})' if outliers[index - 1] else ''
        filename = f'scatter_plot{index}_{tag}(r={r:.3f}){suffix}.png'
        row = parse_filename(filename)
        row.update(target_r=float(n), exact_r=float(r), exact_md=None if md is None else float(md),
                   exact_z=None if z is None else float(z), point_count=251)
        yield row, np.column_stack((xx, yy))


def generate(output, seed=1, z_min=6, z_max=10):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT.parent / 'tribe/cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    if output.exists():
        raise ValueError(f'Choose a new generation directory: {output}')
    (output / 'stimuli').mkdir(parents=True)
    rows, arrays = [], {}
    with plt.rc_context(plt.rcParamsDefault):
        for row, points in generate_points(seed, z_min, z_max):
            fig, ax = plt.subplots(figsize=(2.44, 2.44), dpi=300)
            ax.scatter(points[:, 0], points[:, 1], s=5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            path = output / 'stimuli' / row['filename']
            fig.savefig(path)
            plt.close(fig)
            row['sha256'] = digest(path)
            rows.append(row)
            arrays[str(row['index'])] = points
    np.savez_compressed(output / 'points.npz', **arrays)
    write_csv(output / 'stimulus_metadata.csv', rows)
    (output / 'provenance.json').write_text(json.dumps(dict(
        mode='generate', seed=seed, outlier_z_bounds=[z_min, z_max],
        method='Independent port of the currently checked-in notebook; not an exact pilot-4 regeneration',
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in (NOTEBOOK, Path(__file__))},
        numpy=np.__version__, matplotlib=matplotlib.__version__, stimuli=72), indent=2) + '\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['pilot4', 'generate'], default='pilot4')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--z-min', type=float, default=6)
    parser.add_argument('--z-max', type=float, default=10)
    args = parser.parse_args()
    if args.mode == 'pilot4':
        output = args.output or DEFAULT_OUTPUT
        rows = export_pilot(output)
    else:
        output = args.output or ROOT / f'reproduced/generated_seed_{args.seed}'
        rows = generate(output, args.seed, args.z_min, args.z_max)
    print(f'{args.mode}: {len(rows)} stimuli in {output}')


if __name__ == '__main__':
    main()
