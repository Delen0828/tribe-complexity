#!/usr/bin/env python3
"""Join published annotations to MASSVIS images; Python standard library only."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = ['no_text', 'axes_labels', 'axes_text', 'titles', 'annotations',
        'captions', 'legend_text', 'legend_title', 'text_only']
CHARTS = ['area', 'bar', 'circle', 'diagram', 'distribution', 'grid_or_matrix',
          'line', 'map', 'point', 'table', 'text', 'trees_or_networks']
COUNTS = ['color_count', 'quantitative', 'categorical', 'chart_count', 'chart_types_count']
BINARY = TEXT + CHARTS + ['black_and_white', 'background_color', 'multi_panel', 'single_panel',
                         'news_category', 'infographic_category', 'government_category', 'science_category']
ALIASES = {f'economist_daily_chart_{n}.png': f'economist_daily_chart_{n} (1).png'
           for n in (100, 102, 103, 108, 110, 113)}


def build(root=ROOT):
    source = root / 'label/source/annotations.csv'
    with source.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    ratings_source = root / 'label/source/ratings.csv'
    ratings = defaultdict(list)
    with ratings_source.open(newline='') as handle:
        for rating in csv.DictReader(handle):
            value = float(rating['response'])
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f'Invalid human rating: {rating["image_id"]}')
            ratings[rating['image_id']].append(value)
    if set(ratings) != {row['image_id'] for row in rows}:
        raise ValueError('Individual ratings and annotations must have identical IDs')
    images = {}
    for path in sorted((root / 'data').rglob('*')):
        if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            images.setdefault(path.name, []).append(path)
    records, used, seen, aliases, missing = [], set(), set(), [], []
    for row in rows:
        index = int(Path(row['image_id']).stem)
        if index in seen:
            raise ValueError(f'Duplicate image ID: {index}')
        seen.add(index)
        for key in BINARY + COUNTS:
            row[key] = int(row[key])
            if (key in BINARY and row[key] not in (0, 1)) or row[key] < 0:
                raise ValueError(f'Invalid {key} at ID {index}')
        score = float(row['mean_human_complexity_rating'])
        if not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError(f'Invalid complexity rating at ID {index}')
        responses = ratings[row['image_id']]
        if len(responses) != 10 or not math.isclose(statistics.mean(responses), score, abs_tol=1e-9):
            raise ValueError(f'Human rating count or mean mismatch at ID {index}')
        if sum(row[k] for k in CHARTS) != row['chart_types_count']:
            raise ValueError(f'Chart-type count mismatch at ID {index}')
        if row['single_panel'] + row['multi_panel'] != 1:
            raise ValueError(f'Panel labels inconsistent at ID {index}')
        candidates = images.get(row['filename'], [])
        match = 'exact'
        if not candidates and row['filename'] in ALIASES:
            candidates = images.get(ALIASES[row['filename']], [])
            match = 'documented_alias'
        if len(candidates) > 1:
            raise ValueError(f'Ambiguous image filename: {row["filename"]}')
        path = candidates[0].relative_to(root).as_posix() if candidates else None
        if path:
            if path in used:
                raise ValueError(f'Image assigned more than once: {path}')
            used.add(path)
        else:
            missing.append({'index': index, 'filename': row['filename']})
            match = 'missing'
        if match == 'documented_alias':
            aliases.append({'index': index, 'filename': row['filename'], 'image_path': path})
        records.append({
            'index': index, 'image_id': row['image_id'], 'filename': row['filename'],
            'image_path': path, 'image_match': match, 'category': row['category'],
            'mean_human_complexity_rating': score,
            'human_rating_count': len(responses),
            'human_rating_variance': statistics.variance(responses),
            'human_rating_sd': statistics.stdev(responses),
            'text': {k: row[k] for k in TEXT},
            'color': {k: row[k] for k in ['black_and_white', 'background_color', 'color_count']},
            'data': {k: row[k] for k in ['quantitative', 'categorical']},
            'design': {k: row[k] for k in ['chart_count', 'chart_types_count', 'multi_panel'] + CHARTS},
        })
    if seen != set(range(5800)):
        raise ValueError('Expected published image IDs 0 through 5799')
    records.sort(key=lambda r: r['index'])
    report = {
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'ratings_source_sha256': hashlib.sha256(ratings_source.read_bytes()).hexdigest(),
        'records': len(records), 'local_images': sum(map(len, images.values())),
        'matched_images': len(used), 'aliases': aliases, 'missing_images': missing,
        'unlabeled_images': sorted(p.relative_to(root).as_posix() for paths in images.values()
                                   for p in paths if p.relative_to(root).as_posix() not in used),
    }
    output = root / 'label/output'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'labels.json').write_text(json.dumps(records, separators=(',', ':'), allow_nan=False) + '\n')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    flat = []
    for r in records:
        item = {k: v for k, v in r.items() if not isinstance(v, dict)}
        for group in ['text', 'color', 'data', 'design']:
            item.update({f'{group}.{k}': v for k, v in r[group].items()})
        flat.append(item)
    with (output / 'labels.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(f'{len(records)} labels; {len(used)} images matched; {len(aliases)} aliases; '
          f'{len(missing)} missing; {len(report["unlabeled_images"])} unlabeled local images')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()
