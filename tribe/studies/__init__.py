"""Supplement interpretation and reproducible study manifests."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

REPOSITORY = Path(__file__).resolve().parents[2]
MANIFESTS = Path(__file__).resolve().parent / 'manifests'
ARCHIVES = {'kylie': ('Kylie', 'mm_471300a062_s0.zip'),
            'hayeong': ('Hayeong', 'tvcg-mm-3634785.zip')}


def attribute(key, label, values):
    return dict(key=key, label=label, kind='category',
                values={str(value): text for value, text in values})


DESIGNS = attribute('design_variant', 'Designed complexity variant',
                    [(i, f'Design {i:02d}') for i in range(6)])
CHARTS = attribute('chart_type', 'Visualization type', [('bar', 'Bar'), ('area', 'Area')])
COLORS = attribute('palette', 'Color', [('blackwhite', 'Black and white'), ('color', 'Color')])
STYLE_COLORS = attribute('condition', 'Style and color', [
    ('standard_blackwhite', 'Standard · black and white'),
    ('standard_color', 'Standard · color'),
    ('handdrawn_blackwhite', 'Hand-drawn · black and white'),
    ('handdrawn_color', 'Hand-drawn · color')])
HAND_DRAWN = attribute('condition', 'Drawing and typography', [
    ('standard_color', 'Standard bars + plain font'),
    ('font_color', 'Standard bars + hand-drawn font'),
    ('handdrawn_color', 'Hand-drawn bars + hand-drawn font')])
EXP1_STYLES = attribute('style', 'Experiment 1 style',
                       [('cartoon', 'Cartoon style'), ('image', 'Image style')])
EXP1_CONDITIONS = attribute('condition', 'Style and embellishment', [
    ('cartoon_standard', 'Cartoon topics · standard'),
    ('cartoon_embellished', 'Cartoon topics · embellished'),
    ('image_standard', 'Image topics · standard'),
    ('image_embellished', 'Image topics · embellished')])
EXP2_CONDITIONS = [
    ('standard_blackwhite', 'Standard · black and white'),
    ('standard_color', 'Standard · color'),
    ('font_color', 'Hand-drawn font · color'),
    ('handdrawn_blackwhite', 'Hand-drawn · black and white'),
    ('handdrawn_color', 'Hand-drawn · color'),
    ('image_color', 'Image bars · color')]

PERCEIVED_COMPLEXITY = dict(key='complexity', label='Perceived complexity', kind='bin')
HAYEONG_FEATURES = [dict(key=key, label=key, kind='binary', values={'0': 'No', '1': 'Yes'})
                   for key in ('has_color', 'bar_is_hand_drawn', 'font_is_hand_drawn',
                               'is_comic', 'bar_is_image')]


def study_variables(dataset, rows, attributes):
    """Expose stimulus features, including for selections saved before these fields existed."""
    if dataset not in ('kylie', 'hayeong'):
        return rows, attributes
    updated = []
    for row in rows:
        values = dict(row['attributes'])
        if dataset == 'kylie':
            score = float(row['score'])
            if not math.isfinite(score) or not 0 <= score <= 100:
                raise ValueError('Invalid published complexity rating')
            values['complexity'] = min(int(score // 10), 9)
        else:
            style = values['style']
            if values['experiment'] == 1:
                embellished = values['condition'].endswith('_embellished')
                comic = style == 'cartoon' and embellished
                flags = (embellished, comic, comic, comic, style == 'image' and embellished)
            else:
                hand_drawn = style == 'handdrawn'
                flags = (values['palette'] == 'color', hand_drawn,
                         style in ('font', 'handdrawn'), hand_drawn, style == 'image')
            values.update((definition['key'], int(flag))
                          for definition, flag in zip(HAYEONG_FEATURES, flags))
        updated.append(dict(row, attributes=values))
    additions = [PERCEIVED_COMPLEXITY] if dataset == 'kylie' else HAYEONG_FEATURES
    keys = {definition['key'] for definition in additions}
    existing = [definition for definition in attributes if definition['key'] not in keys]
    definitions = existing + additions if dataset == 'kylie' else additions + existing
    return updated, definitions


def extract_archive(path, target):
    """Preserve meaningful supplement files, ignoring OS/editor artifacts; reject unsafe paths."""
    with ZipFile(path) as archive:
        for entry in archive.infolist():
            parts = PurePosixPath(entry.filename).parts
            if (not parts or '..' in parts or entry.filename.startswith('/')
                    or '\\' in entry.filename or (entry.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f'Unsafe archive path: {entry.filename}')
            if any(part.startswith('.') or part in ('__MACOSX', '__pycache__') for part in parts):
                continue
            if entry.is_dir():
                continue
            destination = target.joinpath(*parts)
            data = archive.read(entry)
            if destination.exists():
                if destination.read_bytes() != data:
                    raise ValueError(f'Extracted file differs from archive: {destination}')
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def source_row(index, source, filename, attributes, **metadata):
    return dict(index=index, source=str(source.relative_to(REPOSITORY)), filename=filename,
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                attributes=attributes, **metadata)


def kylie_manifest(root):
    study = root / 'supp-materials-2026/01. main-study'
    analysis = study / '02. analysis'
    labels = csv_rows(analysis / 'data/stimuli-labels.csv')
    by_filename = defaultdict(list)
    for row in labels:
        by_filename[row['filename']].append(row)
    if len(labels) != 72 or len(by_filename) != 24:
        raise ValueError('Expected 24 Kylie stimuli, each repeated for three source-category settings')
    ratings = defaultdict(list)
    for row in csv_rows(analysis / 'cleaned_data/data_compratings.csv'):
        value = float(row['value'])
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError('Invalid published complexity rating')
        ratings[(row['dataset'], row['dataset_counterbalance'], row['designed_complexity_id'])].append(value)
    rows, aliases = [], []
    for index, (filename, annotations) in enumerate(sorted(by_filename.items()), 1):
        topic, chart, design = Path(filename).stem.split('_')
        features = {k: v for k, v in annotations[0].items()
                    if k not in ('government_category', 'news_category', 'science_category')}
        if any(features != {k: v for k, v in row.items() if k in features} for row in annotations):
            raise ValueError(f'Inconsistent annotations for {filename}')
        group = features['condition_full'].split('-')[-1]
        expected_group = 'group1' if (topic, chart) in [('covid', 'bar'), ('penguins', 'area')] else 'group2'
        if group != expected_group or chart not in ('bar', 'area') or int(design) not in range(6):
            raise ValueError(f'Unexpected Kylie condition: {filename}')
        source = study / '01. stimuli' / topic / chart / filename
        if topic == 'penguins' and chart == 'area':
            source = source.with_name(filename.replace('penguins_', 'penguin_', 1))
            aliases.append(dict(filename=filename, actual_filename=source.name,
                                reason='Published metadata uses penguins_; authoritative area folder uses penguin_.'))
        if not source.is_file():
            raise FileNotFoundError(source)
        values = ratings[('penguin' if topic == 'penguins' else topic, group, design)]
        if not values:
            raise ValueError(f'Missing human ratings for {filename}')
        score = sum(values) / len(values)
        rows.append(source_row(index, source, filename,
                    dict(topic=topic, chart_type=chart, design_variant=int(design)),
                    label=f'{topic.capitalize()} · {chart} · design {design}',
                    score=score, rating_count=len(values),
                    score_label=f'Human complexity {score:.1f} / 100 · n={len(values)}',
                    published_features=features))
    return dict(title='Kylie · Controlled visualization complexity', attributes=[DESIGNS, CHARTS], rows=rows,
                description='Two topics × two visualization types × six designed variants. Design IDs 00–05 identify manipulations; they are not a measured complexity ranking.',
                sampling='All 24 main-study stimuli, with equal weight. Human complexity means use the cleaned study ratings and the published topic/chart counterbalance. The 72 feature rows repeat these images under three source-category settings; images are predicted once.',
                aliases=aliases, comparisons=[
                    dict(id='bar', title='Kylie · Bar-chart design variants',
                         indices=[r['index'] for r in rows if r['attributes']['chart_type'] == 'bar'],
                         attributes=[DESIGNS], description='Six bar-chart variants, each averaged over COVID-19 and penguins. Every design group contains the same two topics.'),
                    dict(id='area', title='Kylie · Area-chart design variants',
                         indices=[r['index'] for r in rows if r['attributes']['chart_type'] == 'area'],
                         attributes=[DESIGNS], description='Six area-chart variants, each averaged over COVID-19 and penguins. Every design group contains the same two topics.')])


def hayeong_manifest(root):
    stimuli = root / 'tvcg-mm-3634785/visulization stimuli'
    rows, aliases = [], []
    for style in ('cartoon', 'image'):
        for source in sorted((stimuli / 'experiment_1' / f'{style}-style').glob('*.png')):
            topic, version = source.stem.split('_')
            condition = 'embellished' if version in ('decor', 'embellished') else 'standard'
            if version not in ('decor', 'embellished', 'standard', 'deploy', 'size'):
                raise ValueError(source)
            rows.append(source_row(len(rows) + 1, source, source.name,
                        dict(experiment=1, topic=topic, style=style, condition=f'{style}_{condition}'),
                        label=f'Experiment 1 · {topic} · {style} · {condition}',
                        score_label=f'Experiment 1 · {style} · {condition}'))
    if len(rows) != 16:
        raise ValueError('Expected 16 experiment-1 stimuli (response summaries are excluded)')
    special_names = {'ice_standard.png': 'standard_blackwhite',
                     'handrawn_medal.png': 'handdrawn_color',
                     'medal_blackwhite.png': 'handdrawn_blackwhite',
                     'medal_color_font.png': 'font_color',
                     'medal_og_final.png': 'image_color'}
    for topic in ('emission', 'ice', 'medal', 'trump'):
        topic_rows = []
        for source in sorted((stimuli / 'experimet_2' / topic).glob('*.png')):
            condition = special_names.get(source.name, source.stem.removeprefix(topic + '_').replace('handrawn', 'handdrawn'))
            if condition not in dict(EXP2_CONDITIONS):
                raise ValueError(f'Unrecognized experiment-2 variant: {source}')
            if source.name in special_names:
                aliases.append(dict(filename=source.name, condition=condition,
                                    reason='Explicit irregular-name mapping, checked against the six-stimulus topic sets and original PNGs.'))
            palette = 'blackwhite' if condition.endswith('blackwhite') else 'color'
            drawing = condition.split('_')[0]
            item = source_row(len(rows) + 1, source, source.name,
                              dict(experiment=2, topic=topic, condition=condition,
                                   palette=palette, style=drawing),
                              label=f'Experiment 2 · {topic} · {dict(EXP2_CONDITIONS)[condition]}',
                              score_label=f'Experiment 2 · {dict(EXP2_CONDITIONS)[condition]}')
            rows.append(item)
            topic_rows.append(item)
        if {r['attributes']['condition'] for r in topic_rows} != set(dict(EXP2_CONDITIONS)) or len(topic_rows) != 6:
            raise ValueError(f'Expected all six variants for {topic}')
    exp1 = [r['index'] for r in rows if r['attributes']['experiment'] == 1]
    color = [r['index'] for r in rows if r['attributes']['experiment'] == 2 and r['attributes']['condition'] in STYLE_COLORS['values']]
    handdrawn = [r['index'] for r in rows if r['attributes']['experiment'] == 2 and r['attributes']['condition'] in HAND_DRAWN['values']]
    return dict(title='Hayeong · Visualization style and color', rows=rows,
                attributes=[attribute('condition', 'Study condition', list(EXP1_CONDITIONS['values'].items()) + EXP2_CONDITIONS),
                            attribute('experiment', 'Experiment', [(1, 'Experiment 1'), (2, 'Experiment 2')])],
                description='Experiment 1 compares standard and embellished charts in cartoon/image topic sets. Experiment 2 contains six drawing, font, image, and color conditions for each of four topics.',
                sampling='All 40 actual visualization stimuli. Eight experiment-1 response-summary PNGs are excluded. Style/color labels come from the supplement folders, filenames, survey, and inspected originals; no complexity ratings are invented.',
                aliases=aliases, comparisons=[
                    dict(id='color', title='Hayeong · Color comparison', indices=color,
                         attributes=[COLORS, STYLE_COLORS],
                         description='Black-and-white versus color in experiment 2, balanced over four topics and standard/hand-drawn styles (eight stimuli per palette). Switch grouping to inspect the four style × color cells. Font-only and image conditions are excluded.'),
                    dict(id='hand_drawn', title='Hayeong · Hand-drawn comparison', indices=handdrawn,
                         attributes=[HAND_DRAWN],
                         description='Three color conditions for the same four topics: standard bars/plain font, standard bars/hand-drawn font, and hand-drawn bars/hand-drawn font. The middle condition separates typography from bar rendering.'),
                    dict(id='cartoon_image', title='Hayeong · Cartoon/image comparison', indices=exp1,
                         attributes=[EXP1_STYLES, EXP1_CONDITIONS],
                         description='Experiment 1: standard and embellished versions within cartoon/image topic sets. Cartoon and image styles use different topics; the between-style difference also includes topic differences and does not isolate a style effect.')])


def prepare_manifest(dataset):
    if dataset == 'arunkumar':
        from .arunkumar import arunkumar_manifest
        result = arunkumar_manifest(REPOSITORY / 'data/Arunkumar')
        provenance = {}
    else:
        folder, filename = ARCHIVES[dataset]
        archive = REPOSITORY / 'data' / folder / filename
        extracted = archive.parent / 'extracted'
        extract_archive(archive, extracted)
        result = (kylie_manifest if dataset == 'kylie' else hayeong_manifest)(extracted)
        provenance = dict(archive=str(archive.relative_to(REPOSITORY)),
                          archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    result['rows'], result['attributes'] = study_variables(dataset, result['rows'], result['attributes'])
    for comparison in result['comparisons']:
        included = set(comparison['indices'])
        _, comparison['attributes'] = study_variables(
            dataset, [row for row in result['rows'] if row['index'] in included], comparison['attributes'])
    result.update(schema_version=1, dataset=dataset, **provenance)
    MANIFESTS.mkdir(exist_ok=True)
    path = MANIFESTS / f'{dataset}.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
