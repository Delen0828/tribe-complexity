"""Interpret the supplied Arunkumar workbooks and both experiments' stimuli."""
from collections import Counter
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from . import REPOSITORY, attribute, source_row

NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
INTERVALS = attribute('image_information_rating', 'Image/information rating', [
    ('I1', 'I1 · Image-like [−10, −6]'),
    ('I2', 'I2 · (−6, −3]'),
    ('I3', 'I3 · (−3, 3)'),
    ('I4', 'I4 · [3, 6)'),
    ('I5', 'I5 · Information-like [6, 10]')])
EXPERIMENTS = attribute('experiment', 'Experiment', [(1, 'Experiment 1'), (2, 'Experiment 2')])
FIELDS = {
    'Provenance and Purpose': [
        ('chart_category', 'Chart category'), ('chart_type', 'Chart type'),
        ('domain', 'Domain'), ('source', 'Source'), ('topic', 'Topic'),
        ('sentiment', 'Sentiment'), ('purpose', 'Purpose')],
    'Visual Elements': [
        ('photorealistic', 'Photorealistic'), ('skeuomorphic', 'Skeuomorphic'),
        ('pictorial_units', 'Pictorial units'),
        ('human_recognizable_objects', 'Human recognizable objects'),
        ('human_depiction', 'Human depiction'), ('glyph_mark_overlay', 'Glyph / mark overlay'),
        ('warped', 'Warped'), ('visual_density', 'Visual density'),
        ('data_ink_ratio', 'Data ink ratio'), ('color_count', 'Number of colors'),
        ('black_and_white', 'Black and white'), ('three_dimensional', '3D'),
        ('background_color', 'Background color')],
    'Data Elements': [
        ('dimension', 'Dimension'), ('multiplicity', 'Multiplicity'), ('temporal', 'Temporal'),
        ('ordering', 'Ordering'), ('data_volume', 'Data volume'), ('text_volume', 'Text volume'),
        ('data_source', 'Data source'), ('axes', 'Axes'), ('title', 'Title'), ('key', 'Key / legend'),
        ('caption', 'Caption'), ('aggregate_data', 'Aggregate data'),
        ('data_redundancy', 'Data redundancy'), ('message_redundancy', 'Message redundancy'),
        ('grid_lines', 'Grid lines'), ('text_labels', 'Text labels'), ('arrows', 'Arrows'),
        ('highlights', 'Highlights'), ('other_aoi', 'Other area of interest')],
}
VALUE_LABELS = {
    'sentiment': {'1': 'Negative', '2': 'Neutral', '3': 'Positive'},
    'purpose': {'1': 'Expose', '2': 'Involve', '3': 'Analyze'},
    'visual_density': {'1': 'Low', '2': 'Medium', '3': 'High'},
    'data_ink_ratio': {'1': 'Low', '2': 'Medium', '3': 'High'},
    'color_count': {'low': 'Low · 2–3', 'medium': 'Medium · 4–6', 'high': 'High · 7+'},
    'background_color': {'1': 'White', '2': 'Non-white'},
    'dimension': {'low': 'Low · <3', 'medium': 'Medium · 3–4', 'high': 'High · >4'},
    'multiplicity': {'1': 'Single panel', '2': 'Multiple panels / charts'},
    'data_volume': {'1': 'Low', '2': 'Medium', '3': 'High'},
    'text_volume': {'1': 'Low', '2': 'Medium', '3': 'High'},
}


def read_workbook(path, sheets):
    """Read selected XLSX sheets, preserving sparse columns and cached cell values.

    These published tables need no Excel engine dependency. Scripts/macros and
    formulas are never executed; formulas must contain a cached numeric value.
    """
    result = {}
    with ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(t.text or '' for t in item.findall('.//s:t', NS))
                       for item in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        relationships = {item.get('Id'): item.get('Target')
                         for item in ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        for sheet in workbook.findall('s:sheets/s:sheet', NS):
            name = sheet.get('name')
            if name not in sheets:
                continue
            target = relationships[sheet.get(f'{{{REL}}}id')]
            parts = PurePosixPath(target).parts
            if '..' in parts:
                raise ValueError('Unsafe workbook sheet path')
            member = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            rows = []
            for row in ET.fromstring(archive.read(member)).findall('s:sheetData/s:row', NS):
                values = []
                for cell in row.findall('s:c', NS):
                    column = re.fullmatch(r'([A-Z]+)[0-9]+', cell.get('r')).group(1)
                    position = 0
                    for letter in column:
                        position = position * 26 + ord(letter) - ord('A') + 1
                    values.extend([None] * max(0, position - len(values)))
                    cached = cell.find('s:v', NS)
                    kind = cell.get('t')
                    if kind == 'inlineStr':
                        value = ''.join(t.text or '' for t in cell.findall('s:is//s:t', NS))
                    elif cached is None:
                        if cell.find('s:f', NS) is not None:
                            raise ValueError(f'Uncached formula in {path}: {cell.get("r")}')
                        value = None
                    elif kind == 's':
                        value = strings[int(cached.text)]
                    elif kind in ('str', 'b'):
                        value = cached.text
                    elif kind == 'e':
                        raise ValueError(f'Excel error in {path}: {cell.get("r")}')
                    else:
                        value = float(cached.text)
                        if not math.isfinite(value):
                            raise ValueError('Non-finite workbook number')
                        if value.is_integer():
                            value = int(value)
                    values[position - 1] = value
                if any(value is not None for value in values):
                    rows.append(values)
            result[name] = rows
    if set(result) != set(sheets):
        raise ValueError(f'Missing workbook sheets in {path}: {set(sheets) - set(result)}')
    return result


def metadata_tables(path):
    tables = read_workbook(path, FIELDS)
    features, raw, definitions = {}, {}, []
    for sheet, fields in FIELDS.items():
        header, *rows = tables[sheet]
        if len(header) != len(fields) + 1 or len(rows) != 500:
            raise ValueError(f'Unexpected metadata table dimensions: {sheet}')
        if {row[0] for row in rows} != set(range(1, 501)):
            raise ValueError(f'Missing or duplicated metadata IDs: {sheet}')
        for column, (key, label) in enumerate(fields, 1):
            values = {str(row[column]).strip() for row in rows}
            binary = values == {'n', 'y'}
            labels = ({'0': 'No', '1': 'Yes'} if binary else VALUE_LABELS.get(key)
                      or {value: 'Not applicable' if value in ('nan', 'N/A') else value
                          for value in sorted(values)})
            if not binary and not values.issubset(labels):
                raise ValueError(f'Unknown metadata values for {key}: {values - labels.keys()}')
            definitions.append(dict(key=key, label=label, kind='binary' if binary else 'category',
                                    values=labels, source_sheet=sheet, source_column=header[column]))
            for row in rows:
                index, value = row[0], str(row[column]).strip()
                if value == 'None':
                    raise ValueError(f'Missing metadata for stimulus {index}: {key}')
                features.setdefault(index, {})[key] = int(value == 'y') if binary else value
                raw.setdefault(index, {}).setdefault(sheet, {})[header[column]] = row[column]
    return features, raw, definitions


def rating_table(path, experiment):
    rows = read_workbook(path, [f'Expt {experiment}'])[f'Expt {experiment}'][2:]
    result, interval = {}, None
    for row in rows:
        filename = str(row[0]).strip()
        if experiment == 1:
            original, rank = int(filename), None
        else:
            match = re.fullmatch(r'(\d+)-(\d+)\.(png|jpg|jpeg)', filename)
            if match is None:
                raise ValueError(f'Invalid experiment-2 filename: {filename}')
            rank, original = map(int, match.group(1, 2))
        interval_column = 3 if experiment == 1 else 7
        if len(row) > interval_column and row[interval_column] is not None:
            interval = str(row[interval_column])
        group = re.match(r'I[1-5]', interval or '')
        if group is None or original in result or original not in range(1, 501):
            raise ValueError(f'Invalid rating membership: {filename}')
        mean, sd = float(row[1]), float(row[2])
        if not math.isfinite(mean) or not -10 <= mean <= 10 or not math.isfinite(sd) or sd < 0:
            raise ValueError(f'Invalid image/information rating: {filename}')
        expected = 'I1' if mean <= -6 else 'I2' if mean <= -3 else 'I3' if mean < 3 else 'I4' if mean < 6 else 'I5'
        if group[0] != expected:
            raise ValueError(f'Published interval disagrees with mean: {filename}')
        rating = dict(mean=mean, stdev=sd, interval=group[0], published_interval=interval)
        if experiment == 2:
            for key, mean_column, sd_column in [('group_a', 3, 4), ('group_b', 5, 6)]:
                group_mean, group_sd = float(row[mean_column]), float(row[sd_column])
                if (not math.isfinite(group_mean) or not -10 <= group_mean <= 10
                        or not math.isfinite(group_sd) or group_sd < 0):
                    raise ValueError(f'Invalid experiment-2 group rating: {filename}')
                rating[key] = dict(mean=group_mean, stdev=group_sd)
        result[original] = dict(filename=filename, rank=rank, rating=rating)
    if len(result) != (500 if experiment == 1 else 100):
        raise ValueError(f'Unexpected experiment-{experiment} rating count')
    if experiment == 2 and {item['rank'] for item in result.values()} != set(range(1, 101)):
        raise ValueError('Invalid experiment-2 ranks')
    return result


def arunkumar_manifest(root):
    metadata = root / 'Stimuli Metadata/500-Stimuli-Metadata.xlsx'
    rating_paths = [root / f'Stimuli Rating Scores/Stimuli Scores Expt {experiment}.xlsx'
                    for experiment in (1, 2)]
    features, published, definitions = metadata_tables(metadata)
    ratings = {experiment: rating_table(path, experiment)
               for experiment, path in enumerate(rating_paths, 1)}
    rows, aliases, canonical = [], [], {}
    for experiment, count in [(1, 500), (2, 100)]:
        folder = root / f'Stimuli/Expt {experiment} - {count} stimuli'
        files = [path for path in folder.iterdir() if not path.name.startswith('.')]
        if len(files) != count or any(path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.gif') for path in files):
            raise ValueError(f'Unexpected experiment-{experiment} stimulus files')
        sources = {path.stem if experiment == 1 else path.name: path for path in files}
        if len(sources) != count:
            raise ValueError('Duplicate source stimulus IDs')
        ordered = sorted(ratings[experiment].items(), key=lambda item: item[0] if experiment == 1 else item[1]['rank'])
        for original, item in ordered:
            source = sources[str(original) if experiment == 1 else item['filename']]
            index = original if experiment == 1 else 500 + item['rank']
            rating = item['rating']
            row = source_row(index, source, source.name,
                             dict(features[original], experiment=experiment,
                                  image_information_rating=rating['interval']),
                             original_stimulus_index=original, experiment_2_rank=item['rank'],
                             label=f'Experiment {experiment} · Stimulus {original}',
                             score=rating['mean'],
                             score_label=f'Image/info {rating["mean"]:+.2f} / [−10, +10] · {rating["interval"]}',
                             ratings={f'experiment_{number}': table[original]['rating']
                                      for number, table in ratings.items() if original in table},
                             published_features=published[original])
            if row['sha256'] in canonical:
                row['reuse_prediction_from'] = canonical[row['sha256']]
                aliases.append(dict(index=index, prediction_index=row['reuse_prediction_from'],
                                    reason='Source bytes match exactly across experiments; reuse the validated prediction.'))
            else:
                canonical[row['sha256']] = index
            rows.append(row)
    documents = {str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in [root / 'README.md', metadata, *rating_paths]}
    digest = hashlib.sha256(json.dumps(dict(documents=documents,
                    stimuli=[(row['source'], row['sha256']) for row in rows]), sort_keys=True).encode()).hexdigest()
    attributes = [INTERVALS, EXPERIMENTS, *definitions]
    comparisons = [dict(id=f'experiment_{experiment}', title=f'Arunkumar · Experiment {experiment}',
                        indices=[row['index'] for row in rows if row['attributes']['experiment'] == experiment],
                        attributes=[INTERVALS, *definitions],
                        description=f'All {count} experiment-{experiment} stimuli, grouped by that experiment’s published image/information means and all 39 stimulus metadata fields.')
                   for experiment, count in [(1, 500), (2, 100)]]
    return dict(title='Arunkumar · Both experiments', rows=rows, attributes=attributes,
                description='Image or Information? Experiment 1 has 500 stimuli; experiment 2 has 100. Ratings range from −10 (image-like) to +10 (information-like).',
                sampling='All 600 experiment-specific stimulus observations, each with equal weight in the combined view. Use the experiment views for separate baselines. Published I1–I5 intervals and all 39 taxonomy fields are preserved. The 97 byte-identical copies reuse predictions; three changed experiment-2 images are predicted separately. The single-frame GIF is presented as a static image.',
                aliases=aliases, comparisons=comparisons, source_documents=documents,
                source_sha256=digest, unique_prediction_count=len(canonical),
                rating_interval_counts={str(experiment): dict(Counter(item['rating']['interval'] for item in table.values()))
                                        for experiment, table in ratings.items()})
