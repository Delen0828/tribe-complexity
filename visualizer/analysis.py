"""Equal-stimulus aggregation for the web analysis grid."""
import numpy as np


def complexity_tertiles(rows, attributes, population):
    """Rank the selected MassVis stimuli into thirds, independent of prediction progress."""
    if not any(definition['key'] == 'complexity' for definition in attributes):
        return rows, attributes
    scores = np.asarray([row['score'] for row in population], dtype=float)
    if not len(scores) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 100)):
        raise ValueError('Invalid perceived complexity scores')
    # Image IDs give tied ratings a stable order while keeping the thirds balanced.
    ranked = [row['index'] for row in sorted(population, key=lambda row: (row['score'], row['index']))]
    labels = {'low': 'Low', 'mid': 'Mid', 'high': 'High'}
    membership = {}
    for value, indices in zip(labels, np.array_split(ranked, 3)):
        membership.update((int(index), value) for index in indices)
    updated = [dict(row, attributes=dict(row['attributes'], complexity=membership[row['index']]))
               for row in rows]
    definitions = [dict(definition, kind='tertile', values=labels)
                   if definition['key'] == 'complexity' else definition for definition in attributes]
    return updated, definitions


def axis_groups(rows, definition):
    if definition is None:
        return [dict(value='', label='All', indices=[r['index'] for r in rows])]
    key = definition['key']
    values = sorted({row['attributes'][key] for row in rows})
    labels = definition.get('values', {})
    if definition['kind'] == 'tertile':
        values = list(labels)
    elif definition['kind'] == 'bin':
        values = list(range(10))
    elif labels:
        values.sort(key=lambda value: list(labels).index(str(value)))
    groups = []
    for value in values:
        label = (f'{value * 10}–{(value + 1) * 10}' if definition['kind'] == 'bin'
                 else labels.get(str(value), 'Yes' if value else 'No')
                 if definition['kind'] == 'binary' else labels.get(str(value), str(value)))
        groups.append(dict(value=value, label=label,
                           indices=[r['index'] for r in rows if r['attributes'][key] == value]))
    return groups


def aggregate_xy(rows, maps, attributes, x, y='', statistic='mean'):
    maps = np.asarray(maps)
    if maps.ndim != 2 or len(maps) != len(rows) or not len(rows) or not np.isfinite(maps).all():
        raise ValueError('Invalid predictions')
    definitions = {item['key']: item for item in attributes}
    if x not in definitions or (y and y not in definitions):
        raise ValueError('Unknown axis variable')
    if statistic not in ('mean', 'contrast'):
        raise ValueError('Unknown statistic')
    axes = dict(x=axis_groups(rows, definitions[x]),
                y=axis_groups(rows, definitions[y] if y else None))
    positions = {row['index']: i for i, row in enumerate(rows)}
    grand = maps.mean(axis=0, dtype=np.float64)
    cells = []
    for y_group in axes['y']:
        members = set(y_group['indices'])
        for x_group in axes['x']:
            indices = [index for index in x_group['indices'] if index in members]
            values = None
            if indices:
                values = maps[[positions[index] for index in indices]].mean(axis=0, dtype=np.float64)
                if statistic == 'contrast':
                    values -= grand
            cells.append(dict(x=x_group['value'], y=y_group['value'], indices=indices,
                              count=len(indices), values=values))
    limit = max(1e-12, max(float(np.abs(cell['values']).max()) for cell in cells if cell['count']))
    return dict(axes=axes, cells=cells, limit=limit, x=x, y=y, statistic=statistic)
