"""Bridge the thesis's reverse-engineering export into the existing study runner."""
import importlib.util
import json
import math

from . import REPOSITORY, attribute, source_row

THESIS = REPOSITORY / 'SemanticsOfDataViz_HAshcraftGTUgThesis'
CORRELATION_CLASS = attribute('correlation_class', 'Correlation class',
                              [('low', 'Low'), ('mid', 'Mid'), ('high', 'High')])


def correlation_class(r):
    """Viewer classes use published |r|, independently of correlation direction."""
    magnitude = abs(float(r))
    if not math.isfinite(magnitude) or magnitude > 1:
        raise ValueError('Correlation must be finite and within [-1, 1]')
    return 'high' if magnitude >= .7 else 'mid' if magnitude >= .3 else 'low'


def ashcraft_variables(rows, attributes):
    """Normalize old saved selections as well as newly prepared study manifests."""
    updated = [dict(row, attributes=dict(row['attributes'],
                    correlation_class=correlation_class(row['published_r']))) for row in rows]
    definitions = [CORRELATION_CLASS if item['key'] == 'correlation_class' else item
                   for item in attributes]
    if not any(item['key'] == 'correlation_class' for item in definitions):
        definitions.append(CORRELATION_CLASS)
    return updated, definitions


def reproduction_module():
    spec = importlib.util.spec_from_file_location('ashcraft_reproduction', THESIS / 'reproduce.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ashcraft_manifest():
    reproduction = reproduction_module()
    metadata = reproduction.export_pilot()
    rows = []
    for item in metadata:
        direction = item['correlation_direction']
        outlier = 'outlier' if item['has_outlier'] else 'no outlier'
        rows.append(source_row(
            item['index'], reproduction.DEFAULT_OUTPUT / 'stimuli' / item['filename'], item['filename'],
            dict(correlation_direction=direction, has_outlier=item['has_outlier'],
                 correlation_class=correlation_class(item['r'])),
            label=f"Scatterplot {item['index']} · {direction} · {outlier}",
            score_label=f"r = {item['r']:+.3f} · {outlier}",
            published_r=item['r'], mahalanobis_distance=item['mahalanobis_distance'],
            outlier_z=item['outlier_z'], ink_density=item['ink_density'],
            original_source=str((reproduction.ORIGINALS / item['filename']).relative_to(REPOSITORY))))
    provenance = json.loads((reproduction.DEFAULT_OUTPUT / 'provenance.json').read_text())
    documents = {str((THESIS / p).relative_to(REPOSITORY)): value
                 for p, value in provenance['source_sha256'].items()}
    return dict(
        title='Ashcraft · Controlled scatterplots',
        description='72 published pilot-4 scatterplots. Correlation direction and outlier presence have balanced marginals, with unequal joint cells (24/12/12/24).',
        sampling='All 72 authoritative pilot-4 PNGs exported without pixel changes. Equal image weight; static 3-second TRIBE protocol. MATLAB attention tasks, prompts, motion, fixation, and block context are omitted; no task-specific or measured fMRI validation is claimed.',
        attributes=[
            attribute('correlation_direction', 'Correlation direction', [('negative', 'Negative'), ('positive', 'Positive')]),
            dict(key='has_outlier', label='Outlier present', kind='binary', values={'0': 'No', '1': 'Yes'}),
            CORRELATION_CLASS],
        rows=rows, comparisons=[], source_documents=documents,
        source_sha256=reproduction.digest(reproduction.WORKBOOK),
        reproduction=provenance,
    )
