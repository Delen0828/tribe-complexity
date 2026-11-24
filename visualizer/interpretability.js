'use strict';
const byId = id => document.getElementById(id);
const directory = '../tribe/outputs/complexity_comparison/ridge_interpretability/';
const descriptions = {
  ablation: 'Each region is removed across all three time points and Ridge is tuned and fitted again. A positive ΔR² means the region adds predictive value given the remaining signals. Gray marks unlabeled cortex.',
  permutation: 'Within each held-out fold, charts exchange all signals in one region together. Positive ΔR² means the fitted model depends on that region. Scores average 30 shuffles, which disrupt correlations between regions. Gray marks unlabeled cortex.',
  haufe: 'Warm colors mark simulated signals that rise with predicted complexity. Cool colors mark signals that fall. Each vertex shows its training-fold covariance with the prediction divided by prediction variance, averaged by training-fold size. All three time points share one scale.'
};
let report;
function showTable(metric) {
  const rows = [...report.regions];
  const sort = byId('sort').value;
  rows.sort((a, b) => sort === 'name' ? a.name.localeCompare(b.name) :
    (sort === 'asc' ? a[metric] - b[metric] : b[metric] - a[metric]) || a.name.localeCompare(b.name));
  byId('regions').replaceChildren(...rows.map(row => {
    const tr = document.createElement('tr');
    for (const value of [...row.name.split(':'), row[metric].toPrecision(7)]) {
      const td = document.createElement('td');
      td.textContent = value;
      tr.append(td);
    }
    return tr;
  }));
}
function update() {
  const metric = byId('metric').value;
  const view = byId('view').value;
  const haufe = metric === 'haufe';
  byId('description').textContent = descriptions[metric];
  byId('time-control').hidden = !haufe;
  byId('region-section').hidden = haufe;
  const key = haufe ? 'haufe-' + byId('time').value : metric;
  const panels = [];
  if (view !== 'four') panels.push('both-ventral');
  if (view !== 'inferior') panels.push('left-lateral', 'right-lateral', 'left-medial', 'right-medial');
  byId('panels').replaceChildren(...panels.map(panel => {
    const img = document.createElement('img');
    img.alt = byId('metric').selectedOptions[0].text + ', ' + panel.replace('-', ' ') +
      (haufe ? ', time point ' + byId('time').value : '');
    img.src = directory + report.run + '/' + key + '-' + panel + '.png';
    return img;
  }));
  const limit = report.limits[metric].toPrecision(4);
  byId('low').textContent = '−' + limit;
  byId('high').textContent = '+' + limit;
  byId('units').textContent = haufe ? 'Simulated response / predicted complexity point' : 'Held-out ΔR²';
  if (!haufe) showTable(metric);
}
async function load() {
  const response = await fetch(directory + 'report.json', {cache: 'no-store'});
  if (!response.ok) throw Error('report');
  report = await response.json();
  byId('summary').textContent = 'Full-model OOF R² ' + report.r2.toFixed(4) + ' · ' +
    report.images.toLocaleString() + ' charts · ' + report.groups.toLocaleString() + ' source groups';
  for (const id of ['metric', 'view', 'time']) byId(id).addEventListener('change', update);
  byId('sort').addEventListener('change', () => showTable(byId('metric').value));
  update();
  byId('results').hidden = false;
  byId('status').hidden = true;
}
load().catch(() => {
  byId('status').textContent = 'Results unavailable. In the tribe environment, run from the repository root: ';
  const cmd = document.createElement('code');
  cmd.textContent = 'python label/stats/interpret_ridge.py';
  byId('status').append(cmd);
});
