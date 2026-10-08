'use strict';
const el = id => document.getElementById(id);
let report = null, baseURL = null, selectedGroup = null, visibleMembers = 0, requestID = 0;
const localReport = document.body.dataset.report;
const catalogPath = document.body.dataset.catalog;
let catalog = null;
const params = new URLSearchParams(location.search);
const legacyViews = ['Left lateral', 'Left medial', 'Right lateral', 'Right medial'].map((label, panel) => ({
  key: label.toLowerCase().replace(' ', '_'), label, image_suffix: 'image', panel, panels: 4,
}));
if (localReport) el('dataset-control').hidden = true;
if (params.get('dataset') === 'massvis_all') el('dataset').value = 'massvis_all';

function asset(path) { return new URL(path, baseURL).href; }
function availableViews() { return report.views || legacyViews; }
function configureViews(preferred) {
  const views = availableViews();
  const hasFour = legacyViews.every(view => views.some(item => item.key === view.key));
  const hasInferior = views.some(view => view.key === 'inferior');
  el('views').replaceChildren();
  for (const [value, label, available] of [
    ['four', 'Four standard views', hasFour],
    ['inferior', 'Inferior / ventral · 1 view', hasInferior],
    ['all', 'All five views', hasFour && hasInferior],
  ]) {
    if (!available) continue;
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    el('views').append(option);
  }
  const choices = [...el('views').options].map(option => option.value);
  const selected = choices.includes(preferred) ? preferred : report.view_layout;
  if (choices.includes(selected)) el('views').value = selected;
}
function groupTitle(attribute, group) {
  if (attribute.key === 'complexity') {
    return `${group.value * 10}–${(group.value + 1) * 10} perceived complexity`;
  }
  // Keep older five-attribute reports readable when they lack count metadata.
  const legacyCountLabels = {
    colors: ['color', 'colors'], charts: ['chart', 'charts'],
    quantitative: ['quantitative variable', 'quantitative variables'],
    categorical: ['categorical variable', 'categorical variables'],
  };
  const countLabels = attribute.count_labels || legacyCountLabels[attribute.key];
  return countLabels
    ? `${group.value} ${countLabels[group.value === 1 ? 0 : 1]}`
    : group.label;
}
function setURL() {
  const url = new URL(location.href);
  if (!localReport) url.searchParams.set('dataset', el('dataset').value);
  url.searchParams.set('attribute', el('attribute').value);
  url.searchParams.set('statistic', el('statistic').value);
  url.searchParams.set('views', el('views').value);
  if (selectedGroup) url.searchParams.set('group', selectedGroup.value);
  history.replaceState(null, '', url);
}
function showMembers(reset = true) {
  if (reset) { visibleMembers = 0; el('members').replaceChildren(); }
  const rowsByIndex = new Map(report.rows.map(row => [row.index, row]));
  const end = Math.min(visibleMembers + 24, selectedGroup.count);
  for (const index of selectedGroup.indices.slice(visibleMembers, end)) {
    const row = rowsByIndex.get(index);
    const link = document.createElement('a');
    link.className = 'member';
    link.href = (localReport || catalogPath) ? asset(`inputs/${index}.png`) : `./?index=${index}`;
    const image = document.createElement('img');
    image.src = asset(`inputs/${index}.png`);
    image.alt = row.filename;
    image.loading = 'lazy';
    const title = document.createElement('strong');
    title.textContent = `Image ${index}`;
    const detail = document.createElement('span');
    detail.textContent = `Complexity ${row.score.toFixed(1)} / 100`;
    link.append(image, title, detail);
    el('members').append(link);
  }
  visibleMembers = end;
  el('member-range').textContent = `${end} of ${selectedGroup.count} stimuli shown`;
  el('more-members').hidden = end >= selectedGroup.count;
}
function showMap() {
  const kind = el('statistic').value;
  const attribute = report.attributes.find(a => a.key === el('attribute').value);
  const groups = report.groups.filter(g => g.attribute === attribute.key);
  const layout = el('views').value;
  const views = availableViews().filter(view => layout === 'all'
    || (layout === 'inferior' ? view.key === 'inferior' : view.key !== 'inferior'));
  el('map-kind').textContent = kind === 'mean' ? 'Mean predicted response' : 'Difference from included-stimulus mean';
  el('map-title').textContent = attribute.label;
  el('group-count').textContent = `${groups.length} groups × ${views.length} ${views.length === 1 ? 'view' : 'views'}`;
  document.querySelector('.brain-grid-scroll').setAttribute('aria-label',
    `Groups across columns, ${views.length} cortical ${views.length === 1 ? 'view' : 'views'} down rows`);
  el('map-description').textContent = kind === 'mean' ? report.aggregation : report.contrast;
  const limit = report.limits[kind];
  el('scale-note').textContent = Number.isFinite(limit)
    ? `Shared scale across every ${kind === 'mean' ? 'group mean' : 'difference map'}: −${limit.toFixed(3)} to +${limit.toFixed(3)} model units. Blue = negative; red = positive. Select a column label to inspect its stimuli.`
    : 'Cortical maps have not been rendered for this report.';
  const grid = el('groups');
  grid.replaceChildren();
  grid.style.minWidth = `${120 + groups.length * 220}px`;
  const header = grid.createTHead().insertRow();
  const corner = document.createElement('th');
  corner.scope = 'col';
  corner.className = 'view-label';
  corner.textContent = 'Cortical view';
  header.append(corner);
  for (const group of groups) {
    const heading = document.createElement('th');
    heading.scope = 'col';
    const sampleSize = document.createElement('div');
    sampleSize.className = 'sample-size-icons';
    sampleSize.setAttribute('role', 'img');
    sampleSize.setAttribute('aria-label', `${group.count} stimuli; one square per stimulus`);
    sampleSize.title = `n = ${group.count} stimuli`;
    for (let i = 0; i < group.count; i++) {
      const square = document.createElement('span');
      square.className = 'sample-square';
      square.setAttribute('aria-hidden', 'true');
      sampleSize.append(square);
    }
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.value = group.value;
    button.setAttribute('aria-pressed', String(group === selectedGroup));
    button.textContent = groupTitle(attribute, group);
    button.setAttribute('aria-label', `${attribute.label}: ${group.label}; ${group.count} stimuli`);
    button.addEventListener('click', () => chooseGroup(group));
    heading.append(sampleSize, button);
    header.append(heading);
  }
  const body = grid.createTBody();
  for (const view of views) {
    const row = body.insertRow();
    const label = document.createElement('th');
    label.scope = 'row';
    label.className = 'view-label';
    label.textContent = view.label;
    row.append(label);
    for (const group of groups) {
      const cell = row.insertCell();
      const frame = document.createElement('div');
      frame.className = 'cortical-view';
      const image = document.createElement('img');
      image.className = 'group-map';
      image.loading = 'lazy';
      // Use report metadata to crop standard strips or show a single bottom map.
      image.style.width = `${100 * view.panels}%`;
      image.style.left = `${-100 * view.panel}%`;
      image.alt = `${view.label}: ${kind === 'mean' ? 'group mean' : 'group minus included-stimulus mean'} for ${attribute.label} ${group.label}, ${group.count} stimuli.`;
      const error = document.createElement('p');
      error.className = 'map-error';
      error.textContent = 'Cannot load this cortical map.';
      error.hidden = true;
      image.addEventListener('error', () => { image.hidden = true; error.hidden = false; });
      const path = group[`${kind}_${view.image_suffix}`];
      if (path) image.src = asset(path);
      else { image.hidden = true; error.hidden = false; }
      frame.append(image, error);
      cell.append(frame);
    }
  }
  setURL();
}
function chooseGroup(group) {
  selectedGroup = group;
  for (const button of el('groups').querySelectorAll('button')) {
    button.setAttribute('aria-pressed', String(button.dataset.value === String(group.value)));
  }
  const attribute = report.attributes.find(a => a.key === group.attribute);
  el('members-heading').textContent = `Stimuli · ${attribute.label}: ${group.label}`;
  setURL();
  showMembers();
}
function showGroups(preferred = null) {
  const groups = report.groups.filter(g => g.attribute === el('attribute').value);
  selectedGroup = groups.find(g => String(g.value) === preferred) || groups[0];
  showMap();
  chooseGroup(selectedGroup);
}
async function loadReport() {
  const id = ++requestID;
  report = null;
  selectedGroup = null;
  el('brain-workspace').hidden = true;
  el('download').hidden = true;
  el('attribute').disabled = true;
  el('statistic').disabled = true;
  el('views').disabled = true;
  el('brain-status').textContent = 'Loading predictions…';
  const defaultPath = catalog
    ? catalog.find(item => item.id === el('dataset').value)?.data
    : `../tribe/outputs/${el('dataset').value}/explorer.json`;
  const path = localReport || new URLSearchParams(location.search).get('data') || defaultPath;
  try {
    const url = new URL(path, location.href);
    if (url.origin !== location.origin) throw new Error('Report must be hosted on this site.');
    const response = await fetch(url, {cache: 'no-cache'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (id !== requestID) return;
    if (data.schema_version !== 1 || !data.groups.length || data.rows.length !== data.count) throw new Error('Invalid report format');
    report = data;
    baseURL = url;
    const currentParams = new URLSearchParams(location.search);
    el('attribute').replaceChildren();
    for (const attribute of report.attributes) {
      const option = document.createElement('option');
      option.value = attribute.key;
      option.textContent = attribute.label;
      el('attribute').append(option);
    }
    if (report.attributes.some(a => a.key === currentParams.get('attribute'))) el('attribute').value = currentParams.get('attribute');
    el('statistic').value = currentParams.get('statistic') === 'contrast' ? 'contrast' : 'mean';
    configureViews(currentParams.get('views'));
    el('attribute').disabled = false;
    el('statistic').disabled = false;
    el('views').disabled = false;
    el('brain-workspace').hidden = false;
    el('brain-status').textContent = `${report.count.toLocaleString()} predictions loaded · ${(report.precision || 'fp32').toUpperCase()}.`;
    el('sample-summary').textContent = `${report.count.toLocaleString()} of ${report.eligible_count.toLocaleString()} available annotated stimuli · ${report.scope === 'sample' ? `Sample seed ${report.seed}` : 'Whole dataset'}`;
    el('sampling-note').textContent = report.scope === 'sample'
      ? 'Balanced sampling across ten complexity bins; groups summarize this selected sample, not the natural dataset distribution. Bins exclude their upper edge except 100. Squares above each column show its sample size: one square per stimulus.'
      : 'All available annotated stimuli. Feature groups use published counts, presence labels, and source categories; group sizes may differ. Bins exclude their upper edge except 100.';
    if (report.complete === false) {
      el('sample-summary').textContent = `Partial run · ${report.count.toLocaleString()} of ${report.selected_count.toLocaleString()} selected stimuli`;
      el('sampling-note').textContent = 'Incomplete inference: only completed predictions are included. These groups may be biased and do not represent the full selection. Contrasts use the mean of completed stimuli.';
    }
    el('download').href = asset('aggregate_maps.npz');
    el('download').hidden = false;
    el('selection-link').href = asset('selection.json');
    el('timing-link').href = asset('timing.json');
    const t = report.timing;
    el('runtime').textContent = t.full_prediction_seconds
      ? `Measured ${t.measured_count} uncached stimuli: ${t.mean_seconds_per_stimulus.toFixed(2)} seconds per stimulus on ${t.device}. Estimated ${t.total_stimuli.toLocaleString()}-stimulus inference run: ${(t.full_prediction_seconds/3600).toFixed(1)} hours. ${t.note}`
      : t.note;
    showGroups(currentParams.get('group'));
  } catch (error) {
    if (id !== requestID) return;
    el('brain-workspace').hidden = true;
    el('download').hidden = true;
    el('brain-status').textContent = 'This report is unavailable. Generate its grouped report and reload. Serve the outputs directory over HTTP when viewing locally.';
    console.error(error);
  }
}
el('dataset').addEventListener('change', () => {
  const url = new URL(location.href);
  url.searchParams.delete('data');
  url.searchParams.set('dataset', el('dataset').value);
  history.replaceState(null, '', url);
  loadReport();
});
el('attribute').addEventListener('change', () => showGroups());
el('statistic').addEventListener('change', showMap);
el('views').addEventListener('change', showMap);
el('more-members').addEventListener('click', () => showMembers(false));
async function initialize() {
  if (catalogPath) {
    try {
      const response = await fetch(catalogPath, {cache: 'no-cache'});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      catalog = await response.json();
      el('dataset').replaceChildren();
      for (const item of catalog) {
        const option = document.createElement('option');
        option.value = item.id;
        option.textContent = item.label;
        el('dataset').append(option);
      }
      if (!catalog.length) throw new Error('No grouped reports available');
      if (catalog.some(item => item.id === params.get('dataset'))) el('dataset').value = params.get('dataset');
    } catch (error) {
      el('brain-status').textContent = 'No grouped reports available. Build reports and serve the outputs directory over HTTP.';
      return;
    }
  }
  await loadReport();
}
initialize();
