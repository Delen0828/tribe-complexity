'use strict';
const el = id => document.getElementById(id);
let report = null, baseURL = null, selectedGroup = null, visibleMembers = 0, requestID = 0;
const params = new URLSearchParams(location.search);
if (params.get('dataset') === 'massvis_all') el('dataset').value = 'massvis_all';

function asset(path) { return new URL(path, baseURL).href; }
function setURL() {
  const url = new URL(location.href);
  url.searchParams.set('dataset', el('dataset').value);
  url.searchParams.set('attribute', el('attribute').value);
  url.searchParams.set('statistic', el('statistic').value);
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
    link.href = `./?index=${index}`;
    const image = document.createElement('img');
    image.src = '../' + row.image_path.split('/').map(encodeURIComponent).join('/');
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
  const attribute = report.attributes.find(a => a.key === selectedGroup.attribute);
  el('map-kind').textContent = kind === 'mean' ? 'Mean predicted response' : 'Difference from selected-set mean';
  el('map-title').textContent = `${attribute.label}: ${selectedGroup.label}`;
  el('group-count').textContent = `n = ${selectedGroup.count}`;
  el('map-description').textContent = (kind === 'mean' ? report.aggregation : report.contrast)
    + (selectedGroup.count < 5 ? ' Fewer than five stimuli in this group.' : '');
  el('map-error').hidden = true;
  el('brain-map').hidden = false;
  el('brain-map').alt = `${kind === 'mean' ? 'Group mean' : 'Group minus selected-set mean'} predicted cortical responses for ${attribute.label} ${selectedGroup.label}, ${selectedGroup.count} stimuli. Left lateral, left medial, right lateral, and right medial views.`;
  el('brain-map').src = asset(selectedGroup[`${kind}_image`]);
  const limit = report.limits[kind];
  el('scale-note').textContent = `Shared scale across every ${kind === 'mean' ? 'group mean' : 'difference map'}: −${limit.toFixed(3)} to +${limit.toFixed(3)} model units. Blue = negative; red = positive. Each hemisphere has lateral and medial views.`;
  setURL();
}
function chooseGroup(group) {
  selectedGroup = group;
  for (const button of el('groups').children) button.setAttribute('aria-pressed', String(Number(button.dataset.value) === group.value));
  showMap();
  showMembers();
}
function showGroups(preferred = null) {
  const attribute = el('attribute').value;
  const groups = report.groups.filter(g => g.attribute === attribute);
  el('groups').replaceChildren();
  for (const group of groups) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.value = group.value;
    button.setAttribute('aria-pressed', 'false');
    button.setAttribute('aria-label', `${group.label}, ${group.count} stimuli`);
    const label = document.createElement('span');
    label.textContent = group.attribute === 'complexity' ? `${group.value*10}–${(group.value+1)*10}` : group.label;
    const count = document.createElement('small');
    count.textContent = `n=${group.count}`;
    button.append(label, count);
    button.addEventListener('click', () => chooseGroup(group));
    el('groups').append(button);
  }
  chooseGroup(groups.find(g => String(g.value) === preferred) || groups[0]);
}
async function loadReport() {
  const id = ++requestID;
  report = null;
  selectedGroup = null;
  el('brain-workspace').hidden = true;
  el('download').hidden = true;
  el('attribute').disabled = true;
  el('statistic').disabled = true;
  el('brain-status').textContent = 'Loading predictions…';
  const defaultPath = `../tribe/outputs/${el('dataset').value}/explorer.json`;
  const path = new URLSearchParams(location.search).get('data') || defaultPath;
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
    el('attribute').disabled = false;
    el('statistic').disabled = false;
    el('brain-workspace').hidden = false;
    el('brain-status').textContent = `${report.count.toLocaleString()} predictions loaded.`;
    el('sample-summary').textContent = `${report.count.toLocaleString()} of ${report.eligible_count.toLocaleString()} available annotated stimuli · ${report.scope === 'sample' ? `Sample seed ${report.seed}` : 'Whole dataset'}`;
    el('sampling-note').textContent = report.scope === 'sample'
      ? 'Balanced sampling across ten complexity bins; groups summarize this selected sample, not the natural dataset distribution. Bins exclude their upper edge except 100. Badge counts show the number of stimuli in each group.'
      : 'All available annotated stimuli. Feature groups use exact published counts; group sizes may differ. Bins exclude their upper edge except 100.';
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
    el('brain-status').textContent = el('dataset').value === 'massvis_all'
      ? 'Whole-dataset results are unavailable. Select the 100-stimulus sample to explore completed predictions.'
      : 'Sample results are unavailable. Generate the dataset report, then reload this page.';
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
el('more-members').addEventListener('click', () => showMembers(false));
el('brain-map').addEventListener('error', () => { el('brain-map').hidden = true; el('map-error').hidden = false; });
loadReport();
