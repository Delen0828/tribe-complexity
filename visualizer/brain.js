'use strict';
const el = id => document.getElementById(id);
let report = null, analysis = null, catalog = [], visibleMembers = 0, memberIndices = [], selection = null;
let reportRequest = 0, analysisRequest = 0, analysisController = null;
const initial = new URLSearchParams(location.search);
const dataPath = initial.get('data');

function sourceParams() {
  const params = new URLSearchParams({dataset: el('dataset').value});
  if (dataPath) {
    const url = new URL(dataPath, location.href);
    if (url.origin !== location.origin) throw new Error('Invalid report URL');
    params.set('data', url.pathname);
  }
  return params;
}
async function json(url, signal) {
  const startupHint = 'Run ./start-visualizer.sh and open the URL it prints.';
  let response;
  try {
    response = await fetch(url, {signal, cache: 'no-store'});
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error(`Cannot reach the visualizer server. ${startupHint}`);
  }
  if ((url === '/api/reports' && response.status === 404) ||
      !(response.headers.get('content-type') || '').includes('application/json')) {
    throw new Error(`Visualizer API unavailable at ${location.origin}. ${startupHint}`);
  }
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Predictions unavailable');
  return data;
}
function setURL() {
  const url = new URL(location.href);
  url.searchParams.set('dataset', el('dataset').value);
  url.searchParams.set('x', el('x-axis').value);
  url.searchParams.set('y', el('y-axis').value);
  url.searchParams.set('statistic', el('statistic').value);
  url.searchParams.set('views', el('views').value);
  url.searchParams.delete('attribute');
  url.searchParams.delete('group');
  for (const key of ['examples', 'xbin', 'ybin']) url.searchParams.delete(key);
  if (selection) {
    url.searchParams.set('examples', selection.axis);
    if (selection.x !== undefined) url.searchParams.set('xbin', selection.x);
    if (selection.y !== undefined) url.searchParams.set('ybin', selection.y);
  }
  history.replaceState(null, '', url);
}
function showMembers(reset = true) {
  if (reset) {visibleMembers = 0; el('members').replaceChildren(); el('members').scrollLeft = 0;}
  const rows = new Map(report.rows.map(row => [row.index, row]));
  const end = Math.min(visibleMembers + 24, memberIndices.length);
  for (const index of memberIndices.slice(visibleMembers, end)) {
    const row = rows.get(index);
    const link = document.createElement('a');
    link.className = 'member';
    link.href = report.dataset === 'massvis' ? `data.html?index=${index}` : row.image;
    const image = document.createElement('img');
    image.src = row.image;
    image.alt = row.filename;
    image.loading = 'lazy';
    const title = document.createElement('strong');
    title.textContent = row.label || `Image ${index}`;
    title.title = title.textContent;
    const detail = document.createElement('span');
    detail.textContent = row.score_label || (Number.isFinite(row.score) ? `Complexity ${row.score.toFixed(1)}` : row.filename);
    detail.title = detail.textContent;
    link.append(image, title, detail);
    el('members').append(link);
  }
  visibleMembers = end;
  el('member-range').textContent = `${memberIndices.length.toLocaleString()} examples`;
  el('members-empty').hidden = memberIndices.length > 0;
}
function chooseExamples(axis, x, y) {
  selection = {axis};
  if (x !== undefined) selection.x = x;
  if (y !== undefined) selection.y = y;
  const xGroup = analysis.axes.x.find(g => String(g.value) === String(x));
  const yGroup = analysis.axes.y.find(g => String(g.value) === String(y));
  let label;
  if (axis === 'x') {memberIndices = xGroup.indices; label = xGroup.label;}
  else if (axis === 'y') {memberIndices = yGroup.indices; label = yGroup.label;}
  else {
    memberIndices = analysis.cells.find(c => String(c.x) === String(x) && String(c.y) === String(y)).indices;
    label = `${xGroup.label} × ${yGroup.label}`;
  }
  for (const button of el('groups').querySelectorAll('button')) {
    const selected = button.dataset.axis === axis &&
      (x === undefined || button.dataset.x === String(x)) && (y === undefined || button.dataset.y === String(y));
    button.setAttribute('aria-pressed', String(selected));
  }
  el('members-heading').textContent = `Examples · ${label}`;
  el('members-heading').title = el('members-heading').textContent;
  el('members-section').hidden = false;
  showMembers();
  setURL();
}
function tick(group, axis) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'axis-tick';
  button.dataset.axis = axis;
  button.dataset[axis] = group.value;
  button.setAttribute('aria-pressed', 'false');
  button.setAttribute('aria-label', `${axis.toUpperCase()}: ${group.label}; ${group.indices.length} examples`);
  button.append(document.createTextNode(group.label));
  const count = document.createElement('small');
  count.textContent = `n = ${group.indices.length}`;
  button.append(count);
  button.addEventListener('click', () => chooseExamples(axis, axis === 'x' ? group.value : undefined,
                                                       axis === 'y' ? group.value : undefined));
  return button;
}
function drawGrid() {
  const grid = el('groups');
  grid.replaceChildren();
  grid.style.minWidth = `${120 + analysis.axes.x.length * 235}px`;
  const head = grid.createTHead().insertRow();
  const corner = document.createElement('th');
  corner.className = 'axis-corner';
  const label = key => report.attributes.find(a => a.key === key)?.label || 'All';
  corner.textContent = `${label(el('y-axis').value)} / ${label(el('x-axis').value)}`;
  head.append(corner);
  for (const group of analysis.axes.x) {
    const th = document.createElement('th'); th.scope = 'col'; th.append(tick(group, 'x')); head.append(th);
  }
  const body = grid.createTBody();
  const cells = new Map(analysis.cells.map(c => [JSON.stringify([c.x, c.y]), c]));
  for (const y of analysis.axes.y) {
    const tr = body.insertRow();
    const th = document.createElement('th'); th.scope = 'row'; th.append(tick(y, 'y')); tr.append(th);
    for (const x of analysis.axes.x) {
      const cell = cells.get(JSON.stringify([x.value, y.value]));
      const td = tr.insertCell();
      td.dataset.count = cell.count;
      if (!cell.count) {td.className = 'empty-cell'; td.textContent = '—'; continue;}
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'cell-map';
      button.dataset.axis = 'cell'; button.dataset.x = x.value; button.dataset.y = y.value;
      button.setAttribute('aria-pressed', 'false');
      button.setAttribute('aria-label', `${x.label} × ${y.label}; ${cell.count} examples`);
      button.addEventListener('click', () => chooseExamples('cell', x.value, y.value));
      const image = document.createElement('img');
      image.className = 'group-map'; image.loading = 'lazy';
      image.width = el('views').value === 'inferior' ? 285 : 570;
      image.height = el('views').value === 'inferior' ? 257 : el('views').value === 'four' ? 513 : 770;
      image.alt = `Predicted cortical response · ${x.label} × ${y.label}`;
      const mapParams = sourceParams();
      for (const [key, value] of Object.entries({id: analysis.id, x: cell.x, y: cell.y, views: el('views').value,
        x_axis: el('x-axis').value, y_axis: el('y-axis').value, statistic: el('statistic').value})) mapParams.set(key, value);
      image.src = '/api/map?' + mapParams;
      const error = document.createElement('span'); error.className = 'map-error'; error.hidden = true;
      error.textContent = 'Map unavailable';
      image.addEventListener('error', () => {image.hidden = true; error.hidden = false;});
      const count = document.createElement('span'); count.className = 'cell-count'; count.textContent = `n = ${cell.count}`;
      button.append(image, error, count); td.append(button);
    }
  }
  el('scale-low').textContent = `−${analysis.limit.toFixed(3)}`;
  el('scale-high').textContent = `+${analysis.limit.toFixed(3)}`;
  if (selection) chooseExamples(selection.axis, selection.x, selection.y);
  else {
    memberIndices = report.rows.map(row => row.index);
    el('members-heading').textContent = 'Examples · All';
    el('members-heading').title = el('members-heading').textContent;
    el('members-section').hidden = false;
    showMembers();
  }
}
function updateFields() {
  for (const button of el('fields').querySelectorAll('button')) {
    button.setAttribute('aria-pressed', String(el(`${button.dataset.axis}-axis`).value === button.dataset.key));
  }
}
async function refreshGrid(restore = null) {
  const request = ++analysisRequest;
  analysisController?.abort();
  analysisController = new AbortController();
  selection = null;
  analysis = null;
  el('members-section').hidden = true;
  el('groups').replaceChildren();
  el('brain-status').textContent = 'Loading…';
  updateFields();
  setURL();
  try {
    const params = sourceParams();
    params.set('x', el('x-axis').value); params.set('y', el('y-axis').value);
    params.set('statistic', el('statistic').value);
    const data = await json('/api/analysis?' + params, analysisController.signal);
    if (request !== analysisRequest) return;
    analysis = data;
    if (restore && ['x', 'y', 'cell'].includes(restore.axis)) {
      const hasX = analysis.axes.x.some(g => String(g.value) === String(restore.x));
      const hasY = analysis.axes.y.some(g => String(g.value) === String(restore.y));
      if ((restore.axis === 'x' && hasX) || (restore.axis === 'y' && hasY) || (restore.axis === 'cell' && hasX && hasY)) selection = restore;
    }
    drawGrid(); setURL();
    el('brain-status').textContent = '';
  } catch (error) {
    if (error.name === 'AbortError' || request !== analysisRequest) return;
    el('brain-status').textContent = error.message;
  }
}
function configureAxes(params) {
  el('field-search').value = '';
  for (const axis of ['x', 'y']) {
    const select = el(`${axis}-axis`); select.replaceChildren();
    if (axis === 'y') select.add(new Option('All', ''));
    for (const attribute of report.attributes) select.add(new Option(attribute.label, attribute.key));
    const preferred = params.get(axis) || (axis === 'x' ? params.get('attribute') : null);
    if ([...select.options].some(o => o.value === preferred)) select.value = preferred;
  }
  el('fields').replaceChildren();
  for (const attribute of report.attributes) {
    const field = document.createElement('div'); field.className = 'field';
    const name = document.createElement('span'); name.textContent = attribute.label; field.append(name);
    for (const axis of ['x', 'y']) {
      const button = document.createElement('button'); button.type = 'button'; button.textContent = axis.toUpperCase();
      button.dataset.axis = axis; button.dataset.key = attribute.key;
      button.setAttribute('aria-label', `Assign ${attribute.label} to ${axis.toUpperCase()}`);
      button.addEventListener('click', () => {el(`${axis}-axis`).value = attribute.key; refreshGrid();});
      field.append(button);
    }
    el('fields').append(field);
  }
}
async function loadReport() {
  const request = ++reportRequest;
  ++analysisRequest; analysisController?.abort();
  report = null; analysis = null; selection = null;
  el('brain-workspace').hidden = true; el('selection-link').hidden = true;
  el('statistic').disabled = true; el('views').disabled = true;
  el('brain-status').textContent = 'Loading predictions…';
  const params = new URLSearchParams(location.search);
  try {
    const data = await json('/api/report?' + sourceParams());
    if (request !== reportRequest) return;
    report = data;
    el('report-title').textContent = report.title; document.title = `TRIBE · ${report.title}`;
    configureAxes(params);
    el('statistic').value = params.get('statistic') === 'contrast' ? 'contrast' : 'mean';
    el('views').value = ['four', 'all'].includes(params.get('views')) ? params.get('views') : 'inferior';
    el('statistic').disabled = false; el('views').disabled = false;
    el('selection-link').href = report.selection; el('selection-link').hidden = false;
    el('sample-summary').textContent = report.complete ? `n = ${report.count.toLocaleString()}`
      : `Partial · ${report.count.toLocaleString()} / ${report.selected_count.toLocaleString()}`;
    el('brain-workspace').hidden = false;
    await refreshGrid({axis: params.get('examples'), x: params.get('xbin') ?? undefined, y: params.get('ybin') ?? undefined});
  } catch (error) {
    if (request !== reportRequest) return;
    el('brain-status').textContent = error.message;
  }
}
el('dataset').addEventListener('change', () => {
  const url = new URL(location.href);
  url.search = new URLSearchParams({dataset: el('dataset').value}); history.replaceState(null, '', url);
  loadReport();
});
for (const id of ['x-axis', 'y-axis']) el(id).addEventListener('change', () => refreshGrid());
el('statistic').addEventListener('change', () => refreshGrid(selection));
el('views').addEventListener('change', () => {if (analysis) {drawGrid(); setURL();}});
el('swap-axes').addEventListener('click', () => {
  const x = el('x-axis').value, y = el('y-axis').value;
  if (!y) return;
  el('x-axis').value = y; el('y-axis').value = x; refreshGrid();
});
el('field-search').addEventListener('input', () => {
  const term = el('field-search').value.toLowerCase();
  for (const field of el('fields').children) field.hidden = !field.textContent.toLowerCase().includes(term);
});
el('members').addEventListener('scroll', () => {
  const gallery = el('members');
  if (visibleMembers < memberIndices.length &&
      gallery.scrollWidth - gallery.scrollLeft - gallery.clientWidth < gallery.clientWidth) showMembers(false);
});
async function initialize() {
  try {
    catalog = await json('/api/reports');
    for (const item of catalog) el('dataset').add(new Option(item.label, item.id));
    if (catalog.some(item => item.id === initial.get('dataset'))) el('dataset').value = initial.get('dataset');
    if (dataPath) el('dataset').disabled = true;
    await loadReport();
  } catch (error) {el('brain-status').textContent = error.message;}
}
initialize();
