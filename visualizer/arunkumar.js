'use strict';
const $ = id => document.getElementById(id);
const intervalEdges = [-10, -6, -3, 3, 6, 10];
let manifest, records = [], current = 1;
const ratingFor = record => record.ratings[`experiment_${record.attributes.experiment}`];
const signed = value => value > 0 ? `+${value.toFixed(2)}` : value.toFixed(2);

function drawHistogram(record) {
  const svg = $('histogram');
  svg.replaceChildren();
  const add = (tag, attrs, text, parent = svg) => {
    const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
    if (text !== undefined) element.textContent = text;
    parent.append(element);
    return element;
  };
  const bins = Array.from({length: 20}, () => ({count: 0, index: Infinity, distance: Infinity}));
  for (const row of records) {
    const mean = ratingFor(row).mean;
    const position = Math.min(19, Math.floor(mean + 10));
    const bin = bins[position];
    bin.count++;
    const distance = Math.abs(mean - (-10 + position + 0.5));
    const tied = Math.abs(distance - bin.distance) < 1e-9;
    if ((!tied && distance < bin.distance) || (tied && row.index < bin.index)) {
      bin.distance = distance;
      bin.index = row.index;
    }
  }
  const max = Math.max(10, Math.ceil(Math.max(...bins.map(bin => bin.count)) / 10) * 10);
  const left = 42, width = 500, top = 42, height = 128, baseline = top + height;
  const x = value => left + (value + 10) / 20 * width;
  const rating = ratingFor(record), mean = rating.mean, sd = rating.stdev;
  add('title', {}, `${records.length} experiment-specific means in one-point bins; current mean ${signed(mean)}, SD ${sd.toFixed(2)}. I1–I5 boundaries: −6, −3, 3, 6.`);
  for (const count of [0, max / 2, max]) {
    const y = baseline - count / max * height;
    add('line', {x1: left, x2: left + width, y1: y, y2: y, class: 'hist-grid'});
    add('text', {x: left - 7, y: y + 4, 'text-anchor': 'end', class: 'hist-label'}, String(count));
  }
  bins.forEach((bin, i) => {
    const start = -10 + i, end = start + 1, midpoint = start + 0.5;
    const bar = add('rect', {x: x(start) + 1, y: baseline - bin.count / max * height,
      width: width / 20 - 2, height: bin.count / max * height, class: 'hist-bar',
      'data-start': start, 'data-end': end, 'data-count': bin.count, 'data-index': bin.count ? bin.index : ''});
    const label = `[${start}, ${end}${i === 19 ? ']' : ')'}: ${bin.count} observations`;
    add('title', {}, label + (bin.count ? `. Open index ${bin.index}: mean closest to ${midpoint}.` : ''), bar);
    if (bin.count) {
      bar.setAttribute('role', 'button');
      bar.setAttribute('tabindex', '0');
      bar.setAttribute('aria-label', `${label}. Open visualization ${bin.index}, whose mean is closest to the bin midpoint ${midpoint}.`);
      bar.addEventListener('click', () => show(bin.index));
      bar.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          show(bin.index);
          svg.querySelectorAll('.hist-bar')[i].focus();
        }
      });
    }
  });
  const low = Math.max(-10, mean - sd), high = Math.min(10, mean + sd);
  add('rect', {x: x(low), y: top, width: x(high) - x(low), height, class: 'hist-shade'});
  for (const boundary of intervalEdges.slice(1, -1)) {
    const line = add('line', {x1: x(boundary), x2: x(boundary), y1: top, y2: baseline,
      class: 'hist-boundary', 'data-value': boundary});
    add('title', {}, `Published I1–I5 boundary at ${boundary}`, line);
    add('text', {x: x(boundary), y: top - 8, 'text-anchor': 'middle', class: 'hist-boundary-label'}, String(boundary));
  }
  const labels = manifest.attributes.find(attribute => attribute.key === 'image_information_rating').values;
  $('rating-intervals').replaceChildren();
  intervalEdges.slice(0, -1).forEach((start, i) => {
    const id = `I${i + 1}`, active = rating.interval === id;
    const text = add('text', {x: x((start + intervalEdges[i + 1]) / 2), y: 16,
      'text-anchor': 'middle', class: `hist-interval${active ? ' current' : ''}`}, id);
    add('title', {}, labels[id], text);
    const item = document.createElement('li');
    item.textContent = labels[id];
    if (active) {item.className = 'current'; item.setAttribute('aria-current', 'true');}
    $('rating-intervals').append(item);
  });
  add('line', {x1: x(mean), x2: x(mean), y1: top, y2: baseline, class: 'hist-mean'});
  add('circle', {cx: x(mean), cy: top, r: 4, class: 'hist-dot'});
  for (const tick of [-10, -5, 0, 5, 10]) {
    add('text', {x: x(tick), y: baseline + 20, 'text-anchor': 'middle', class: 'hist-label'}, String(tick));
  }
  add('text', {x: left, y: 212, class: 'hist-label'}, 'Image-like');
  add('text', {x: left + width, y: 212, 'text-anchor': 'end', class: 'hist-label'}, 'Information-like');
  $('rating-spread').textContent = `Experiment ${record.attributes.experiment} · SD ${sd.toFixed(2)} · Variance ${(sd * sd).toFixed(2)}`;
  $('group-ratings').hidden = !rating.group_a;
  $('group-ratings').textContent = rating.group_a
    ? `Group A: ${signed(rating.group_a.mean)} (SD ${rating.group_a.stdev.toFixed(2)}) · Group B: ${signed(rating.group_b.mean)} (SD ${rating.group_b.stdev.toFixed(2)})` : '';
}

function drawFeatures(record) {
  $('dimensions').replaceChildren();
  const groups = new Map();
  for (const attribute of manifest.attributes) {
    if (!attribute.source_sheet) continue;
    if (!groups.has(attribute.source_sheet)) groups.set(attribute.source_sheet, []);
    groups.get(attribute.source_sheet).push(attribute);
  }
  for (const [group, attributes] of groups) {
    const section = document.createElement('section');
    section.className = 'dimension';
    const heading = document.createElement('h2');
    heading.textContent = group;
    const table = document.createElement('table');
    table.setAttribute('aria-label', `${group} attributes`);
    const body = document.createElement('tbody');
    for (const attribute of attributes) {
      const value = record.attributes[attribute.key];
      const row = document.createElement('tr');
      row.dataset.key = attribute.key;
      const term = document.createElement('th');
      term.scope = 'row';
      term.textContent = attribute.label;
      term.title = attribute.source_column;
      const detail = document.createElement('td');
      detail.textContent = value == null ? 'Unavailable' : attribute.values[String(value)] ?? String(value);
      if (attribute.kind === 'binary') detail.className = value ? 'yes' : 'no';
      row.append(term, detail);
      body.append(row);
    }
    table.append(body);
    section.append(heading, table);
    $('dimensions').append(section);
  }
}

function show(value, updateURL = true) {
  const position = records.findIndex(row => row.index === Number(value));
  if (!/^\d+$/.test(String(value)) || position < 0) {
    $('status').textContent = `Enter a whole-number visualization index from ${$('index').min} to ${$('index').max}.`;
    return;
  }
  const record = records[position], rating = ratingFor(record);
  current = record.index;
  $('index').value = current;
  $('workspace').hidden = false;
  $('status').textContent = `Viewing visualization ${current} · Experiment ${record.attributes.experiment} · ${records.length} observations in this distribution.`;
  $('category').textContent = `Experiment ${record.attributes.experiment} · ${record.attributes.chart_category}`;
  $('image-title').textContent = record.filename;
  $('image-id').textContent = `Original stimulus ${record.original_stimulus_index}`;
  $('match').textContent = record.experiment_2_rank == null ? 'Original experiment 1 image' : `Experiment 2 rank ${record.experiment_2_rank} of 100`;
  const imageURL = '../' + record.source.split('/').map(encodeURIComponent).join('/');
  $('image').hidden = false;
  $('image-error').hidden = true;
  $('image').alt = `${record.label}: ${record.filename}`;
  $('image').src = imageURL;
  $('original').href = imageURL;
  $('score').textContent = signed(rating.mean);
  $('current-interval').textContent = rating.interval;
  drawHistogram(record);
  drawFeatures(record);
  $('previous').disabled = position === 0;
  $('next').disabled = position === records.length - 1;
  document.title = `Arunkumar ${current} · Dataset explorer`;
  if (updateURL) {
    const url = new URL(location.href);
    url.searchParams.set('experiment', $('experiment').value);
    url.searchParams.set('index', current);
    history.pushState(null, '', url);
  }
}

function selectExperiment(experiment) {
  $('experiment').value = ['all', '1', '2'].includes(experiment) ? experiment : 'all';
  records = manifest.rows.filter(row => $('experiment').value === 'all' ||
    String(row.attributes.experiment) === $('experiment').value);
  $('index').min = records[0].index;
  $('index').max = records.at(-1).index;
  $('index-range').textContent = `${$('index').min}–${$('index').max}`;
  $('population-label').textContent = $('experiment').value === 'all' ? 'Both experiments' : `Experiment ${$('experiment').value}`;
  $('distribution-note').textContent = 'Histogram bins are one point wide: [−10, −9), [−9, −8), …, [9, 10]. '
    + ($('experiment').value === 'all'
      ? 'Both experiments include 600 observations; repeated stimuli retain their separate experiment ratings.'
      : `Each of the ${records.length} stimuli uses its experiment ${$('experiment').value} rating.`);
}

function restoreURL() {
  const params = new URLSearchParams(location.search);
  selectExperiment(params.get('experiment') ?? 'all');
  show(params.get('index') ?? records[0].index, false);
}

$('image').addEventListener('error', () => {
  $('image').hidden = true;
  $('image-error').hidden = false;
});
$('lookup').addEventListener('submit', event => {event.preventDefault(); show($('index').value);});
$('previous').addEventListener('click', () => show(records[records.findIndex(row => row.index === current) - 1]?.index));
$('next').addEventListener('click', () => show(records[records.findIndex(row => row.index === current) + 1]?.index));
$('experiment').addEventListener('change', () => {
  const original = manifest.rows.find(row => row.index === current).original_stimulus_index;
  selectExperiment($('experiment').value);
  const record = records.find(row => row.index === current)
    ?? records.find(row => row.original_stimulus_index === original) ?? records[0];
  show(record.index);
});
window.addEventListener('popstate', () => {if (manifest) restoreURL();});

async function start() {
  try {
    const response = await fetch('../tribe/studies/manifests/arunkumar.json');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    manifest = await response.json();
    $('experiment').disabled = false;
    $('lookup').querySelector('button').disabled = false;
    restoreURL();
  } catch (error) {
    $('status').textContent = 'Arunkumar annotations unavailable.';
    console.error(error);
  }
}
start();
