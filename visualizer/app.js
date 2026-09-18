'use strict';
const $ = id => document.getElementById(id);
const names = {
  no_text: 'No text', axes_labels: 'Axis labels', axes_text: 'Axis text', titles: 'Titles',
  annotations: 'Annotations', captions: 'Captions', legend_text: 'Legend text', legend_title: 'Legend titles',
  text_only: 'Text only', black_and_white: 'Black and white', background_color: 'Non-white background',
  color_count: 'Distinct colors', quantitative: 'Quantitative variables', categorical: 'Categorical variables',
  chart_count: 'Charts', chart_types_count: 'Chart types', multi_panel: 'Multiple panels',
  area: 'Area', bar: 'Bar', circle: 'Circle', diagram: 'Diagram', distribution: 'Distribution',
  grid_or_matrix: 'Grid & matrix', line: 'Line', map: 'Map', point: 'Point', table: 'Table',
  text: 'Text', trees_or_networks: 'Trees & networks',
};
const counts = new Set(['color_count', 'quantitative', 'categorical', 'chart_count', 'chart_types_count']);
const categories = {S: 'Science', N: 'News', G: 'Government', I: 'Infographic'};
let records = [], current = 0;
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
  const bins = Array(20).fill(0);
  const midpointIndices = Array(20).fill(Infinity);
  const midpointDistances = Array(20).fill(Infinity);
  for (const r of records) {
    const bin = Math.min(19, Math.floor(r.mean_human_complexity_rating / 5));
    bins[bin]++;
    const distance = Math.abs(r.mean_human_complexity_rating - (bin * 5 + 2.5));
    const tied = Math.abs(distance - midpointDistances[bin]) < 1e-9;
    if ((!tied && distance < midpointDistances[bin]) || (tied && r.index < midpointIndices[bin])) {
      midpointDistances[bin] = distance;
      midpointIndices[bin] = r.index;
    }
  }
  const max = Math.ceil(Math.max(...bins) / 100) * 100;
  const left = 42, width = 500, top = 20, height = 126, baseline = top + height;
  const x = value => left + value / 100 * width;
  const mean = record.mean_human_complexity_rating, sd = record.human_rating_sd;
  add('title', {}, `All ${records.length} image means; current mean ${mean.toFixed(1)}, standard deviation ${sd.toFixed(1)} from ${record.human_rating_count} human ratings.`);
  for (const count of [0, max / 2, max]) {
    const y = baseline - count / max * height;
    add('line', {x1: left, x2: left + width, y1: y, y2: y, class: 'hist-grid'});
    add('text', {x: left - 7, y: y + 4, 'text-anchor': 'end', class: 'hist-label'}, String(count));
  }
  add('text', {x: left, y: 12, class: 'hist-label'}, 'Images');
  bins.forEach((count, i) => {
    const bar = add('rect', {x: x(i * 5) + 1, y: baseline - count / max * height,
      width: width / 20 - 2, height: count / max * height, class: 'hist-bar'});
    const label = `${i * 5}–${(i + 1) * 5}${i === 19 ? ' (inclusive)' : ' (upper bound excluded)'}: ${count} images`;
    add('title', {}, label + (count ? `. Open index ${midpointIndices[i]}: mean closest to ${i * 5 + 2.5}` : ''), bar);
    if (count) {
      bar.setAttribute('role', 'button');
      bar.setAttribute('tabindex', '0');
      bar.setAttribute('aria-label', `${label}. Open visualization ${midpointIndices[i]}, whose mean is closest to the bin midpoint ${i * 5 + 2.5}.`);
      bar.addEventListener('click', () => show(midpointIndices[i]));
      bar.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          show(midpointIndices[i]);
          // Redrawing replaces the bars; preserve keyboard focus on this bin.
          svg.querySelectorAll('.hist-bar')[i].focus();
        }
      });
    }
  });
  const low = Math.max(0, mean - sd), high = Math.min(100, mean + sd);
  add('rect', {x: x(low), y: top, width: x(high) - x(low), height, class: 'hist-shade'});
  add('line', {x1: x(mean), x2: x(mean), y1: top, y2: baseline, class: 'hist-mean'});
  add('circle', {cx: x(mean), cy: top, r: 4, class: 'hist-dot'});
  for (const tick of [0, 25, 50, 75, 100]) {
    add('text', {x: x(tick), y: baseline + 20, 'text-anchor': 'middle', class: 'hist-label'}, String(tick));
  }
  $('rating-spread').textContent = `Human ratings: n = ${record.human_rating_count} · SD ${sd.toFixed(1)} · Sample variance ${record.human_rating_variance.toFixed(1)} points². Shading shows rating spread, not a confidence interval${mean - sd < 0 || mean + sd > 100 ? '; clipped to 0–100' : ''}.`;
}
function show(value, updateURL = true) {
  if (!/^\d+$/.test(String(value)) || !records[Number(value)]) {
    $('status').textContent = 'Enter a whole-number visualization index from 0 to 5799.';
    return;
  }
  current = Number(value);
  const r = records[current];
  $('index').value = current;
  $('workspace').hidden = false;
  $('status').textContent = `Viewing visualization ${current} of indices 0–5799.`;
  $('image-title').textContent = r.filename;
  $('category').textContent = categories[r.category] || r.category;
  $('image-id').textContent = `Study ID ${r.image_id}`;
  $('match').textContent = r.image_match === 'documented_alias' ? 'Matched to local filename alias' : 'Original MASSVIS image';
  const img = $('image');
  img.hidden = !r.image_path;
  $('image-error').hidden = Boolean(r.image_path);
  $('original').hidden = !r.image_path;
  img.alt = `MASSVIS visualization ${current}: ${r.filename}`;
  if (r.image_path) {
    const url = '../' + r.image_path.split('/').map(encodeURIComponent).join('/');
    img.src = url;
    $('original').href = url;
  } else {
    img.removeAttribute('src');
    $('original').removeAttribute('href');
  }
  $('score').textContent = r.mean_human_complexity_rating.toFixed(1);
  drawHistogram(r);
  $('dimensions').replaceChildren();
  for (const [group, title] of [['text', 'Text'], ['color', 'Color'], ['data', 'Data'], ['design', 'Design']]) {
    const section = document.createElement('section');
    section.className = 'dimension';
    const heading = document.createElement('h2');
    heading.textContent = title;
    section.append(heading);
    const list = document.createElement('table');
    list.setAttribute('aria-label', `${title} attributes`);
    const body = document.createElement('tbody');
    for (const [key, value] of Object.entries(r[group])) {
      const row = document.createElement('tr');
      const term = document.createElement('th');
      term.scope = 'row';
      const detail = document.createElement('td');
      term.textContent = names[key];
      detail.textContent = value == null ? 'Unavailable' : counts.has(key) ? value : value ? 'Yes' : 'No';
      if (!counts.has(key)) detail.className = value ? 'yes' : 'no';
      row.append(term, detail);
      body.append(row);
    }
    list.append(body);
    section.append(list);
    $('dimensions').append(section);
  }
  $('previous').disabled = current === 0;
  $('next').disabled = current === records.length - 1;
  document.title = `MASSVIS ${current} · Complexity explorer`;
  if (updateURL) {
    const url = new URL(location.href);
    url.searchParams.set('index', current);
    history.pushState(null, '', url);
  }
}
$('image').addEventListener('error', () => {
  $('image').hidden = true;
  $('image-error').hidden = false;
});
$('lookup').addEventListener('submit', event => { event.preventDefault(); show($('index').value); });
$('previous').addEventListener('click', () => show(current - 1));
$('next').addEventListener('click', () => show(current + 1));
window.addEventListener('popstate', () => show(new URLSearchParams(location.search).get('index') ?? '0', false));
async function start() {
  try {
    const response = await fetch('../label/output/labels.json');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    records = await response.json();
    show(new URLSearchParams(location.search).get('index') ?? '0', false);
  } catch (error) {
    $('status').textContent = 'Cannot load annotations. Run python3 label/build.py, then serve the repository with python3 -m http.server 8000.';
    console.error(error);
  }
}
start();
