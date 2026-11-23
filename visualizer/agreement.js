'use strict';

const $ = id => document.getElementById(id);
const format = new Intl.NumberFormat('en-US');
const spreadGroupSize = 200;
const ratingsPlot = $('ratings-plot');
const spreadPlot = $('spread-plot');
let ranked = [], spreadGroups = [], sdMax = 10, ratingCount = 0;
let firstRank = 1, windowSize = 0, selectedRank = 0;

function rankRatings(csv) {
  const [header, ...lines] = csv.trim().split(/\r?\n/);
  if (header !== 'image_id,response' || !lines.length) throw new Error('Missing rating data');
  const byImage = new Map();
  for (const line of lines) {
    const [imageId, response] = line.split(',');
    const score = Number(response);
    if (!/^\d+\.png$/.test(imageId) || !response?.trim() || !Number.isFinite(score) || score < 0 || score > 100) {
      throw new Error(`Invalid rating for ${imageId}`);
    }
    const index = Number(imageId.slice(0, -4));
    if (!byImage.has(index)) byImage.set(index, []);
    const ratings = byImage.get(index);
    if (ratings.length === 10) throw new Error(`More than 10 ratings for ${imageId}`);
    ratings.push(score);
  }
  return Array.from(byImage, ([index, ratings]) => {
    ratings.sort((a, b) => a - b);
    const mean = ratings.reduce((sum, score) => sum + score, 0) / ratings.length;
    const sumOfSquares = ratings.reduce((sum, score) => sum + (score - mean) ** 2, 0);
    const sd = ratings.length > 1 ? Math.sqrt(sumOfSquares / (ratings.length - 1)) : null;
    const counts = new Map();
    for (const score of ratings) counts.set(score, (counts.get(score) ?? 0) + 1);
    return {index, ratings, mean, sd, counts};
  }).sort((a, b) => a.mean - b.mean || a.index - b.index);
}

function summarizeSpread(records) {
  const groups = [];
  for (let start = 0; start < records.length; start += spreadGroupSize) {
    const values = records.slice(start, start + spreadGroupSize)
      .map(record => record.sd).filter(sd => sd !== null).sort((a, b) => a - b);
    const middle = Math.floor(values.length / 2);
    groups.push(values.length ? (values[middle] + values[Math.floor((values.length - 1) / 2)]) / 2 : null);
  }
  return groups;
}

function lastRank() {
  return Math.min(ranked.length, firstRank + windowSize - 1);
}

function windowStart(rank) {
  return Math.floor((rank - 1) / windowSize) * windowSize + 1;
}

function plotArea(width, height) {
  const left = 40, right = width - 23, top = 10, bottom = height - 28;
  return {left, right, top, bottom, width: right - left};
}

function rankTicks(last, width) {
  const interval = (last - firstRank + 1) / Math.max(2, Math.floor(width / 100));
  const magnitude = 10 ** Math.floor(Math.log10(interval));
  const step = Math.max(1, [1, 2, 5, 10].map(value => value * magnitude).find(value => value >= interval));
  const clearance = (last - firstRank + 1) * 50 / width;
  const ticks = [firstRank];
  for (let rank = Math.ceil(firstRank / step) * step; rank < last; rank += step) {
    if (rank - firstRank >= clearance && last - rank >= clearance) ticks.push(rank);
  }
  if (last > firstRank) ticks.push(last);
  return ticks;
}

function drawAxes(ctx, area, x, y, last, max, increment) {
  ctx.font = '14px system-ui, sans-serif';
  ctx.fillStyle = '#52667d';
  ctx.strokeStyle = '#dce3eb';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let tick = 0; tick <= max; tick += increment) {
    const py = y(tick);
    ctx.beginPath();
    ctx.moveTo(area.left, py);
    ctx.lineTo(area.right, py);
    ctx.stroke();
    ctx.fillText(tick, area.left - 8, py);
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const tick of rankTicks(last, area.width)) {
    const px = x(tick);
    ctx.beginPath();
    ctx.moveTo(px, area.bottom);
    ctx.lineTo(px, area.bottom + 5);
    ctx.stroke();
    ctx.fillText(format.format(tick), px, area.bottom + 8);
  }
}

function drawSummary(ctx, isSpread, x, y, last) {
  ctx.strokeStyle = '#aa4b14';
  ctx.lineWidth = 2;
  ctx.beginPath();
  if (isSpread) {
    const firstGroup = Math.floor((firstRank - 1) / spreadGroupSize);
    for (let group = firstGroup; group * spreadGroupSize < last; group++) {
      const median = spreadGroups[group];
      if (median === null) continue;
      const start = Math.max(firstRank, group * spreadGroupSize + 1);
      const end = Math.min(last, (group + 1) * spreadGroupSize);
      ctx.moveTo(x(start - 0.5) + 0.5, y(median));
      ctx.lineTo(x(end + 0.5) - 0.5, y(median));
    }
  } else {
    ctx.moveTo(x(firstRank - 0.5), y(ranked[firstRank - 1].mean));
    for (let rank = firstRank; rank <= last; rank++) ctx.lineTo(x(rank), y(ranked[rank - 1].mean));
    ctx.lineTo(x(last + 0.5), y(ranked[last - 1].mean));
  }
  ctx.stroke();
}

function dot(ctx, x, y, radius) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, 2 * Math.PI);
  ctx.fill();
}

function drawPlot(canvas, isSpread) {
  const {width, height} = canvas.getBoundingClientRect();
  const area = plotArea(width, height);
  if (area.width <= 0 || !windowSize) return;
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext('2d');
  ctx.scale(scale, scale);
  const last = lastRank();
  const spacing = area.width / (last - firstRank + 1);
  const max = isSpread ? sdMax : 100;
  const x = rank => area.left + (rank - firstRank + 0.5) * spacing;
  const y = score => area.bottom - score / max * (area.bottom - area.top);
  drawAxes(ctx, area, x, y, last, max, isSpread ? 10 : 25);

  const hasSelection = selectedRank >= firstRank && selectedRank <= last;
  if (hasSelection) {
    ctx.strokeStyle = '#7c9ffa';
    ctx.beginPath();
    ctx.moveTo(x(selectedRank), area.top);
    ctx.lineTo(x(selectedRank), area.bottom);
    ctx.stroke();
  }
  drawSummary(ctx, isSpread, x, y, last);

  const radius = Math.max(1.1, Math.min(3.5, spacing / 4));
  ctx.fillStyle = '#215bd5';
  ctx.globalAlpha = spacing < 2 ? 0.22 : 0.72;
  function drawRecord(rank, size) {
    const record = ranked[rank - 1];
    if (isSpread) {
      if (record.sd !== null) dot(ctx, x(rank), y(record.sd), size);
      return;
    }
    for (const [score, count] of record.counts) {
      const gap = Math.min(8, spacing * 0.85 / count);
      const dotSize = count === 1 ? size : Math.max(1, Math.min(size, gap * 0.44));
      for (let i = 0; i < count; i++) dot(ctx, x(rank) + (i - (count - 1) / 2) * gap, y(score), dotSize);
    }
  }
  for (let rank = firstRank; rank <= last; rank++) {
    if (rank !== selectedRank) drawRecord(rank, radius);
  }
  if (hasSelection) {
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#153f99';
    drawRecord(selectedRank, Math.max(radius, 2.4));
  }
}

function drawCharts() {
  drawPlot(ratingsPlot, false);
  drawPlot(spreadPlot, true);
}

function updateView() {
  $('previous-window').disabled = firstRank === 1;
  $('next-window').disabled = lastRank() === ranked.length;
  $('status').textContent = `${format.format(ranked.length)} visualizations · ${format.format(ratingCount)} ratings · Showing ranks ${format.format(firstRank)}–${format.format(lastRank())}`;
  drawCharts();
}

function selectRank(rank) {
  const record = ranked[rank - 1];
  selectedRank = rank;
  firstRank = windowStart(rank);
  $('rank').value = rank;
  const spread = record.sd === null ? 'SD unavailable' : `SD ${record.sd.toFixed(2)}`;
  $('rank-details').textContent = `Rank ${format.format(rank)} · Image ${record.index}.png · Mean ${record.mean.toFixed(2)} · ${spread} · ${record.ratings.length} ${record.ratings.length === 1 ? 'rating' : 'ratings'} (${record.ratings.join(', ')})`;
  $('image-link').href = `data.html?index=${record.index}`;
  $('image-link').hidden = false;
  updateView();
}

for (const canvas of [ratingsPlot, spreadPlot]) {
  canvas.addEventListener('click', event => {
    const rect = canvas.getBoundingClientRect();
    const area = plotArea(rect.width, rect.height);
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    if (x < area.left || x > area.right || y < area.top || y > area.bottom) return;
    const offset = Math.floor((x - area.left) / area.width * (lastRank() - firstRank + 1));
    selectRank(Math.min(lastRank(), firstRank + offset));
  });
}

$('rank-window').addEventListener('change', event => {
  const anchor = selectedRank >= firstRank && selectedRank <= lastRank() ? selectedRank : firstRank;
  windowSize = Math.min(Number(event.target.value) || ranked.length, ranked.length);
  firstRank = windowStart(anchor);
  updateView();
});
$('previous-window').addEventListener('click', () => {
  firstRank -= windowSize;
  updateView();
});
$('next-window').addEventListener('click', () => {
  firstRank += windowSize;
  updateView();
});
$('rank-lookup').addEventListener('submit', event => {
  event.preventDefault();
  selectRank($('rank').valueAsNumber);
});

async function start() {
  try {
    const response = await fetch('../label/source/ratings.csv');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    ranked = rankRatings(await response.text());
    spreadGroups = summarizeSpread(ranked);
    const largestSd = ranked.reduce((max, record) => Math.max(max, record.sd ?? 0), 0);
    sdMax = Math.max(10, Math.ceil(largestSd / 10) * 10);
    ratingCount = ranked.reduce((sum, record) => sum + record.ratings.length, 0);
    windowSize = ranked.length;
    $('rank').max = ranked.length;
    $('rating-workspace').hidden = false;
    updateView();
    const observer = new ResizeObserver(drawCharts);
    observer.observe(ratingsPlot);
    observer.observe(spreadPlot);
  } catch (error) {
    $('status').textContent = 'MASSVIS ratings could not be loaded.';
    console.error(error);
  }
}

start();
