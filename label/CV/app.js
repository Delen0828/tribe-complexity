/* Static file-compatible viewer. Results load as scripts; no fetch/server needed. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const data = window.CV_DATA;
  if (!data) { $('detail-title').textContent = 'Generate results with run.py render'; return; }
  window.CV_FIGURES = window.CV_FIGURES || {};
  const key = `mark-atlas-reviews:${data.run_id}`;
  let reviews = {}, storageAvailable = true;
  try { reviews = JSON.parse(localStorage.getItem(key) || '{}'); } catch { storageAvailable = false; }
  let selected = null, detail = null, mode = 'overlay', band = null, instance = null;
  const label = s => (s || '').replaceAll('_', ' ');
  const countLabel = e => e.count !== null ? String(e.count) : e.status === 'not_applicable' ? 'N/A' : e.status === 'failed' ? 'Failed' : '—';
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const quality = id => reviews[id]?.quality || 'unreviewed';
  const number = n => Number(n || 0).toLocaleString();
  $('model').textContent = data.model;
  $('selected-total').textContent = data.entries.length;
  $('retained-total').textContent = number(data.summary.total_retained);
  $('ambiguous-total').textContent = number(data.summary.ambiguous);
  $('count-definition').textContent = data.count_definition;
  $('provenance').textContent = `${data.model} · ${data.provenance.gpu || 'GPU run pending'} · ${Math.round(data.summary.runtime_seconds / 60)} inference minutes · no additional tiled inference.`;
  $('run-label').textContent = `Run ${data.run_id.slice(0,12)} · Seed 0 · Original-resolution references`;
  $('about-toggle').onclick = () => { $('about').hidden = !$('about').hidden; };
  [...new Set(data.entries.flatMap(e => e.families))].sort().forEach(f => { const o = document.createElement('option'); o.value=f; o.textContent=label(f); $('family').append(o); });
  function renderBands() {
    $('bands').innerHTML = data.summary.bands.map(b => `<button class="band ${band === b.band ? 'active' : ''}" data-band="${b.band}" aria-pressed="${band === b.band}"><span>${b.band*10}–${(b.band+1)*10}</span><div class="band-bar"></div><strong>${b.selected} figures</strong></button>`).join('');
    $('bands').querySelectorAll('button').forEach(b => b.onclick = () => { const n=Number(b.dataset.band); band=band===n?null:n;renderBands();renderGallery(); });
  }
  function filtered() {
    const q=$('search').value.toLowerCase().trim(), f=$('family').value, s=$('status').value, r=$('review-filter').value;
    return data.entries.filter(e => (band===null||e.band===band) && (!q||`${e.image_id} ${e.filename}`.toLowerCase().includes(q)) && (!f||e.families.includes(f)) && (!s||e.status===s) && (!r||quality(e.image_id)===r));
  }
  function reviewTotal() { $('review-total').textContent = `${Object.values(reviews).filter(r=>r.quality!=='unreviewed').length} / ${data.entries.length}`; }
  function renderGallery() {
    const entries=filtered(); $('visible-total').textContent=`${entries.length} figures`; $('empty').hidden=!!entries.length;
    $('gallery').innerHTML=entries.map(e=>`<button class="card ${selected?.image_id===e.image_id?'active':''}" data-id="${esc(e.image_id)}" aria-label="Image ${esc(e.image_id)}, complexity ${e.mean_human_complexity_rating.toFixed(1)}, ${esc(countLabel(e))} marks"><div class="card-img"><img src="${esc(e.assets.thumbnail||e.assets.original)}" loading="lazy" alt="Visualization ${esc(e.image_id)}"></div><div class="card-caption"><div class="card-title"><strong>${esc(e.image_id)}</strong><span>${esc(countLabel(e))}</span></div><div class="card-meta"><span>Score ${e.mean_human_complexity_rating.toFixed(1)}</span><i class="review-dot ${quality(e.image_id)}" title="${quality(e.image_id)}"></i></div></div></button>`).join('');
    $('gallery').querySelectorAll('button').forEach(b=>b.onclick=()=>select(b.dataset.id));
    if (!entries.some(e=>e.image_id===selected?.image_id)) {
      if (entries.length) select((entries.find(e=>e.count>0)||entries[0]).image_id); else { selected=null;detail=null;$('detail-content').hidden=true;$('detail-empty').hidden=false; }
    }
    reviewTotal();
  }
  async function select(id) {
    selected=data.entries.find(e=>e.image_id===id); detail=null;instance=null;
    $('detail-content').hidden=false;$('detail-empty').hidden=true;
    $('gallery').querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.id===id));
    $('detail-title').textContent=`Visualization ${id}`;
    $('detail-filename').textContent=selected.filename;
    const categories={S:'SCIENCE',N:'NEWS',G:'GOVERNMENT',I:'INFOGRAPHIC'};
    $('detail-category').textContent=`${categories[selected.category]||selected.category} / COMPLEXITY ${selected.band*10}–${(selected.band+1)*10}`;
    $('detail-score').textContent=selected.mean_human_complexity_rating.toFixed(1);
    $('detail-count').textContent=countLabel(selected);
    $('detail-ambiguous').textContent=number(selected.ambiguous_count);
    $('detail-size').textContent=`${selected.width} × ${selected.height}`;
    $('detail-status').className=`status ${selected.status}`;
    $('detail-status').textContent=label(selected.status);
    $('json-link').href=selected.detail_json;
    $('instance-detail').hidden=true;
    $('quality-note').hidden=true;
    const saved=reviews[id]||{quality:'unreviewed',issues:[],notes:''};
    $('review-quality').value=saved.quality;
    $('review-notes').value=saved.notes;
    document.querySelectorAll('[name=issue]').forEach(b=>b.checked=saved.issues.includes(b.value));
    $('review-message').textContent='';
    updateImage();
    if (!window.CV_FIGURES[id]) {
      try { await new Promise((resolve,reject)=>{const s=document.createElement('script');s.src=selected.detail_script;s.onload=resolve;s.onerror=reject;document.head.append(s);}); }
      catch { if(selected?.image_id===id)$('image-hint').textContent='The detail data file could not be loaded. Regenerate results or check the static folder.';return; }
    }
    if(selected?.image_id!==id)return;
    detail=window.CV_FIGURES[id];
    $('quality-note').hidden=!detail.triage_notes;
    $('quality-note').textContent=detail.triage_notes||'';
    $('panel-list').innerHTML=detail.panels.map(p=>`<div class="panel-row"><div><strong>${esc(p.panel_id)} · ${esc(p.mark_families.map(label).join(', ')||'No discrete unit')}</strong><small>${esc(p.notes)}${p.preparation_status==='unresolved'?' · Preparation unresolved':''}</small></div><strong>${p.count===null?'N/A':p.count}${p.ambiguous_count?` <small>+ ${p.ambiguous_count} unresolved</small>`:''}</strong></div>`).join('')||'<p class="subtitle">No panel result available yet.</p>';
    renderInstances();
  }
  function updateImage() {
    if(!selected)return;
    $('main-image').src=selected.assets[mode]||selected.assets.original;
    $('main-image').alt=`${label(mode)} view of visualization ${selected.image_id}`;
    const native=$('size-mode').value==='native';
    $('image-stage').classList.toggle('native',native);
    if(native){$('image-stage').style.width=`${selected.width}px`;$('image-stage').style.height=`${selected.height}px`;}
    else {const maxHeight=620;const w=Math.min(selected.width,Math.max(200,$('image-viewport').clientWidth-34),maxHeight*selected.width/selected.height);$('image-stage').style.width=`${w}px`;$('image-stage').style.height='auto';}
    document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));
    $('image-hint').textContent=`${native?'Native resolution · 1 image pixel per CSS pixel. Scroll to inspect large figures.':'Fit view · scaled for browsing; use Native · 100% for the reference pixel size.'} Click a mark or instance ID to inspect.`;
    renderBoundaries();
  }
  function renderBoundaries() {
    const layer=$('instance-layer');layer.innerHTML='';if(!selected||!detail)return;
    layer.setAttribute('viewBox',`0 0 ${selected.width} ${selected.height}`);
    const state=$('instance-state').value;
    detail.instances.filter(c=>c.decision===state).forEach(c=>{const r=document.createElementNS('http://www.w3.org/2000/svg','rect');const b=c.bbox;r.setAttribute('x',b[0]);r.setAttribute('y',b[1]);r.setAttribute('width',b[2]-b[0]);r.setAttribute('height',b[3]-b[1]);r.classList.toggle('selected',instance===c.candidate_id);r.setAttribute('aria-label',`${c.candidate_id}: ${c.reason}`);r.onclick=()=>inspect(c.candidate_id);layer.append(r);});
  }
  function renderInstances() {
    if(!detail)return;const state=$('instance-state').value;const items=detail.instances.filter(c=>c.decision===state);
    $('instance-list').innerHTML=items.map(c=>`<button data-instance="${esc(c.candidate_id)}" class="${instance===c.candidate_id?'selected':''}" style="border-left-color:rgb(${c.color.join(',')})">${esc(c.candidate_id)}</button>`).join('')||'<small>No masks in this category.</small>';
    $('instance-list').querySelectorAll('button').forEach(b=>b.onclick=()=>inspect(b.dataset.instance));renderBoundaries();
  }
  function inspect(id) {
    instance=id;const c=detail.instances.find(c=>c.candidate_id===id);if(!c)return;
    const box=$('instance-detail');box.hidden=false;box.innerHTML=`<strong>${esc(id)} · ${esc(label(c.family||c.decision))}</strong><br>${esc(c.reason)}<br>Mask area ${number(c.area)} px · model quality ${c.score.toFixed(3)}<br>Native box [${c.bbox.join(', ')}] · ${esc(c.origin)}<br><small>Model quality is not a probability of human visibility.</small>`;
    const canvas=document.createElement('canvas'),[h,w]=c.rle.size;canvas.width=w;canvas.height=h;const ctx=canvas.getContext('2d');const img=new Image();
    // Use a separate alpha canvas: no source-pixel readback, so file:// remains supported.
    img.onload=()=>{ctx.drawImage(img,c.bbox[0],c.bbox[1],w,h,0,0,w,h);const alpha=document.createElement('canvas');alpha.width=w;alpha.height=h;const ac=alpha.getContext('2d');let cursor=0;ac.fillStyle='black';c.rle.counts.forEach((count,i)=>{if(i%2){let remaining=count,pos=cursor;while(remaining){const x=Math.floor(pos/h),y=pos%h,run=Math.min(remaining,h-y);ac.fillRect(x,y,1,run);pos+=run;remaining-=run;}}cursor+=count;});ctx.globalCompositeOperation='destination-in';ctx.drawImage(alpha,0,0);};img.src=selected.assets.original;canvas.style.maxWidth='160px';canvas.style.maxHeight='130px';canvas.style.width='auto';canvas.style.height='auto';canvas.style.marginTop='8px';box.append(canvas);renderInstances();
  }
  function saveReview() {
    if(!selected)return;reviews[selected.image_id]={quality:$('review-quality').value,issues:[...document.querySelectorAll('[name=issue]:checked')].map(b=>b.value),notes:$('review-notes').value,updated_at:new Date().toISOString()};
    try{localStorage.setItem(key,JSON.stringify(reviews));$('save-state').textContent='Saved in this browser for this run.';}catch{storageAvailable=false;$('save-state').textContent='Browser storage unavailable. Export reviews to keep them.';}
    reviewTotal();$('gallery').querySelectorAll('.card').forEach(c=>{const dot=c.querySelector('.review-dot');dot.className=`review-dot ${quality(c.dataset.id)}`;dot.title=quality(c.dataset.id);});
  }
  function download(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  $('export-reviews').onclick=()=>download({schema_version:1,run_id:data.run_id,selection_id:data.selection_id,exported_at:new Date().toISOString(),reviews},`mark-atlas-reviews-${data.run_id.slice(0,12)}.json`);
  $('import-reviews').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;const imported=JSON.parse(await file.text());if(imported.run_id!==data.run_id||imported.selection_id!==data.selection_id)throw Error('This review file belongs to a different run.');if(!imported.reviews||typeof imported.reviews!=='object'||Array.isArray(imported.reviews))throw Error('Invalid review file.');for(const[id,r]of Object.entries(imported.reviews)){if(!data.entries.some(x=>x.image_id===id)||!['unreviewed','good','mixed','poor'].includes(r.quality)||!Array.isArray(r.issues)||r.issues.some(x=>!['missed','merged','split','decoration'].includes(x))||typeof r.notes!=='string')throw Error('Invalid review entry.');}reviews={...reviews,...imported.reviews};try{localStorage.setItem(key,JSON.stringify(reviews));}catch{storageAvailable=false;}const id=selected?.image_id;renderGallery();if(id&&selected?.image_id===id)select(id);$('review-message').textContent=`Imported ${Object.keys(imported.reviews).length} reviews.`;}catch(err){$('review-message').textContent=err.message;}finally{e.target.value='';}};
  ['search','family','status','review-filter'].forEach(id=>$(id).addEventListener(id==='search'?'input':'change',renderGallery));
  ['review-quality','review-notes'].forEach(id=>$(id).addEventListener(id==='review-notes'?'input':'change',saveReview));document.querySelectorAll('[name=issue]').forEach(b=>b.onchange=saveReview);
  document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;updateImage();});$('size-mode').onchange=updateImage;$('instance-state').onchange=()=>{instance=null;$('instance-detail').hidden=true;renderInstances();};window.addEventListener('resize',updateImage);
  if(!storageAvailable)$('save-state').textContent='Browser storage unavailable. Export reviews to keep them.';
  renderBands();renderGallery();
})();
