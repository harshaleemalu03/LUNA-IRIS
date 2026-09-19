/* ============================================================
   API SERVICE LAYER
   Swap REGISTRATION_ENDPOINT / register() internals to go live.
   The rest of the UI only ever talks to RegistrationAPI.register().
   ============================================================ */
/* ============================================================
   API SERVICE LAYER
   Real Luna-tics backend integration.
   ============================================================ */

const RegistrationAPI = (() => {

  const REGISTRATION_ENDPOINT =
    'http://127.0.0.1:8000/api/register';


  async function register({ source, reference, sensor }, onStep) {

    const body = new FormData();

    // Files expected by FastAPI
    body.append('source', source);
    body.append('reference', reference);

    // Sensor selected from the frontend
    body.append('sensor', sensor);


    // Show initial processing step
    onStep && onStep(0);


    try {

      const response = await fetch(
        REGISTRATION_ENDPOINT,
        {
          method: 'POST',
          body: body
        }
      );


      // Backend returned an error
      if (!response.ok) {

        let message = 'Registration failed';

        try {
          const error = await response.json();
          message = error.detail || message;
        } catch (_) {
          // Response was not JSON
        }

        throw new Error(message);
      }


      // Tell UI that backend processing completed
      onStep && onStep(5);


      const result = await response.json();

      onStep && onStep(6);

      return result;

    } catch (error) {

      console.error(
        'Luna-tics registration failed:',
        error
      );

      // IMPORTANT:
      // Do NOT fall back to fake/demo results.
      throw error;
    }
  }


  return {
    register
  };

})();

/* ============================================================
   NAV
   ============================================================ */
const nav = document.getElementById('nav');
window.addEventListener('scroll', () => { nav.classList.toggle('scrolled', window.scrollY > 40); }, { passive: true });
const navToggle = document.getElementById('navToggle');
const mobileMenu = document.getElementById('mobileMenu');
const menuClose = document.getElementById('menuClose');
function openMenu(){ mobileMenu.classList.add('open'); navToggle.setAttribute('aria-expanded','true'); }
function closeMenu(){ mobileMenu.classList.remove('open'); navToggle.setAttribute('aria-expanded','false'); }
navToggle.addEventListener('click', openMenu);
menuClose.addEventListener('click', closeMenu);
mobileMenu.querySelectorAll('a').forEach(a => a.addEventListener('click', closeMenu));

/* ============================================================
   CUSTOM CURSOR
   ============================================================ */
const cursorDot = document.querySelector('.cursor-dot');
const cursorRing = document.querySelector('.cursor-ring');
if (window.matchMedia('(hover:hover)').matches) {
  let mx = 0, my = 0, rx = 0, ry = 0;
  window.addEventListener('mousemove', e => { mx = e.clientX; my = e.clientY; cursorDot.style.left = mx+'px'; cursorDot.style.top = my+'px'; });
  function loop(){ rx += (mx-rx)*0.18; ry += (my-ry)*0.18; cursorRing.style.left = rx+'px'; cursorRing.style.top = ry+'px'; requestAnimationFrame(loop); }
  loop();
  function bindCursor(sel, cls){ document.querySelectorAll(sel).forEach(el => {
    el.addEventListener('mouseenter', () => cursorRing.classList.add(cls));
    el.addEventListener('mouseleave', () => cursorRing.classList.remove(cls));
  });}
  bindCursor('a, button, .upload-card, input[type=file]', 'active');
  bindCursor('.compare-handle', 'data');
}

/* ============================================================
   SCROLL REVEALS
   ============================================================ */
const io = new IntersectionObserver((entries) => {
  entries.forEach(en => { if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); } });
}, { threshold: 0.15 });
document.querySelectorAll('.reveal').forEach(el => io.observe(el));

if (window.gsap && window.ScrollTrigger) {
  gsap.registerPlugin(ScrollTrigger);
  gsap.to('.orbit-stage', { y: 40, ease: 'none', scrollTrigger: { trigger: '.hero', start: 'top top', end: 'bottom top', scrub: true } });
  gsap.fromTo('.hero-grid', { opacity: 0, y: 24 }, { opacity: 1, y: 0, duration: 1.1, ease: 'power2.out', delay: 0.15 });
}

document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const id = a.getAttribute('href');
    const target = document.querySelector(id);
    if (target) { e.preventDefault(); target.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  });
});
document.getElementById('heroCta').addEventListener('click', () => document.getElementById('register').scrollIntoView({ behavior: 'smooth' }));
document.getElementById('emptyBeginBtn').addEventListener('click', () => document.getElementById('fileSource').click());

/* ============================================================
   COMPARE SLIDER (reusable)
   ============================================================ */
function initCompareSlider(stageId, handleId) {
  const stage = document.getElementById(stageId);
  const handle = document.getElementById(handleId);
  const top = stage.querySelector('.layer-top');
  let dragging = false;

  function setPct(pct) {
    pct = Math.max(0, Math.min(100, pct));
    top.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    handle.style.left = pct + '%';
  }
  function fromEvent(e) {
    const rect = stage.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    setPct((x / rect.width) * 100);
  }
  handle.addEventListener('mousedown', () => dragging = true);
  handle.addEventListener('touchstart', () => dragging = true, { passive: true });
  window.addEventListener('mouseup', () => dragging = false);
  window.addEventListener('touchend', () => dragging = false);
  window.addEventListener('mousemove', e => { if (dragging) fromEvent(e); });
  window.addEventListener('touchmove', e => { if (dragging) fromEvent(e); }, { passive: true });
  stage.addEventListener('click', fromEvent);
  setPct(50);
  return { setPct };
}

const inputCompareCtl =
  initCompareSlider('inputCompareStage', 'inputCompareHandle');

/* ============================================================
   UPLOAD CARDS
   ============================================================ */
const state = { source: null, reference: null, sensor: null };

/* ============================================================
   SENSOR SELECT (custom dropdown)
   ============================================================ */
const sensorSelect = document.getElementById('sensorSelect');
const sensorTrigger = document.getElementById('sensorTrigger');
const sensorTriggerLabel = document.getElementById('sensorTriggerLabel');
const sensorMenu = document.getElementById('sensorMenu');
const sensorOptions = Array.from(sensorMenu.querySelectorAll('.sensor-option'));

function toggleSensor(force) {
  const open = force !== undefined ? force : !sensorSelect.classList.contains('open');
  sensorSelect.classList.toggle('open', open);
  sensorTrigger.setAttribute('aria-expanded', String(open));
}
sensorTrigger.addEventListener('click', () => toggleSensor());
document.addEventListener('click', e => { if (!sensorSelect.contains(e.target)) toggleSensor(false); });
sensorOptions.forEach(opt => {
  opt.addEventListener('click', () => {
    sensorOptions.forEach(o => o.classList.remove('selected'));
    opt.classList.add('selected');
    sensorTriggerLabel.textContent = opt.dataset.value;
    sensorTriggerLabel.classList.remove('sensor-placeholder');
    state.sensor = opt.dataset.value;
    toggleSensor(false);
    updateSummary();
  });
});
sensorTrigger.addEventListener('keydown', e => {
  if (e.key === 'ArrowDown') { e.preventDefault(); toggleSensor(true); sensorOptions[0].focus(); }
});
sensorOptions.forEach((opt, i) => {
  opt.addEventListener('keydown', e => {
    if (e.key === 'ArrowDown') { e.preventDefault(); (sensorOptions[i+1]||sensorOptions[0]).focus(); }
    if (e.key === 'ArrowUp') { e.preventDefault(); (sensorOptions[i-1]||sensorOptions[sensorOptions.length-1]).focus(); }
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); opt.click(); }
    if (e.key === 'Escape') { toggleSensor(false); sensorTrigger.focus(); }
  });
});

function setupUploadCard(cardId, inputId, key, roleLabel) {
  const card = document.getElementById(cardId);
  const input = document.getElementById(inputId);
  card.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  input.addEventListener('change', () => { if (input.files[0]) handleFile(input.files[0]); });
  ['dragenter','dragover'].forEach(evt => card.addEventListener(evt, e => { e.preventDefault(); card.classList.add('drag'); }));
  ['dragleave','drop'].forEach(evt => card.addEventListener(evt, e => { e.preventDefault(); card.classList.remove('drag'); }));
  card.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) handleFile(f); });

  function handleFile(file) {
    if (!file.type.startsWith('image/')) { flashInvalid(); return; }
    if (file.size > 25 * 1024 * 1024) { flashInvalid(); return; }
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      state[key] = { file, url, w: img.naturalWidth, h: img.naturalHeight };
      renderFilled(card, file, url, img.naturalWidth, img.naturalHeight, key, cardId, inputId, roleLabel);
      updateSummary();
      updateInputCompare();
    };
    img.src = url;
  }
  function flashInvalid() { card.style.borderColor = 'var(--err)'; setTimeout(() => card.style.borderColor = '', 500); }
}

function renderFilled(card, file, url, w, h, key, cardId, inputId, roleLabel) {
  card.classList.add('filled');
  const idxLabel = key === 'source' ? 'SOURCE' : 'REFERENCE';
  card.innerHTML = `
    <input type="file" id="${inputId}" accept="image/*" aria-label="Upload ${idxLabel}">
    <img class="preview-img" src="${url}" alt="Preview of ${idxLabel}">
    <div class="top-row">
      <span class="idx">${idxLabel}</span>
      <span class="status-chip">READY</span>
    </div>
    <div class="preview-overlay">
      <div class="file-meta">
        <div class="name">${file.name}</div>
        <div class="dims">${w} × ${h} · ${(file.size/1024/1024).toFixed(1)}MB</div>
      </div>
      <div class="file-actions">
        <button type="button" class="replace-btn" aria-label="Replace ${idxLabel}" title="Replace">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M1 7a6 6 0 0110-4.2M13 7a6 6 0 01-10 4.2" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/><path d="M11 1v2.8h-2.8M3 13v-2.8h2.8" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </button>
        <button type="button" class="remove-btn" aria-label="Remove ${idxLabel}" title="Remove">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><line x1="2" y1="2" x2="12" y2="12" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/><line x1="12" y1="2" x2="2" y2="12" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>
        </button>
      </div>
    </div>`;
  const newInput = card.querySelector('input[type=file]');
  newInput.addEventListener('change', () => { if (newInput.files[0]) reuploadHandler(newInput.files[0]); });
  card.querySelector('.replace-btn').addEventListener('click', (e) => { e.preventDefault(); newInput.click(); });
  card.querySelector('.remove-btn').addEventListener('click', (e) => { e.preventDefault(); resetCard(); });

  function reuploadHandler(file) {
    const url2 = URL.createObjectURL(file);
    const img2 = new Image();
    img2.onload = () => {
      state[key] = { file, url: url2, w: img2.naturalWidth, h: img2.naturalHeight };
      renderFilled(card, file, url2, img2.naturalWidth, img2.naturalHeight, key, cardId, inputId, roleLabel);
      updateSummary();
      updateInputCompare();
    };
    img2.src = url2;
  }
  function resetCard() {
    state[key] = null;
    card.classList.remove('filled');
    card.innerHTML = originalCardHTML(idxLabel, inputId, roleLabel, key);
    setupUploadCard(cardId, inputId, key, roleLabel);
    updateSummary();
    updateInputCompare();
  }
}

function originalCardHTML(idxLabel, inputId, roleLabel, key) {
  return `
    <input type="file" id="${inputId}" accept="image/png,image/jpeg,image/tiff,image/webp" aria-label="Upload ${idxLabel}">
    <div class="top-row"><span class="idx">${idxLabel}</span><span class="role-tag">${roleLabel}</span></div>
    <div class="center">
      <svg viewBox="0 0 40 40" fill="none"><path d="M20 26V10M20 10l-7 7M20 10l7 7" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><line x1="9" y1="32" x2="31" y2="32" stroke-width="1.6" stroke-linecap="round"/></svg>
      <div class="drop-label">Drop ${key} image</div>
      <div class="hint">or click to browse</div>
    </div>
    <div class="bottom-row"><span class="fmt">JPG · PNG · TIFF — MAX 25MB</span></div>`;
}

setupUploadCard('cardSource', 'fileSource', 'source', 'MOVING');
setupUploadCard('cardRef', 'fileRef', 'reference', 'FIXED');

function updateInputCompare() {
  const block = document.getElementById('inputCompare');
  if (state.source && state.reference) {
    document.getElementById('cmpTopImg').src = state.source.url;
    document.getElementById('cmpBottomImg').src = state.reference.url;
    block.classList.add('active');
  } else {
    block.classList.remove('active');
  }
}

/* ============================================================
   SUMMARY / READINESS
   ============================================================ */
const sumSource = document.getElementById('sumSource');
const sumRef = document.getElementById('sumRef');
const sumSensor = document.getElementById('sumSensor');
const readyStatus = document.getElementById('readyStatus');
const readyText = document.getElementById('readyText');
const runBtn = document.getElementById('runBtn');

function updateSummary() {
  sumSource.textContent = state.source ? 'Uploaded' : 'Not uploaded';
  sumSource.classList.toggle('pending', !state.source);
  sumRef.textContent = state.reference ? 'Uploaded' : 'Not uploaded';
  sumRef.classList.toggle('pending', !state.reference);
  sumSensor.textContent = state.sensor || 'Not selected';
  sumSensor.classList.toggle('pending', !state.sensor);
  const ready = !!state.source && !!state.reference && !!state.sensor;
  readyStatus.classList.toggle('ready', ready);
  readyText.textContent = ready ? 'READY FOR REGISTRATION' : 'AWAITING INPUT';
  runBtn.disabled = !ready;
}
updateSummary();

/* ============================================================
   MATCH-POINT CANVAS VISUALIZATION
   ============================================================ */
function drawMatchPoints(canvas, srcImgEl, refImgEl, opts = {}) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const gap = 6;
  const halfW = (w - gap) / 2;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#08090c';
  ctx.fillRect(0, 0, w, h);

  function drawCover(img, x, targetW, targetH) {
    if (!img || !img.naturalWidth) return;
    const ir = img.naturalWidth / img.naturalHeight;
    const tr = targetW / targetH;
    let sx, sy, sw, sh;
    if (ir > tr) { sh = img.naturalHeight; sw = sh * tr; sx = (img.naturalWidth - sw) / 2; sy = 0; }
    else { sw = img.naturalWidth; sh = sw / tr; sx = 0; sy = (img.naturalHeight - sh) / 2; }
    ctx.drawImage(img, sx, sy, sw, sh, x, 0, targetW, targetH);
  }
  drawCover(srcImgEl, 0, halfW, h);
  drawCover(refImgEl, halfW + gap, halfW, h);
  ctx.fillStyle = 'rgba(8,9,12,0.28)';
  ctx.fillRect(0, 0, halfW, h);
  ctx.fillRect(halfW + gap, 0, halfW, h);

  const count = opts.count || 36;
  const progress = opts.progress != null ? opts.progress : 1;
  const inlierRatio = opts.inlierRatio != null ? opts.inlierRatio / 100 : 0.9;
  const visible = Math.round(count * progress);

  const pts = canvas._matchPts || (canvas._matchPts = Array.from({ length: count }, () => {
    const y = 0.1 + Math.random() * 0.8;
    const xs = 0.15 + Math.random() * 0.7;
    const jitter = (Math.random() - 0.5) * 0.06;
    return { xs, y, xr: Math.min(0.95, Math.max(0.05, xs + jitter)), yr: Math.min(0.95, Math.max(0.05, y + jitter * 0.6)) };
  }));

  for (let i = 0; i < visible; i++) {
    const p = pts[i];
    const isInlier = (i / count) < inlierRatio;
    const color = isInlier ? '54, 214, 201' : '201, 99, 74';
    const x1 = p.xs * halfW;
    const y1 = p.y * h;
    const x2 = halfW + gap + p.xr * halfW;
    const y2 = p.yr * h;

    ctx.strokeStyle = `rgba(${color}, 0.55)`;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();

    ctx.fillStyle = `rgba(${color}, 0.95)`;
    ctx.beginPath(); ctx.arc(x1, y1, 2.6, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(x2, y2, 2.6, 0, Math.PI * 2); ctx.fill();
  }

  ctx.strokeStyle = 'rgba(240,238,232,0.12)';
  ctx.beginPath(); ctx.moveTo(halfW + gap/2, 0); ctx.lineTo(halfW + gap/2, h); ctx.stroke();
}

function animateMatchPoints(canvas, srcImgEl, refImgEl, inlierRatio, duration = 1600) {
  canvas._matchPts = null;
  const start = performance.now();
  function step(now) {
    const p = Math.min(1, (now - start) / duration);
    drawMatchPoints(canvas, srcImgEl, refImgEl, { progress: p, inlierRatio });
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ============================================================
   ANALYSIS FLOW (state machine)
   ============================================================ */
const form = document.getElementById('registerForm');
const processingBlock = document.getElementById('processingBlock');
const emptyState = document.getElementById('emptyState');
const errorState = document.getElementById('errorState');
const resultsSection = document.getElementById('results');
const procStepsEls = Array.from(document.querySelectorAll('.proc-step'));
const procCanvas = document.getElementById('procCanvas');

function setProcStep(i) {
  procStepsEls.forEach((el, idx) => {
    el.classList.toggle('active', idx === i);
    el.classList.toggle('done', idx < i);
  });
  if (i === 1) {
    const srcImg = new Image(); srcImg.src = state.source.url;
    const refImg = new Image(); refImg.src = state.reference.url;
    srcImg.onload = refImg.onload = () => { if (srcImg.complete && refImg.complete) animateMatchPoints(procCanvas, srcImg, refImg, 90, 1400); };
  }
}

let isProcessing = false;
async function runRegistration() {
  if (isProcessing) return;
  isProcessing = true;
  runBtn.disabled = true;
  runBtn.textContent = 'INITIALIZING…';
  emptyState.classList.remove('active');
  errorState.classList.remove('active');
  resultsSection.classList.remove('active');
  processingBlock.classList.add('active');
  processingBlock.scrollIntoView({ behavior: 'smooth', block: 'center' });
  setProcStep(-1);
  drawMatchPoints(procCanvas, null, null, { progress: 0 });

  try {
    setTimeout(() => { runBtn.textContent = 'PROCESSING…'; }, 300);
    const response = await RegistrationAPI.register(
      { source: state.source.file, reference: state.reference.file, sensor: state.sensor },
      (step) => setProcStep(step)
    );
    setProcStep(6);
    await new Promise(r => setTimeout(r, 400));
    processingBlock.classList.remove('active');
    renderResults(response);
    resultsSection.classList.add('active');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    processingBlock.classList.remove('active');
    errorState.classList.add('active');
    errorState.scrollIntoView({ behavior: 'smooth', block: 'center' });
  } finally {
    isProcessing = false;
    runBtn.disabled = false;
    runBtn.innerHTML = 'RUN REGISTRATION <span class="arrow">→</span>';
  }
}
form.addEventListener('submit', e => { e.preventDefault(); runRegistration(); });
document.getElementById('retryBtn').addEventListener('click', () => { errorState.classList.remove('active'); runRegistration(); });

/* ============================================================
   RESULTS RENDERING
   ============================================================ */
function renderResults(res) {
  console.log("BACKEND RESPONSE:", res);

  const outputImg = document.getElementById('outputImg');

  if (!outputImg) {
    console.error('outputImg element not found');
    return;
  }

  if (!res.output_image) {
    console.error('No output_image in backend response');
    return;
  }

  const imageUrl =
    'http://127.0.0.1:8000' + res.output_image;

  console.log("MATCHES IMAGE URL:", imageUrl);

  outputImg.onload = () => {
    console.log(
      'MATCHES IMAGE LOADED:',
      outputImg.naturalWidth,
      'x',
      outputImg.naturalHeight
    );
  };

  outputImg.onerror = () => {
    console.error(
      'MATCHES IMAGE FAILED TO LOAD:',
      imageUrl
    );
  };

  outputImg.src = imageUrl;
}


/* initial idle canvas frame */
drawMatchPoints(procCanvas, null, null, { progress: 0 });
