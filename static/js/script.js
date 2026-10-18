(() => {
  const form = document.getElementById('upload-form');
  if (!form) return; // only runs on the dashboard

  const $ = (id) => document.getElementById(id);
  const ACCEPT = {
    image: '.jpg,.jpeg,.png,.webp',
    video: '.mp4,.avi,.mov,.mkv,.webm',
    audio: '.wav,.mp3,.flac,.ogg,.m4a',
  };
  const LOADING = {
    image: 'Cropping the face and running the CNN...',
    video: 'Sampling frames and scoring each one...',
    audio: 'Extracting MFCC features and classifying...',
  };
  const ICON = { image: 'fa-image', video: 'fa-video', audio: 'fa-music' };

  let mediaType = 'image';
  let file = null;
  let previewUrl = null;

  const fileInput = $('file-input'), dropZone = $('drop-zone'), analyzeBtn = $('analyze-btn');

  function show(el, on = true) { el.classList.toggle('hidden', !on); }
  function setText(id, value) { $(id).textContent = value; }

  // ------------------------------------------------ tabs
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      mediaType = btn.dataset.type;
      $('media-type').value = mediaType;
      fileInput.accept = ACCEPT[mediaType];
      resetSelection();
    });
  });

  function resetSelection() {
    file = null;
    fileInput.value = '';
    setText('file-info', 'No file selected.');
    analyzeBtn.disabled = true;
    show($('media-preview-container'), false);
    show($('result-container'), false);
    show($('empty-result'), true);
    show($('error-box'), false);
    if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
  }

  // ------------------------------------------------ file selection
  function setFile(f) {
    if (!f) return;
    file = f;
    setText('file-info', `${f.name} (${(f.size / 1048576).toFixed(1)} MB)`);
    analyzeBtn.disabled = false;
    show($('error-box'), false);
    show($('result-container'), false);
    show($('empty-result'), true);
    renderPreview(f);
  }

  function renderPreview(f) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(f);
    const holder = $('media-preview');
    holder.replaceChildren();
    const tag = mediaType === 'image' ? 'img' : mediaType === 'video' ? 'video' : 'audio';
    const el = document.createElement(tag);
    el.src = previewUrl;
    if (tag !== 'img') el.controls = true;
    holder.appendChild(el);
    show($('media-preview-container'), true);
  }

  fileInput.addEventListener('change', () => setFile(fileInput.files[0]));
  ['dragenter', 'dragover'].forEach((ev) => dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); }));
  ['dragleave', 'drop'].forEach((ev) => dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); }));
  dropZone.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files[0];
    if (f) { fileInput.value = ''; setFile(f); }
  });

  // ------------------------------------------------ submit
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!file) return;
    const body = new FormData();
    body.append('media_type', mediaType);
    body.append('file', file);

    analyzeBtn.disabled = true;
    setText('loading-text', LOADING[mediaType]);
    show($('loading-state'), true);
    show($('error-box'), false);
    show($('result-container'), false);

    try {
      const res = await fetch('/upload', { method: 'POST', body });
      let data = {};
      try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
      if (res.status === 401) { window.location.href = '/login'; return; }
      if (!res.ok || data.error) throw new Error(data.error || `Server error (${res.status})`);
      renderResult(data.result);
      loadHistory();
    } catch (err) {
      const box = $('error-box');
      box.textContent = err.message;
      show(box, true);
    } finally {
      show($('loading-state'), false);
      analyzeBtn.disabled = !file;
    }
  });

  // ------------------------------------------------ results
  function renderResult(r) {
    show($('empty-result'), false);
    show($('result-container'), true);

    const cls = r.label === 'FAKE' ? 'fake' : 'real';
    const label = $('result-label');
    label.textContent = r.label;
    label.className = `result-stamp ${cls}`;
    setText('media-type-badge', mediaType.toUpperCase());
    setText('processing-time', new Date().toLocaleTimeString());
    show($('low-conf'), !!r.low_confidence);

    const bar = $('confidence-bar');
    bar.className = `confidence-bar-fill ${cls}`;
    bar.style.width = '0%';
    requestAnimationFrame(() => { bar.style.width = `${r.confidence}%`; });
    setText('confidence-text', `${r.confidence}%  (P fake = ${r.fake_probability}%)`);

    setText('result-details', r.details || '');

    // video
    const isVideo = mediaType === 'video';
    show($('temporal-analysis'), isVideo);
    if (isVideo) {
      setText('frames-analyzed', `${r.frames_analyzed} (${r.faces_found} with a face)`);
      setText('temporal-score', `${r.temporal_score}%`);
    }
    renderFrames(isVideo ? r.frame_results : null);

    // audio
    const isAudio = mediaType === 'audio';
    show($('acoustic-analysis'), isAudio);
    if (isAudio) {
      setText('acoustic-score', `${r.acoustic_score}%`);
      setText('features-used', (r.features_used || []).length + ' groups');
      const img = $('spectrogram');
      if (r.spectrogram) { img.src = r.spectrogram; show(img, true); } else { show(img, false); }
    }
    renderFeatures(isAudio ? r.feature_analysis : null);

    // preprocessing
    const pre = r.preprocessing;
    show($('preprocessing-info'), !!pre);
    if (pre) {
      setText('enhancement-status', pre.enhancement);
      setText('preprocessing-time', pre.processing_time);
    }
  }

  function renderFrames(frames) {
    const wrap = $('frame-analysis'), grid = $('frame-breakdown');
    grid.replaceChildren();
    show(wrap, !!(frames && frames.length));
    (frames || []).forEach((f) => {
      const d = document.createElement('div');
      d.className = `frame-item ${f.label === 'FAKE' ? 'fake' : 'real'}`;
      const top = document.createElement('em');
      top.textContent = f.time != null ? `${f.time}s` : `#${f.frame}`;
      const strong = document.createElement('strong');
      strong.textContent = `${f.label} ${f.fake_probability}%`;
      const foot = document.createElement('em');
      foot.textContent = f.face_found ? 'face found' : 'no face';
      d.append(top, strong, foot);
      grid.appendChild(d);
    });
  }

  function renderFeatures(obj) {
    const wrap = $('feature-analysis'), grid = $('feature-breakdown');
    grid.replaceChildren();
    const entries = obj ? Object.entries(obj) : [];
    show(wrap, entries.length > 0);
    entries.forEach(([k, v]) => {
      const d = document.createElement('div');
      d.className = 'feature-item';
      const s = document.createElement('span');
      s.textContent = k;
      d.append(s, document.createTextNode(v));
      grid.appendChild(d);
    });
  }

  // ------------------------------------------------ history
  async function loadHistory() {
    const list = $('history-list');
    try {
      const res = await fetch('/api/history');
      if (!res.ok) throw new Error();
      const rows = await res.json();
      list.replaceChildren();
      if (!rows.length) {
        const p = document.createElement('p');
        p.className = 'text-sm text-center mt-3';
        p.textContent = 'No recent scans found.';
        list.appendChild(p);
        return;
      }
      rows.forEach((r) => {
        const item = document.createElement('div');
        item.className = 'history-item';

        const icon = document.createElement('div');
        icon.className = 'history-icon';
        const i = document.createElement('i');
        i.className = `fa-solid ${ICON[r.media_type] || 'fa-file'}`;
        icon.appendChild(i);

        const details = document.createElement('div');
        details.className = 'history-details';
        const name = document.createElement('p');
        name.className = 'history-filename';
        name.textContent = r.original_filename || '';
        const date = document.createElement('p');
        date.className = 'history-date';
        date.textContent = r.created_at ? new Date(r.created_at).toLocaleString() : '';
        details.append(name, date);

        const badge = document.createElement('div');
        badge.className = `history-badge ${r.prediction === 'REAL' ? 'badge-real' : 'badge-fake'}`;
        badge.textContent = `${r.prediction} (${Number(r.confidence).toFixed(1)}%)`;

        item.append(icon, details, badge);
        list.appendChild(item);
      });
    } catch (_) {
      list.textContent = 'Could not load history.';
    }
  }

  loadHistory();
})();
