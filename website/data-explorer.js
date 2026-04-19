/* ═══════════════════════════════════════════════════════════
   OASIS DATA EXPLORER — JavaScript (Globe + Charts + Modal)
   ═══════════════════════════════════════════════════════════ */

let DATA = null;
let ndviChartInstance = null;
let bandChartInstance = null;
const FONT = "'Inter', sans-serif";

/* ── Boot ─────────────────────────────────────────────────── */
async function init() {
  const resp = await fetch('data-assets/data_stats.json');
  DATA = await resp.json();
  populateSummary();
  initGlobe();
  populateTileGrid();
  initFilters();
  initModal();
  initTimelapse();
  initDeforestChart();
}
document.addEventListener('DOMContentLoaded', init);

/* ── Summary cards ────────────────────────────────────────── */
function populateSummary() {
  const s = DATA.summary;
  el('statTrainTiles').textContent = s.total_train;
  el('statTestTiles').textContent  = s.total_test;
  el('statBands').textContent      = s.s2_bands;
  el('statMonths').textContent     = s.months;
  el('headerStats').textContent    = `${s.total_train + s.total_test} tiles · ${s.s2_bands} bands · ${s.months} months`;
}

/* ═══════════════════════════════════════════════════════════
   THREE.JS GLOBE
   ═══════════════════════════════════════════════════════════ */
function initGlobe() {
  const wrap = el('globeWrap');
  const canvas = el('globeCanvas');
  const tooltip = el('globeTooltip');
  const W = wrap.clientWidth;
  const H = wrap.clientHeight;

  // Scene
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 100);
  camera.position.set(0, 0, 4.5);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  // Lighting
  const ambient = new THREE.AmbientLight(0xffffff, 0.25);
  scene.add(ambient);
  const sun = new THREE.DirectionalLight(0xffffff, 1.1);
  sun.position.set(5, 3, 5);
  scene.add(sun);

  // Earth group
  const earthGroup = new THREE.Group();
  scene.add(earthGroup);

  const RADIUS = 1.6;
  const earthGeom = new THREE.SphereGeometry(RADIUS, 64, 64);
  const textureLoader = new THREE.TextureLoader();

  const earthMat = new THREE.MeshPhongMaterial({
    color: 0xffffff,
    shininess: 20,
    specular: new THREE.Color(0x222222),
  });

  textureLoader.load('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg', tex => {
    earthMat.map = tex; earthMat.needsUpdate = true;
  });
  textureLoader.load('https://unpkg.com/three-globe/example/img/earth-topology.png', tex => {
    earthMat.bumpMap = tex; earthMat.bumpScale = 0.015; earthMat.needsUpdate = true;
  });

  const earth = new THREE.Mesh(earthGeom, earthMat);
  earthGroup.add(earth);

  // Atmosphere glow
  const atmosGeom = new THREE.SphereGeometry(RADIUS * 1.06, 64, 64);
  const atmosMat = new THREE.ShaderMaterial({
    vertexShader: `
      varying vec3 vNormal;
      void main() {
        vNormal = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: `
      varying vec3 vNormal;
      void main() {
        float intensity = pow(0.62 - dot(vNormal, vec3(0,0,1)), 3.0);
        vec3 col = mix(vec3(0.06,0.72,0.5), vec3(0.02,0.71,0.83), intensity);
        gl_FragColor = vec4(col, intensity * 0.5);
      }`,
    blending: THREE.AdditiveBlending,
    side: THREE.BackSide,
    transparent: true,
    depthWrite: false
  });
  earthGroup.add(new THREE.Mesh(atmosGeom, atmosMat));

  // Stars
  const starVerts = [];
  for (let i = 0; i < 3000; i++) {
    const r = 20 + Math.random() * 40;
    const th = Math.random() * Math.PI * 2;
    const ph = Math.acos(2 * Math.random() - 1);
    starVerts.push(r * Math.sin(ph) * Math.cos(th), r * Math.sin(ph) * Math.sin(th), r * Math.cos(ph));
  }
  const starGeom = new THREE.BufferGeometry();
  starGeom.setAttribute('position', new THREE.Float32BufferAttribute(starVerts, 3));
  const starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.08, sizeAttenuation: true });
  scene.add(new THREE.Points(starGeom, starMat));

  // ── Tile markers ──
  const markers = [];
  const markerGroup = new THREE.Group();
  earthGroup.add(markerGroup);

  function latLonToVec3(lat, lon, r) {
    const phi = (90 - lat) * Math.PI / 180;
    const theta = (lon + 180) * Math.PI / 180;
    return new THREE.Vector3(
      -r * Math.sin(phi) * Math.cos(theta),
       r * Math.cos(phi),
       r * Math.sin(phi) * Math.sin(theta)
    );
  }

  function addMarker(id, info, color) {
    const pos = latLonToVec3(info.lat, info.lon, RADIUS + 0.01);

    // Main dot
    const dotGeom = new THREE.SphereGeometry(0.025, 12, 12);
    const dotMat = new THREE.MeshBasicMaterial({ color });
    const dot = new THREE.Mesh(dotGeom, dotMat);
    dot.position.copy(pos);
    markerGroup.add(dot);

    // Glow ring
    const ringGeom = new THREE.RingGeometry(0.03, 0.05, 24);
    const ringMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.4, side: THREE.DoubleSide });
    const ring = new THREE.Mesh(ringGeom, ringMat);
    ring.position.copy(pos);
    ring.lookAt(0, 0, 0);
    markerGroup.add(ring);

    // Vertical beam
    const normal = pos.clone().normalize();
    const beamGeom = new THREE.CylinderGeometry(0.003, 0.003, 0.12, 4);
    const beamMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.5 });
    const beam = new THREE.Mesh(beamGeom, beamMat);
    const beamPos = pos.clone().add(normal.clone().multiplyScalar(0.06));
    beam.position.copy(beamPos);
    // Orient beam along the normal
    beam.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), normal);
    markerGroup.add(beam);

    markers.push({ id, info, mesh: dot, ring, beam, pos, color });
  }

  for (const [id, info] of Object.entries(DATA.train)) addMarker(id, info, 0x22c55e);
  for (const [id, info] of Object.entries(DATA.test))  addMarker(id, info, 0xf59e0b);

  // ── Interaction: drag to rotate ──
  let isDragging = false;
  let prevMouse = { x: 0, y: 0 };
  let rotVel = { x: 0, y: 0 };
  let autoRotate = true;

  canvas.addEventListener('pointerdown', e => {
    isDragging = true;
    autoRotate = false;
    prevMouse = { x: e.clientX, y: e.clientY };
    canvas.setPointerCapture(e.pointerId);
  });

  canvas.addEventListener('pointermove', e => {
    if (isDragging) {
      const dx = e.clientX - prevMouse.x;
      const dy = e.clientY - prevMouse.y;
      rotVel.x = dy * 0.003;
      rotVel.y = dx * 0.003;
      earthGroup.rotation.x += rotVel.x;
      earthGroup.rotation.y += rotVel.y;
      prevMouse = { x: e.clientX, y: e.clientY };
    }

    // Tooltip on hover
    updateTooltip(e);
  });

  canvas.addEventListener('pointerup', () => {
    isDragging = false;
    setTimeout(() => { autoRotate = true; }, 3000);
  });

  // Scroll to zoom
  wrap.addEventListener('wheel', e => {
    e.preventDefault();
    camera.position.z = Math.max(2.5, Math.min(8, camera.position.z + e.deltaY * 0.003));
  }, { passive: false });

  // Click to open modal
  canvas.addEventListener('click', e => {
    const hit = raycastMarker(e);
    if (hit) {
      openModal(hit.id, hit.info);
    }
  });

  // ── Raycasting ──
  const raycaster = new THREE.Raycaster();
  raycaster.params.Points = { threshold: 0.1 };
  const mouse = new THREE.Vector2();

  function getMouseNDC(e) {
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    return mouse;
  }

  function raycastMarker(e) {
    getMouseNDC(e);
    raycaster.setFromCamera(mouse, camera);
    let closest = null;
    let minDist = 0.08;
    for (const m of markers) {
      // Get world position of marker
      const wp = new THREE.Vector3();
      m.mesh.getWorldPosition(wp);
      const d = raycaster.ray.distanceToPoint(wp);
      if (d < minDist) {
        minDist = d;
        closest = m;
      }
    }
    return closest;
  }

  function updateTooltip(e) {
    const hit = raycastMarker(e);
    if (hit) {
      const deforest = hit.info.labels?.radd?.deforest_pct;
      let html = `<strong>${hit.id}</strong><br>`;
      html += `${hit.info.region} · ${hit.info.split}<br>`;
      html += `S2: ${hit.info.s2_count || '—'} files`;
      if (deforest !== undefined) html += ` · ${deforest}% def.`;
      html += `<br><span style="color:#22c55e;font-size:.68rem">Click for details</span>`;
      tooltip.innerHTML = html;
      tooltip.classList.add('visible');
      // Position near cursor
      const rect = wrap.getBoundingClientRect();
      let tx = e.clientX - rect.left + 15;
      let ty = e.clientY - rect.top - 10;
      if (tx + 200 > rect.width) tx = e.clientX - rect.left - 210;
      tooltip.style.left = tx + 'px';
      tooltip.style.top  = ty + 'px';
      canvas.style.cursor = 'pointer';
    } else {
      tooltip.classList.remove('visible');
      canvas.style.cursor = isDragging ? 'grabbing' : 'grab';
    }
  }

  // ── Pulse animation for rings ──
  let time = 0;

  // ── Animate ──
  function animate() {
    requestAnimationFrame(animate);
    time += 0.016;

    if (autoRotate) {
      earthGroup.rotation.y += 0.001;
    }

    // Dampen velocity
    rotVel.x *= 0.92;
    rotVel.y *= 0.92;
    if (!isDragging) {
      earthGroup.rotation.x += rotVel.x;
      earthGroup.rotation.y += rotVel.y;
    }

    // Pulse marker rings
    for (const m of markers) {
      const s = 1 + 0.3 * Math.sin(time * 2.5 + m.pos.x * 10);
      m.ring.scale.set(s, s, s);
      m.ring.material.opacity = 0.2 + 0.2 * Math.sin(time * 2.5 + m.pos.x * 10);
    }

    renderer.render(scene, camera);
  }
  animate();

  // Resize
  window.addEventListener('resize', () => {
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
}

/* ═══════════════════════════════════════════════════════════
   TILE GRID
   ═══════════════════════════════════════════════════════════ */
function populateTileGrid() {
  const grid = el('tileGrid');
  grid.innerHTML = '';

  const allTiles = [
    ...Object.entries(DATA.train).map(([id, info]) => ({ id, ...info })),
    ...Object.entries(DATA.test).map(([id, info]) => ({ id, ...info }))
  ];

  for (const tile of allTiles) {
    const card = document.createElement('div');
    card.className = 'de-tile-card';
    card.dataset.split  = tile.split;
    card.dataset.region = tile.region;
    card.dataset.id     = tile.id;

    const regionShort = tile.region === 'South America' ? 'Amazon'
                      : tile.region === 'Southeast Asia' ? 'SE Asia' : 'Africa';

    card.innerHTML = `
      <img class="de-tile-thumb" src="data-assets/thumbs/${tile.id}_s2_rgb_first.png" alt="${tile.id}" loading="lazy">
      <div class="de-tile-info">
        <div class="de-tile-name">${tile.id}</div>
        <div class="de-tile-meta">
          <span class="de-tile-badge ${tile.split}">${tile.split}</span>
          <span class="de-tile-badge region">${regionShort}</span>
          ${tile.labels?.radd?.deforest_pct !== undefined
            ? `<span class="de-tile-badge" style="color:var(--red);border-color:rgba(239,68,68,.25)">${tile.labels.radd.deforest_pct}%</span>`
            : ''}
        </div>
      </div>`;

    card.addEventListener('click', () => {
      openModal(tile.id, DATA.train[tile.id] || DATA.test[tile.id]);
    });
    grid.appendChild(card);
  }
}

/* ═══════════════════════════════════════════════════════════
   FILTERS
   ═══════════════════════════════════════════════════════════ */
function initFilters() {
  const btns = document.querySelectorAll('.de-filter');
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const filter = btn.dataset.filter;
      document.querySelectorAll('.de-tile-card').forEach(card => {
        card.style.display = (filter === 'all' || card.dataset.split === filter || card.dataset.region === filter) ? '' : 'none';
      });
    });
  });
}

/* ═══════════════════════════════════════════════════════════
   MODAL — Fixed z-index, proper event handling
   ═══════════════════════════════════════════════════════════ */
function initModal() {
  el('modalClose').addEventListener('click', closeModal);
  el('modalOverlay').addEventListener('click', e => {
    if (e.target === el('modalOverlay')) closeModal();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeModal();
  });
}

function closeModal() {
  el('modalOverlay').classList.remove('open');
  document.body.style.overflow = '';
}

function openModal(id, info) {
  const overlay = el('modalOverlay');
  overlay.classList.add('open');
  document.body.style.overflow = 'hidden';

  // Title + badges
  el('modalTitle').textContent = id;
  const regionShort = info.region === 'South America' ? 'Amazon'
                    : info.region === 'Southeast Asia' ? 'SE Asia' : 'Africa';
  el('modalBadges').innerHTML = `
    <span class="de-tile-badge ${info.split}">${info.split}</span>
    <span class="de-tile-badge region">${regionShort}</span>
    <span class="de-tile-badge">${info.lat.toFixed(2)}°, ${info.lon.toFixed(2)}°</span>`;

  buildImageTabs(id, info);
  buildTimeCompare(id);
  buildStats(info);
  buildNDVIChart(info);
  buildBandChart(info);
  buildLabels(info);

  overlay.scrollTop = 0;
}

/* ── Image Tabs (rebuilt fresh each time, no clone bugs) ── */
function buildImageTabs(id, info) {
  const wrap = el('imgTabs');
  const img  = el('modalImg');
  const cap  = el('imgCaption');
  const isTrainTile = info.split === 'train';

  const views = [
    { key: 'rgb',    label: 'True Color',  src: `data-assets/thumbs/${id}_s2_rgb_first.png`, cap: 'Sentinel-2 True Color (B04, B03, B02) — Jan 2020',             enabled: true },
    { key: 'fc',     label: 'False Color',  src: `data-assets/thumbs/${id}_s2_fc.png`,       cap: 'False Color (NIR, Red, Green) — Vegetation appears red',        enabled: true },
    { key: 'ndvi',   label: 'NDVI',         src: `data-assets/thumbs/${id}_ndvi.png`,        cap: 'NDVI — Green = dense canopy, Brown = bare soil',                enabled: true },
    { key: 's1',     label: 'SAR (S1)',     src: `data-assets/thumbs/${id}_s1.png`,          cap: 'Sentinel-1 SAR Backscatter (VV) — Cyan = radar return',         enabled: true },
    { key: 'aef',    label: 'AEF Embed',    src: `data-assets/thumbs/${id}_aef.png`,         cap: 'AlphaEarth Embeddings (3 of 64 dims as RGB)',                   enabled: true },
    { key: 'radd',   label: 'RADD Labels',  src: `data-assets/thumbs/${id}_radd.png`,        cap: 'RADD Radar Alerts (post-2020) — Red = deforestation',           enabled: isTrainTile },
    { key: 'glads2', label: 'GLAD-S2',      src: `data-assets/thumbs/${id}_glads2.png`,      cap: 'GLAD-S2 — Red = high conf, Orange = medium, Yellow = low',      enabled: isTrainTile },
  ];

  // Build fresh buttons
  wrap.innerHTML = '';
  views.forEach((v, i) => {
    const btn = document.createElement('button');
    btn.className = 'de-img-tab' + (i === 0 ? ' active' : '') + (!v.enabled ? ' disabled' : '');
    btn.textContent = v.label;
    btn.addEventListener('click', () => {
      if (!v.enabled) return;
      wrap.querySelectorAll('.de-img-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      img.src = v.src;
      cap.textContent = v.cap;
    });
    wrap.appendChild(btn);
  });

  // Default
  img.src = views[0].src;
  cap.textContent = views[0].cap;
}

function buildTimeCompare(id) {
  el('compareFirst').src = `data-assets/thumbs/${id}_s2_rgb_first.png`;
  el('compareMid').src   = `data-assets/thumbs/${id}_s2_rgb_mid.png`;
  el('compareLast').src  = `data-assets/thumbs/${id}_s2_rgb_last.png`;
}

function buildStats(info) {
  const items = [
    { k: 'S2 Files', v: info.s2_count || '—' },
    { k: 'S1 Files', v: info.s1_count || '—' },
    { k: 'AEF Files', v: info.aef_count || '—' },
  ];
  if (info.s2_shape) items.push({ k: 'S2 Pixels', v: `${info.s2_shape[0]}×${info.s2_shape[1]}` });
  if (info.s1_shape) items.push({ k: 'S1 Pixels', v: `${info.s1_shape[0]}×${info.s1_shape[1]}` });
  if (info.aef_bands) items.push({ k: 'AEF Dims', v: info.aef_bands });
  items.push({ k: 'Region', v: info.region });
  items.push({ k: 'Lat / Lon', v: `${info.lat.toFixed(3)}° / ${info.lon.toFixed(3)}°` });
  if (info.labels?.radd) {
    items.push({ k: 'RADD Alerts', v: fmtNum(info.labels.radd.post2020_alerts) });
    items.push({ k: 'Deforestation', v: info.labels.radd.deforest_pct + '%' });
  }
  el('modalStats').innerHTML = items.map(i =>
    `<div class="de-ms-item"><div class="de-ms-val">${i.v}</div><div class="de-ms-key">${i.k}</div></div>`
  ).join('');
}

/* ── Charts ─────────────────────────────────────────────── */
function buildNDVIChart(info) {
  const wrap = el('ndviChartWrap');
  if (!info.ndvi_ts || !info.ndvi_ts.length) { wrap.style.display = 'none'; return; }
  wrap.style.display = '';

  const labels = info.ndvi_ts.map(d => `${d.year}-${String(d.month).padStart(2,'0')}`);
  const means = info.ndvi_ts.map(d => d.ndvi_mean);
  const upper = info.ndvi_ts.map(d => d.ndvi_mean + d.ndvi_std);
  const lower = info.ndvi_ts.map(d => d.ndvi_mean - d.ndvi_std);

  if (ndviChartInstance) ndviChartInstance.destroy();
  ndviChartInstance = new Chart(el('ndviChart'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'NDVI', data: means, borderColor: '#22c55e', borderWidth: 2, pointRadius: 0, fill: false, tension: .3 },
        { label: '+σ', data: upper, borderColor: 'transparent', backgroundColor: 'rgba(34,197,94,.08)', pointRadius: 0, fill: '+1', tension: .3 },
        { label: '-σ', data: lower, borderColor: 'transparent', pointRadius: 0, fill: false, tension: .3 },
      ]
    },
    options: chartOpts({ yMin: 0, yMax: 1, yLabel: 'NDVI', xLimit: 12 })
  });
}

function buildBandChart(info) {
  const wrap = el('bandChartWrap');
  if (!info.band_means || !info.band_means.length) { wrap.style.display = 'none'; return; }
  wrap.style.display = '';

  const names = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12'];
  const colors = ['#a78bfa','#60a5fa','#34d399','#f87171','#fb923c','#f97316','#ea580c','#dc2626','#b91c1c','#06b6d4','#818cf8','#f472b6','#e879f9'];

  if (bandChartInstance) bandChartInstance.destroy();
  bandChartInstance = new Chart(el('bandChart'), {
    type: 'bar',
    data: {
      labels: names.slice(0, info.band_means.length),
      datasets: [{
        label: 'Reflectance',
        data: info.band_means,
        backgroundColor: colors.slice(0, info.band_means.length).map(c => c + '88'),
        borderColor: colors.slice(0, info.band_means.length),
        borderWidth: 1.5,
        borderRadius: 3
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1a2235', borderColor: '#22c55e', borderWidth: 1,
          callbacks: { label: ctx => `Mean: ${Math.round(ctx.raw)} · σ: ${Math.round(info.band_stds[ctx.dataIndex])}` }
        }
      },
      scales: {
        x: { ticks: { color: '#94a3b8', font: { family: FONT, size: 10 } }, grid: { display: false } },
        y: { title: { display: true, text: 'Reflectance', color: '#94a3b8', font: { family: FONT } },
             ticks: { color: '#64748b', font: { family: FONT, size: 10 } },
             grid: { color: 'rgba(255,255,255,.04)' } }
      }
    }
  });
}

function buildLabels(info) {
  const section = el('labelSection');
  const grid = el('labelGrid');
  if (info.split !== 'train' || !info.labels) { section.style.display = 'none'; return; }
  section.style.display = '';
  grid.innerHTML = '';

  if (info.labels.radd?.total_alerts > 0) {
    const r = info.labels.radd;
    grid.innerHTML += `<div class="de-label-card"><h4>RADD (Radar)</h4>
      ${lr('Total',r.total_alerts)}${lr('Post-2020',r.post2020_alerts)}${lr('High conf',r.high_conf)}${lr('Low conf',r.low_conf)}
      ${lr('Deforestation',r.deforest_pct+'%','var(--red)')}
      <div class="de-deforest-bar"><div class="de-deforest-fill" style="width:${Math.min(r.deforest_pct*2,100)}%;background:var(--red)"></div></div></div>`;
  }
  if (info.labels.gladl?.total > 0) {
    const g = info.labels.gladl;
    let rows = '';
    for (const [yr, c] of Object.entries(g)) { if (yr !== 'total') rows += lr(yr, c); }
    grid.innerHTML += `<div class="de-label-card"><h4>GLAD-L (Landsat)</h4>${rows}${lr('Total',g.total)}</div>`;
  }
  if (info.labels.glads2?.total > 0) {
    const g = info.labels.glads2;
    grid.innerHTML += `<div class="de-label-card"><h4>GLAD-S2</h4>
      ${lr('Total',g.total)}${lr('High conf',g.high_conf)}${lr('Medium',g.med_conf)}${lr('Low',g.low_conf)}</div>`;
  }
}

/* ── Time-lapse ─────────────────────────────────────────── */
function initTimelapse() {
  const slider = el('tlSlider');
  const img = el('tlImage');
  const label = el('tlYear');
  const btn = el('tlPlay');
  let interval = null;

  slider.addEventListener('input', () => {
    img.src = `data-assets/thumbs/timelapse_${slider.value}.png`;
    label.textContent = slider.value;
  });

  btn.addEventListener('click', () => {
    if (interval) { clearInterval(interval); interval = null; btn.textContent = '▶ Play'; btn.classList.remove('playing'); return; }
    btn.textContent = '⏸ Pause'; btn.classList.add('playing');
    let yr = 2020;
    interval = setInterval(() => {
      img.src = `data-assets/thumbs/timelapse_${yr}.png`;
      label.textContent = yr;
      slider.value = yr;
      yr = yr >= 2025 ? 2020 : yr + 1;
    }, 1000);
  });
}

/* ── Deforestation Chart ─────────────────────────────────── */
function initDeforestChart() {
  const tiles = [], raddPcts = [], gladlK = [], glads2K = [];
  for (const [id, info] of Object.entries(DATA.train)) {
    tiles.push(id.replace(/_/g, ' '));
    raddPcts.push(info.labels?.radd?.deforest_pct || 0);
    gladlK.push((info.labels?.gladl?.total || 0) / 1000);
    glads2K.push((info.labels?.glads2?.total || 0) / 1000);
  }

  new Chart(el('deforestChart'), {
    type: 'bar',
    data: {
      labels: tiles,
      datasets: [
        { label: 'RADD %', data: raddPcts, backgroundColor: 'rgba(239,68,68,.55)', borderColor: '#ef4444', borderWidth: 1, borderRadius: 2, yAxisID: 'y' },
        { label: 'GLAD-L (k)', data: gladlK, backgroundColor: 'rgba(245,158,11,.45)', borderColor: '#f59e0b', borderWidth: 1, borderRadius: 2, yAxisID: 'y1' },
        { label: 'GLAD-S2 (k)', data: glads2K, backgroundColor: 'rgba(59,130,246,.45)', borderColor: '#3b82f6', borderWidth: 1, borderRadius: 2, yAxisID: 'y1' },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { family: FONT, size: 11 } } },
        tooltip: { backgroundColor: '#1a2235', borderColor: 'rgba(255,255,255,.12)', borderWidth: 1 }
      },
      scales: {
        x: { ticks: { color: '#64748b', font: { family: FONT, size: 7.5 }, maxRotation: 50 }, grid: { display: false } },
        y:  { position: 'left', title: { display: true, text: 'RADD %', color: '#ef4444', font: { family: FONT } }, ticks: { color: '#ef4444', font: { family: FONT, size: 10 } }, grid: { color: 'rgba(255,255,255,.04)' } },
        y1: { position: 'right', title: { display: true, text: 'Alerts (×1k)', color: '#3b82f6', font: { family: FONT } }, ticks: { color: '#3b82f6', font: { family: FONT, size: 10 } }, grid: { drawOnChartArea: false } }
      }
    }
  });
}

/* ═══ HELPERS ═══ */
function el(id) { return document.getElementById(id); }
function fmtNum(n) {
  if (n == null) return '—';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(n);
}
function lr(label, val, color) {
  return `<div class="de-label-row"><span>${label}</span><span class="de-label-val"${color ? ` style="color:${color}"` : ''}>${fmtNum(val)}</span></div>`;
}
function chartOpts({ yMin, yMax, yLabel, xLimit }) {
  return {
    responsive: true, maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { labels: { color: '#94a3b8', font: { family: FONT, size: 10 } } },
      tooltip: { backgroundColor: '#1a2235', borderColor: '#22c55e', borderWidth: 1 }
    },
    scales: {
      x: { ticks: { color: '#64748b', font: { family: FONT, size: 8 }, maxTicksLimit: xLimit || 20, maxRotation: 40 }, grid: { color: 'rgba(255,255,255,.03)' } },
      y: { min: yMin, max: yMax, title: { display: true, text: yLabel, color: '#94a3b8', font: { family: FONT } }, ticks: { color: '#64748b', font: { family: FONT, size: 10 } }, grid: { color: 'rgba(255,255,255,.04)' } }
    }
  };
}
