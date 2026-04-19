/* ═══════════════════════════════════════
   OASIS — Ultra Premium Interactive JS
   Real Earth Globe + GSAP + 20 Upgrades
   ═══════════════════════════════════════ */

gsap.registerPlugin(ScrollTrigger);

/* ═══ PRELOADER ═══ */
function initPreloader() {
  const counter = document.getElementById('preCount');
  const fill = document.getElementById('preFill');
  const preloader = document.getElementById('preloader');
  let count = { val: 0 };

  gsap.to(count, {
    val: 100, duration: 2.4, ease: 'power2.inOut',
    onUpdate() {
      const v = Math.round(count.val);
      counter.textContent = v;
      fill.style.width = v + '%';
    },
    onComplete() {
      gsap.to(preloader, {
        yPercent: -100, duration: 0.9, ease: 'power3.inOut', delay: 0.3,
        onComplete() { preloader.style.display = 'none'; animateHero(); }
      });
    }
  });
}

/* ═══ CURSOR ═══ */
function initCursor() {
  const dot = document.getElementById('curDot');
  const ring = document.getElementById('curRing');
  if (!dot || !ring) return;
  let mx = innerWidth / 2, my = innerHeight / 2;
  let dx = mx, dy = my, rx = mx, ry = my;

  addEventListener('mousemove', e => { mx = e.clientX; my = e.clientY; });

  const hoverSel = 'a,button,[data-magnetic],[data-tilt],.hcard,.ts-btn,.dtab,.fcat-head,.cg,.back-top';
  document.addEventListener('mouseover', e => { if (e.target.closest(hoverSel)) ring.classList.add('hover'); });
  document.addEventListener('mouseout', e => { if (e.target.closest(hoverSel)) ring.classList.remove('hover'); });

  (function tick() {
    dx += (mx - dx) * 0.15; dy += (my - dy) * 0.15;
    rx += (mx - rx) * 0.07; ry += (my - ry) * 0.07;
    dot.style.transform = `translate(${dx - 4}px,${dy - 4}px)`;
    ring.style.transform = `translate(${rx - 22}px,${ry - 22}px)`;
    requestAnimationFrame(tick);
  })();
}

/* ═══ MAGNETIC ═══ */
function initMagnetic() {
  document.querySelectorAll('[data-magnetic]').forEach(el => {
    el.addEventListener('mousemove', e => {
      const r = el.getBoundingClientRect();
      const dx = (e.clientX - r.left - r.width / 2) * 0.12;
      const dy = (e.clientY - r.top - r.height / 2) * 0.12;
      gsap.to(el, { x: dx, y: dy, duration: 0.4, ease: 'power2.out' });
    });
    el.addEventListener('mouseleave', () => {
      gsap.to(el, { x: 0, y: 0, duration: 0.7, ease: 'elastic.out(1,0.4)' });
    });
  });
}

/* ═══ THREE.JS — REAL EARTH GLOBE + CLOUD LAYER ═══ */
function initGlobe() {
  const canvas = document.getElementById('globeCanvas');
  if (!canvas || typeof THREE === 'undefined') return;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 1000);
  camera.position.z = 4.2;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;

  // Lighting
  const ambient = new THREE.AmbientLight(0x334455, 0.4);
  scene.add(ambient);

  const sun = new THREE.DirectionalLight(0xffffff, 1.8);
  sun.position.set(5, 3, 5);
  scene.add(sun);

  const rimLight = new THREE.DirectionalLight(0x10b981, 0.5);
  rimLight.position.set(-5, 0, -3);
  scene.add(rimLight);

  const earthGroup = new THREE.Group();
  scene.add(earthGroup);

  // Earth Sphere
  const earthGeom = new THREE.SphereGeometry(1.6, 64, 64);
  const textureLoader = new THREE.TextureLoader();
  const earthTexURL = 'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg';
  const bumpURL = 'https://unpkg.com/three-globe/example/img/earth-topology.png';

  const earthMat = new THREE.MeshPhongMaterial({
    color: 0xffffff,
    shininess: 25,
    specular: new THREE.Color(0x333333),
  });

  textureLoader.load(earthTexURL, tex => {
    earthMat.map = tex;
    earthMat.needsUpdate = true;
  });

  textureLoader.load(bumpURL, tex => {
    earthMat.bumpMap = tex;
    earthMat.bumpScale = 0.02;
    earthMat.needsUpdate = true;
  });

  const earth = new THREE.Mesh(earthGeom, earthMat);
  earthGroup.add(earth);

  // Cloud Layer
  const cloudGeom = new THREE.SphereGeometry(1.63, 48, 48);
  const cloudMat = new THREE.MeshPhongMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.15,
    depthWrite: false,
  });
  textureLoader.load('https://unpkg.com/three-globe/example/img/earth-water.png', tex => {
    cloudMat.alphaMap = tex;
    cloudMat.needsUpdate = true;
  });
  const clouds = new THREE.Mesh(cloudGeom, cloudMat);
  earthGroup.add(clouds);

  // Atmosphere glow (Fresnel)
  const atmosGeom = new THREE.SphereGeometry(1.72, 64, 64);
  const atmosMat = new THREE.ShaderMaterial({
    vertexShader: `
      varying vec3 vNormal;
      void main() {
        vNormal = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      varying vec3 vNormal;
      void main() {
        float intensity = pow(0.65 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 3.0);
        vec3 color = mix(vec3(0.063, 0.725, 0.506), vec3(0.024, 0.714, 0.831), intensity);
        gl_FragColor = vec4(color, intensity * 0.55);
      }
    `,
    blending: THREE.AdditiveBlending,
    side: THREE.BackSide,
    transparent: true,
    depthWrite: false
  });
  earthGroup.add(new THREE.Mesh(atmosGeom, atmosMat));

  // Stars
  const starCount = 4000;
  const starPos = new Float32Array(starCount * 3);
  const starSizes = new Float32Array(starCount);
  for (let i = 0; i < starCount; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const r = 20 + Math.random() * 80;
    starPos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    starPos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    starPos[i * 3 + 2] = r * Math.cos(phi);
    starSizes[i] = 0.5 + Math.random() * 1.5;
  }
  const starGeom = new THREE.BufferGeometry();
  starGeom.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
  starGeom.setAttribute('size', new THREE.BufferAttribute(starSizes, 1));
  const starMat = new THREE.ShaderMaterial({
    vertexShader: `
      attribute float size;
      varying float vAlpha;
      void main() {
        vAlpha = 0.3 + size * 0.3;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * (200.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: `
      varying float vAlpha;
      void main() {
        float d = length(gl_PointCoord - vec2(0.5));
        if (d > 0.5) discard;
        float a = smoothstep(0.5, 0.0, d) * vAlpha;
        gl_FragColor = vec4(1.0, 1.0, 1.0, a);
      }
    `,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending
  });
  scene.add(new THREE.Points(starGeom, starMat));

  // Location Markers
  const locations = [
    { lat: -3, lon: -60, label: 'Amazon' },
    { lat: 2, lon: 110, label: 'SE Asia' },
    { lat: 0, lon: 25, label: 'Africa' }
  ];

  locations.forEach(loc => {
    const phi = (90 - loc.lat) * Math.PI / 180;
    const theta = (loc.lon + 180) * Math.PI / 180;
    const r = 1.62;
    const x = -r * Math.sin(phi) * Math.cos(theta);
    const y = r * Math.cos(phi);
    const z = r * Math.sin(phi) * Math.sin(theta);

    const dotGeom = new THREE.SphereGeometry(0.025, 8, 8);
    const dotMat = new THREE.MeshBasicMaterial({ color: 0x10b981 });
    const dot = new THREE.Mesh(dotGeom, dotMat);
    dot.position.set(x, y, z);
    earthGroup.add(dot);

    const ringGeom = new THREE.RingGeometry(0.04, 0.055, 24);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.6, side: THREE.DoubleSide });
    const ring = new THREE.Mesh(ringGeom, ringMat);
    ring.position.set(x, y, z);
    ring.lookAt(0, 0, 0);
    earthGroup.add(ring);

    gsap.to(ring.scale, { x: 3, y: 3, z: 3, duration: 2.5, repeat: -1, ease: 'power1.out' });
    gsap.to(ringMat, { opacity: 0, duration: 2.5, repeat: -1, ease: 'power1.out' });

    const beamGeom = new THREE.CylinderGeometry(0.003, 0.003, 0.3, 4);
    const beamMat = new THREE.MeshBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.3 });
    const beam = new THREE.Mesh(beamGeom, beamMat);
    beam.position.set(x * 1.08, y * 1.08, z * 1.08);
    beam.lookAt(0, 0, 0);
    beam.rotateX(Math.PI / 2);
    earthGroup.add(beam);
  });

  // Mouse interaction
  let targetRotY = 0, targetRotX = 0;
  addEventListener('mousemove', e => {
    targetRotY = ((e.clientX / innerWidth) - 0.5) * 0.4;
    targetRotX = ((e.clientY / innerHeight) - 0.5) * 0.2;
  });

  // Animation loop
  function animate() {
    requestAnimationFrame(animate);
    earth.rotation.y += 0.001;
    clouds.rotation.y += 0.0004;
    clouds.rotation.x += 0.0001;
    earthGroup.rotation.y += (targetRotY - earthGroup.rotation.y) * 0.015;
    earthGroup.rotation.x += (targetRotX - earthGroup.rotation.x) * 0.015;
    renderer.render(scene, camera);
  }
  animate();

  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  // Scroll: push globe back
  gsap.to(earthGroup.scale, {
    x: 0.6, y: 0.6, z: 0.6,
    scrollTrigger: { trigger: '#hero', start: 'top top', end: 'bottom top', scrub: 1.5 }
  });
  gsap.to(earthGroup.position, {
    y: -1,
    scrollTrigger: { trigger: '#hero', start: 'top top', end: 'bottom top', scrub: 1.5 }
  });
}

/* ═══ HERO ENTRANCE ═══ */
function animateHero() {
  const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });
  tl.to('.hero-badge', { opacity: 1, duration: 0.8 }, 0.1)
    .to('.tl span', { y: 0, duration: 1.1, stagger: 0.13 }, 0.3)
    .to('.hero-sub', { opacity: 1, duration: 0.9 }, 0.9)
    .to('.hero-cta', { opacity: 1, duration: 0.9 }, 1.1)
    .to('.hero-stats', { opacity: 1, duration: 0.8 }, 1.3)
    .to('.scroll-cue', { opacity: 1, duration: 0.8 }, 1.5);

  document.querySelectorAll('.hstat-num').forEach(el => {
    const target = +el.dataset.count;
    gsap.to(el, { innerText: target, duration: 2.2, delay: 1.6, snap: { innerText: 1 }, ease: 'power2.out' });
  });
}

/* ═══ SCROLL PROGRESS BAR ═══ */
function initScrollProgress() {
  const bar = document.getElementById('scrollProgress');
  if (!bar) return;
  addEventListener('scroll', () => {
    const pct = scrollY / (document.documentElement.scrollHeight - innerHeight) * 100;
    bar.style.width = pct + '%';
  }, { passive: true });
}

/* ═══ NAV + ACTIVE SECTION HIGHLIGHTING ═══ */
function initNav() {
  const nav = document.getElementById('nav');
  let lastY = 0;
  ScrollTrigger.create({
    start: 'top -80',
    onUpdate: self => {
      const sy = self.scroll();
      if (sy > 80) {
        nav.classList.add('scrolled');
        nav.classList.toggle('hidden', sy > lastY && sy > 400);
      } else {
        nav.classList.remove('scrolled', 'hidden');
      }
      lastY = sy;
    }
  });

  // Mobile menu
  const menuBtn = document.getElementById('menuBtn');
  const mobileMenu = document.getElementById('mobileMenu');
  if (menuBtn && mobileMenu) {
    menuBtn.addEventListener('click', () => { menuBtn.classList.toggle('open'); mobileMenu.classList.toggle('open'); });
    mobileMenu.querySelectorAll('a').forEach(a => a.addEventListener('click', () => { menuBtn.classList.remove('open'); mobileMenu.classList.remove('open'); }));
  }

  // Active nav link via IntersectionObserver
  const navLinks = document.querySelectorAll('.nav-link');
  const sections = document.querySelectorAll('.sec');
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        navLinks.forEach(link => {
          link.classList.toggle('active', link.getAttribute('href') === '#' + id);
        });
      }
    });
  }, { rootMargin: '-30% 0px -60% 0px' });
  sections.forEach(sec => observer.observe(sec));
}

/* ═══ BACK TO TOP ═══ */
function initBackToTop() {
  const btn = document.getElementById('backTop');
  if (!btn) return;
  addEventListener('scroll', () => {
    btn.classList.toggle('visible', scrollY > innerHeight);
  }, { passive: true });
  btn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}

/* ═══ SCROLL ANIMATIONS ═══ */
function initScrollAnimations() {
  // Section headers — stagger children
  gsap.utils.toArray('.sec-head').forEach(head => {
    gsap.from(head.children, {
      y: 60, opacity: 0, stagger: 0.15, duration: 1, ease: 'power3.out',
      scrollTrigger: { trigger: head, start: 'top 82%', once: true }
    });
  });

  // Glass cards, model cards, inno cards, etc. (exclude pipe-cards — handled by initPipeline)
  gsap.utils.toArray('.glass,.inno,.mcard,.pp-step,.fcat,.cg').forEach(el => {
    if (el.closest('.pipe-step')) return; // pipeline has its own animation
    gsap.from(el, {
      y: 50, opacity: 0, duration: 0.9, ease: 'power3.out',
      scrollTrigger: { trigger: el, start: 'top 87%', once: true }
    });
  });

  // Apple-style image reveals
  gsap.utils.toArray('.reveal-image').forEach(img => {
    gsap.to(img, {
      clipPath: 'inset(0% 0% round 24px)', ease: 'none',
      scrollTrigger: { trigger: img, start: 'top 82%', end: 'top 25%', scrub: 1 }
    });
    const inner = img.querySelector('img');
    if (inner) {
      gsap.to(inner, {
        y: -50, ease: 'none',
        scrollTrigger: { trigger: img, start: 'top bottom', end: 'bottom top', scrub: 1 }
      });
    }
  });

  // Sub titles
  gsap.utils.toArray('.sub-title,.sub-desc').forEach(el => {
    gsap.from(el, { y: 30, opacity: 0, duration: 0.8, ease: 'power3.out', scrollTrigger: { trigger: el, start: 'top 85%', once: true } });
  });

  // Ensemble flow
  const ensFlow = document.querySelector('.ens-flow');
  if (ensFlow) gsap.from(ensFlow, { y: 40, opacity: 0, duration: 1, scrollTrigger: { trigger: ensFlow, start: 'top 82%', once: true } });

  // Section head parallax
  gsap.utils.toArray('.sec-head').forEach(head => {
    gsap.to(head, {
      y: -20, ease: 'none',
      scrollTrigger: { trigger: head, start: 'top bottom', end: 'bottom top', scrub: 1.5 }
    });
  });

  // Band stagger entrance — use immediate stagger, not scroll-based
  // (scroll triggers are unreliable after horizontal scroll pin)
  const bandStack = document.querySelector('.band-stack');
  if (bandStack) {
    Array.from(bandStack.children).forEach((band, i) => {
      band.style.opacity = '0';
      band.style.transform = `translateX(${-40 + i * 4}px)`;
    });
    // Trigger when data section enters
    ScrollTrigger.create({
      trigger: '#data', start: 'top 70%', once: true,
      onEnter() {
        Array.from(bandStack.children).forEach((band, i) => {
          gsap.to(band, {
            opacity: 0.3, x: i * 4, duration: 0.5, delay: i * 0.05, ease: 'power2.out'
          });
        });
      }
    });
  }

  // Footer metrics count up
  document.querySelectorAll('.fm span').forEach(el => {
    const text = el.textContent.replace('~', '');
    const num = parseInt(text);
    if (isNaN(num)) return;
    const prefix = el.textContent.includes('~') ? '~' : '';
    el.textContent = '0';
    ScrollTrigger.create({
      trigger: el, start: 'top 90%', once: true,
      onEnter() {
        gsap.to(el, { innerText: num, duration: 2, snap: { innerText: 1 }, ease: 'power2.out',
          onUpdate() { if (prefix) el.textContent = prefix + el.textContent; }
        });
      }
    });
  });
}

/* ═══ HORIZONTAL SCROLL + PROGRESS ═══ */
function initHorizontalScroll() {
  const wrap = document.getElementById('hscrollWrap');
  const track = document.getElementById('hscrollTrack');
  const fill = document.getElementById('hscrollFill');
  if (!wrap || !track) return;

  const st = gsap.to(track, {
    x: () => -(track.scrollWidth - innerWidth),
    ease: 'none',
    scrollTrigger: {
      trigger: wrap, start: 'top top', end: () => '+=' + track.scrollWidth,
      pin: true, scrub: 1, invalidateOnRefresh: true, anticipatePin: 1,
      onUpdate: self => {
        if (fill) fill.style.width = (self.progress * 100) + '%';
      }
    }
  });

  // Parallax on card images
  gsap.utils.toArray('.hcard-img img').forEach(img => {
    gsap.to(img, {
      x: -30, ease: 'none',
      scrollTrigger: { trigger: wrap, start: 'top top', end: () => '+=' + track.scrollWidth, scrub: 1 }
    });
  });
}

/* ═══ 3D TILT CARDS ═══ */
function initTiltCards() {
  document.querySelectorAll('[data-tilt]').forEach(card => {
    let targetRX = 0, targetRY = 0, currentRX = 0, currentRY = 0, hovering = false;

    card.addEventListener('mouseenter', () => { hovering = true; });
    card.addEventListener('mouseleave', () => { hovering = false; targetRX = 0; targetRY = 0; });
    card.addEventListener('mousemove', e => {
      const r = card.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width - 0.5;
      const y = (e.clientY - r.top) / r.height - 0.5;
      targetRX = -y * 10;
      targetRY = x * 10;
    });

    (function animate() {
      currentRX += (targetRX - currentRX) * 0.08;
      currentRY += (targetRY - currentRY) * 0.08;
      if (Math.abs(currentRX) > 0.01 || Math.abs(currentRY) > 0.01 || hovering) {
        card.style.transform = `perspective(1000px) rotateX(${currentRX}deg) rotateY(${currentRY}deg)`;
      }
      requestAnimationFrame(animate);
    })();
  });
}

/* ═══ MODEL CARD CURSOR GLOW ═══ */
function initModelCardGlow() {
  document.querySelectorAll('.mcard').forEach(card => {
    card.addEventListener('mousemove', e => {
      const r = card.getBoundingClientRect();
      card.style.setProperty('--glow-x', (e.clientX - r.left) + 'px');
      card.style.setProperty('--glow-y', (e.clientY - r.top) + 'px');
    });
  });
}

/* ═══ DATA TABS + SLIDING INDICATOR ═══ */
function initDataTabs() {
  const tabs = document.querySelectorAll('.dtab');
  const panels = document.querySelectorAll('.dpanel');
  const indicator = document.getElementById('dtabIndicator');

  function moveIndicator(tab) {
    if (!indicator || !tab) return;
    const container = tab.parentElement;
    const cr = container.getBoundingClientRect();
    const tr = tab.getBoundingClientRect();
    indicator.style.left = (tr.left - cr.left) + 'px';
    indicator.style.width = tr.width + 'px';
  }

  // Initialize indicator position
  requestAnimationFrame(() => {
    const active = document.querySelector('.dtab.active');
    if (active) moveIndicator(active);
  });

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const t = tab.dataset.panel;
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      panels.forEach(p => { p.classList.remove('active'); if (p.id === 'dpanel-' + t) p.classList.add('active'); });
      moveIndicator(tab);
    });
  });

  // Embed grid
  const grid = document.getElementById('embedGrid');
  if (grid) {
    for (let i = 0; i < 64; i++) {
      const c = document.createElement('div');
      const h = 140 + Math.random() * 60;
      const l = 18 + Math.random() * 40;
      c.style.cssText = `background:hsl(${h},60%,${l}%);border-radius:4px;aspect-ratio:1;transition:transform .3s,box-shadow .3s`;
      c.addEventListener('mouseenter', () => { c.style.transform = 'scale(1.3)'; c.style.boxShadow = `0 0 12px hsl(${h},60%,${l}%)`; });
      c.addEventListener('mouseleave', () => { c.style.transform = 'scale(1)'; c.style.boxShadow = 'none'; });
      grid.appendChild(c);
    }
  }

  // Reposition indicator on resize
  addEventListener('resize', () => {
    const active = document.querySelector('.dtab.active');
    if (active) moveIndicator(active);
  });
}

/* ═══ FEATURE ACCORDION (GSAP smooth) ═══ */
function initFeatureAccordion() {
  document.querySelectorAll('.fcat-head').forEach(head => {
    head.addEventListener('click', () => {
      const fcat = head.parentElement;
      const body = fcat.querySelector('.fcat-body');
      const wasOpen = fcat.classList.contains('open');

      // Close all
      document.querySelectorAll('.fcat').forEach(fc => {
        if (fc === fcat) return;
        const fb = fc.querySelector('.fcat-body');
        if (fc.classList.contains('open')) {
          fc.classList.remove('open');
          gsap.to(fb, { height: 0, opacity: 0, duration: 0.4, ease: 'power2.inOut',
            onComplete() { fb.style.display = 'none'; }
          });
        }
      });

      if (wasOpen) {
        fcat.classList.remove('open');
        gsap.to(body, { height: 0, opacity: 0, duration: 0.4, ease: 'power2.inOut',
          onComplete() { body.style.display = 'none'; }
        });
      } else {
        fcat.classList.add('open');
        body.style.display = 'block';
        body.style.height = '0px';
        body.style.opacity = '0';
        const autoH = body.scrollHeight;
        gsap.to(body, { height: autoH, opacity: 1, duration: 0.5, ease: 'power2.out',
          onComplete() { body.style.height = 'auto'; }
        });
      }
    });
  });

  // Init: close all except .open, and set their body display properly
  document.querySelectorAll('.fcat').forEach(fc => {
    const body = fc.querySelector('.fcat-body');
    if (!fc.classList.contains('open')) {
      body.style.display = 'none';
      body.style.height = '0px';
    }
  });
}

/* ═══ DONUT CHART ═══ */
function initDonut() {
  const canvas = document.getElementById('donutChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height, cx = w / 2, cy = h / 2, radius = 160, thick = 34;

  const segs = [
    { value: 84, color: '#10b981' },
    { value: 20, color: '#06b6d4' },
    { value: 387, color: '#8b5cf6' },
    { value: 17, color: '#f59e0b' }
  ];
  const total = segs.reduce((s, d) => s + d.value, 0);
  let prog = { val: 0 };

  function draw(p) {
    ctx.clearRect(0, 0, w, h);
    ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255,255,255,0.03)'; ctx.lineWidth = thick; ctx.stroke();

    let start = -Math.PI / 2;
    segs.forEach(s => {
      const sweep = (s.value / total) * Math.PI * 2 * p;
      ctx.beginPath(); ctx.arc(cx, cy, radius, start, start + sweep);
      ctx.strokeStyle = s.color; ctx.lineWidth = thick; ctx.lineCap = 'round';
      ctx.shadowColor = s.color; ctx.shadowBlur = 16; ctx.stroke(); ctx.shadowBlur = 0;
      start += sweep;
    });
  }

  ScrollTrigger.create({
    trigger: canvas, start: 'top 78%', once: true,
    onEnter() {
      gsap.to(prog, { val: 1, duration: 1.6, ease: 'power2.out', onUpdate: () => draw(prog.val) });
      const numEl = document.querySelector('.donut-num');
      if (numEl) gsap.to(numEl, { innerText: 508, duration: 2.2, snap: { innerText: 1 }, ease: 'power2.out' });
    }
  });
}

/* ═══ TIME SERIES + CROSSHAIR ═══ */
function initTimeSeries() {
  const canvas = document.getElementById('tsChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const pad = { top: 30, right: 20, bottom: 40, left: 50 };

  const series = {
    forest: { color: '#10b981', data: [] },
    cleared: { color: '#ef4444', data: [] },
    gradual: { color: '#f59e0b', data: [] }
  };

  for (let i = 0; i < 72; i++) {
    const s = 0.08 * Math.sin(i * Math.PI / 6);
    series.forest.data.push(0.82 + s + (Math.random() - 0.5) * 0.04);
    series.cleared.data.push(i < 36 ? 0.78 + s + (Math.random() - 0.5) * 0.05 : i < 42 ? 0.78 - (i - 36) * 0.08 : 0.25 + (Math.random() - 0.5) * 0.04);
    series.gradual.data.push(0.80 - i * 0.005 + s * 0.5 + (Math.random() - 0.5) * 0.04);
  }

  let curr = 'forest', dp = { val: 0 };
  let mouseX = -1, mouseY = -1;

  function draw(p) {
    ctx.clearRect(0, 0, w, h);
    const data = series[curr].data, color = series[curr].color;
    const pw = (w - pad.left - pad.right) / (data.length - 1);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)'; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (h - pad.top - pad.bottom) * i / 4;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
    }
    // Labels
    ctx.fillStyle = 'rgba(255,255,255,0.22)'; ctx.font = '11px Inter'; ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) ctx.fillText((1 - i * 0.25).toFixed(2), pad.left - 8, pad.top + (h - pad.top - pad.bottom) * i / 4 + 4);
    ctx.textAlign = 'center';
    for (let yr = 2020; yr <= 2025; yr++) ctx.fillText(yr + '', pad.left + (yr - 2020) * 12 * pw, h - 10);

    const n = Math.floor(data.length * p);
    if (n < 2) return;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = pad.left + i * pw, y = pad.top + (1 - data[i]) * (h - pad.top - pad.bottom);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color; ctx.lineWidth = 2.5;
    ctx.shadowColor = color; ctx.shadowBlur = 10; ctx.stroke(); ctx.shadowBlur = 0;

    const lastX = pad.left + (n - 1) * pw;
    const grad = ctx.createLinearGradient(0, pad.top, 0, h - pad.bottom);
    grad.addColorStop(0, color + '18'); grad.addColorStop(1, color + '00');
    ctx.lineTo(lastX, h - pad.bottom); ctx.lineTo(pad.left, h - pad.bottom); ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();

    if (curr === 'cleared' && p > 0.5) {
      const ex = pad.left + 36 * pw;
      ctx.strokeStyle = 'rgba(239,68,68,0.35)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(ex, pad.top); ctx.lineTo(ex, h - pad.bottom); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#ef4444'; ctx.font = 'bold 11px Inter'; ctx.textAlign = 'center';
      ctx.fillText('Clearing Event', ex, pad.top - 8);
    }

    // Crosshair
    if (mouseX >= pad.left && mouseX <= w - pad.right && p >= 1) {
      const idx = Math.round((mouseX - pad.left) / pw);
      if (idx >= 0 && idx < data.length) {
        const cx = pad.left + idx * pw;
        const cy = pad.top + (1 - data[idx]) * (h - pad.top - pad.bottom);

        // Vertical line
        ctx.strokeStyle = 'rgba(255,255,255,.15)'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(cx, pad.top); ctx.lineTo(cx, h - pad.bottom); ctx.stroke(); ctx.setLineDash([]);

        // Dot
        ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = color; ctx.fill();
        ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();

        // Tooltip
        const year = 2020 + Math.floor(idx / 12);
        const month = (idx % 12) + 1;
        const val = data[idx].toFixed(3);
        const label = `${year}-${String(month).padStart(2, '0')}  NDVI: ${val}`;

        ctx.font = 'bold 11px Inter';
        const tw = ctx.measureText(label).width + 16;
        const tx = Math.min(cx - tw / 2, w - pad.right - tw);
        const ty = cy - 30;

        ctx.fillStyle = 'rgba(0,0,0,.8)';
        ctx.beginPath();
        const rr = 6, rx = tx, ry = ty - 10, rw = tw, rh = 22;
        ctx.moveTo(rx + rr, ry); ctx.lineTo(rx + rw - rr, ry); ctx.quadraticCurveTo(rx + rw, ry, rx + rw, ry + rr);
        ctx.lineTo(rx + rw, ry + rh - rr); ctx.quadraticCurveTo(rx + rw, ry + rh, rx + rw - rr, ry + rh);
        ctx.lineTo(rx + rr, ry + rh); ctx.quadraticCurveTo(rx, ry + rh, rx, ry + rh - rr);
        ctx.lineTo(rx, ry + rr); ctx.quadraticCurveTo(rx, ry, rx + rr, ry); ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,.15)'; ctx.lineWidth = 1; ctx.stroke();

        ctx.fillStyle = '#fff'; ctx.textAlign = 'left';
        ctx.fillText(label, tx + 8, ty + 5);
      }
    }
  }

  ScrollTrigger.create({
    trigger: canvas, start: 'top 78%', once: true,
    onEnter() { gsap.to(dp, { val: 1, duration: 2, ease: 'power2.out', onUpdate: () => draw(dp.val) }); }
  });

  // Crosshair mouse tracking
  canvas.addEventListener('mousemove', e => {
    const r = canvas.getBoundingClientRect();
    mouseX = (e.clientX - r.left) * (w / r.width);
    mouseY = (e.clientY - r.top) * (h / r.height);
    if (dp.val >= 1) draw(1);
  });
  canvas.addEventListener('mouseleave', () => {
    mouseX = -1; mouseY = -1;
    if (dp.val >= 1) draw(1);
  });

  document.querySelectorAll('.ts-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.ts-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      curr = btn.dataset.series; dp.val = 0;
      gsap.to(dp, { val: 1, duration: 1.5, ease: 'power2.out', onUpdate: () => draw(dp.val) });
    });
  });
}

/* ═══ PIPELINE + DOT PULSE ═══ */
function initPipeline() {
  const fill = document.getElementById('pipeFill');
  if (!fill) return;

  gsap.to(fill, {
    height: '100%', ease: 'none',
    scrollTrigger: { trigger: '.pipe-track', start: 'top 60%', end: 'bottom 40%', scrub: 1 }
  });

  gsap.utils.toArray('.pipe-step').forEach(step => {
    const card = step.querySelector('.pipe-card');
    const fromX = step.classList.contains('right') ? 70 : -70;
    gsap.from(card, {
      x: fromX, opacity: 0, duration: 0.9, ease: 'power3.out',
      scrollTrigger: { trigger: step, start: 'top 82%', once: true }
    });
    const dot = step.querySelector('.pipe-dot');
    if (dot) {
      ScrollTrigger.create({
        trigger: step, start: 'top 62%', once: true,
        onEnter() {
          dot.classList.add('active');
          // Spawn pulse ring
          const ring = document.createElement('div');
          ring.style.cssText = 'position:absolute;inset:-6px;border-radius:50%;border:1.5px solid #10b981;pointer-events:none';
          dot.style.position = 'relative';
          dot.appendChild(ring);
          gsap.fromTo(ring,
            { scale: 1, opacity: 0.7 },
            { scale: 3.5, opacity: 0, duration: 1.2, ease: 'power2.out', onComplete() { ring.remove(); } }
          );
        }
      });
    }
  });
}

/* ═══ POST-PROCESSING CANVASES ═══ */
function initPostProcess() {
  const green = [16, 185, 129];

  // 1. Raw Probs — noise gradient
  const c1 = document.getElementById('ppProbs');
  if (c1) {
    const ctx = c1.getContext('2d');
    const w = c1.width, h = c1.height;
    const imgData = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const cx = w / 2, cy = h / 2;
        const dx1 = x - cx * 0.6, dy1 = y - cy * 0.8;
        const dist = Math.sqrt(dx1 * dx1 + dy1 * dy1) / (w * 0.4);
        const dx2 = x - cx * 1.4, dy2 = y - cy * 1.2;
        const dist2 = Math.sqrt(dx2 * dx2 + dy2 * dy2) / (w * 0.35);
        const prob = Math.max(0, Math.min(1, (1 - dist) * 0.8 + Math.random() * 0.3)) * 0.7 +
                     Math.max(0, Math.min(1, (1 - dist2) * 0.6 + Math.random() * 0.2)) * 0.3;
        const a = Math.max(0, Math.min(255, prob * 255));
        imgData.data[i] = green[0]; imgData.data[i+1] = green[1]; imgData.data[i+2] = green[2]; imgData.data[i+3] = a * 0.7;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  // 2. Binary threshold
  const c2 = document.getElementById('ppBinary');
  if (c2) {
    const ctx = c2.getContext('2d');
    const w = c2.width, h = c2.height;
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = `rgb(${green.join(',')})`;
    // Draw blocky shapes
    const shapes = [[15,20,45,35],[55,50,35,30],[20,65,25,20],[70,15,20,25]];
    shapes.forEach(([x, y, sw, sh]) => {
      ctx.fillRect(x * w / 120, y * h / 120, sw * w / 120, sh * h / 120);
    });
  }

  // 3. Morphology — smoothed
  const c3 = document.getElementById('ppMorph');
  if (c3) {
    const ctx = c3.getContext('2d');
    const w = c3.width, h = c3.height;
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = `rgb(${green.join(',')})`;
    ctx.beginPath();
    ctx.ellipse(w * 0.35, h * 0.4, w * 0.22, h * 0.18, -0.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(w * 0.7, h * 0.65, w * 0.16, h * 0.14, 0.3, 0, Math.PI * 2);
    ctx.fill();
  }

  // 4. Area filter — only large blob remains
  const c4 = document.getElementById('ppFilter');
  if (c4) {
    const ctx = c4.getContext('2d');
    const w = c4.width, h = c4.height;
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = `rgb(${green.join(',')})`;
    ctx.beginPath();
    ctx.ellipse(w * 0.4, h * 0.45, w * 0.24, h * 0.2, -0.15, 0, Math.PI * 2);
    ctx.fill();
  }

  // 4b. Erode — tightened boundaries (smaller than morphology output)
  const cErode = document.getElementById('ppErode');
  if (cErode) {
    const ctx = cErode.getContext('2d');
    const w = cErode.width, h = cErode.height;
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = `rgb(${green.join(',')})`;
    ctx.beginPath();
    ctx.ellipse(w * 0.35, h * 0.4, w * 0.17, h * 0.13, -0.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(w * 0.7, h * 0.65, w * 0.11, h * 0.09, 0.3, 0, Math.PI * 2);
    ctx.fill();
    // Show shrinkage indicator
    ctx.strokeStyle = 'rgba(16,185,129,0.25)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.ellipse(w * 0.35, h * 0.4, w * 0.22, h * 0.18, -0.2, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // 5. Vectorize — polygon outline
  const c5 = document.getElementById('ppVector');
  if (c5) {
    const ctx = c5.getContext('2d');
    const w = c5.width, h = c5.height;
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = `rgb(${green.join(',')})`;
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.beginPath();
    const pts = [[0.2, 0.25], [0.55, 0.2], [0.65, 0.35], [0.6, 0.6], [0.35, 0.65], [0.15, 0.5]];
    pts.forEach(([px, py], i) => {
      i === 0 ? ctx.moveTo(px * w, py * h) : ctx.lineTo(px * w, py * h);
    });
    ctx.closePath();
    ctx.stroke();
    // Fill with low opacity
    ctx.fillStyle = `rgba(${green.join(',')}, 0.15)`;
    ctx.fill();
    // Vertices
    ctx.fillStyle = `rgb(${green.join(',')})`;
    pts.forEach(([px, py]) => {
      ctx.beginPath();
      ctx.arc(px * w, py * h, 3, 0, Math.PI * 2);
      ctx.fill();
    });
    // Label
    ctx.fillStyle = 'rgba(255,255,255,.4)';
    ctx.font = '10px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('GeoJSON', w / 2, h - 10);
  }
}

/* ═══ SMOOTH ANCHORS ═══ */
function initSmoothAnchors() {
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();
      const href = a.getAttribute('href');
      if (href === '#') return;
      const target = document.querySelector(href);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

/* ═══ MUSIC PLAYER ═══ */
function initMusicPlayer() {
  const audio = document.getElementById('bgAudio');
  const toggle = document.getElementById('mpToggle');
  const playIcon = document.getElementById('mpPlayIcon');
  const pauseIcon = document.getElementById('mpPauseIcon');
  const vol = document.getElementById('mpVol');
  const viz = document.getElementById('mpViz');
  const titleEl = document.getElementById('mpTitle');
  const nextBtn = document.getElementById('mpNext');
  if (!audio || !toggle) return;

  const tracks = [
    { src: 'track-floating.mp3', title: 'Floating Cities', sub: 'Interstellar Vibes' },
    { src: 'track-ossuary.mp3', title: 'Ossuary 6 — Air', sub: 'Cinematic Ambient' },
    { src: 'track-martian.mp3', title: 'Martian Cowboy', sub: 'The Martian Vibes' }
  ];
  let trackIdx = 0;
  let isPlaying = false;

  audio.volume = 0.3;

  function loadTrack(idx) {
    trackIdx = idx % tracks.length;
    const t = tracks[trackIdx];
    audio.src = t.src;
    if (titleEl) titleEl.textContent = t.title;
    if (isPlaying) {
      audio.play().catch(() => {});
    }
  }

  // Auto-advance to next track when current ends
  audio.addEventListener('ended', () => {
    loadTrack(trackIdx + 1);
  });

  toggle.addEventListener('click', () => {
    if (audio.paused) {
      audio.play().then(() => {
        isPlaying = true;
        playIcon.style.display = 'none';
        pauseIcon.style.display = 'block';
        viz.classList.add('playing');
      }).catch(() => {});
    } else {
      audio.pause();
      isPlaying = false;
      playIcon.style.display = 'block';
      pauseIcon.style.display = 'none';
      viz.classList.remove('playing');
    }
  });

  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      loadTrack(trackIdx + 1);
    });
  }

  if (vol) {
    vol.addEventListener('input', () => {
      audio.volume = vol.value / 100;
    });
  }
}

/* ═══ INIT ═══ */
document.addEventListener('DOMContentLoaded', () => {
  initPreloader();
  initCursor();
  initGlobe();
  initNav();
  initScrollProgress();
  initBackToTop();
  initMagnetic();
  initTiltCards();
  initModelCardGlow();
  initHorizontalScroll();
  initDataTabs();
  initFeatureAccordion();
  initDonut();
  initTimeSeries();
  initPipeline();
  initPostProcess();
  initScrollAnimations();
  initSmoothAnchors();
  initMusicPlayer();
});
