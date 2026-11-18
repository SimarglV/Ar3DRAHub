// The studio's one 3D viewer: the same controls for every asset and every step.
// Wraps the shared model viewport; owns the render loop, axis map, tools and clip bar.

import * as THREE from 'three';
import { createModelViewport, disposeModelViewport } from '../components/model-viewport.js';
import { setCameraView } from '../components/camera-view-controls.js';
import { AnimationPlayer } from '../animation/player.js';
import { projectAxes, AXES } from './gizmo.js';
import { createBoneDisplay } from './bones.js';
import { createFloor, standOnFloor, homeView, facingAngle } from './floor.js';
import { applyLook, meshesOf, faceLabel } from './looks.js';
import { attachZoom } from './zoom.js';

const ICON = {
  reset: '<svg viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></svg>',
  wire: '<svg viewBox="0 0 24 24"><path d="M12 2 3 7v10l9 5 9-5V7z"/><path d="M3 7l9 5 9-5M12 12v10"/></svg>',
  grid: '<svg viewBox="0 0 24 24"><path d="M3 15l9 5 9-5M3 15l9-5 9 5M7.5 12.5l9 5M16.5 12.5l-9 5"/></svg>',
  clay: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><path d="M8 9a5 5 0 0 1 5-2"/></svg>',
  normal: '<svg viewBox="0 0 24 24"><path d="M4 18h16"/><path d="M12 18V5M8 9l4-4 4 4"/></svg>',
  albedo: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><path d="M12 4v8l6 6"/><circle cx="12" cy="12" r="1"/></svg>',
  normalmap: '<svg viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 10h16M4 16h16M10 4v16M16 4v16"/></svg>',
  bones: '<svg viewBox="0 0 24 24"><circle cx="12" cy="4" r="2"/><path d="M12 6v8M12 9l-5 3M12 9l5 3M12 14l-3 7M12 14l3 7"/></svg>',
  fix: '<svg viewBox="0 0 24 24"><path d="M14 4l6 6-9 9H5v-6z"/><path d="M12 6l6 6"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M7 4l13 8-13 8z"/></svg>',
  pause: '<svg viewBox="0 0 24 24"><path d="M7 5h3v14H7zM14 5h3v14h-3z"/></svg>',
};

export function set3dToolsMode(tools, is3d) {
  tools.hidden = !is3d;
  tools.inert = !is3d;
}

export function pictureOverlayTop(stageRect, headerRect, gap = 12) {
  return Math.max(24, Math.ceil(headerRect.bottom - stageRect.top + gap));
}

export function pictureMaxHeight(stageRect, headerRect, controlsHeight, captionHeight, gap = 10, breathing = 12) {
  const top = pictureOverlayTop(stageRect, headerRect);
  return Math.max(0, Math.floor(stageRect.height - top - controlsHeight - captionHeight - gap * 2 - breathing));
}

export function createStudioViewer({ stage, logoUrl, onClipChange, fixSession = null }) {
  const pane = document.createElement('div');
  pane.className = 'sv-pane';
  pane.innerHTML = '<span class="stats" hidden></span>';
  stage.appendChild(pane);

  const overlay = document.createElement('div');
  overlay.className = 'sv-overlay';
  stage.appendChild(overlay);
  const stageHeader = stage.querySelector?.('.hud.tl');
  const resizePictureOverlay = () => {
    if (!stageHeader?.getBoundingClientRect || !stage.getBoundingClientRect || !overlay.style?.setProperty) return;
    const stageRect = stage.getBoundingClientRect();
    const headerRect = stageHeader.getBoundingClientRect();
    const top = pictureOverlayTop(stageRect, headerRect);
    const controlsHeight = overlay.querySelector?.('.zoombar')?.getBoundingClientRect?.().height || 0;
    const captionHeight = overlay.querySelector?.('figcaption')?.getBoundingClientRect?.().height || 0;
    overlay.style.setProperty('--sv-picture-top', `${top}px`);
    overlay.style.setProperty('--sv-picture-max-height', `${pictureMaxHeight(stageRect, headerRect, controlsHeight, captionHeight)}px`);
  };
  const pictureLayoutObserver = stageHeader && typeof ResizeObserver !== 'undefined' ? new ResizeObserver(resizePictureOverlay) : null;
  pictureLayoutObserver?.observe(stageHeader);
  pictureLayoutObserver?.observe(stage);

  const tools = document.createElement('div');
  tools.className = 'sv-tools';
  tools.innerHTML = `
    <div class="sv-gizmo" aria-label="Look from an axis"><svg viewBox="-56 -56 112 112" aria-hidden="true"><g class="lines"></g></svg></div>
    <div class="sv-toolrow">
      <button class="tool" data-act="reset" data-tip="Reset view: back to the starting camera (Home)" aria-label="Reset view">${ICON.reset}</button>
      <button class="tool" data-act="wire" aria-pressed="false" data-tip="Wireframe: see the triangles" aria-label="Wireframe">${ICON.wire}</button>
      <button class="tool" data-act="grid" aria-pressed="true" data-tip="Floor grid and the origin (0,0,0): red line is X, blue is Z" aria-label="Floor grid">${ICON.grid}</button>
      <button class="tool" data-act="clay" aria-pressed="false" data-tip="Clay: plain grey, so you judge the shape without the paint" aria-label="Clay view">${ICON.clay}</button>
      <button class="tool" data-act="normal" aria-pressed="false" data-tip="Normals: the surface's direction as colour; dents, lumps and flipped faces stand out" aria-label="Normal view">${ICON.normal}</button>
      <button class="tool" data-act="normalmap" aria-pressed="false" data-tip="Normal map: the baked tangent-space surface detail (Marigold Finish adds one). Blue is flat, colour is relief" aria-label="Normal map view">${ICON.normalmap}</button>
      <button class="tool" data-act="albedo" aria-pressed="false" data-tip="Albedo: the raw base colour with the lights off, to see what is painted (and any shadows baked into the texture)" aria-label="Albedo view">${ICON.albedo}</button>
      <button class="tool" data-act="bones" aria-pressed="false" data-tip="Show skeleton: the bones the auto-rig placed" aria-label="Show skeleton">${ICON.bones}</button>
      <button class="tool" data-act="fix" aria-pressed="false" data-tip="Fix a spot: click a hole or a messy patch to smooth it over and repaint it from your picture. Size it with the slider or [ and ]. Saves a new version" aria-label="Fix a spot">${ICON.fix}</button>
    </div>`;
  stage.appendChild(tools);
  tools.hidden = true;

  const clipBar = document.createElement('div');
  clipBar.className = 'sv-clips';
  clipBar.hidden = true;
  stage.appendChild(clipBar);

  let view = null, player = null, skeleton = null, renderPending = false, loadToken = 0;
  let clips = [], clipIndex = -1;
  const gizmo = tools.querySelector('.sv-gizmo');
  const axisButtons = new Map(AXES.map((end) => {
    const button = document.createElement('button');
    button.className = `${end.axis}${end.key[0] === '-' ? ' neg' : ''} tip-left`;
    button.textContent = end.key[0] === '+' ? end.axis.toUpperCase() : '';
    button.dataset.tip = `Look from: ${end.label}`;
    button.setAttribute('aria-label', `Look from ${end.label}`);
    button.onclick = () => snap(end.view);
    gizmo.appendChild(button);
    return [end.key, button];
  }));

  function requestRender() {
    if (renderPending || !view) return;
    renderPending = true;
    requestAnimationFrame(renderFrame);
  }
  function renderFrame() {
    renderPending = false;
    if (!view) return;
    const moving = view.controls.update();
    skeleton?.update();
    view.renderer.render(view.scene, view.camera);
    drawGizmo();
    if (moving) requestRender();
  }
  function resize() {
    if (!view) return;
    const { clientWidth: w, clientHeight: h } = pane;
    if (!w || !h) return;
    view.renderer.setSize(w, h);
    view.camera.aspect = w / h;
    view.camera.updateProjectionMatrix();
    requestRender();
  }
  new ResizeObserver(resize).observe(pane);

  function drawGizmo() {
    const enabled = !!view?.root;
    gizmo.classList.toggle('off', !enabled);
    const direction = enabled
      ? view.camera.position.clone().sub(view.controls.target).normalize().toArray()
      : [0.5, 0.35, 0.8];
    const up = enabled ? view.camera.up.toArray() : [0, 1, 0];
    const ends = projectAxes(direction, up);
    const radius = 38;
    gizmo.querySelector('.lines').innerHTML = ends.filter((e) => e.key[0] === '+')
      .map((e) => `<line class="${e.axis}" x1="0" y1="0" x2="${e.x * radius}" y2="${e.y * radius}"/>`).join('');
    // Buttons are made once and only moved: rebuilding them every animation frame swallowed clicks.
    ends.forEach((end, order) => {
      const button = axisButtons.get(end.key);
      button.style.left = `${56 + end.x * radius}px`;
      button.style.top = `${56 + end.y * radius}px`;
      button.style.zIndex = String(order + 1);  // nearer ends on top
      button.disabled = !enabled;
    });
  }

  let home = homeView(1);
  function snap(name) {
    if (!view) return;
    if (name === 'reset') {
      // our own home: looking at the middle of a model that stands on the floor
      view.controls.target.set(...home.target);
      view.camera.position.set(...home.position);
      view.camera.up.set(0, 1, 0);
      view.camera.lookAt(view.controls.target);
      view.controls.update();
      requestRender();
      return;
    }
    if (setCameraView(view, name)) requestRender();
  }

  // one clay and one normal material for the viewer's life, shared by every mesh; albedo is
  // per-mesh, because it shows each mesh's own base-colour map with the lights off
  const looks = {
    clay: new THREE.MeshStandardMaterial({ color: 0x8a8580, roughness: 0.7, metalness: 0 }),
    normal: new THREE.MeshNormalMaterial(),
    albedo: (paint) => new THREE.MeshBasicMaterial({ map: paint?.map ?? null }),
    normalmap: (paint) => new THREE.MeshBasicMaterial({ map: paint?.normalMap ?? null }),
  };

  function setWireframe(on) {
    if (!view) return;
    for (const entry of view.materials) entry.original.wireframe = on;
    for (const material of Object.values(looks)) material.wireframe = on;
    requestRender();
  }

  function setLook(mode) {
    if (!view) return;
    applyLook(meshesOf(view.materials), mode, looks);
    requestRender();
  }

  function setSkeleton(on) {
    skeleton?.dispose();
    skeleton = null;
    if (on && view?.rig?.bones?.length) {
      skeleton = createBoneDisplay(view.rig.bones);
      view.scene.add(skeleton.group);
      skeleton.update();
    }
    requestRender();
  }

  tools.addEventListener('click', (event) => {
    const button = event.target.closest('[data-act]');
    if (!button) return;
    const act = button.dataset.act;
    if (act === 'reset') return snap('reset');
    const on = button.getAttribute('aria-pressed') !== 'true';
    if (act === 'bones' && on && !view?.rig?.bones?.length) {
      overlayNote('No skeleton yet: rig this model first.');
      return;
    }
    button.setAttribute('aria-pressed', String(on));
    if (act === 'clay' || act === 'normal' || act === 'normalmap' || act === 'albedo') {
      // one look at a time: pressing one lets the others go
      if (on) ['clay', 'normal', 'normalmap', 'albedo'].filter((a) => a !== act)
        .forEach((a) => tools.querySelector(`[data-act="${a}"]`).setAttribute('aria-pressed', 'false'));
      setLook(on ? act : 'paint');
    }
    if (act === 'wire') setWireframe(on);
    if (act === 'grid' && view?.floor) { view.floor.visible = on; requestRender(); }
    if (act === 'bones') setSkeleton(on);
    if (act === 'fix') {
      if (!on && session.preview && !confirm('Close the brush and throw away the patch you have not kept?')) {
        button.setAttribute('aria-pressed', 'true');
        return;
      }
      pane.classList.toggle('fixing', on);
      brushPanel.hidden = !on;
      if (!on) { hideRing(); endSession(true); }
      if (on) overlayNote('Mark the spots to fix, then Patch. Size: the slider, or [ and ].');
    }
  });

  // Fix a spot, as one session: clicks mark spots (rings that stay), Patch makes a preview,
  // Before / After flips between the model and the preview with the camera held still, Keep
  // saves one version and Discard drops it. The brush is a share of the model's height; the
  // server takes the same range (viewer/fixspot_api.py).
  const BRUSH_MIN = 0.005, BRUSH_MAX = 0.08;
  // Smooth's strength range matches the server (Ar3DRAHub/spot_fix.py)
  const STRENGTH_MIN = 0.1, STRENGTH_MAX = 1.0;
  let brushShare = 0.02, brushMode = 'fill', brushStrength = 0.6;
  const RING_COLOURS = { fill: 0xffffff, smooth: 0xff9a3c, repaint: 0x58c4ff };
  const brushPanel = document.createElement('div');
  brushPanel.className = 'sv-brush';
  brushPanel.hidden = true;
  brushPanel.innerHTML = `
    <div class="row"><span>Brush</span>
      <button class="chip" data-mode="fill" aria-pressed="true" data-tip="Fill hole: cut out a pocket or dent and patch it smooth">Fill hole</button>
      <button class="chip" data-mode="smooth" aria-pressed="false" data-tip="Smooth: iron out bumps, ridges and dents, then repaint from your picture; nothing is cut">Smooth</button>
      <button class="chip" data-mode="repaint" aria-pressed="false" data-tip="Repaint: paint only, from your picture; the shape stays as it is">Repaint</button></div>
    <label class="row"><span>Size</span><input type="range" min="${BRUSH_MIN}" max="${BRUSH_MAX}" step="0.0025" value="${brushShare}" aria-label="Brush size"><kbd>[ ]</kbd></label>
    <label class="row" data-strength hidden><span>Strength</span><input type="range" min="${STRENGTH_MIN}" max="${STRENGTH_MAX}" step="0.05" value="${brushStrength}" aria-label="Smooth strength"></label>
    <div class="row"><span data-count>Click to mark spots</span><button class="ghost" data-fix="undo" disabled>Undo</button>
      <button class="chip" data-fix="patch" disabled>Patch</button></div>
    <div class="row" data-review hidden><button class="ghost" data-fix="compare">Showing: after</button>
      <button class="chip" data-fix="keep">Keep</button><button class="ghost" data-fix="discard">Discard</button></div>`;
  tools.appendChild(brushPanel);
  const brushSlider = brushPanel.querySelector('input[aria-label="Brush size"]');
  const strengthRow = brushPanel.querySelector('[data-strength]');
  strengthRow.querySelector('input').addEventListener('input', (event) => { brushStrength = Number(event.target.value); });
  let ring = null, lastHit = null;
  const session = { marks: [], preview: null, beforeUrl: null, afterUrl: null, showing: 'after', busy: false };
  function setBrush(share) {
    brushShare = Math.min(BRUSH_MAX, Math.max(BRUSH_MIN, share));
    brushSlider.value = String(brushShare);
    if (lastHit) showRing(lastHit);
  }
  brushSlider.addEventListener('input', () => setBrush(Number(brushSlider.value)));
  brushPanel.querySelectorAll('[data-mode]').forEach((chip) => chip.addEventListener('click', () => {
    brushMode = chip.dataset.mode;
    strengthRow.hidden = brushMode !== 'smooth';
    brushPanel.querySelectorAll('[data-mode]').forEach((c) => c.setAttribute('aria-pressed', String(c === chip)));
  }));
  window.addEventListener('keydown', (event) => {
    if (brushPanel.hidden || event.target.closest?.('input, textarea, select')) return;
    if (event.key === '[') setBrush(brushShare / 1.25);
    if (event.key === ']') setBrush(brushShare * 1.25);
  });
  function ringMesh(color, opacity) {
    const mesh = new THREE.Mesh(new THREE.RingGeometry(0.86, 1, 64),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthTest: false, side: THREE.DoubleSide }));
    mesh.renderOrder = 999;
    return mesh;
  }
  function placeRing(mesh, hit, share) {
    const height = new THREE.Box3().setFromObject(view.root).getSize(new THREE.Vector3()).y;
    const normal = hit.face.normal.clone().transformDirection(hit.object.matrixWorld);
    mesh.position.copy(hit.point).addScaledVector(normal, height * 0.002);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
    mesh.scale.setScalar(share * height);
  }
  function hideRing() {
    pane.classList.remove('ringing');
    lastHit = null;
    if (ring) { ring.removeFromParent(); ring.geometry.dispose(); ring.material.dispose(); ring = null; requestRender(); }
  }
  function pick(event) {
    const rect = view.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, view.camera);
    const marks = new Set(session.marks.map((m) => m.mesh));
    return ray.intersectObject(view.root, true).find((h) => h.object.isMesh && h.object !== ring && !marks.has(h.object)) || null;
  }
  function showRing(hit) {
    if (!ring) { ring = ringMesh(0xff7a1a, 0.95); view.scene.add(ring); }
    placeRing(ring, hit, brushShare);
    pane.classList.add('ringing');
    lastHit = hit;
    requestRender();
  }
  function clearMarks() {
    for (const mark of session.marks) { mark.mesh.removeFromParent(); mark.mesh.geometry.dispose(); mark.mesh.material.dispose(); }
    session.marks = [];
  }
  function syncPanel() {
    const n = session.marks.length;
    brushPanel.querySelector('[data-count]').textContent = session.busy ? 'Patching…'
      : n ? `${n} spot${n === 1 ? '' : 's'} marked` : session.preview ? 'Mark more, or keep it' : 'Click to mark spots';
    brushPanel.querySelector('[data-fix="undo"]').disabled = !n || session.busy;
    brushPanel.querySelector('[data-fix="patch"]').disabled = !n || session.busy;
    brushPanel.querySelector('[data-fix="patch"]').textContent = n ? `Patch ${n}` : 'Patch';
    brushPanel.querySelector('[data-review]').hidden = !session.preview || session.busy;
    brushPanel.querySelector('[data-fix="compare"]').textContent = `Showing: ${session.showing}`;
    brushPanel.querySelector('[data-fix="compare"]').disabled = n > 0;   // patch or undo the new marks first
  }
  function endSession(discard) {
    if (discard && session.preview) fixSession?.discard(session.preview);
    clearMarks();
    Object.assign(session, { preview: null, beforeUrl: null, afterUrl: null, showing: 'after', busy: false });
    syncPanel();
    requestRender();
  }
  brushPanel.addEventListener('click', async (event) => {
    const act = event.target.closest('[data-fix]')?.dataset.fix;
    if (!act || !fixSession) return;
    if (act === 'undo') {
      const mark = session.marks.pop();
      mark?.mesh.removeFromParent();
      syncPanel();
      requestRender();
    }
    if (act === 'patch') {
      session.busy = true;
      syncPanel();
      const spots = session.marks.map(({ point, share, mode, strength }) => ({ point, radius: share, mode, strength }));
      try {
        const made = await fixSession.patch(spots, session.preview);
        if (!session.beforeUrl) session.beforeUrl = currentUrl;
        session.preview = made.preview;
        session.afterUrl = made.url;
        session.showing = 'after';
        clearMarks();
        load(made.url, { label: 'patched (not kept yet)', keepCamera: true });
      } catch (error) {
        overlayNote(String(error.message || error));
      }
      session.busy = false;
      syncPanel();
    }
    if (act === 'compare' && session.preview) {
      session.showing = session.showing === 'after' ? 'before' : 'after';
      load(session.showing === 'after' ? session.afterUrl : session.beforeUrl, { label: `showing: ${session.showing}`, keepCamera: true });
      syncPanel();
    }
    if (act === 'keep' && session.preview) {
      const preview = session.preview;
      endSession(false);
      await fixSession.keep(preview);
    }
    if (act === 'discard' && session.preview) {
      const before = session.beforeUrl;
      endSession(true);
      if (before) load(before, { label: 'as it was', keepCamera: true });
    }
  });
  pane.addEventListener('pointermove', (event) => {
    const fixing = tools.querySelector('[data-act="fix"]').getAttribute('aria-pressed') === 'true';
    if (!fixing || !view?.root || event.buttons) { if (!fixing) hideRing(); return; }
    const hit = pick(event);
    if (hit) showRing(hit); else hideRing();
  });
  pane.addEventListener('pointerleave', hideRing);
  let downAt = null;
  pane.addEventListener('pointerdown', (event) => { downAt = [event.clientX, event.clientY]; });
  pane.addEventListener('pointerup', (event) => {
    const fixing = tools.querySelector('[data-act="fix"]').getAttribute('aria-pressed') === 'true';
    if (!fixing || !view?.root || !downAt || session.busy) return;
    if (Math.hypot(event.clientX - downAt[0], event.clientY - downAt[1]) > 4) return;   // that was an orbit
    if (!fixSession?.canFix()) { overlayNote(fixSession?.blockedNote?.() ?? 'Fix a spot works on a finished model.'); return; }
    if (session.preview && session.showing === 'before') { overlayNote('Switch to "after" to mark more spots.'); return; }
    const hit = pick(event);
    if (!hit) { overlayNote('Missed the model: click right on it.'); return; }
    // the GLB's own coordinates: the mesh's local space (Finish writes no node transforms)
    const local = hit.object.worldToLocal(hit.point.clone());
    const mesh = ringMesh(RING_COLOURS[brushMode], 0.9);
    placeRing(mesh, hit, brushShare);
    view.scene.add(mesh);
    session.marks.push({ point: [local.x, local.y, local.z], share: brushShare, mode: brushMode, strength: brushStrength, mesh });
    syncPanel();
    requestRender();
  });

  let noteTimer = null;
  function overlayNote(text) {
    const note = document.createElement('div');
    note.className = 'sv-note';
    note.textContent = text;
    stage.appendChild(note);
    clearTimeout(noteTimer);
    noteTimer = setTimeout(() => note.remove(), 2400);
  }

  function setOverlay(content) {
    if (content && content.nodeType === 1) {
      overlay.innerHTML = '';
      overlay.appendChild(content);
    } else {
      overlay.innerHTML = content;
    }
    overlay.hidden = !content;
  }

  function dispose() {
    if (ring) { ring.geometry.dispose(); ring.material.dispose(); ring = null; }
    pane.querySelector('.stats').hidden = true;   // a picture or an empty stage has no face count
    player?.dispose?.();
    player = null;
    skeleton?.dispose();
    skeleton = null;
    handRig = null;   // its meshes are disposed with the scene; the reference must not dangle
    if (view) {
      disposeModelViewport(view);
      view.renderer.domElement.remove();
      view = null;
    }
  }

  function setToolsMode(is3d) {
    set3dToolsMode(tools, is3d);
  }

  let currentUrl = null;
  function load(url, { label = '', autoplay = false, keepCamera = false } = {}) {
    setToolsMode(true);
    const token = ++loadToken;
    const camera = keepCamera && view ? { position: view.camera.position.clone(), target: view.controls.target.clone() } : null;
    if (!keepCamera) endSession(false);
    currentUrl = url;
    dispose();
    setOverlay(`<div class="ph"><img src="${logoUrl}" alt=""><span>Loading model…</span></div>`);
    const created = createModelViewport({
      pane,
      spec: { url, ext: 'glb', label },
      onChange: requestRender,
      onLoaded: (loaded) => {
        if (token !== loadToken) return;
        setOverlay('');
        // turn a rigged character to face the camera (its feet say which way is forward), then
        // stand it on a floor with the origin marked, so it is not floating in limbo
        loaded.root.rotation.y += facingAngle(facing);
        const height = standOnFloor(loaded.root);
        home = homeView(height);
        loaded.floor = createFloor();
        loaded.floor.visible = tools.querySelector('[data-act="grid"]').getAttribute('aria-pressed') === 'true';
        loaded.scene.add(loaded.floor);
        snap('reset');
        if (camera) {   // Before / After: the same view, so the change is what moves
          view.camera.position.copy(camera.position);
          view.controls.target.copy(camera.target);
          view.controls.update();
        }
        if (loaded.rig.animations.length) {
          player = new AnimationPlayer(new THREE.AnimationMixer(loaded.modelRoot), {
            onFrame: (time, duration) => { requestRender(); syncScrub(time, duration); },
            onStateChange: (playing) => syncPlayButton(playing),
          });
          player.select(loaded.rig.animations[0]);
          player.action.setLoop(THREE.LoopRepeat, Infinity);
          if (autoplay) player.play();
        }
        // keep wireframe and skeleton on across models and moves
        const pressed = (act) => tools.querySelector(`[data-act="${act}"]`).getAttribute('aria-pressed') === 'true';
        if (pressed('wire')) setWireframe(true);
        if (pressed('clay')) setLook('clay');
        if (pressed('normal')) setLook('normal');
        const stats = pane.querySelector('.stats');
        stats.textContent = faceLabel(stats.textContent);
        stats.hidden = !stats.textContent;
        if (pressed('bones')) {
          if (loaded.rig.bones.length) setSkeleton(true);
          else tools.querySelector('[data-act="bones"]').setAttribute('aria-pressed', 'false');
        }
        resize();
        drawGizmo();
      },
      onError: (error) => {
        if (token !== loadToken) return;
        const message = String(error?.message || error).slice(0, 140);
        const placeholder = document.createElement('div');
        placeholder.className = 'ph';
        const img = document.createElement('img');
        img.src = logoUrl;
        img.alt = '';
        placeholder.appendChild(img);
        const line = document.createElement('span');
        line.textContent = `This model didn't load: ${message}`;
        placeholder.appendChild(line);
        setOverlay(placeholder);
      },
    });
    view = created;
    resize();
  }

  /** One picture, or a front and back pair side by side (`back` given). */
  // `labels` names the two pictures: front and back by default, original and new for a skin
  function showImage(url, caption, back = null, labels = ['Front', 'Back']) {
    setToolsMode(false);
    ++loadToken;
    dispose();
    const pictures = back
      ? `<div class="pair"><div><div class="zoombox"><img src="${url}" alt="${labels[0]}"></div><span>${labels[0]}</span></div><div><div class="zoombox"><img src="${back}" alt="${labels[1]}"></div><span>${labels[1]}</span></div></div>`
      : `<div class="zoombox"><img src="${url}" alt=""></div>`;
    setOverlay(`<figure class="sv-picture${back ? ' two' : ''}">${pictures}<div class="zoombar" role="group" aria-label="Zoom"></div><figcaption>${caption} <span class="zoomhint">Pick a zoom, or scroll over a picture; drag to move, double-click to reset.</span></figcaption></figure>`);
    resizePictureOverlay();
    attachZoom(overlay);
    resizePictureOverlay();
    drawGizmo();
    clipBar.hidden = true;
  }

  function showEmpty(text) {
    setToolsMode(false);
    ++loadToken;
    dispose();
    setOverlay(`<div class="ph"><img src="${logoUrl}" alt=""><span>${text}</span></div>`);
    drawGizmo();
    clipBar.hidden = true;
  }

  // ------------------------------------------------------------------ clips
  function syncPlayButton(playing) {
    const button = clipBar.querySelector('[data-clip-act="play"]');
    if (button) {
      button.innerHTML = playing ? ICON.pause : ICON.play;
      button.dataset.tip = playing ? 'Pause' : 'Play';
    }
  }

  /** Show a row of moves (or, with `still`, of props). `items` are {name, url}. */
  // which way the next rigged character's feet point; null leaves models as they come
  let facing = null;
  function setFacing(forward) { facing = forward || null; }

  function setClips(items, { restUrl = null, index = 0, still = false } = {}) {
    clips = items;
    clipBar.hidden = !items.length && !restUrl;
    if (clipBar.hidden) { clipBar.innerHTML = ''; return; }
    clipIndex = items.length ? index : -1;
    clipBar.innerHTML = (still ? '' : `
      <button class="tool" data-clip-act="play" data-tip="Play or pause" aria-label="Play or pause">${ICON.pause}</button>
      <button class="ghost" data-clip-act="rest" data-tip="Reset pose: stop and show the character in its rest pose">Reset pose</button>
      <span class="sep"></span>`) + `
      ${items.map((clip, i) => `<button class="chip" data-clip="${i}" aria-pressed="${i === clipIndex}">${clip.name}</button>`).join('')}`
      + (still ? '' : `<span class="scrubber"><input type="range" class="scrub" data-scrub min="0" max="1000" value="0" aria-label="Scrub through the move"><span class="meta frame" data-frame>frame 0</span></span>`);
    if (clipIndex >= 0) onClipChange?.(items[clipIndex]);
    clipBar.dataset.rest = restUrl || '';
    if (clipIndex >= 0) load(items[clipIndex].url, { label: items[clipIndex].name, autoplay: true });
    else if (restUrl) load(restUrl, { label: 'rest pose' });
  }

  // the scrubber: shows where the move is; dragging pauses and moves to that moment
  const FPS = 30;
  function syncScrub(time = 0, duration = 0) {
    const scrub = clipBar.querySelector('[data-scrub]'), label = clipBar.querySelector('[data-frame]');
    if (!scrub || scrub.matches(':active')) return;
    scrub.value = duration ? String(Math.round((time / duration) * 1000)) : '0';
    label.textContent = `frame ${Math.round(time * FPS)} / ${Math.round(duration * FPS)}`;
  }
  clipBar.addEventListener('input', (event) => {
    if (!event.target.matches('[data-scrub]') || !player) return;
    player.pause();
    syncPlayButton(false);
    const t = (Number(event.target.value) / 1000) * player.duration;
    player.seek(t);
    clipBar.querySelector('[data-frame]').textContent = `frame ${Math.round(t * FPS)} / ${Math.round(player.duration * FPS)}`;
    requestRender();
  });

  clipBar.addEventListener('click', (event) => {
    const chip = event.target.closest('[data-clip]');
    if (chip) {
      clipIndex = Number(chip.dataset.clip);
      clipBar.querySelectorAll('[data-clip]').forEach((b) => b.setAttribute('aria-pressed', String(b === chip)));
      load(clips[clipIndex].url, { label: clips[clipIndex].name, autoplay: true });
      onClipChange?.(clips[clipIndex]);
      return;
    }
    const act = event.target.closest('[data-clip-act]')?.dataset.clipAct;
    if (act === 'play' && player) player.playing ? player.pause() : player.play();
    if (act === 'rest') {
      player?.pause();
      view?.resetPose();
      requestRender();
      syncPlayButton(false);
    }
  });

  window.addEventListener('keydown', (event) => {
    if (event.target.closest?.('input, textarea, select, [contenteditable]')) return;
    const keys = { 1: 'front', 3: 'right', 7: 'top', Home: 'reset' };
    if (keys[event.key]) snap(keys[event.key]);
  });

  // ------------------------------------------------------------------ rig by hand
  // A skeleton placed by hand: the model goes translucent and the 52 joints are draggable
  // spheres joined by bone lines. Joint fractions travel as {fx left/right, fy front/back,
  // fz up} in the same Blender-up space the server's human_rig module uses; here we convert
  // to the GLB's own glTF space (x right, y up, z front) only to place and read the markers.
  // Bones are coloured by part (head pink, spine green, arms amber, hands orange, legs blue);
  // a dragged joint rides a plane facing the camera, so it moves even off the model. Click a
  // joint to select it (shift-click to add more); dragging moves the whole selection. Selected
  // joints get a white outline. Name tags show on the selected and the hovered joint by default;
  // the "All names" toggle shows every one.
  let handRig = null;   // { group, joints, order, min, size, midX, labelLift, defaults, mirrorOn, showAllNames, hovered, selection, undo }
  let handRigWired = false;   // the drag listeners are attached once, on the first session
  const handRay = new THREE.Raycaster();
  const toGltf = ([fx, fy, fz]) => [fx, fz, 1 - fy];
  const toBlender = ([gx, gy, gz]) => [gx, 1 - gz, gy];
  const GROUP_COLOUR = { head: 0xe64f9a, spine: 0x7ac943, arm: 0xffc107, hand: 0xff6b35, leg: 0x3d8bfd, other: 0xff7a1a };
  const SELECT_COLOUR = 0xffffff;

  // One white ring texture for every selected joint's outline (billboarded, so it always
  // faces the camera). Shared across joints and sessions: it is 64 px and never disposed.
  const RING_TEXTURE = (() => {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 64;
    const c = canvas.getContext('2d');
    c.strokeStyle = '#ffffff';
    c.lineWidth = 10;
    c.beginPath();
    c.arc(32, 32, 24, 0, Math.PI * 2);
    c.stroke();
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    return texture;
  })();

  function boneLabel(name) {
    return name.replace('ForeArm', 'Forearm').replace('UpLeg', 'Thigh').replace('ToeBase', 'Toe');
  }

  function makeLabelSprite(text) {
    const fontPx = 44;
    const canvas = document.createElement('canvas');
    const measure = canvas.getContext('2d');
    measure.font = `600 ${fontPx}px system-ui, -apple-system, "Segoe UI", sans-serif`;
    const padX = fontPx * 0.45, padY = fontPx * 0.22;
    const textW = measure.measureText(text).width;
    canvas.width = Math.max(1, Math.ceil(textW + padX * 2));
    canvas.height = Math.max(1, Math.ceil(fontPx * 1.2 + padY * 2));
    const c = canvas.getContext('2d');
    c.font = `600 ${fontPx}px system-ui, -apple-system, "Segoe UI", sans-serif`;
    c.textBaseline = 'middle';
    c.fillStyle = 'rgba(16, 13, 12, 0.82)';
    if (c.roundRect) { c.beginPath(); c.roundRect(0, 0, canvas.width, canvas.height, canvas.height / 2); c.fill(); }
    else c.fillRect(0, 0, canvas.width, canvas.height);
    c.fillStyle = '#ece7e1';
    c.fillText(text, padX, canvas.height / 2);
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    texture.generateMipmaps = false;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true }));
    sprite.renderOrder = 1001;
    sprite.userData.aspect = canvas.width / canvas.height;
    return sprite;
  }

  function makeBoneLine(from, to, color) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.75, depthTest: false }));
    line.renderOrder = 998;
    writeBoneLine(line, from, to);
    return line;
  }
  function writeBoneLine(line, from, to) {
    const pos = line.geometry.attributes.position.array;
    pos[0] = from.x; pos[1] = from.y; pos[2] = from.z;
    pos[3] = to.x; pos[4] = to.y; pos[5] = to.z;
    line.geometry.attributes.position.needsUpdate = true;
  }
  function refreshBones(name) {
    const joint = handRig.joints.get(name);
    if (joint.line) writeBoneLine(joint.line, handRig.joints.get(joint.parent).mesh.position, joint.mesh.position);
    for (const child of joint.children) {
      const c = handRig.joints.get(child);
      writeBoneLine(c.line, joint.mesh.position, c.mesh.position);
    }
  }
  function syncJointDecorations(name) {
    const joint = handRig.joints.get(name);
    const p = joint.mesh.position;
    joint.label.position.set(p.x, p.y + handRig.labelLift, p.z);
    joint.ring.position.copy(p);
  }
  function placeJoint(name, pos, moved) {
    const joint = handRig.joints.get(name);
    if (moved.has(name)) return;
    moved.add(name);
    joint.mesh.position.copy(pos);
    syncJointDecorations(name);
    refreshBones(name);
    const mirrorName = joint.mirror;
    if (handRig.mirrorOn && mirrorName && !handRig.selection.has(mirrorName) && !moved.has(mirrorName)) {
      const twin = handRig.joints.get(mirrorName);
      moved.add(mirrorName);
      twin.mesh.position.set(handRig.midX * 2 - pos.x, pos.y, pos.z);
      syncJointDecorations(mirrorName);
      refreshBones(mirrorName);
    }
  }
  function syncJointSelection(name) {
    const joint = handRig.joints.get(name);
    const selected = handRig.selection.has(name);
    joint.mesh.material.color.set(selected ? SELECT_COLOUR : joint.baseColor);
    joint.mesh.scale.setScalar(selected ? 1.35 : 1);
    joint.ring.visible = selected;
    if (joint.line) joint.line.material.color.set(selected ? SELECT_COLOUR : joint.baseColor);
  }
  function syncAllSelection() {
    for (const name of handRig.order) syncJointSelection(name);
  }
  function jointLabelVisible(name) {
    return handRig.showAllNames || handRig.selection.has(name) || handRig.hovered === name;
  }
  function syncJointLabel(name) {
    handRig.joints.get(name).label.visible = jointLabelVisible(name);
  }
  function syncAllLabels() {
    for (const name of handRig.order) syncJointLabel(name);
  }
  function setHover(name) {
    if (!handRig || handRig.hovered === name) return;
    const prev = handRig.hovered;
    handRig.hovered = name;
    if (prev) syncJointLabel(prev);
    if (name) syncJointLabel(name);
    requestRender();
  }
  function selectOnly(name) {
    handRig.selection.clear();
    handRig.selection.add(name);
    syncAllSelection();
    syncAllLabels();
    requestRender();
  }
  function toggleSelection(name) {
    if (handRig.selection.has(name)) handRig.selection.delete(name); else handRig.selection.add(name);
    syncAllSelection();
    syncAllLabels();
    requestRender();
  }
  function clearSelection() {
    if (!handRig.selection.size) return;
    handRig.selection.clear();
    syncAllSelection();
    syncAllLabels();
    requestRender();
  }

  function enterHandRig(url, joints) {
    exitHandRig();
    setToolsMode(true);
    const token = ++loadToken;
    currentUrl = url;
    dispose();
    setOverlay(`<div class="ph"><img src="${logoUrl}" alt=""><span>Loading model…</span></div>`);
    const created = createModelViewport({
      pane,
      spec: { url, ext: 'glb', label: 'rig by hand' },
      onChange: requestRender,
      onLoaded: (loaded) => {
        if (token !== loadToken) return;
        setOverlay('');
        // raw glTF space: no facing rotation and no floor, so fractions match the server's bbox
        const box = new THREE.Box3().setFromObject(loaded.root);
        const min = box.min, size = box.getSize(new THREE.Vector3());
        const translucent = [];
        for (const entry of loaded.materials) {
          const m = entry.original;
          m.transparent = true; m.opacity = 0.35; m.depthWrite = false; m.needsUpdate = true;
          translucent.push(m);
        }
        const group = new THREE.Group();
        loaded.scene.add(group);
        const at = (joint) => {
          const [gx, gy, gz] = toGltf(joint.position);
          return new THREE.Vector3(min.x + gx * size.x, min.y + gy * size.y, min.z + gz * size.z);
        };
        const radius = size.length() * 0.012;
        const labelLift = radius * 2.4, labelHeight = radius * 3.1;
        const ringMaterial = new THREE.SpriteMaterial({ map: RING_TEXTURE, depthTest: false, transparent: true });
        const jointMap = new Map();
        const defaults = new Map();
        for (const joint of joints) {
          const pos = at(joint);
          defaults.set(joint.name, pos.clone());
          const baseColor = GROUP_COLOUR[joint.group] || GROUP_COLOUR.other;
          const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 16, 16),
            new THREE.MeshBasicMaterial({ color: baseColor, depthTest: false, transparent: true, opacity: 0.95 }));
          mesh.position.copy(pos);
          mesh.renderOrder = 999;
          mesh.userData.joint = joint.name;
          group.add(mesh);
          const label = makeLabelSprite(boneLabel(joint.name));
          label.scale.set(labelHeight * label.userData.aspect, labelHeight, 1);
          label.position.set(pos.x, pos.y + labelLift, pos.z);
          label.userData.joint = joint.name;   // hovering or clicking the tag acts on the joint
          group.add(label);
          const ring = new THREE.Sprite(ringMaterial);
          ring.scale.setScalar(radius * 4.5);
          ring.position.copy(pos);
          ring.renderOrder = 1002;
          ring.visible = false;
          group.add(ring);
          jointMap.set(joint.name, { name: joint.name, group: joint.group, mirror: joint.mirror || null,
            parent: joint.parent || null, baseColor, mesh, line: null, label, ring, children: [] });
        }
        for (const [name, joint] of jointMap) {
          if (!joint.parent) continue;
          const parent = jointMap.get(joint.parent);
          parent.children.push(name);
          const line = makeBoneLine(parent.mesh.position, joint.mesh.position, joint.baseColor);
          group.add(line);
          joint.line = line;
        }
        handRig = { group, joints: jointMap, order: joints.map((j) => j.name), min, size,
                    midX: min.x + size.x * 0.5, labelLift, defaults, translucent,
                    mirrorOn: false, showAllNames: false, hovered: null, selection: new Set(), undo: [] };
        syncAllLabels();
        wireHandRigDrag();
        snap('reset');
        resize();
        requestRender();
      },
      onError: (error) => {
        if (token !== loadToken) return;
        setOverlay(`<div class="ph"><img src="${logoUrl}" alt=""><span>${String(error?.message || error).slice(0, 140)}</span></div>`);
      },
    });
    view = created;
    resize();
  }

  function exitHandRig() {
    if (!handRig) return;
    // The model went translucent for the skeleton to show through: make it opaque again.
    for (const material of handRig.translucent || []) {
      material.transparent = false; material.opacity = 1; material.depthWrite = true; material.needsUpdate = true;
    }
    handRig.group.traverse((obj) => { obj.geometry?.dispose?.(); obj.material?.dispose?.(); });
    handRig.group.removeFromParent();
    handRig = null;
    requestRender();
  }

  function collectHandRigJoints() {
    if (!handRig) return null;
    const out = {};
    for (const [name, joint] of handRig.joints) {
      const p = joint.mesh.position;
      const [fx, fy, fz] = toBlender([
        (p.x - handRig.min.x) / handRig.size.x,
        (p.y - handRig.min.y) / handRig.size.y,
        (p.z - handRig.min.z) / handRig.size.z,
      ]);
      out[name] = [Math.round(fx * 10000) / 10000, Math.round(fy * 10000) / 10000, Math.round(fz * 10000) / 10000];
    }
    return out;
  }

  function wireHandRigDrag() {
    if (handRigWired) return;   // the listeners read handRig and view through the closure, so one set is enough
    handRigWired = true;
    const toNdc = (event) => {
      const rect = pane.getBoundingClientRect();
      return new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1);
    };
    const pickMarker = (event) => {
      if (!handRig || !view) return null;
      handRay.setFromCamera(toNdc(event), view.camera);
      const hit = handRay.intersectObjects(handRig.group.children, false)
        .find((h) => h.object.userData.joint);
      const name = hit?.object.userData.joint;
      return name ? handRig.joints.get(name).mesh : null;   // the sphere, even when the tag was hit
    };
    const snapshot = () => new Map([...handRig.joints].map(([name, joint]) => [name, joint.mesh.position.clone()]));
    let dragging = null;   // { plane, start, snap }
    const camDir = new THREE.Vector3(), target = new THREE.Vector3();
    pane.addEventListener('pointerdown', (event) => {
      if (!view) return;
      const marker = pickMarker(event);
      const additive = event.shiftKey || event.ctrlKey || event.metaKey;
      if (!marker) { if (!additive) clearSelection(); return; }
      const name = marker.userData.joint;
      if (additive) { toggleSelection(name); return; }
      if (!handRig.selection.has(name)) selectOnly(name);
      handRig.undo.push(snapshot());
      if (handRig.undo.length > 100) handRig.undo.shift();
      view.camera.getWorldDirection(camDir);
      dragging = {
        plane: new THREE.Plane(camDir, -camDir.dot(marker.position)),
        start: marker.position.clone(),
        snap: new Map([...handRig.selection].map((n) => [n, handRig.joints.get(n).mesh.position.clone()])),
      };
      if (view.controls) view.controls.enabled = false;
      event.stopPropagation();
    });
    pane.addEventListener('pointermove', (event) => {
      if (!handRig || !view) return;
      const hoverMarker = pickMarker(event);
      setHover(hoverMarker ? hoverMarker.userData.joint : null);
      if (!dragging) return;
      if (handRay.ray.intersectPlane(dragging.plane, target)) {
        const delta = target.clone().sub(dragging.start);
        const moved = new Set();
        for (const name of handRig.selection) {
          placeJoint(name, dragging.snap.get(name).clone().add(delta), moved);
        }
        requestRender();
      }
    });
    pane.addEventListener('pointerleave', () => setHover(null));
    window.addEventListener('pointerup', () => {
      if (!dragging) return;
      dragging = null;
      if (view?.controls) view.controls.enabled = true;
    });
  }

  function handRigMirror(on) {
    if (handRig) handRig.mirrorOn = !!on;
  }

  function handRigSetNames(on) {
    if (!handRig) return;
    handRig.showAllNames = !!on;
    syncAllLabels();
    requestRender();
  }

  function handRigUndo() {
    if (!handRig) return;
    const snap = handRig.undo.pop();
    if (!snap) return;
    for (const [name, pos] of snap) handRig.joints.get(name).mesh.position.copy(pos);
    for (const name of handRig.order) { syncJointDecorations(name); refreshBones(name); }
    requestRender();
  }

  function handRigReset() {
    if (!handRig) return;
    for (const [name, joint] of handRig.joints) joint.mesh.position.copy(handRig.defaults.get(name));
    for (const name of handRig.order) { syncJointDecorations(name); refreshBones(name); }
    requestRender();
  }

  drawGizmo();
  return { load, showImage, showEmpty, setClips, setFacing, snap, dispose,
           enterHandRig, exitHandRig, collectHandRigJoints,
           handRigMirror, handRigUndo, handRigReset, handRigSetNames };
}
