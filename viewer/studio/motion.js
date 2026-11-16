// UniMate motion on a rig the person brings: a built-in example, an uploaded rigged
// GLB/glTF/FBX, or a path on this machine (a rigged GLB, or an already-adapted rig folder).
// This is the non-humanoid path: no auto-rig, UniMate adapts (rig_preprocess) and animates
// any skeleton that already has one. Create's "A rig" source runs it.

import { followJob, statusUrlFor, jobEnd } from './jobs.js';

const esc = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/**
 * Post a motion request to /api/motion/import (a JSON string or a FormData), draw progress
 * into `out` and hand the finished GLB to `onResult(url, name, path)`. Never throws:
 * failures are written into `out`.
 */
export async function runMotionImport({ body, out, onResult }) {
  out.hidden = false;
  out.innerHTML = '<div class="job"><div class="bigbar"><span></span></div><p class="job-msg">Starting…</p></div>';
  const bar = out.querySelector('.bigbar span'), msg = out.querySelector('.job-msg');
  let start;
  try {
    const response = await fetch('/api/motion/import', { method: 'POST',
      ...(typeof body === 'string' ? { headers: { 'Content-Type': 'application/json' }, body } : { body }) });
    start = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(start.error || `HTTP ${response.status}`);
  } catch (error) {
    out.innerHTML = `<p class="job-msg">${esc(error.message || 'Could not start.')}</p>`;
    return;
  }
  const job = followJob(statusUrlFor('motion', start), { interval: 1200, onUpdate: ({ event, percent }) => {
    msg.textContent = event.message || event.phase || 'Working…';
    if (percent != null) bar.style.width = `${percent}%`; else bar.classList.add('busy');
  } });
  const final = await job.done;
  if (final.status !== 'done') {
    out.innerHTML = `<p class="job-msg">${esc(jobEnd(final).why)}</p>`;
    return;
  }
  const done = final.last_event || {};
  const name = (done.path || '').split('/').pop() || 'motion';
  const videos = (done.media || []).filter((m) => m.endsWith('.mp4'));
  const files = (done.files || []).map((f) => ({ name: f.split('/').pop(), url: `/${f}` }));
  out.innerHTML = `<p class="job-msg">Done</p>` +
    videos.map((v) => `<video class="motion-video" controls muted loop src="${esc(`/${v}`)}"></video>`).join('') +
    `<div class="row">${files.map((f) => `<a class="ghost small" href="${esc(f.url)}" download="${esc(f.name)}">${esc(f.name)}</a>`).join('')}</div>`;
  onResult(done.result_url, name, done.path);
}
