# Changelog

All notable changes to the studio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **The project is MIT.** A root `LICENSE` is added and every model record, backend card,
  bootstrap text and document now names MIT instead of the upstream terms. Model licences
  are no longer carried in the provenance profiles: each profile is `commercial-clear`, so
  nothing is refused for being conditional, and the records keep only the useful conditions
  (the DINOv3 use bans, the Qwen credit, Hi3D's scale notice). `output/` no longer sorts a
  run by licence class — everything lands in the commercial folder.

### Fixed
- **Marigold albedo left the old preview in place**: the de-lighted picture replaces the front
  under the same name, so the Library thumbnail and the picture view kept showing the cached
  one. The thumbnail URL now carries the asset's timestamp, and the de-light forces the
  picture view to re-fetch it.
- **Multi-view back and sides mirrored instead of turning**: "Turn to back view" and
  "Change the perspective to the left side view" read to the model as "keep it roughly the
  same", so the back kept the front's palms and the sides kept the front's pose. The prompts
  now name the view directly — "Back view", "left side view", "right side view" — the form
  a Flux multi-view rig proved works (`Ar3DRAHub/back_view.py`).
- **UltraTex choice never reached the finish job**: the viewer's finish-settings whitelist
  (`activity_api.FINISH_KEYS`) lacked `texture`/`ultratex_steps`/`ultratex_seed`, so a run
  requested with `texture=ultratex` (from the UI or an agent) silently ran the plain finish.
  The keys now pass through, and a regression test pins it.

### Changed
- **Back and side pictures are drawn with Flux by default**: the studio and
  `scripts/make_back_view.py` now draw the back and both sides with FLUX.2 [klein] 4B
  (Apache-2.0) instead of Qwen-Image, which needs its vision projector. `--engine qwen`
  (CLI) or picking Qwen keeps the old route.
- **The side views can be drawn whenever a picture engine can draw them**: the Steps panel
  no longer gates "Draw the left/right side" on Pixal3D's multi-view weights — Flux (or Qwen
  with its vision projector) alone is enough, and the model step still asks for the weights
  when it builds from them. The side drawing also gained its own picture-engine choice, so
  every view can be drawn with whichever installed model you want, like the back already had.
- **Finish settings now relate to each other**: Marigold de-lights the photo paint before
  the UltraTex repaint runs (the generated paint is never de-lit); Marigold normals and the
  Pixel Match switches (edge fade, nudge, fade normals) nest under their parent in the
  panel and are dropped by the server when the parent is off; a geometry-only Hi3D refine
  (no texture transfer) keeps the UltraTex repaint off, since there is no texture to
  repaint onto. The panel groups them so a sub-knob cannot pretend its parent is on.
- **Machine presets now cover every process**: beyond the picture engine, back view,
  finish detail, texture size and motion effort, each preset now also carries the finish
  quality extras — Marigold de-lighting (+ normals), Hi3D refine (+ resolution and texture
  transfer), the UltraTex repaint (+ steps and the 12-view guided pass) and the watertight
  rebuild. Minimal and Normal keep everything off; High adds Marigold and Hi3D 512; Ultra
  turns the full ladder on. An explicitly set knob always wins, off extras send nothing,
  and the server's own guards still apply.

### Added
- **Dropped picture models** (`models/pictures/`): drop a GGUF diffusion model, its text
  encoder and its VAE into `models/pictures/<name>/` with a small `model.json` that names the
  three files, and the model appears in the picture-engine picker everywhere — Create, back
  and side redraws, the brush. Its licence is whatever the manifest declares, or "verify it
  yourself" when it says nothing, and that travels with every picture's record. See
  `docs/custom-models.md`.
- **Diffusers picture runtime** (`scripts/bootstrap_diffusers.py`,
  `scripts/diffusers_generate.py`): draws pictures from raw `.safetensors`/`.ckpt` checkpoints
  (or diffusers folders) dropped into `models/pictures/`, through a torch + diffusers venv —
  the same checkpoints ComfyUI and Stable Diffusion WebUI load. Text-to-image, image-to-image
  (the back and side views) and inpainting (the brush); a model's manifest may swap in its own
  VAE, text encoder and (for SDXL / FLUX) second text encoder. The Setup card installs only
  the software, never model weights, and loading is pinned offline.
- **Marigold albedo as a picture post-process**: the Picture step gains "Marigold albedo",
  which de-lights a view through Marigold V2 (its own venv, installed from Setup & Status)
  and makes the de-lit result that view — every view the asset has is offered (front, then
  the drawn back, left and right), the old one is kept beside it, and the model can be built
  again from the de-lit views. The same post-process accepts `normals` and `depth` through
  `POST /api/image/marigold`. Finish's capabilities now report whether Marigold V2 is
  installed, so the button shows only when it can run.
- **UltraTex in the studio UI**: a Setup & Status card (install/remove, the four weight
  sets with sizes and the non-commercial caveat) and Finish-panel controls — "UltraTex
  repaint" with "12-view guided" and the steps pick — gated on the server's capabilities.
  Until now the texture choice existed only for agents; a person had to run the bootstrap
  by hand.
- **UltraTex 12-view guided pass** (`ultratex_guided`): a second albedo run over six
  45-degree offset views (four diagonal sides + two 45-degree elevated), baked together
  with the canonical six so more of the mesh takes the generated paint. The offset poses
  are the dataset's own camera family, not an invention: the ring formula reproduces the
  four canonical side poses verbatim, and the elevated tilt reproduces top/bottom at 90°.
- **UltraTex fast path**: the empty prompt is now encoded once before the 9B denoiser
  loads and the Qwen text encoder is released right after, so it never shares VRAM with
  the big model — and CFG doubling is skipped (one forward per step instead of two, exact
  for empty prompts). `scripts/patch_ultratex_speed.py` keeps the checkout patched and the
  backend replays it before every run.
- **Watertight rebuild, external engine** (`--tool wativo`): WTiVo can do the rebuild
  instead of the built-in module — much faster, because it runs on the GPU. The studio adds
  a second checkbox, "Faster rebuild", beside "Watertight", and says nothing about licences
  in the interface: it is a speed choice, and the record carries the rest. The sidecar gains
  an `external_tools` entry naming the tool and the GPL its native extension brings (via
  CGAL); the result still keeps the input's licence and classification, because the tool
  ran and relicensed nothing. Its own card in **Setup** installs it into `vendor/wativo`
  with a venv of its own — its extension is CPython 3.12 with torch 2.8.0, which the
  studio's 3.11 environment cannot host — and the Watertight block in Finish grows a
  "Faster rebuild" checkbox once it is there, with a detail pick of its own — 512 (fast,
  a lighter file) or 1536 (the tool's full detail, a much bigger file) — instead of
  mapping the CPU voxel knob onto it. The studio and the command line find the same
  install (`vendor/wativo`, then `$WTIVO_HOME`/`$WTIVO_DIR`, with `$WTIVO_PYTHON` to
  override the interpreter), and a wrong-torch venv is refused by a real extension
  import, not a version print; without an install the run says so and stops rather than
  falling back silently. Whatever WTiVo leaves detached (at 512: three 12-face specks)
  is dropped by the same edge-connectivity rule the CPU engine uses, and the sidecar
  records the cleanup under its own `postprocess` key, MIT, beside the tool's entry.
- **Watertight rebuild** (`Ar3DRAHub/watertight.py`, `scripts/watertight.py`): closes the
  hairline gaps between shells and merges them into one closed body, so a generated model
  can be sliced, Boolean'd or simulated. A "Watertight" choice in the Finish step writes
  `watertight.glb` beside the result, with the resolution in Advanced. A thin shell stays
  a shell — it is never poured solid — and the interior of a hollow body is kept as air.
  It resamples, so detail below the voxel size is smoothed away and the extra model has
  no texture; the finished model is kept as it is beside it.
  MIT throughout, with **no CGAL and no GPL**: numpy, scipy, trimesh and scikit-image only.
  The rebuild measures free memory before each big allocation and stops with a sentence
  you can act on, never an OOM kill.
  The sidecar records the rebuild as an MIT component, so the output keeps its input's
  licence and classification.
- **Four-view Pixal3D builds** (`scripts/pixal3d_generate.py` `--left`/`--right`,
  `viewer/studio` "Four views"): the model is built from the front, back, left and right
  on the canonical Pixal3D turntable (`turntable_rig`, the same matrices as
  raven38/pixal3d.cpp's `calibration.js`). The studio draws the back and both sides with
  Qwen (the side prompt lives beside `BACK_PROMPT` in `Ar3DRAHub/back_view.py`), and the
  `create` MCP tool takes `sides=True`. The multi-view weights already fetched for
  "front + back" are the same ones; nothing new downloads. `pixel_match_result` reports
  "on (4 views)" and the provenance records `views: "four views"`.

### Changed
- CI lint gate: `ruff.toml` now names the rules it enforces (`E9` syntax, `F` pyflakes)
  instead of inheriting ruff's defaults, so a ruff upgrade cannot turn CI red on its own.
- The 3 MB cap in `check-added-large-files` no longer applies to `assets/`: the promo
  art and the example meshes are deliberately bigger than that.

### Fixed
- **Hi3D lost whole slabs of the model at 1024/1536** — real missing geometry, not
  shading. Two causes. The dominant one was **the SDPA attention port**
  (`scripts/patch_hi3d_sdpa.py`): it padded every span up to a mask the size of the
  whole voxel sequence, so PyTorch left the flash/memory-efficient kernels (fp32
  softmax) for the math kernel (fp16 softmax), which is not what the weights were
  measured against. It now runs one fused SDPA call per span — no mask — so the fused
  kernels apply again; 1024 went from 89% of the input surface rebuilt to ~96%, and
  Cancel now stops the refine (the subprocess was never registered on the job).
- **The 1536 rung stays unreliable** (~63% even after the port fix — its third pass at
  structure 192 is beyond the checkpoint's reach on this route) and would still ship
  holes. So every refine now measures the result against the input
  (`hi3d_backend.surface_coverage`, area-weighted samples, unit-radius frame) and, below
  93%, steps down one rung and says so in the log: a 1536 request lands on 1024. The
  run reports the rung that actually produced the mesh; the requested one is kept
  beside it; if even 512 fails, the run errors instead of presenting a holey model.
- **Watertight was silently skipped after a Hi3D geometry-only refine.** That route ends
  before the retopo worker, and the rebuild lived only in the worker's success path, so
  a ticked Watertight produced no `watertight.glb` and no message. Both routes now call
  one `watertight_extra` helper; a failed rebuild still leaves the finished model intact
  and is reported in the run's events and log.
- **Agents could not ask for Watertight at all.** The step chain keeps an allow-list of
  Finish keys (`activity_api.FINISH_KEYS`) and the four watertight ones were missing, so
  `continue_asset(..., advanced={"watertight": True})` quietly produced a plain Finish.
  They are listed now, with a test that walks the agent's own Advanced table.

## [0.5.9] - 2026-11-18

### Added
- **Hi3D v1.0 geometry refinement** (`--hi3d`): refine a coarse mesh against its
  reference image into geometry-only GLBs. Two routes, both NVIDIA-only: Linux runs
  the official measured stack, Windows runs this repo's own SDPA attention port
  (`scripts/patch_hi3d_sdpa.py`, no flash-attn). Installed by
  `scripts/bootstrap_hi3d.py` (about 10 GB of weights, Hi3D Open License v1.0).
- **Hi3D refine in Finish**: a "Hi3D refine" choice in the Finish step rebuilds the
  model's geometry before it is rigged (NVIDIA only). Geometry only by default — no
  textures — with the record chaining the Hi3D licence with the source's. Opt-in
  "Keep the texture" (`scripts/blender_hi3d_transfer.py`) bakes the original texture
  onto the refined mesh and then runs the usual Finish (Detail, Pixel Match,
  Marigold, compress). A Hi3D card on Setup & Status installs the route.
- **Marigold V2** de-light in Finish: Marigold V2 (Apache-2.0, Qwen-Image-Edit-2509
  backbone) removes the photo's own shadows and highlights from the finished texture, so
  the engine's lighting does not double up. Runs on front + back Pixal3D models, projected
  through the same saved camera Pixel Match uses. Install from the Setup page
  (`scripts/bootstrap_marigold_v2.py`, ~22 GB, NVIDIA + CUDA).
- **Opt-in surface normals.** "Marigold normals" (bakes a Marigold normal map) and "Fade
  normals where painted" (Pixel Match flattens the normal map where it painted) are both
  off by default, so neither step touches the normal map unless asked.

### Changed
- Pixel Match no longer flattens the baked normal map where it painted by default: the
  full normal relief is kept everywhere, with the old behaviour behind the opt-in "Fade
  normals where painted".
- Marigold Finish de-lights only by default; the normal map it can bake is a separate
  opt-in.
- Marigold's normal map is baked against the mesh's own `NORMAL`/`TANGENT` attributes and
  the GLB is rewritten at the byte level, so Finish no longer loses smooth shading (the
  previous trimesh round-trip dropped those attributes and left models flat-shaded).
