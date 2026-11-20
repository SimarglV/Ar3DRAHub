# Changelog

All notable changes to the studio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Changed
- CI lint gate: `ruff.toml` now names the rules it enforces (`E9` syntax, `F` pyflakes)
  instead of inheriting ruff's defaults, so a ruff upgrade cannot turn CI red on its own.
- The 3 MB cap in `check-added-large-files` no longer applies to `assets/`: the promo
  art and the example meshes are deliberately bigger than that.

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
