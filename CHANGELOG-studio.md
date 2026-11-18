# Changelog

All notable changes to the studio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
