# Hi3D v1.0 geometry refinement

Hi3D v1.0 refines a coarse mesh against a reference image of the same object into
high-resolution **geometry-only** GLBs. No textures are generated, and the output is
recentered and normalized to unit radius, so world scale is not preserved. Severe
errors in the coarse mesh can persist — Hi3D adds detail, it does not fix wrong
structure.

## Routes

Both routes are NVIDIA-only (compute capability 8.0+); there is no CPU path and no
Metal backend.

| Route | Stack | Notes |
|---|---|---|
| Linux (`vendor/hi3d/`) | upstream's measured stack: torch 2.5.0+cu124, flash-attn from source, spconv-cu120 | The stack the authors validated on a 96 GB H20 |
| Windows (`vendor/hi3d/`) | this repo's own port: torch 2.11.0+cu128, spconv-cu126 2.3.8, triton-windows, **no flash-attn** | flash-attn has no Windows build; `scripts/patch_hi3d_sdpa.py` replaces it with an SDPA fallback. Our port, not an author-validated stack. |

## Install

```bash
python scripts/bootstrap_hi3d.py        # announces route, sizes, licence; asks
python scripts/bootstrap_hi3d.py --yes  # non-interactive
```

Weights, about 10 GB total: Hi3D v1.0 artifacts (7.2 GB), DINOv2-large (1.2 GB),
CLIP ViT-L/14 (1.6 GB). All public. The Hi3D artifacts are materialized into
`models/hi3d/` as real files: the official loader refuses symlinks it does not
expect, and the hub cache snapshots are symlinked on Windows.

## Run

```bash
python pipeline.py IMAGE --hi3d --coarse-mesh coarse.glb [--resolution 512]
python pipeline.py --run-manifest manifests/example-hi3d.json
```

- `--resolution 512|1024|1536` — more passes, more detail, much more VRAM. Start at 512.
- Sequential CPU offload is on by default (`--no-hi3d-offload` to disable). The authors
  measured on a 96 GB H20; on smaller cards offload is the memory-safe path.
- The image may be a cut-out (alpha kept) or a plain photo; the backend mattes
  uncut inputs with the shared remover.

## Quality gate

The high-resolution rungs are not reliable on every route. On this repo's Windows SDPA
port, before the attention fix `scripts/patch_hi3d_sdpa.py` shipped (2026-11-21), the
1024 rung rebuilt 89% of the input's surface and 512 rebuilt 100% — whole slabs of the
body were absent, which reads as holes. The two causes and their fixes:

- **The attention port padded every span to a whole-sequence mask.** The transformer's
  self-attention is one span of every voxel, so the mask was tens of gigabytes and
  forced PyTorch off its flash/memory-efficient kernels (fp32 softmax) into the math
  kernel (fp16 softmax) — not what the weights were measured against. The port now runs
  one fused SDPA call *per span*, no mask, so the fused kernels apply again. This
  repaired 1024 (now ~96%).
- **The 1536 rung is still unreliable** (~63% even after the fix): its third pass, at
  structure resolution 192, is beyond what the checkpoint reliably produces on this
  route. Every refine measures the result against the input
  (`hi3d_backend.surface_coverage`: area-weighted samples of the input, distance to the
  refined surface in Hi3D's own unit-radius frame) and, if less than 93% of that surface
  came back, steps down one rung and says so in the log — so a 1536 request lands on
  1024, not on a holey mesh. The run reports the rung that actually produced the mesh;
  the requested one is kept beside it in the record. If even 512 fails, the run errors
  with the measured share instead of presenting the mesh as finished.

The gate is about missing geometry, not seams: a hairline gap moves no sampled point
far from the surface, while a missing chest leaves every sample there with nothing
nearby.

## In the studio

Finish offers **Hi3D refine** (NVIDIA only, the model must have its reference
picture). It rebuilds the geometry before rigging. Two modes:

- **Geometry only** (default): the refined mesh as-is — no textures, re-centred to
  unit size, named `<name>_hi3d.glb`.
- **Keep the texture** (`hi3d_texture`): `scripts/blender_hi3d_transfer.py` bakes the
  original texture onto the refined mesh and re-aligns it to the source, then the
  ordinary Finish runs on it (Detail, Pixel Match, Marigold, compress). Front-facing
  details can drift slightly: the alignment is by bounding boxes, not surface
  matching.

## Licence

Hi3D Open License Agreement v1.0: commercial use permitted; above 200,000 monthly
active users or USD 1,000,000 monthly revenue, notify the Licensor and apply for a
separate written licence within 60 days; commercial integrations must display
"Powered by Hi3D 1.0" with a link to hi3d.ai. Standalone outputs containing no Hi3D
Materials or Model Derivative do not trigger those obligations. Encoders: DINOv2-large
(Apache-2.0), CLIP ViT-L/14 (MIT). Every run records all of this in its
`.provenance.json`, chained with the coarse mesh's own record.
