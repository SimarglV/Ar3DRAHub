# Watertight: one closed body, from triangle soup

Every model this studio generates is a **display** surface, not a solid. Pixal3D, TRELLIS.2
and Hi3D all emit triangle soup: a renderer is happy with it, and a slicer, a physics engine
or a Boolean is not. Measured on the knight of Rome used for the Hi3D work — 957,610 faces,
`is_watertight` false, **97 separate shells**. Ninety-six hairline seams that no eye sees and
every downstream tool trips on.

**Watertight** rebuilds the model as one closed body.

## What it does

- **Closes the gaps.** The surface is resampled on a voxel grid, so a gap narrower than a
  voxel simply disappears.
- **Merges the pieces.** What comes back is one shell, not 97.
- **Keeps a hollow model hollow.** A shell a fraction of a millimetre thick stays a shell.
  The interior of a body is air, and it stays air.

## What it costs

- **It resamples.** Detail smaller than a voxel is smoothed away. `across` is the knob: how
  many voxels span the model's longest side. Too coarse melts fingers and sword blades.
- **The result has no texture.** The rebuild works on shape only. Carry the texture across
  with `scripts/blender_hi3d_transfer.py`, or bake again afterwards.
- **Sharp edges soften**, because marching cubes contours a smooth field.
- **A thin wall comes back one voxel thick**, so the volume goes *up* even though nothing was
  added inside. On the knight: 0.00268 in, 0.00600 out — but the output encloses 3.2% of its
  bounding box against the input's 1.4%. A closed surface thinner than a voxel cannot be
  built; that is the resolution cost, made visible.

## In the studio

Finish step, a **Watertight** checkbox. It runs last, after Detail, Pixel Match and the bake
— a surface that has been cut up and re-welded is exactly where seams reappear. The result is
written **beside** your finished model as `watertight.glb`, never in place of it: it has no
UVs, so it is a different deliverable, not a better one. The resolution sits in Advanced.

Nothing happens if the rebuild fails: the finished model is the deliverable, and the run
carries on.

## From the command line

```bash
python scripts/watertight.py model.glb model-watertight.glb --across 300
python scripts/watertight.py model.glb --check-only      # just report the condition
```

`--check-only` prints the shell count and the open edge count and exits without rebuilding —
useful for asking "is this model even broken?" before spending three minutes.

## Licensing

The maths is **MIT**, in `Ar3DRAHub/watertight.py`. It runs on numpy, scipy, trimesh and
scikit-image — all MIT. **No external tool, nothing fetched from the network.**

The sidecar records the rebuild as an MIT *component*, never as a new licence term: MIT
material adds no condition, so the output keeps its input's licence and classification.

### The external engine (`--tool wativo`)

If **WTiVo** is installed, it can do the rebuild instead — much faster on a CUDA machine,
because it runs on the GPU. Install it from **Setup** ("WTiVo (faster watertight rebuild)",
about 7 GB), or run `python scripts/bootstrap_wativo.py`. Either way it lands in
`vendor/wativo` with a `.venv` of its own: its native extension is built for CPython 3.12
and pins torch 2.8.0, neither of which the studio's own 3.11 environment can host. The
studio then finds both by itself, and the Watertight block in Finish grows a second
checkbox labelled **Faster rebuild**.

WTiVo's own code is **MIT**, and so is its native extension. The tool runs as a
separate process, and **the result keeps the input's licence**.

What changes is the record. `provenance.json` gains an `external_tools` entry:

```json
"external_tools": [{
  "component": "WTiVo (WatertightVoxel, SimarglV)",
  "license": "MIT",
  "copyleft": false
}]
```

That is the honest position: the output is yours, and the file says how it was made. It is
deliberately **not** merged into `license.name` — that would read as a licence on the
output, which it is not. Anyone who needs to know, looks; nobody is nagged.

WTiVo takes `--input`, `--output` and its resolution as `--input-res` / `--final-res`.
The "Faster rebuild" checkbox carries its own detail pick — **512 (fast, a lighter
file)** or **1536 (the tool's full detail, a much bigger file)** — instead of mapping the
CPU engine's `--across` knob onto it, because the two engines' resolutions are different
scales. There is no voxel-size flag, so `--voxel` is reported as unusable rather than
silently dropped. Both the studio and the command line find the same install:
`vendor/wativo` first, then a checkout named by `$WTIVO_HOME` / `$WTIVO_DIR`, with
`$WTIVO_PYTHON` to override which 3.12 interpreter runs it. No install, no silent
fallback: the run says what is missing and stops.

WTiVo at 512 can leave a few microscopic detached specks beside the body (measured on
the knight: three 12-face shells). After the external tool finishes, the studio drops
whole shells below a tenth of the biggest one's size with the same edge-connectivity
rule the CPU engine uses — the tool's geometry is otherwise untouched, and a clean result
is kept byte for byte. The sidecar records both facts: the tool under `external_tools`,
and the cleanup under a `postprocess` key of its own, MIT.

## How it works

The same idea WTiVo uses — resample to a volume, extract one surface — built from
permissive parts:

1. **Near-surface distances only.** Cells within a couple of voxels of the surface get their
   exact distance to the nearest triangle, via a k-d tree over triangles. Only that band is
   filled, so a 700-voxel model costs a fraction of the dense grid.
2. **An inside/outside mask from the surface's own voxelisation.** `trimesh.voxelized` marks
   the cells a triangle crosses — a skin, never a filled body — and that skin *is* the solid.
3. **A parity test, so a genuinely solid body still fills.** What the skin encloses is either
   material or sealed air, and the skin alone cannot tell them apart. A region the flood fill
   cannot reach from the grid border is labelled, and a ray from it counts the wall *runs* it
   crosses: odd means material behind it, even means enclosed air. A hollow box keeps its
   cavity; a solid cube gets its middle filled.
4. **One field, then marching cubes.** Positive distance outside the mask, negative inside,
   `skimage.measure.marching_cubes` contouring at zero. A mask, not a winding test, is what
   turns a shell into a solid — so an inside-out triangle no longer punches a hole.
5. **Keep the body and its cavities, drop the crumbs**, by shared *edges* rather than shared
   vertices: edge sharing is the rule a slicer uses.

## Measuring it honestly

Two numbers say more than a volume alone:

- **Inside ratio** — the share of the bounding box the surface encloses. If it barely moves,
  nothing was poured in.
- **Volume over area** — for a solid, roughly the half-thickness; a shell reads far lower.
