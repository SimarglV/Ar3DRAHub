# Four-view Pixal3D builds

A model can be built from **four views** of the same subject — front, back, left and
right — on the canonical Pixal3D turntable, instead of from one picture or from front +
back. The multi-view weights are the ones "front + back" already fetched
(`raven38/pixal3d-q8_0-v1`); nothing new downloads.

The turntable cameras are raven38/pixal3d.cpp's `CANONICAL_RIG` verbatim: azimuths
0/90/180/270, elevation 0, FOV 20°, distance 3.1192049980163574.

## Run it

```bash
# Studio: Create -> a picture/idea -> Pixal3D -> tick "Four views".
# CLI: draw the three extra views, then build.
python scripts/make_back_view.py front.png back.png
python scripts/make_back_view.py front.png view_090.png --side left
python scripts/make_back_view.py front.png view_270.png --side right
python scripts/pixal3d_generate.py front.png out.glb --back back.png \
    --left view_090.png --right view_270.png
```

Agents: `create(..., sides=True)`.

## The one thing to check by eye

The view order is front → left side → back → right side:

| file | camera | shows |
|---|---|---|
| `view_000.png` | azim 0 | the front (the reference) |
| `view_090.png` | azim 90 | the subject's **left** side |
| `view_180.png` | azim 180 | the back |
| `view_270.png` | azim 270 | the subject's **right** side |

This mapping was derived from the camera matrices, not from a rendered check, and it is
entangled with the GLB ↔ view-space frame (`Ar3DRAHub/photo_paint.py`'s
`PIXAL3D_FRAME`). **Validate it once** on a clearly asymmetric subject:

1. Draw or find a picture of a character with a one-sided detail (a sword in the right
   hand, a badge on one side of the chest).
2. Run a four-view build.
3. Check the finished model: the sword should be on the subject's right, the badge on the
   correct side.

If they come out mirrored, swap `left` and `right` in the mapping — the file names and
the `--left`/`--right` help text in `scripts/pixal3d_generate.py`, and the side prompts'
targets. The pose-consistency test (`tests/test_pixal3d_rig.py`) locks the *order* of the
cameras, not the handedness, so it stays green either way.

## Known limits

- **Side views are the weak spot for characters**: silhouette IoU is ~0.92 on the
  front/back but ~0.67–0.76 on the sides, so the sides improve a model but won't be as
  faithful as the front.
- **Thin protruding parts can double** (one tail becomes two), a property of the model
  itself.
- A four-view build takes three extra picture drawings, so it is slower than front + back.
