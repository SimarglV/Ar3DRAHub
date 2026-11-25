"""Bake UltraTex's generated view canvases onto the mesh's own UVs.

    python scripts/ultratex_bake_views.py MESH.glb CANVAS.png OUT.glb [--mr MR.png] [--resolution 1024]

**Why.** UltraTex denoises the mesh's six canonical views and returns them as one
2x3 canvas; the texture becomes real only when those views are projected back
onto the surface. This reuses photo_paint's rasterizer: it splits the canvas
into the six tiles (the same [0,1,3]/[2,4,5] order the dataset's load_and_concat
builds), turns each tile into a View with the canonical TexVerse camera, and
paints the mesh's texture at exactly the pixels the model generated them for --
the same normalisation the renderer applied (centre at the origin, largest
dimension 0.6), the same lens, the same poses, so the bake cannot drift from the
render. Texels no view can see keep the mesh's existing paint.

The painted texture replaces the GLB's base colour and is also written beside
the output as PNG.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Ar3DRAHub import photo_paint  # noqa: E402
from Ar3DRAHub.photo_paint import View, read_glb, replace_base_colour  # noqa: E402
from Ar3DRAHub.ultratex_backend import GUIDED_POSES, POSES, TARGET_SIZE  # noqa: E402

#: The tile order the dataset's load_and_concat builds: row 1 takes [0, 1, 3],
#: row 2 takes [2, 4, 5] of the 2x3 canvas.
TILE_PLACEMENT = ((0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (1, 2))

#: The bake's trust settings: the views are clean renders generated for exactly
#: this mesh, so there is nothing to align or palette-match -- the model's paint
#: is only kept where no view looks.
BAKE_SETTINGS = photo_paint.Settings(
    align=False,
    match_colour=False,
    local_colour=False,
    cutout_edge=0.0,
    inner_edge=4 / 1024,
    edge_px=3,
)


def split_canvas(canvas: np.ndarray, placement: tuple = TILE_PLACEMENT) -> list[np.ndarray]:
    """The six tiles of a 2x3 UltraTex canvas, in dataset order 000..005."""
    height, width = canvas.shape[:2]
    tile_h, tile_w = height // 2, width // 3
    return [canvas[py * tile_h:(py + 1) * tile_h, px * tile_w:(px + 1) * tile_w]
            for py, px in placement]


def normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre the mesh at the origin and scale its largest dimension to TARGET_SIZE.

    Returns (normalised, transform) where transform is (offset, scale); the inverse
    of the transform maps the normalised points back onto the GLB's own coordinates.
    """
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    centre = (minimum + maximum) / 2.0
    extent = float((maximum - minimum).max())
    if extent <= 0.0:
        raise ValueError("the mesh has zero extent; nothing to bake")
    scale = TARGET_SIZE / extent
    return (points - centre) * scale, (centre, scale)


def view_fov(resolution: int) -> float:
    """The horizontal field of view of the canonical camera at `resolution` pixels."""
    focal_px = (6400.0 / 4096.0) * resolution     # fx scales with the render size
    return float(2.0 * np.arctan((resolution / 2.0) / focal_px))


def build_views(tiles: list[np.ndarray], resolution: int,
                poses: tuple = POSES) -> list[View]:
    """A photo_paint View per tile, with the matching TexVerse camera."""
    fov = view_fov(resolution)
    views = []
    for index, tile in enumerate(tiles):
        pose = poses[index]
        cam_to_world = np.array(
            [pose[0], pose[1], pose[2], (0.0, 0.0, 0.0, 1.0)], dtype=np.float64)
        views.append(View(np.asarray(tile), cam_to_world, fov, f"view_{index:03d}"))
    return views


def pack_metallic_roughness(painted: np.ndarray) -> np.ndarray:
    """The glTF MR map: green = roughness, blue = metallic, alpha opaque.

    The UltraTex MR canvas stores roughness in the red channel and metallic in
    the blue (green duplicates red); the baked texture is RGB, so the packed
    map is built as RGBA with red unused.
    """
    packed = np.zeros(painted.shape[:2] + (4,), dtype=np.uint8)
    packed[..., 1] = painted[..., 0]
    packed[..., 2] = painted[..., 2]
    packed[..., 3] = 255
    return packed


def load_tiles(canvas_path: Path) -> list[np.ndarray]:
    """The six tiles of a 2x3 canvas, as RGBA arrays in dataset order."""
    return split_canvas(np.asarray(Image.open(canvas_path).convert("RGBA")))


def albedo_views(canvas: Path, resolution: int,
                 guided_canvas: Path | None = None) -> list[View]:
    """The views the albedo canvas (and the guided canvas) bake from.

    The 12-view guided mode merges the six canonical views with the six
    45-degree offset views into one paint pass, so the weight map blends all
    twelve viewpoints instead of last-wins.
    """
    views = build_views(load_tiles(canvas), resolution)
    if guided_canvas is not None:
        views += build_views(load_tiles(guided_canvas), resolution, GUIDED_POSES)
    return views


def bake_canvas(glb: Path, canvas: Path, output: Path, resolution: int,
                mr_canvas: Path | None = None,
                guided_canvas: Path | None = None) -> dict:
    """Paint `canvas` onto `glb`'s UVs and write `output`; report the coverage.

    With `guided_canvas`, the albedo is baked from the twelve views of the
    guided mode (canonical six plus the 45-degree offset six). With
    `mr_canvas`, the metallic-roughness views are baked the same way: the
    dataset packs roughness in the red channel (green duplicates it) and
    metallic in the blue channel, so the baked result's R/B channels become the
    glTF map's green (roughness) and blue (metallic), attached as the material's
    metallicRoughnessTexture.
    """
    positions, uvs, faces, texture = read_glb(glb)
    normalised, _ = normalize(positions)

    def paint(views: list[View]) -> tuple[np.ndarray, np.ndarray]:
        return photo_paint.paint_texture(
            texture, normalised, uvs, faces, views,
            settings=BAKE_SETTINGS)

    painted, weight = paint(albedo_views(canvas, resolution, guided_canvas))
    result = replace_base_colour(glb.read_bytes(), photo_paint.encode_png(painted))
    report: dict = {
        "albedo": str(output.with_suffix(".albedo.png")),
        "coverage": round(float((weight > 0.5).mean()), 3),
    }
    if mr_canvas is not None:
        rough_metallic, _ = paint(build_views(load_tiles(mr_canvas), resolution))
        packed = pack_metallic_roughness(rough_metallic)
        result = photo_paint.add_metallic_roughness_texture(
            result, photo_paint.encode_png(packed))
        output.with_suffix(".roughness-metallic.png").write_bytes(
            photo_paint.encode_png(packed))
        report["roughness_metallic"] = str(output.with_suffix(".roughness-metallic.png"))
    output.write_bytes(result)
    output.with_suffix(".albedo.png").write_bytes(photo_paint.encode_png(painted))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mesh", type=Path, help="the low-poly GLB to paint")
    parser.add_argument("canvas", type=Path, help="the UltraTex 2x3 albedo canvas (PNG)")
    parser.add_argument("output", type=Path, help="the textured GLB to write")
    parser.add_argument("--mr", type=Path, default=None,
                        help="the UltraTex 2x3 metallic-roughness canvas (PNG)")
    parser.add_argument("--guided", type=Path, default=None,
                        help="the second 2x3 albedo canvas of the 45-degree offset views (PNG)")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="the resolution the views were generated at")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.mesh, args.canvas):
        if not path.is_file():
            raise SystemExit(f"not found: {path}")
    report = bake_canvas(args.mesh, args.canvas, args.output, args.resolution,
                         args.mr, args.guided)
    extra = f" + MR {report['roughness_metallic']}" if "roughness_metallic" in report else ""
    guided = f" + guided {args.guided}" if args.guided else ""
    print(f"[ultratex-bake] {report['coverage']:.0%} of the texture from the views -> "
          f"{args.output}{extra}{guided}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
