"""Prepare the six-view UltraTex dataset for a mesh: render, tiles, reference.

    python scripts/ultratex_views.py MESH.glb IMAGE.png OUT_DIR [--resolution 1024] [--blender PATH]

**Why.** UltraTex's dataset reader expects the G-buffer TexVerse layout: the six
canonical views as world-space normal tiles (``bump_normal_world/``), a target
tile folder whose alpha is the per-view silhouette (``albedo/`` for the colour
task, ``roughness_metallic/`` for the metallic-roughness task), and the reference
picture aligned to the front view in ``render_ref_*``. This script builds all of
it for one asset: it runs the headless Blender renderer
(``blender_render_texverse_views.py``) for the normal tiles, fills the target
tiles from the silhouettes, mats and frames the reference the way the
PixelArtistry workflow does (cut out, crop to the subject with padding, square),
and writes the eval JSON that ``ultratex_backend`` hands to ``inference_flux2.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PIL import Image  # noqa: E402

TARGET_SUBDIRS = ("albedo", "roughness_metallic")
REFERENCE_DIRS = tuple(f"render_ref_{index}" for index in range(4))
REFERENCE_FILES = 4     # the dataset picks one of 000..003 at random per index


def render_command(mesh: Path, output: Path, resolution: int,
                   blender: Path, guided: bool = False) -> list[str]:
    """The headless Blender invocation that renders six normal tiles.

    With `guided`, the 45-degree offset view set is rendered instead of the
    canonical six (the second pass of the 12-view guided mode).
    """
    command = [
        str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
        "--python", str(REPO / "scripts" / "blender_render_texverse_views.py"), "--",
        str(mesh), str(output), str(resolution),
    ]
    if guided:
        command.append("guided")
    return command


def build_target_tiles(renders: Path, subdirs: tuple[str, ...] = TARGET_SUBDIRS) -> int:
    """White tiles whose alpha is the per-view silhouette, for every target folder.

    The dataset reader only uses the target tiles' alpha (the silhouette) at
    inference; the RGB content never reaches the model. Reuses the rendered
    bump_normal_world tiles as the silhouette source. Returns how many were written.
    """
    written = 0
    for index in range(6):
        source = renders / "bump_normal_world" / f"{index:03d}.webp"
        with Image.open(source).convert("RGBA") as normal:
            alpha = normal.getchannel("A")
        for subdir in subdirs:
            destination = renders / subdir / f"{index:03d}.webp"
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.merge("RGBA", (Image.new("L", alpha.size, 255),
                                 Image.new("L", alpha.size, 255),
                                 Image.new("L", alpha.size, 255), alpha)).save(destination)
            written += 1
    return written


def frame_reference(rgba: Image.Image, pad: float, size: int) -> Image.Image:
    """Crop the cut-out subject to its alpha bounding box, pad, and square it.

    The dataset resizes the reference to the view resolution itself; the framing
    here is the PixelArtistry workflow's: the subject fills the frame with a
    ``pad`` margin (their crop uses 1.2). The RGB is composited on white the way
    the dataset composites its own renders, while the alpha keeps the subject
    mask -- the reader uses that alpha as the foreground mask.
    """
    bbox = rgba.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError("the reference has no visible subject")
    left, top, right, bottom = bbox
    width, height = right - left, bottom - top
    margin_x, margin_y = int(width * (pad - 1.0) / 2), int(height * (pad - 1.0) / 2)
    cropped = rgba.crop((
        max(0, left - margin_x), max(0, top - margin_y),
        min(rgba.width, right + margin_x), min(rgba.height, bottom + margin_y),
    ))
    rgb = Image.new("RGB", cropped.size, (255, 255, 255))
    rgb.paste(cropped.convert("RGB"), (0, 0), cropped.getchannel("A"))
    framed = rgb.resize((size, size), Image.Resampling.LANCZOS)
    alpha = cropped.getchannel("A").resize((size, size), Image.Resampling.LANCZOS)
    framed.putalpha(alpha)
    return framed


def reference_tiles(framed: Image.Image, renders: Path,
                    dirs: tuple[str, ...] = REFERENCE_DIRS,
                    files: int = REFERENCE_FILES) -> int:
    """The framed reference in every render_ref_* bucket, 000..003 each.

    The dataset randomises which bucket and which of its four files an index
    reads, and the albedo and MR readers use different bucket counts, so every
    bucket carries the same image.
    """
    written = 0
    for folder in dirs:
        for index in range(files):
            path = renders / folder / f"{index:03d}.webp"
            path.parent.mkdir(parents=True, exist_ok=True)
            framed.save(path)
            written += 1
    return written


def eval_entry(output_dir: Path) -> dict:
    """The one-entry eval JSON the backend hands to ``inference_flux2.py``."""
    return {
        "id": output_dir.name,
        "prompt": "",     # image-guided texturing: the PixelArtistry workflow's mode
        "image_dir": str(output_dir.resolve()),
    }


def guided_dir(output: Path) -> Path:
    """The sibling folder the 45-degree offset views render into."""
    return output.with_name(output.name + "-guided")


def guided_json_path(json_path: Path) -> Path:
    """The eval JSON for the guided folder, beside the canonical one."""
    return json_path.with_name(json_path.stem + "-guided.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mesh", type=Path, help="the low-poly GLB to texture")
    parser.add_argument("image", type=Path, help="the reference picture")
    parser.add_argument("output", type=Path, help="views directory to build")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--blender", type=Path, default=None,
                        help="Blender executable; found automatically when omitted")
    parser.add_argument("--json", type=Path, default=None,
                        help="where to write the eval JSON (default: beside the views)")
    parser.add_argument("--pad", type=float, default=1.2,
                        help="reference crop padding, as a factor of the subject")
    parser.add_argument("--guided", action="store_true",
                        help="also render the six 45-degree offset views (12-view guided mode)")
    return parser


def prepare_views(mesh: Path, image: Path, output: Path, resolution: int,
                  blender: Path, pad: float, json_path: Path,
                  guided: bool) -> list[Path]:
    """Render the view folders, fill their tiles, and write the eval JSON(s).

    Returns every eval JSON written (the canonical one, plus the guided one).
    """
    from Ar3DRAHub.matte import cut_out, is_matted

    subprocess.run(
        render_command(mesh, output, resolution, blender),
        check=True,
    )
    build_target_tiles(output)
    with Image.open(image) as opened:
        cut = (opened.convert("RGBA") if is_matted(opened)
               else cut_out(opened.convert("RGB")))
    framed = frame_reference(cut, pad, resolution)
    reference_tiles(framed, output)
    json_path.write_text(json.dumps([eval_entry(output)], indent=2) + "\n",
                         encoding="utf-8")
    written = [json_path]
    if guided:
        second_dir = guided_dir(output)
        subprocess.run(
            render_command(mesh, second_dir, resolution, blender, guided=True),
            check=True,
        )
        build_target_tiles(second_dir)
        reference_tiles(framed, second_dir)
        second = guided_json_path(json_path)
        second.write_text(json.dumps([eval_entry(second_dir)], indent=2) + "\n",
                          encoding="utf-8")
        written.append(second)
    return written


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from Ar3DRAHub.blender import find_blender, missing_help

    blender = args.blender or find_blender()
    if blender is None or not Path(blender).is_file():
        raise SystemExit(missing_help())
    for path in (args.mesh, args.image):
        if not path.is_file():
            raise SystemExit(f"not found: {path}")
    args.output.mkdir(parents=True, exist_ok=True)

    label = "twelve guided views" if args.guided else "six views"
    print(f"[ultratex-views] rendering {label} at {args.resolution}px ...", flush=True)
    json_path = args.json or args.output.parent / f"{args.output.name}.json"
    for written in prepare_views(args.mesh, args.image, args.output, args.resolution,
                                 Path(blender), args.pad, json_path, args.guided):
        print(f"[ultratex-views] eval JSON -> {written}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
