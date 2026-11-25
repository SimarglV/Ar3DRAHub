#!/usr/bin/env python3
"""Draw the back or a side of a subject from its front picture, for a multi-view 3D build.

    python scripts/make_back_view.py front.png back.png [--seed 3]
    python scripts/make_back_view.py front.png view_090.png --side left

Flux.2 [klein] 4B by default — the studio's picture engine — reads the front (cut out, on
plain grey) and draws the same subject from behind or from one side. `--engine qwen` uses
Qwen-Image in edit mode instead. The prompt and the reference rules live in
`Ar3DRAHub/back_view.py`.

Qwen's edit mode needs its vision projector (`mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`, 0.75 GB)
beside the Qwen weights. This never downloads it: with it missing, it says what it is and
how to get it, and stops. A `<back>.provenance.json` records the run like any picture.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "viewer")]

import image_api
from PIL import Image

from Ar3DRAHub import back_view, matte
from Ar3DRAHub.provenance import write_provenance

# Every back in the 2026-10-12 eight-character test was drawn with this seed.
DEFAULT_SEED = 3
VISION_GB = 0.75


def reference_for(front: Path, destination: Path) -> Path:
    """Write the front's cut-out on plain grey, cutting it out first unless it already is."""
    with Image.open(front) as opened:
        cut = opened.convert("RGBA") if matte.is_matted(opened) else matte.cut_out(
            opened.convert("RGB"))[0]
    back_view.reference_on_grey(cut).save(destination)
    return destination


def missing_vision_message() -> str:
    directory, filename = image_api.VISION_FILE
    repo = directory.removeprefix("models--").replace("--", "/")
    return (f"The back picture needs Qwen's vision projector, {filename} ({VISION_GB} GB),\n"
            f"from {repo}. Fetch it with:\n"
            f"    python scripts/bootstrap_qwen_image.py --vision")


def command_for(reference: Path, back: Path, seed: int, weights: dict[str, Path],
                vision: Path | None, binary: Path = image_api.BINARY,
                prompt: str = back_view.BACK_PROMPT,
                engine: image_api.PictureEngine = image_api.FLUX) -> list[str]:
    settings = image_api.clean_settings({"seed": seed}, engine)
    return image_api.build_command(prompt, settings, back, weights,
                                   binary=binary, references=[reference], vision=vision,
                                   engine=engine)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("front", type=Path)
    parser.add_argument("back", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--engine", choices=("flux", "flux-light", "qwen"), default="flux",
        help="which picture engine draws the view (default: flux)",
    )
    parser.add_argument(
        "--side", choices=("left", "right"), default=None,
        help="draw the subject from its left or right side (a quarter turn) instead of "
             "behind; the output path is still the second argument",
    )
    args = parser.parse_args(argv)
    if not args.front.is_file():
        raise SystemExit(f"not found: {args.front}")
    engine = image_api.ENGINES[args.engine]
    vision = image_api.resolve_vision(engine)
    if engine.vision_file is not None and vision is None:
        print(missing_vision_message())
        return 1
    weights = image_api.resolve_weights(engine)
    args.back.parent.mkdir(parents=True, exist_ok=True)
    reference = reference_for(args.front, args.back.with_name(f"{args.back.stem}__reference.png"))
    prompt = back_view.side_prompt(args.side) if args.side else back_view.BACK_PROMPT
    command = command_for(reference, args.back, args.seed, weights, vision, prompt=prompt,
                          engine=engine)
    started = time.time()
    result = subprocess.run(command, capture_output=True, text=True, cwd=str(REPO), check=False)
    if result.returncode or not args.back.exists():
        print((result.stdout + result.stderr)[-1500:])
        return 1
    seconds = time.time() - started
    record = image_api.provenance(prompt, image_api.clean_settings(
        {"seed": args.seed}, engine), seconds, args.back, engine)
    mode = f"edit ({args.side} view)" if args.side else "edit (back view)"
    record["reference"] = {"front": str(args.front), "mode": mode}
    write_provenance(args.back.with_suffix(".provenance.json"), record)
    print(f"[back-view] {seconds:.0f}s -> {args.back}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
