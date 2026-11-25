#!/usr/bin/env python3
"""Predict Marigold V2 depth, normals, and albedo from one image, natively.

    python scripts/marigold_v2_predict.py input.png --modality all --out out/

Runs the reference Marigold V2 predictors (MIT) on a local Qwen-Image-Edit-2509
GGUF backbone. Needs its own venv: `python scripts/bootstrap_marigold_v2.py` (torch,
diffusers, peft). Weights live in models/marigold-v2/ (or MARIGOLD_V2_MODELS_DIR).

One backbone load serves every requested modality; switching swaps the LoRA adapter and
the VAE decoder in place. Writes <stem>_<modality>.npy and a PNG beside it, plus a
.provenance.json sidecar recording the licence of every model it loaded.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Ar3DRAHub import marigold_v2 as m
from Ar3DRAHub.provenance import sha256_file, write_provenance

# Marigold V2 is MIT. Both are named in every sidecar.
MARIGOLD_LICENSE = "MIT"
MARIGOLD_LICENSE_URL = "https://huggingface.co/huawei-bayerlab/marigold-v2-0"
QWEN_BACKBONE_LICENSE = "MIT"
QWEN_BACKBONE_URL = "https://huggingface.co/Qwen/Qwen-Image-Edit-2509"


def sidecar(image: Path, out_dir: Path, modality: str, files: dict[str, Path],
            params: dict, seconds: float) -> Path:
    """The run's provenance record, written beside the predictions."""
    outputs = {name: {"path": str(path), "sha256": sha256_file(path)}
               for name, path in files.items()}
    record = {
        "schema_version": 1,
        "backend": "marigold-v2",
        "modality": modality,
        "input": {"path": str(image), "sha256": sha256_file(image)},
        "output": outputs,
        "parameters": params,
        "license": {"name": MARIGOLD_LICENSE, "url": MARIGOLD_LICENSE_URL},
        "components": [
            {"component": "huawei-bayerlab/marigold-v2-0", "purpose": "dense predictor",
             "license": MARIGOLD_LICENSE, "license_url": MARIGOLD_LICENSE_URL},
            {"component": "Qwen/Qwen-Image-Edit-2509", "purpose": "frozen DiT backbone",
             "license": QWEN_BACKBONE_LICENSE, "license_url": QWEN_BACKBONE_URL},
        ],
        "timings_seconds": {"total": round(seconds, 1)},
    }
    return write_provenance(out_dir / "marigold_v2.provenance.json", record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--modality", choices=("depth", "normals", "albedo", "all"),
                        default="depth")
    parser.add_argument("--depth-variant", default=m.DEFAULT_DEPTH_VARIANT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--models", type=Path, default=None,
                        help=f"weights folder (default: models/marigold-v2 or ${m.MODELS_DIR_ENV})")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args(argv)

    if not args.image.is_file():
        raise SystemExit(f"not found: {args.image}")
    models = args.models or m.models_dir()
    out_dir = args.out or args.image.with_name(f"{args.image.stem}__marigold")
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = m.MODALITIES if args.modality == "all" else (args.modality,)

    # Pre-flight: every file the run needs, before the slow load starts.
    missing = sorted({name for mod in requested
                      for name in m.missing_files(models, mod, args.depth_variant)})
    if missing:
        raise SystemExit("Marigold V2 weights are missing:\n  " + "\n  ".join(missing)
                         + f"\n  under {models}. Run scripts/bootstrap_marigold_v2.py.")

    started = time.time()
    predictor = m.Predictor(models, requested[0], args.depth_variant)
    files: dict[str, Path] = {}
    for modality in requested:
        if modality != predictor.spec.modality:
            predictor.use(modality, args.depth_variant)
            print(f"[marigold-v2] switched to {modality}", flush=True)
        t = time.time()
        array = predictor.predict(args.image, max_side=args.max_side, seed=args.seed)
        npy, png = m.save_prediction(array, out_dir, args.image.stem, modality)
        files[f"{modality}.npy"] = npy
        files[f"{modality}.png"] = png
        print(f"[marigold-v2] {modality}: {array.shape} in {time.time() - t:.0f}s -> "
              f"{npy.name}, {png.name}", flush=True)

    record = sidecar(args.image, out_dir, args.modality, files,
                     {"max_side": args.max_side, "seed": args.seed,
                      "depth_variant": args.depth_variant,
                      "modalities": list(requested)},
                     time.time() - started)
    print(f"[marigold-v2] done in {time.time() - started:.0f}s; "
          f"manifest {record.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
