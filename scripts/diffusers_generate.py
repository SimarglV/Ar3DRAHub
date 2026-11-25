#!/usr/bin/env python3
"""Draw one picture with a raw diffusion checkpoint, through ``diffusers``.

This is the runner behind the studio's "diffusers" picture runtime: it loads a checkpoint
the person dropped into `models/pictures/` (a single `.safetensors` / `.ckpt` file, or a
diffusers-format folder with a `model_index.json`) and draws from it, text-to-image or, with
a reference picture, image-to-image (the back and side views). It runs in its own venv,
installed by `scripts/bootstrap_diffusers.py`.

It never downloads: loading is pinned to local files, and a remote Hugging Face id is refused.

    python scripts/diffusers_generate.py --model models/pictures/my-model/model.safetensors \\
        --prompt "a knight" -W 1024 -H 1024 --steps 20 --cfg 4.5 --seed 42 -o out.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# A download must never happen by surprise: this pins every load to what is already on disk.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True,
                        help="a single .safetensors/.ckpt checkpoint, or a diffusers folder")
    parser.add_argument("--text-encoder", type=Path, default=None,
                        help="a text encoder to use instead of the checkpoint's own: a "
                             "diffusers folder or a single file")
    parser.add_argument("--text-encoder-2", type=Path, default=None,
                        help="a second text encoder (SDXL's CLIP-G, FLUX's T5) to use instead "
                             "of the checkpoint's own: a diffusers folder or a single file")
    parser.add_argument("--vae", type=Path, default=None,
                        help="a VAE to use instead of the checkpoint's own: a diffusers "
                             "folder or a single file")
    parser.add_argument("-p", "--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("-W", "--width", type=int, default=1024)
    parser.add_argument("-H", "--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-r", "--reference", type=Path, default=None,
                        help="a reference picture: draw an image-to-image edit from it")
    parser.add_argument("--mask", type=Path, default=None,
                        help="a mask (white = repaint) over the reference picture: inpaint "
                             "only that area (the brush)")
    parser.add_argument("--strength", type=float, default=0.8,
                        help="image-to-image / inpainting strength (how much of the reference "
                             "is redrawn)")
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser


def load_pipeline(model: Path, *, mode: str, text_encoder: Path | None = None,
                  text_encoder_2: Path | None = None, vae: Path | None = None):
    """Load the checkpoint as a diffusers pipeline, never touching the network.

    A single file is a full checkpoint (Stable Diffusion 1.x / SDXL, with the text encoder and
    VAE bundled); a folder is a diffusers-format model with its own `model_index.json`. `mode`
    is one of "txt2img", "img2img" or "inpaint". `text_encoder`, `text_encoder_2` and `vae`
    swap those parts of the pipeline for the given folders or single files.
    """
    import torch
    from diffusers import AutoencoderKL, AutoPipelineForImage2Image, AutoPipelineForInpainting, AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    cls = {"txt2img": AutoPipelineForText2Image,
           "img2img": AutoPipelineForImage2Image,
           "inpaint": AutoPipelineForInpainting}[mode]
    if model.is_file():
        pipeline = cls.from_single_file(str(model), torch_dtype=dtype)
    elif model.is_dir():
        pipeline = cls.from_pretrained(str(model), torch_dtype=dtype, local_files_only=True)
    else:
        raise SystemExit(f"model not found: {model}")
    if vae is not None:
        pipeline.vae = _load_component(AutoencoderKL, vae, dtype)
    if text_encoder is not None:
        pipeline.text_encoder = _load_component(type(pipeline.text_encoder), text_encoder, dtype)
    if text_encoder_2 is not None:
        if not hasattr(pipeline, "text_encoder_2"):
            raise SystemExit("this model has no second text encoder (SDXL / FLUX do)")
        pipeline.text_encoder_2 = _load_component(type(pipeline.text_encoder_2), text_encoder_2, dtype)
    pipeline.to(device)
    if device == "mps":
        pipeline.enable_attention_slicing()
    return pipeline


def _load_component(cls, path: Path, dtype):
    """Load one pipeline component from a diffusers folder or a single file, offline."""
    if path.is_dir():
        return cls.from_pretrained(str(path), torch_dtype=dtype, local_files_only=True)
    return cls.from_single_file(str(path), torch_dtype=dtype)


def generate(args: argparse.Namespace) -> None:
    """Load the model and write the picture. Prints a JSON line with the result on success."""
    from PIL import Image

    if args.mask is not None:
        mode = "inpaint"
    elif args.reference is not None:
        mode = "img2img"
    else:
        mode = "txt2img"
    pipeline = load_pipeline(args.model, mode=mode, text_encoder=args.text_encoder,
                             text_encoder_2=args.text_encoder_2, vae=args.vae)
    if mode == "inpaint":
        with Image.open(args.reference) as opened:
            image = opened.convert("RGB").resize((args.width, args.height))
        with Image.open(args.mask) as opened:
            mask_image = opened.convert("L").resize((args.width, args.height))
        result = pipeline(
            prompt=args.prompt,
            image=image,
            mask_image=mask_image,
            strength=args.strength,
            negative_prompt=args.negative_prompt or None,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=_generator(pipeline, args.seed),
        ).images[0]
    elif mode == "img2img":
        with Image.open(args.reference) as opened:
            reference = opened.convert("RGB").resize((args.width, args.height))
        result = pipeline(
            prompt=args.prompt,
            image=reference,
            strength=args.strength,
            negative_prompt=args.negative_prompt or None,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=_generator(pipeline, args.seed),
        ).images[0]
    else:
        result = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt or None,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            width=args.width,
            height=args.height,
            generator=_generator(pipeline, args.seed),
        ).images[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(json.dumps({"output": str(args.output)}), flush=True)


def _generator(pipeline, seed: int):
    import torch
    return torch.Generator(device=getattr(pipeline, "device", "cpu")).manual_seed(seed)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.prompt.strip():
        print("error: a prompt is required", file=sys.stderr)
        return 2
    if args.mask is not None and args.reference is None:
        print("error: --mask needs --reference (the picture to inpaint)", file=sys.stderr)
        return 2
    started = time.time()
    try:
        generate(args)
    except SystemExit:
        raise
    except Exception as exc:  # the viewer reads the last non-progress line as the reason
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"generated in {time.time() - started:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
