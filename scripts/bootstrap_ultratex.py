#!/usr/bin/env python3
"""Install UltraTex texturing: the official repo, its venv, and about 36 GB of weights.

UltraTex (SIGGRAPH Asia 2026, yiboz2001/UltraTex) paints a textured 3D asset's
albedo and metallic-roughness maps from the reference picture: it renders the mesh's
six canonical views, denoises them through FLUX.2-klein-base-9B with the UltraTex
LoRA, and the views are baked onto the asset's UVs. This script clones the official
repo at a pinned commit into `vendor/ultratex/`, builds its own Python 3.12 venv on
the stack this repo's NVIDIA routes already use (torch 2.11.0+cu128, which knows
RTX 50-series cards; no Triton — the dense attention path needs none), and fetches
the weights:

- base models through Hugging Face, into the project's models folder
  (models/huggingface, or wherever the Setup page points ASSETFURNACE_MODELS):
  FLUX.2-klein-base-9B (about 18.2 GB), the FLUX.2 autoencoder (about 0.4 GB) and
  Qwen3-8B (about 16.4 GB);
- the UltraTex LoRAs (albedo + metallic-roughness) and the foreground-aware VAE
  decoder (about 1.4 GB) from ModelScope, the only host of the official files,
  into `models/ultratex/checkpoints/` in the layout the official code reads.

`AGENTS.md`: a download path must name the backend, name the route, state the size,
and require an affirmative answer. This prints all of that and stops, unless
`--yes` is given for non-interactive use. Defaulting to yes is not allowed, so it
does not.

    python scripts/bootstrap_ultratex.py            # says what it wants, then asks
    python scripts/bootstrap_ultratex.py --yes      # for the viewer and for agents
    python scripts/bootstrap_ultratex.py --code-only
    python scripts/bootstrap_ultratex.py --weights-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Ar3DRAHub import host, models_home  # noqa: E402, RUF100
from Ar3DRAHub.ultratex_backend import LORA_FILES, checkpoints_dir  # noqa: E402, RUF100

VENDOR = REPO / "vendor" / "ultratex"
CHECKOUT = VENDOR / "UltraTex"
BUILT_MARKER = VENDOR / ".ultratex-build-complete"
UPSTREAM = "https://github.com/yiboz2001/UltraTex.git"
PINNED_COMMIT = "a726678f3f3631fe2c5e3d1fb4597114784c7a8b"
PYTHON_VERSION = "3.12"

# (repo, allow_patterns or None, approximate gigabytes). The UltraTex LoRA and
# decoder come from ModelScope separately, not from the Hub.
WEIGHTS = [
    ("black-forest-labs/FLUX.2-klein-base-9B",
     ["flux-2-klein-base-9b.safetensors", "*.json", "LICENSE.md"], 18.2),
    ("black-forest-labs/FLUX.2-dev", ["ae.safetensors"], 0.4),
    ("Qwen/Qwen3-8B",
     ["*.json", "model-*.safetensors", "tokenizer.json", "vocab.json",
      "merges.txt", "LICENSE", "*.md"], 16.4),
]
MODELSCOPE_MODEL = "Yibo-Zhang/UltraTex"
MODELSCOPE_SIZE = 1.4

# The stack this repo's NVIDIA routes already run: cu128 is the first CUDA build
# that knows RTX 50-series cards. UltraTex's own pin (torch 2.5.1+cu124) predates
# Blackwell and would not load on this card. triton-windows runs the paper's
# sparse-attention kernel (the same pin the Hi3D route proved on Windows);
# deepspeed and tensorboard are left out: nothing in the inference imports them.
TORCH = "2.11.0+cu128"
TORCHVISION = "0.26.0+cu128"
PINS = [
    "triton-windows==3.6.0.post26",
    "accelerate==1.11.0",
    "diffusers==0.30.1",
    "einops==0.8.0",
    "peft==0.19.1",
    "safetensors==0.7.0",
    "transformers==5.2.0",
    "numpy==1.26.4",
    "Pillow==11.3.0",
    "opencv-python-headless==4.11.0.86",
    "scikit-image==0.25.2",
    "tqdm==4.67.3",
    "PyYAML==6.0.3",
    "sentencepiece==0.2.0",
    "protobuf==6.33.1",
    "huggingface-hub==1.4.1",
    "modelscope==1.34.0",
]

LICENCE = (
    "MIT.\n"
    "  Do not host the model as a service and do not train a competing model.\n"
    "  The UltraTex LoRAs and decoder follow the FLUX weights."
)


def total_gb() -> float:
    return sum(size for _, _, size in WEIGHTS) + MODELSCOPE_SIZE


def venv_python(root: Path = VENDOR) -> Path:
    """The checkout venv's interpreter, named the way this OS names it."""
    return host.venv_python(root)


def choose_route(platform_id: str | None = None, family: str | None = None) -> tuple[str | None, str]:
    """("cuda" | None, why). UltraTex runs wherever torch cu128 runs: NVIDIA, dense."""
    platform_id = platform_id or host.host_platform()
    family = family or host.os_family()
    if platform_id != host.NVIDIA:
        return None, ("UltraTex runs on an NVIDIA GPU only: its stack is torch cu128 "
                      "and there is no CPU or Metal path")
    if family not in ("linux", "windows"):
        return None, f"UltraTex runs on Linux or Windows with an NVIDIA card; this is {family}"
    return "cuda", ("torch 2.11.0+cu128, sparse attention on triton-windows (the "
                    "paper's speed mode); no deepspeed; RTX 50-series cards are supported")


def announcement(route: str | None, why: str, code: bool = True,
                 weights: bool = True) -> str:
    lines = ["", "About to install:", "",
             "  backend: UltraTex texturing (yiboz2001/UltraTex)",
             f"  route:   {route or 'none'}: {why}"]
    if code:
        lines += [
            f"  code:    yiboz2001/UltraTex @ {PINNED_COMMIT[:7]} -> vendor/ultratex/",
            f"           plus its own Python {PYTHON_VERSION} venv (several GB of packages)",
        ]
    if weights:
        root = models_home.chosen()
        lines.append(f"  weights: {total_gb():.1f} GB total -> {root}")
        for repo, _, size in WEIGHTS:
            lines.append(f"             {size:>6.2f} GB  {repo} (Hugging Face)")
        lines.append(f"             {MODELSCOPE_SIZE:>6.2f} GB  {MODELSCOPE_MODEL} "
                     "(ModelScope: the UltraTex LoRAs + decoder)")
    lines += ["", "  licence: " + LICENCE, ""]
    return "\n".join(lines)


def pip_commands(uv: str, python: Path) -> list[list[str]]:
    """Every package install, in order. Pure, so the order and flags can be tested."""
    pip = [uv, "pip", "install", "--python", str(python)]
    return [
        [*pip, f"torch=={TORCH}", f"torchvision=={TORCHVISION}",
         "--index-url", host.TORCH_INDEX + "cu128"],
        [*pip, *PINS],
    ]


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def build_present() -> bool:
    return BUILT_MARKER.is_file()


def fetch_checkout() -> None:
    if (CHECKOUT / ".git").is_dir():
        return
    VENDOR.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", UPSTREAM, str(CHECKOUT)])
    run(["git", "checkout", "--detach", PINNED_COMMIT], cwd=CHECKOUT)


def install_code() -> None:
    uv = host.find_uv()
    if uv is None:
        raise SystemExit("uv is required: https://docs.astral.sh/uv/")
    fetch_checkout()
    if not venv_python().is_file():
        run([uv, "venv", str(VENDOR / ".venv"), "--python", PYTHON_VERSION])
    env = host.patient_downloads(os.environ)
    for command in pip_commands(uv, venv_python()):
        run(command, env=env)
    BUILT_MARKER.write_text(json.dumps({"route": "cuda", "commit": PINNED_COMMIT}) + "\n")


def lora_files_present() -> list[str]:
    """Which of the three UltraTex files are still missing, by name."""
    root = checkpoints_dir()
    return [relative for relative in LORA_FILES if not (root / relative).is_file()]


def locate_modelscope_files(scratch: Path) -> dict[str, Path]:
    """Find the three FLUX.2 UltraTex files in a ModelScope download, by name.

    The ModelScope repo ships both backbones (flux1/, flux2/, flux2_mr/), so the
    search is constrained to the FLUX.2 folders: exactly one `flux2/decoder.pt`,
    one `flux2/lora/dit_lora.safetensors` and one `flux2_mr/lora/dit_lora.safetensors`.
    Returns {"albedo": ..., "mr": ..., "decoder": ...}.
    """
    relative = {p: p.relative_to(scratch).parts
                for p in scratch.rglob("*") if p.is_file()}

    def find(name: str, folder: str) -> Path:
        hits = [p for p, parts in relative.items() if p.name == name and folder in parts]
        if len(hits) != 1:
            raise RuntimeError(
                f"expected exactly one {folder}/{name} in the ModelScope download, "
                f"found {len(hits)}: {[str(p) for p in hits]}")
        return hits[0]

    return {
        "albedo": find("dit_lora.safetensors", "flux2"),
        "mr": find("dit_lora.safetensors", "flux2_mr"),
        "decoder": find("decoder.pt", "flux2"),
    }


def modelscope_cli() -> Path:
    """The modelscope console script beside the venv interpreter, named the way this
    OS names it (`modelscope.exe` on Windows, `modelscope` elsewhere)."""
    sibling = host.venv_python(VENDOR).with_name(
        "modelscope.exe" if host.os_family() == "windows" else "modelscope")
    return sibling


def install_ultratex_files() -> None:
    """The UltraTex LoRAs + decoder: ModelScope, searched, then placed in the
    official layout under models/ultratex/checkpoints/."""
    root = checkpoints_dir()
    missing = lora_files_present()
    if not missing:
        print("UltraTex LoRAs + decoder already present.", flush=True)
        return
    scratch = root.parent / ".modelscope-ultratex"
    if scratch.is_dir():
        print("Reusing the existing ModelScope download.", flush=True)
    else:
        scratch.mkdir(parents=True)
        run([str(modelscope_cli()), "download",
             "--model", MODELSCOPE_MODEL, "--local_dir", str(scratch)],
            env=host.patient_downloads(os.environ))
    found = locate_modelscope_files(scratch)
    destinations = {
        "albedo": root / "flux2" / "lora" / "dit_lora.safetensors",
        "mr": root / "flux2_mr" / "lora" / "dit_lora.safetensors",
        "decoder": root / "flux2" / "decoder.pt",
    }
    for key, destination in destinations.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == found[key].stat().st_size:
            continue
        shutil.copyfile(found[key], destination)
    shutil.rmtree(scratch)
    print(f"UltraTex files in place under {root}", flush=True)


def install_weights() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is not installed. uv sync") from exc
    for repo, patterns, size in WEIGHTS:
        print(f"\nFetching {repo} ({size:.2f} GB)...", flush=True)
        snapshot_download(repo, allow_patterns=patterns)
    install_ultratex_files()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the viewer and for agents.")
    parser.add_argument("--code-only", action="store_true",
                        help="Install the code and its venv, leaving the weights.")
    parser.add_argument("--weights-only", action="store_true",
                        help="Fetch the weights only.")
    args = parser.parse_args(argv)

    # The studio redirects HF_HOME to its own models folder; the bootstrap must land
    # the weights where the pipeline will look for them, not in the user cache.
    models_home.apply()

    route, why = choose_route()
    if route is None:
        print(f"Cannot install UltraTex here: {why}. Nothing downloaded.")
        return 1

    code = not args.weights_only
    weights = not args.code_only
    print(announcement(route, why, code=code, weights=weights))

    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1

    if code:
        install_code()
    if weights:
        install_weights()
    print("\nDone. The UltraTex stage lands in the finish pipeline next; for now the "
          "checkout and weights are ready in vendor/ultratex/ and models/ultratex/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
