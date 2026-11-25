#!/usr/bin/env python3
"""Install the diffusers picture runtime: its own venv with torch and diffusers.

This runtime draws pictures from raw diffusion checkpoints (a single `.safetensors` /
`.ckpt` file, or a diffusers-format folder with `model_index.json`) dropped into
`models/pictures/`. It installs only the software — torch and diffusers and their friends —
never model weights; the models are the files you drop.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops, unless `--yes` is given
for non-interactive use. Defaulting to yes is not allowed, so it does not.

    python scripts/bootstrap_diffusers.py           # says what it wants, then asks
    python scripts/bootstrap_diffusers.py --yes     # for the viewer and for agents
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Ar3DRAHub import host  # noqa: E402

VENDOR = REPO / "vendor" / "diffusers"
BUILT_MARKER = VENDOR / ".diffusers-build-complete"
PYTHON_VERSION = "3.12"

# The versions the other torch routes here already run (bootstrap_ultratex proved cu128 on
# RTX 50-series cards), so one stack serves every route.
TORCH = "2.11.0"
TORCHVISION = "0.26.0"
PINS = [
    "diffusers==0.30.1",
    "safetensors==0.7.0",
    "transformers==5.2.0",
    "accelerate==1.11.0",
    "Pillow==11.3.0",
    "numpy==1.26.4",
    "huggingface-hub==1.4.1",
    "sentencepiece==0.2.0",
    "protobuf==6.33.1",
]

# Packages are a few GB unpacked; the size is a floor, not a promise.
SIZE_GB = 4.0


def torch_install(platform_id: str | None = None, driver=None) -> tuple[str | None, str, str]:
    """(index-url or None, torch spec, torchvision spec) for this machine.

    NVIDIA gets a CUDA build from PyTorch's own index; Apple and everything else get the
    PyPI default (Metal/MPS on Apple Silicon, CPU elsewhere).
    """
    platform_id = platform_id or host.host_platform()
    if platform_id == host.NVIDIA:
        index = host.torch_cuda_index(driver) or (host.TORCH_INDEX + "cu128")
        tag = index.rstrip("/").rsplit("/", 1)[-1]
        return index, f"torch=={TORCH}+{tag}", f"torchvision=={TORCHVISION}+{tag}"
    return None, "torch", "torchvision"


def pip_commands(uv: str, python: Path, platform_id: str | None = None, driver=None) -> list[list[str]]:
    """Every package install, in order. Pure, so the order and flags can be tested."""
    pip = [uv, "pip", "install", "--python", str(python)]
    index, torch, torchvision = torch_install(platform_id, driver)
    torch_line = [*pip, torch, torchvision]
    if index:
        torch_line += ["--index-url", index]
    return [torch_line, [*pip, *PINS]]


def announcement(platform_id: str | None = None, driver=None) -> str:
    index, torch, _ = torch_install(platform_id, driver)
    route = f"torch {torch}" + (f" from {index}" if index else " from PyPI")
    return "\n".join([
        "", "About to install:", "",
        "  backend: the diffusers picture runtime (text-to-image from dropped checkpoints)",
        f"  route:   {route}",
        f"  code:    its own Python {PYTHON_VERSION} venv under vendor/diffusers/",
        f"           (about {SIZE_GB:.1f} GB of packages)",
        "",
        "  weights: none — the models are the files you drop into models/pictures/.",
        "",
        "  licence: MIT.",
        "           Each model you drop carries its own licence; the studio records it.",
        "",
    ])


def venv_python() -> Path:
    return host.venv_python(VENDOR)


def build_present() -> bool:
    return BUILT_MARKER.is_file()


def install() -> None:
    uv = host.find_uv()
    if uv is None:
        raise SystemExit("uv is required: https://docs.astral.sh/uv/")
    if not venv_python().is_file():
        VENDOR.mkdir(parents=True, exist_ok=True)
        subprocess.run([uv, "venv", str(VENDOR / ".venv"), "--python", PYTHON_VERSION], check=True)
    env = host.patient_downloads(os.environ)
    platform_id, driver = host.host_platform(), host.driver_cuda_version()
    for command in pip_commands(uv, venv_python(), platform_id, driver):
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True, env=env)
    BUILT_MARKER.write_text(json.dumps({"torch": TORCH, "platform": platform_id}) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args(argv)
    if build_present():
        print("The diffusers runtime is already installed.")
        return 0
    print(announcement())
    if not args.yes:
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing installed.")
            return 1
    install()
    print("\nDone. Drop a checkpoint into models/pictures/ and it appears in the picture picker.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
