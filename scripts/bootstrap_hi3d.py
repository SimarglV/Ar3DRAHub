#!/usr/bin/env python3
"""Install Hi3D v1.0: the official Linux stack, or this repo's own Windows port.

Hi3D v1.0 refines a coarse mesh against a reference image into high-resolution
geometry-only GLBs. The official stack is Linux + NVIDIA only: torch 2.5.0+cu124,
spconv-cu120 and flash-attn built from source. This script adds a second route for
Windows + NVIDIA: the same official code with this repo's SDPA attention port
(`patch_hi3d_sdpa.py`) replacing flash-attn, on a pinned stack that has Windows
builds (torch 2.11.0+cu128, spconv-cu126 2.3.8, triton-windows). The Windows route
is our own port, not a stack the Hi3D authors measured; the Linux route is their
verified one.

The checkout is cloned at a pinned commit into `vendor/hi3d/`, gets its own Python
3.10 venv (upstream requires >=3.10,<3.11), then the weights: Hi3D v1.0 artifacts
(about 7.2 GB), DINOv2-large (about 1.2 GB) and CLIP ViT-L/14 (about 1.6 GB), about
10 GB total, all public on Hugging Face.

`AGENTS.md`: a download path must name the backend, name the route, state the size,
and require an affirmative answer. This prints all of that and stops, unless
`--yes` is given for non-interactive use. Defaulting to yes is not allowed, so it
does not.

    python scripts/bootstrap_hi3d.py            # says what it wants, then asks
    python scripts/bootstrap_hi3d.py --yes      # for the viewer and for agents
    python scripts/bootstrap_hi3d.py --code-only
    python scripts/bootstrap_hi3d.py --weights-only
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
sys.path.insert(0, str(REPO / "scripts"))

import patch_hi3d_sdpa as sdpa_patch  # noqa: E402, RUF100

from Ar3DRAHub import host, models_home  # noqa: E402, RUF100
from Ar3DRAHub.hi3d_backend import REQUIRED_FILES  # noqa: E402, RUF100

VENDOR = REPO / "vendor" / "hi3d"
CHECKOUT = VENDOR / "Hi3Dv1.0"
BUILT_MARKER = VENDOR / ".hi3d-build-complete"
UPSTREAM = "https://github.com/hi3d-official/Hi3Dv1.0.git"
PINNED_COMMIT = "13657ad09afaa92e350b461a1cd66399e44a59c4"
PYTHON_VERSION = "3.10"

# (repo, allow_patterns or None, approximate gigabytes)
WEIGHTS = [
    ("hi3d-official/Hi3Dv1.0", list(REQUIRED_FILES), 7.2),
    ("facebook/dinov2-large", ["*.json", "model.safetensors"], 1.2),
    ("openai/clip-vit-large-patch14",
     ["*.json", "model.safetensors", "vocab.json", "merges.txt"], 1.6),
]

# The Windows route's stack. torch/torchvision come from the cu128 index because
# PyPI's default Windows torch is CPU-only; cu128 is the first build that knows
# RTX 50-series cards, and 2.11.0 pairs with the torchvision below.
TORCH_WIN = "2.11.0+cu128"
TORCHVISION_WIN = "0.26.0+cu128"
# Everything else the checkout's own pyproject declares, on pins with Windows wheels.
WINDOWS_PINS = [
    "triton-windows==3.6.0.post26",
    "spconv-cu126==2.3.8",
    "transformers==4.46.3",
    "accelerate==1.6.0",
    "numpy==1.26.4",
    "safetensors==0.5.3",
    "huggingface-hub==0.34.4",
    "pyfqmr==0.4.0",
    "Pillow==11.2.1",
    "trimesh==4.6.8",
]

# The Linux route installs upstream's own measured stack: torch cu124, then the
# [inference] extra with the constraints file, which pulls flash-attn (source
# build, needs nvcc 12.x) and spconv-cu120. The SDPA patch is applied on Linux too
# but stays unused while flash-attn is present.
TORCH_LINUX = "2.5.0+cu124"
TORCHVISION_LINUX = "0.20.0+cu124"

LICENCE = (
    "Hi3D v1.0 code + weights: Hi3D Open License Agreement v1.0\n"
    "  (commercial use allowed; >200k MAU or >USD 1M/month revenue needs a separate\n"
    "  written licence; commercial integrations must show 'Powered by Hi3D 1.0').\n"
    "DINOv2-large: Apache-2.0. CLIP ViT-L/14: MIT."
)


def total_gb() -> float:
    return sum(size for _, _, size in WEIGHTS)


def venv_python(root: Path = VENDOR) -> Path:
    """The checkout venv's interpreter, named the way this OS names it."""
    return host.venv_python(root)


def choose_route(family: str | None = None, capability: str | None = None,
                 driver: tuple[int, int] | None = None,
                 platform_id: str | None = None) -> tuple[str | None, str]:
    """("linux-official" | "windows-sdpa" | None, why)."""
    platform_id = platform_id or host.host_platform()
    family = family or host.os_family()
    if platform_id != host.NVIDIA:
        return None, "Hi3D v1.0 needs an NVIDIA GPU; there is no CPU path"
    if family == "linux":
        if driver is None:
            driver = host.driver_cuda_version()
        if driver is not None and driver < (12, 4):
            return None, (f"the driver supports CUDA {driver[0]}.{driver[1]}; the "
                          "measured stack is torch 2.5.0+cu124, which needs CUDA 12.4")
        return "linux-official", ("upstream's measured stack: torch 2.5.0+cu124, "
                                  "flash-attn built from source, spconv-cu120")
    if family == "windows":
        if capability is None:
            capability = host.compute_capability()
        if capability is None or not capability.isdigit() or int(capability) < 80:
            return None, ("compute capability 8.0 or newer is required; this card "
                          f"reports {capability or 'nothing'}")
        return "windows-sdpa", ("this repo's SDPA port: torch 2.11.0+cu128, "
                                "spconv-cu126 2.3.8, triton-windows, no flash-attn")
    return None, ("Hi3D v1.0 runs on Linux or Windows with an NVIDIA card; this is "
                  f"{family or 'an unknown platform'}")


def announcement(route: str | None, why: str, code: bool = True,
                 weights: bool = True) -> str:
    lines = ["", "About to install:", "",
             "  backend: Hi3D v1.0 (hi3d-official)", 
             f"  route:   {route or 'none'}: {why}"]
    if code:
        lines += [
            f"  code:    hi3d-official/Hi3Dv1.0 @ {PINNED_COMMIT[:7]} -> vendor/hi3d/",
            f"           plus its own Python {PYTHON_VERSION} venv (several GB of packages)",
        ]
        if route == "linux-official":
            lines.append("           flash-attn compiles from source: count on 30-60 minutes")
        else:
            lines.append("           SDPA attention port applied (scripts/patch_hi3d_sdpa.py)")
    if weights:
        destination = (Path(os.environ["HF_HOME"]) / "hub").resolve() \
            if os.environ.get("HF_HOME") else Path.home() / ".cache" / "huggingface" / "hub"
        lines.append(f"  weights: {total_gb():.1f} GB total -> {destination}")
        for repo, _, size in WEIGHTS:
            lines.append(f"             {size:>6.2f} GB  {repo}")
    lines += ["", "  licence: " + LICENCE, "",
              "  and:     VRAM. The authors measured on a 96 GB H20; on smaller cards",
              "           start at 512 with CPU offload (the CLI default)."]
    return "\n".join(lines)


def pip_commands(route: str, uv: str, python: Path,
                 checkout: Path = CHECKOUT) -> list[list[str]]:
    """Every package install, in order. Pure, so the order and flags can be tested."""
    pip = [uv, "pip", "install", "--python", str(python)]
    if route == "windows-sdpa":
        return [
            [*pip, f"torch=={TORCH_WIN}", f"torchvision=={TORCHVISION_WIN}",
             "--index-url", host.TORCH_INDEX + "cu128"],
            [*pip, *WINDOWS_PINS],
            [*pip, "--no-deps", "-e", str(checkout)],
        ]
    if route == "linux-official":
        constraints = checkout / "docker" / "constraints.txt"
        return [
            [*pip, f"torch=={TORCH_LINUX}", f"torchvision=={TORCHVISION_LINUX}",
             "--index-url", host.TORCH_INDEX + "cu124"],
            [*pip, "--no-build-isolation", "flash-attn==2.7.4.post1"],
            [*pip, "-c", str(constraints), "-e", f"{checkout}[inference]"],
        ]
    raise ValueError(f"no install plan for route {route!r}")


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


def install_code(route: str) -> None:
    uv = host.find_uv()
    if uv is None:
        raise SystemExit("uv is required: https://docs.astral.sh/uv/")
    fetch_checkout()
    changed = sdpa_patch.apply(CHECKOUT)
    print(f"SDPA fallback: {'applied' if changed else 'already present'}", flush=True)
    if not venv_python().is_file():
        run([uv, "venv", str(VENDOR / ".venv"), "--python", PYTHON_VERSION])
    env = host.patient_downloads(os.environ)
    for command in pip_commands(route, uv, venv_python()):
        run(command, env=env)
    BUILT_MARKER.write_text(json.dumps({"route": route, "commit": PINNED_COMMIT}) + "\n")


def materialize(snapshot: Path, target: Path) -> int:
    """Copy the loader's required files out of the symlinked hub snapshot.

    The official loader rejects symlinks it does not expect, and the hub cache
    snapshots on Windows are symlinked, so the eight required artifacts are
    copied into the route's own models folder as real files. Idempotent: skips
    files already present at the same size. Returns how many were written.
    """
    written = 0
    for relative in REQUIRED_FILES:
        source = snapshot / relative
        destination = target / relative
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as handle, destination.open("wb") as out:
            out.write(handle.read())
        written += 1
    return written


def install_weights() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is not installed. uv sync") from exc
    for repo, patterns, size in WEIGHTS:
        print(f"\nFetching {repo} ({size:.2f} GB)...", flush=True)
        snapshot_download(repo, allow_patterns=patterns)
    # The Hi3D snapshot lands as symlinks; materialize what the loader consumes.
    snapshot = snapshot_download(
        "hi3d-official/Hi3Dv1.0", allow_patterns=list(REQUIRED_FILES),
        local_files_only=True,
    )
    target = models_home.vendor_models("hi3d")
    written = materialize(Path(snapshot), target)
    print(f"Materialized {written} model file(s) into {target}", flush=True)


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
        print(f"Cannot install Hi3D v1.0 here: {why}. Nothing downloaded.")
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
        install_code(route)
    if weights:
        install_weights()
    print("\nDone. Refine a coarse mesh with:\n"
          "    python pipeline.py IMAGE --hi3d --coarse-mesh coarse.glb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
