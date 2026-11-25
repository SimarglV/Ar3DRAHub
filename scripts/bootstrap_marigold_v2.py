#!/usr/bin/env python3
"""Install Marigold V2's native predictor: its own venv, then the weights if missing.

The predictor (Ar3DRAHub/marigold_v2.py) runs the reference Marigold V2 single-step
depth / normals / albedo models over a Qwen-Image-Edit-2509 backbone. The backbone is a
GGUF build of the same 2509 DiT, so nothing is compiled and no NF4 pass runs at load. It
needs a Python environment with torch, diffusers and peft, which the studio's own
environment does not carry, so this makes a venv beside the route and installs pinned
versions there.

The **weights** (~22 GB) live in models/marigold-v2/ (the repo's models home): the Qwen
VAE and DiT config, the Q6_K GGUF, the three Marigold checkpoints and their prompt
embeddings. Already present -- nothing is fetched. Missing and this installs them.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops, unless `--yes` is given.

    python scripts/bootstrap_marigold_v2.py            # says what it wants, then asks
    python scripts/bootstrap_marigold_v2.py --yes      # for the viewer and for agents
    python scripts/bootstrap_marigold_v2.py --code-only
    python scripts/bootstrap_marigold_v2.py --weights-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Ar3DRAHub import host, marigold_v2 as m

VENDOR = REPO / "vendor" / "marigold-v2"
SOURCE_PYTHON = "3.12"

# PyTorch for the driver's CUDA, pinned to the version proven on this route. The rest are
# the exact versions the predictor was validated against (torch 2.14.1+cu130, diffusers
# 0.37, peft 0.21, bitsandbytes 0.50): the GGUF path does not quantize at load, so
# bitsandbytes is only present for diffusers' 4-bit config types, not exercised.
TORCH = "2.14.1"
PACKAGES = [
    "bitsandbytes==0.50.0",
    "diffusers==0.37.0",
    "peft==0.21.1",
    "transformers==5.22.2",
    "accelerate==1.12.0",
    "safetensors==0.8.0",
    "huggingface_hub",
    "gguf>=0.10.0",  # diffusers needs it to load the Q6_K GGUF backbone
    "numpy",
    "pillow",
]

# (repo, allow_patterns or None, approximate gigabytes). The first three map onto the
# models/marigold-v2 layout the predictor reads; a test holds them against the catalogue.
QWEN_REPO = "Qwen/Qwen-Image-Edit-2509"
GGUF_REPO = "QuantStack/Qwen-Image-Edit-2509-GGUF"
MARIGOLD_REPO = "huawei-bayerlab/marigold-v2-0"
WEIGHTS = [
    (QWEN_REPO, ["model_index.json", "vae/*", "transformer/config.json"], 0.24),
    (GGUF_REPO, ["Qwen-Image-Edit-2509-Q6_K.gguf"], 16.8),
    (MARIGOLD_REPO, ["depth/Log-stage2/*", "normals/*", "albedo/*",
                     "qwen_text_embeddings/*"], 5.6),
]

LICENCE = (
    "Marigold V2 code and models: MIT.\n"
    f"  Qwen-Image-Edit-2509 backbone: its own licence (see {QWEN_REPO})."
)


def venv_python() -> Path:
    return host.venv_python(VENDOR)


def total_gb() -> float:
    return sum(size for _, _, size in WEIGHTS)


def announcement(code: bool = True, weights: bool = True) -> str:
    """Exactly what is about to be fetched, before anything is."""
    lines = ["", "About to install:", "",
             "  backend: Marigold V2 (depth, normals, albedo)",
             "  route:   GGUF via diffusers + PyTorch CUDA, on an NVIDIA card"]
    if code:
        lines += [
            f"  venv:    {venv_python()}",
            f"           torch=={TORCH} (CUDA) + diffusers/peft/bitsandbytes, pinned",
        ]
    if weights:
        lines += [f"  weights: {total_gb():.1f} GB total -> models/marigold-v2/"]
        for repo, _, size in WEIGHTS:
            lines.append(f"             {size:>5.2f} GB  {repo}")
    lines += ["", "  licence: " + LICENCE, ""]
    return "\n".join(lines)


def build_env(uv: str) -> None:
    """Make the venv and install the pinned packages into it."""
    if not venv_python().is_file():
        subprocess.run([uv, "venv", str(VENDOR / ".venv"), "--python", SOURCE_PYTHON],
                       check=True)
    python = venv_python()
    index = host.torch_cuda_index(host.driver_cuda_version())
    if index is None:
        raise SystemExit("No PyTorch CUDA build matches this driver; update the NVIDIA driver.")
    subprocess.run([uv, "pip", "install", "--python", str(python),
                    f"torch=={TORCH}", "--index-url", index], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(python), *PACKAGES], check=True)


def install_weights() -> None:
    """Fetch every weight the predictor needs but does not have, into models/marigold-v2."""
    models = m.models_dir()
    missing = {name for mod in m.MODALITIES for name in m.missing_files(models, mod)}
    if not missing:
        print(f"weights are already in {models}; nothing fetched.", flush=True)
        return
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is not installed. uv sync") from exc

    def need(*parts: str) -> bool:
        return "/".join(parts) in missing

    qwen_dir = models / m.QWEN_DIR_NAME
    if need("vae", "config.json") or need("transformer", "config.json"):
        snapshot_download(repo_id=QWEN_REPO, repo_type="model", local_dir=str(qwen_dir),
                          allow_patterns=["model_index.json", "vae/*", "transformer/config.json"])
    gguf = models / "gguf" / m.GGUF_NAME
    if need("gguf", m.GGUF_NAME):
        gguf.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=GGUF_REPO, filename=m.GGUF_NAME, repo_type="model",
                        local_dir=str(models / "gguf"))
    if any(name.startswith(m.CHECKPOINTS_DIR_NAME + "/") for name in missing):
        snapshot_download(repo_id=MARIGOLD_REPO, repo_type="model",
                          local_dir=str(models / m.CHECKPOINTS_DIR_NAME),
                          allow_patterns=["depth/Log-stage2/*", "normals/*", "albedo/*",
                                          "qwen_text_embeddings/*"])
    print(f"weights installed into {models}.", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the viewer and for scripts.")
    parser.add_argument("--code-only", action="store_true",
                        help="Make the venv and stop, leaving the weights.")
    parser.add_argument("--weights-only", action="store_true",
                        help="Fetch the weights only, assuming the venv is already present.")
    args = parser.parse_args(argv)

    code = not args.weights_only
    weights = not args.code_only
    print(announcement(code=code, weights=weights))

    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing installed.")
            return 1

    if code:
        uv = host.find_uv()
        if uv is None:
            raise SystemExit("uv is required. Install it (install.ps1 / install.sh), then rerun.")
        build_env(uv)
    if weights:
        install_weights()
    print("\nDone. Predict depth/normals/albedo with:")
    print(f"    {venv_python()} scripts/marigold_v2_predict.py input.png --modality all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
