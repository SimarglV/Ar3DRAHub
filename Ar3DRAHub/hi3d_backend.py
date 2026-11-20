"""Refine a coarse mesh with Hi3D v1.0: image + coarse mesh -> geometry-only GLB.

Hi3D v1.0 (hi3d-official/Hi3Dv1.0) takes a reference image and an existing coarse
mesh of the same object and refines the geometry through sparse latent passes
(512/1024/1536). It produces **geometry only**: no textures, no vertex colors, and
the output is recentered and normalized to unit radius, so world scale is not
preserved. This backend is the thin runner around the checkout that
``scripts/bootstrap_hi3d.py`` builds in ``vendor/hi3d/``; it runs upstream's own
``demo/refine_mesh.py``, so the pipeline calls can never drift from what the
authors release.

Two routes exist, both NVIDIA-only (no CPU path, no Metal backend). Linux runs the
official measured stack (flash-attn from source, spconv-cu120, CUDA 12.4). Windows
runs this repo's own SDPA attention port because flash-attn has no Windows build
(``scripts/patch_hi3d_sdpa.py``) -- that route is ours, not a stack the authors
measured. Generation never fetches weights: all three model repos (Hi3D artifacts,
DINOv2-large, CLIP ViT-L/14) must already sit in the Hugging Face cache.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Hi3DOptions:
    repo: Path
    resolution: int = 512
    steps: int = 25
    seed: int = 0
    cpu_offload: bool = True


#: The three model repos the official loader consumes, with approximate sizes.
WEIGHTS = (
    ("hi3d-official/Hi3Dv1.0", "about 7.2 GB"),
    ("facebook/dinov2-large", "about 1.2 GB"),
    ("openai/clip-vit-large-patch14", "about 1.6 GB"),
)

CHECKOUT_MARKER = ".hi3d-build-complete"

#: The eight artifacts upstream's loader requires, in their fixed layout. The loader
#: rejects symlinks it does not expect, so the bootstrap materializes these into a
#: real folder (`models/hi3d/`) instead of pointing at the symlinked hub snapshot.
REQUIRED_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "vae/config.json",
    "vae/model.safetensors",
    "transformer_512/config.json",
    "transformer_512/model.safetensors",
    "transformer_1024/config.json",
    "transformer_1024/model.safetensors",
)


def venv_python(repo: Path) -> Path:
    """The checkout venv's interpreter, named the way this OS names it."""
    from Ar3DRAHub import host

    return host.venv_python(repo)


def setup_command() -> str:
    """The script that installs Hi3D on this machine: Linux official or Windows port."""
    return "scripts/bootstrap_hi3d.py"


def platform_refusal() -> str | None:
    """Why this machine cannot run Hi3D, or None.

    Two routes exist. Linux + NVIDIA runs the official measured stack. Windows +
    NVIDIA runs this repo's SDPA attention port (`scripts/patch_hi3d_sdpa.py`)
    because flash-attn has no Windows build; that route is ours, not a stack the
    Hi3D authors measured. Everything else (macOS included) has no path: there is
    no CPU fallback and no Metal backend.
    """
    from Ar3DRAHub import host

    if host.host_platform() != host.NVIDIA:
        return "Hi3D v1.0 needs an NVIDIA GPU with compute capability 8.0 or newer."
    if host.os_family() not in ("linux", "windows"):
        return ("Hi3D v1.0 has no CPU path and no Metal backend; it runs on Linux or "
                "Windows with an NVIDIA card only.")
    return None


def verify_paths(repo: Path) -> list[str]:
    """Cheap filesystem checks that the checkout is present and finished building."""
    from Ar3DRAHub import host

    checks = {
        "venv python": host.venv_python(repo),
        "refine_mesh.py": repo / "Hi3Dv1.0" / "demo" / "refine_mesh.py",
        f"completed build ({setup_command()})": repo / CHECKOUT_MARKER,
    }
    if host.os_family() == "windows":
        checks["SDPA port applied"] = repo / "Hi3Dv1.0" / "sparc3d" / "_sdpa_varlen.py"
    return [f"missing {name}: {path}" for name, path in checks.items() if not path.exists()]


def build_refine_command(
    python: Path,
    refine_mesh: Path,
    image: Path,
    coarse_mesh: Path,
    output: Path,
    options: Hi3DOptions,
    model_dir: str,
    dino_dir: str,
    clip_dir: str,
) -> list[str]:
    """The argv handed to upstream's ``demo/refine_mesh.py``.

    Pulled out of ``generate_hi3d`` so the flags can be asserted without a GPU run.
    ``--local-files-only`` is always passed: generation never fetches weights, and
    the model directories come from the weight gate, which pins the process offline.
    """
    command = [
        str(python),
        str(refine_mesh),
        "--model",
        model_dir,
        "--image",
        str(image),
        "--coarse-mesh",
        str(coarse_mesh),
        "--output",
        str(output),
        "--resolution",
        str(options.resolution),
        "--num-inference-steps",
        str(options.steps),
        "--seed",
        str(options.seed),
        "--local-files-only",
        "--dino-model",
        dino_dir,
        "--clip-model",
        clip_dir,
    ]
    if options.cpu_offload:
        # Off by default upstream, on by default here: the only card upstream
        # validated (96 GB H20) is bigger than anything this repo has run on, and
        # offload is the memory-safe path on 24-32 GB cards.
        command.append("--sequential-cpu-offload")
    return command


def prepare_image(image: Path, output_dir: Path) -> Path:
    """The image to condition on, cut out by our own remover if it has no matte.

    Hi3D conditions on the subject, and upstream's own background removal is opt-in
    rembg (installed only with the demo extra, which we do not install). A real
    cut-out goes straight in; anything else is matted here.
    """
    from PIL import Image

    from Ar3DRAHub.matte import cut_out, is_matted

    with Image.open(image) as opened:
        if is_matted(opened):
            destination = output_dir / f"{image.stem}_hi3d_input.png"
            opened.convert("RGBA").save(destination)
            return destination
        cut, _model = cut_out(opened.convert("RGB"))
        destination = output_dir / f"{image.stem}_hi3d_input.png"
        cut.save(destination)
        print(f"  [hi3d] matted input -> {destination}", file=sys.stderr)
        return destination


def model_dir_present() -> tuple[bool, Path, list[str]]:
    """Whether the materialized model folder has every required artifact.

    The official loader rejects symlinks it does not expect, and the hub cache
    snapshots on Windows are symlinked, so the bootstrap materializes the eight
    required files into the route's own models folder instead.
    """
    from Ar3DRAHub import models_home

    root = models_home.vendor_models("hi3d")
    missing = [relative for relative in REQUIRED_FILES
               if not (root / relative).is_file()]
    return not missing, root, missing


def generate_hi3d(
    image: Path,
    coarse_mesh: Path,
    output_dir: Path,
    options: Hi3DOptions,
    log: Any | None = None,
) -> Path:
    """Refine `coarse_mesh` against `image` and return the refined GLB.

    Raises RuntimeError with a reason when the platform, the checkout or the
    weights are not ready. The weights gate (require_local) refuses with sizes
    when any of the three repos is not already on disk. `log(line)` receives
    the refinement's own output line by line, when the caller wants a live log.
    """
    refusal = platform_refusal()
    if refusal:
        raise RuntimeError(refusal)
    repo = options.repo.expanduser().resolve()
    problems = verify_paths(repo)
    if problems:
        raise RuntimeError(
            f"Hi3D environment not found at {repo}: {', '.join(problems)}. "
            f"Run {setup_command()} first."
        )
    if not coarse_mesh.is_file():
        raise ValueError(f"coarse mesh does not exist: {coarse_mesh}")

    from Ar3DRAHub.weight_gate import refuse_hub_download, require_local

    present, model_dir, missing = model_dir_present()
    if not present:
        refuse_hub_download(
            backend="Hi3D v1.0",
            route="--hi3d (models/hi3d)",
            size="about 7.2 GB",
            detail=f"missing: {', '.join(missing)}. "
                   f"Run {setup_command()} --weights-only.",
        )
    dino_dir = require_local(
        "facebook/dinov2-large",
        backend="Hi3D v1.0",
        route="--hi3d (vendor/hi3d)",
        size=WEIGHTS[1][1],
    )
    clip_dir = require_local(
        "openai/clip-vit-large-patch14",
        backend="Hi3D v1.0",
        route="--hi3d (vendor/hi3d)",
        size=WEIGHTS[2][1],
    )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_image(image, output_dir)
    output_path = output_dir / f"{image.stem}_hi3d.glb"
    python = venv_python(repo)
    command = build_refine_command(
        python,
        repo / "Hi3Dv1.0" / "demo" / "refine_mesh.py",
        prepared,
        coarse_mesh,
        output_path,
        options,
        str(model_dir),
        dino_dir,
        clip_dir,
    )
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    if log is None:
        try:
            subprocess.run(command, cwd=repo, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Hi3D refinement failed with exit code {exc.returncode}"
            ) from exc
    else:
        process = subprocess.Popen(
            command, cwd=repo, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
            text=True,
        )
        assert process.stdout is not None
        for raw in process.stdout:
            log(raw.rstrip("\n"))
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"Hi3D refinement failed with exit code {code}")
    if not output_path.is_file():
        raise RuntimeError(f"Hi3D reported success but did not create {output_path}")
    return output_path
