"""Texture an asset with UltraTex: the reference picture + six rendered views.

UltraTex (SIGGRAPH Asia 2026, yiboz2001/UltraTex) denoises the six canonical
views of a mesh (4 sides + top + bottom, as world-space normal maps) into albedo
and metallic-roughness images through FLUX.2-klein-base-9B with the UltraTex
LoRA, conditioned on the normals and the reference picture. Baking the resulting
views onto the asset's UVs is a separate stage (`scripts/ultratex_bake_views.py`);
this backend is the thin runner around the checkout that
``scripts/bootstrap_ultratex.py`` builds in ``vendor/ultratex/``: it runs
upstream's own ``inference_flux2.py`` through a written config, so the pipeline
calls can never drift from what the authors release.

NVIDIA only: the stack is torch 2.11.0+cu128 with the paper's own speed modes
on -- background-token dropping and the vendored sparse-attention kernel
(triton-windows on this OS) -- so no deepspeed is needed. Generation never
fetches weights: the base models must sit in the Hugging Face cache (the
weight gate pins the process offline) and the UltraTex LoRAs + decoder must
sit in the materialized layout under models/ultratex/checkpoints/.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class UltraTexOptions:
    repo: Path
    resolution: int = 1024
    steps: int = 25
    seed: int = 42
    guidance: float = 4.0
    task: str = "albedo"      # "albedo" or "mr"


@dataclass(frozen=True)
class UltraTexResult:
    output_dir: Path
    canvas: Path              # the 2x3 view canvas the model wrote
    task: str


#: The three base-model repos the official loader consumes, with approximate sizes.
WEIGHTS = (
    ("black-forest-labs/FLUX.2-klein-base-9B", "about 18.2 GB"),
    ("black-forest-labs/FLUX.2-dev", "about 0.4 GB"),
    ("Qwen/Qwen3-8B", "about 16.4 GB"),
)

CHECKOUT_MARKER = ".ultratex-build-complete"

#: The six canonical cameras as G-buffer TexVerse pose matrices (camera-to-world:
#: columns are the camera's axes -- right, up, backward -- and the fourth column is
#: the position; the camera looks down its own -Z, i.e. forward is -column 2).
#: Copied verbatim from the demo pose files shipped in the UltraTex checkout, so the
#: renders and the UV bake match what the model saw during training.
POSES = (
    ((1.0, 0.0, 0.0, 0.0), (0.0, -0.0, -1.0, -1.67), (0.0, 1.0, -0.0, 0.0)),
    ((-0.0, 0.0, -1.0, -1.67), (-1.0, -0.0, 0.0, 0.0), (0.0, 1.0, -0.0, 0.0)),
    ((-1.0, 0.0, 0.0, 0.0), (0.0, -0.0, 1.0, 1.67), (0.0, 1.0, -0.0, 0.0)),
    ((-0.0, 0.0, 1.0, 1.67), (1.0, -0.0, 0.0, 0.0), (0.0, 1.0, -0.0, 0.0)),
    ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.67)),
    ((-1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, -1.0, -1.67)),
)

#: Lens in mm (fx 6400 at 4096 px on a 36 mm sensor, from the dataset's
#: intrinsics.npy); the view focal in pixels scales with the render resolution.
LENS = 56.25

#: The object normalisation the dataset uses: largest dimension, in world units,
#: at a camera distance of 1.67 (measured 84% frame fill on the demo side view).
TARGET_SIZE = 0.6

#: The UltraTex LoRA + decoder files, in the layout the official code reads under
#: models/<chosen>/ultratex/checkpoints/. The bootstrap materializes them there
#: from ModelScope; this tuple is the single source both sides import.
LORA_FILES = ("flux2/lora/dit_lora.safetensors",
              "flux2_mr/lora/dit_lora.safetensors",
              "flux2/decoder.pt")

#: The task a LoRA belongs to: "albedo" reads flux2/lora, "mr" reads flux2_mr/lora.
TASKS = ("albedo", "mr")

#: The object's vertical axis: the ring views orbit this, and R_x tilts about it.
_CAMERA_DISTANCE = 1.67


def _rotate_x(pose: tuple[tuple[float, float, float, float], ...],
              degrees: float) -> tuple[tuple[float, float, float, float], ...]:
    """Rotate a pose about the world X axis, column by column.

    Each column of the pose tuple is one vector (right, up, backward, position),
    so the rotation applies to whole vectors, never across them.
    """
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    columns = []
    for index in range(4):
        x, y, z = pose[0][index], pose[1][index], pose[2][index]
        columns.append((x, y * c - z * s, y * s + z * c))
    return tuple(tuple(columns[c][i] for c in range(4)) for i in range(3))


def ring_pose(azimuth: float) -> tuple[tuple[float, float, float, float], ...]:
    """A horizontal ring view at `azimuth` degrees, from the dataset's own family.

    The four canonical side poses (0/90/180/270) are exactly this family, so the
    45-degree diagonals are the same family rather than an invention: the camera
    orbits at ``_CAMERA_DISTANCE`` with its up axis following the orbit.
    """
    theta = math.radians(azimuth)
    c, s = math.cos(theta), math.sin(theta)
    return (
        (c, 0.0, -s, -_CAMERA_DISTANCE * s),
        (-s, 0.0, -c, -_CAMERA_DISTANCE * c),
        (0.0, 1.0, 0.0, 0.0),
    )


def elevated_pose(azimuth: float, elevation: float) -> tuple[tuple[float, float, float, float], ...]:
    """A ring pose tilted `elevation` degrees above the horizon about the world X axis.

    ``elevated_pose(0, 90)`` is the canonical top pose verbatim; the guided pass
    uses 45 degrees, looking down at the subject from front and back.
    """
    return _rotate_x(ring_pose(azimuth), -elevation)


#: The guided 45-degree pass (the PixelArtistry "12-view guided" idea, ported
#: natively): four diagonal sides and two 45-degree-elevated views. Together
#: with the canonical POSES these are twelve viewpoints; the bake merges both.
GUIDED_POSES = (
    ring_pose(45),
    ring_pose(135),
    ring_pose(225),
    ring_pose(315),
    elevated_pose(0, 45),
    elevated_pose(180, 45),
)


def checkpoints_dir() -> Path:
    """models/<chosen>/ultratex/checkpoints, the layout the official code reads."""
    from Ar3DRAHub import models_home

    return models_home.vendor_models("ultratex") / "checkpoints"


def setup_command() -> str:
    """The script that installs UltraTex on this machine."""
    return "scripts/bootstrap_ultratex.py"


def ensure_speed_patch(repo: Path) -> list[str]:
    """Replay the empty-prompt fast path over the checkout, idempotently.

    The checkout is upstream's tree (git-ignored, re-cloned on bootstrap), so
    the patch lives in ``scripts/patch_ultratex_speed.py`` and is replayed
    here before every run. Fails closed when upstream changed shape, which
    surfaces a clear error before an expensive run instead of after it.
    """
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "patch_ultratex_speed.py"
    spec = importlib.util.spec_from_file_location("patch_ultratex_speed", script)
    patcher = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(patcher)
    return patcher.apply((repo / "UltraTex").resolve())


def platform_refusal() -> str | None:
    """Why this machine cannot run UltraTex, or None.

    The stack is torch cu128 with dense attention: NVIDIA on Linux or Windows.
    There is no CPU path and no Metal backend, so everything else is refused.
    """
    from Ar3DRAHub import host

    if host.host_platform() != host.NVIDIA:
        return "UltraTex needs an NVIDIA GPU with compute capability 8.0 or newer."
    if host.os_family() not in ("linux", "windows"):
        return ("UltraTex has no CPU path and no Metal backend; it runs on Linux or "
                "Windows with an NVIDIA card only.")
    return None


def verify_paths(repo: Path) -> list[str]:
    """Cheap filesystem checks that the checkout is present and finished building."""
    from Ar3DRAHub import host

    checks = {
        "venv python": host.venv_python(repo),
        "inference_flux2.py": repo / "UltraTex" / "inference_flux2.py",
        f"completed build ({setup_command()})": repo / CHECKOUT_MARKER,
    }
    return [f"missing {name}: {path}" for name, path in checks.items() if not path.exists()]


def lora_path(root: Path, task: str) -> Path:
    """The LoRA directory the official loader reads for `task`."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    return root / ({"albedo": "flux2/lora", "mr": "flux2_mr/lora"}[task])


def venv_python(repo: Path) -> Path:
    """The checkout venv's interpreter, named the way this OS names it."""
    from Ar3DRAHub import host

    return host.venv_python(repo)


def write_config(
    path: Path,
    options: UltraTexOptions,
    *,
    views_json: Path,
    project_dir: Path,
    lora: Path,
    decoder: Path,
    seed: int | None = None,
) -> dict[str, Any]:
    """The run's settings, written as JSON beside it: a record of what ran.

    Not fed to upstream's parser (transformers 5 reads `--config` files as flat
    whitespace tokens, which breaks on anything but `--key value` lines); the
    command line carries the flags, this file carries the record.
    """
    config = {
        "task": options.task,
        "resolution": int(options.resolution),
        "inference_num_steps": int(options.steps),
        "guidance": float(options.guidance),
        "seed": int(seed if seed is not None else options.seed),
        "eval_data_json": str(views_json),
        "resume_from_checkpoint": str(lora),
        "decoder_ckpt": str(decoder),
        "project_dir": str(project_dir),
        # Background Token Dropping on: prunes the background tokens the
        # silhouettes mark (the ~200k sequence shrinks to tens of thousands,
        # which is most of the paper's speedup). Sparse attention stays OFF:
        # the vendored Triton kernel deadlocks on sm_120 + triton-windows
        # (measured 2026-11-23: CPU idle, GPU pinned 99%, never returns).
        "sparse_attention": False,
        "inference_drop_background_tokens": True,
        "inference_remove_background": True,
        # Empty-prompt fast path: the empty prompt is encoded once before the
        # 9B denoiser loads, the Qwen encoder is then released (it never
        # shares VRAM with the 9B model, which is what made encoder calls take
        # ~30 s instead of ~1 s), and CFG doubling is skipped -- one forward
        # per step instead of two. Exact for empty prompts.
        "fast_empty_prompt": True,
        "save_raw_result": True,
        "save_composite": False,
        "output_format": "png",
        # The dataset expands each object to four indices that differ only in
        # which render_ref_* bucket they read; with identical references those
        # four runs produce the same canvas. One index is one generation.
        "max_samples": 1,
        "num_workers": 0,       # Windows + one job: no worker pool, no spawn cost
    }
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def build_command(
    python: Path,
    inference: Path,
    options: UltraTexOptions,
    *,
    views_json: Path,
    project_dir: Path,
    lora: Path,
    decoder: Path,
) -> list[str]:
    """The argv handed to upstream's ``inference_flux2.py``, as plain flags.

    Booleans ride as `--flag false`/`--flag true` (upstream's parser converts
    them with its string_to_bool). Background-token dropping stays on (the
    paper's speed mode that works here); sparse attention stays off until the
    vendored Triton kernel is fixed for sm_120 -- without BTD the dense path
    over the ~200k-token sequence costs on the order of an hour per task.
    """
    return [
        str(python), str(inference),
        "--task", options.task,
        "--resolution", str(int(options.resolution)),
        "--inference_num_steps", str(int(options.steps)),
        "--guidance", str(float(options.guidance)),
        "--seed", str(int(options.seed)),
        "--eval_data_json", str(views_json),
        "--resume_from_checkpoint", str(lora),
        "--decoder_ckpt", str(decoder),
        "--project_dir", str(project_dir),
        "--sparse_attention", "false",
        "--inference_drop_background_tokens", "true",
        "--inference_remove_background", "true",
        "--fast_empty_prompt", "true",
        "--save_raw_result", "true",
        "--save_composite", "false",
        "--output_format", "png",
        "--max_samples", "1",
        "--num_workers", "0",
    ]


def run_environment() -> dict[str, str]:
    """The environment upstream's loader reads its base models from, pinned offline."""
    from Ar3DRAHub.weight_gate import require_local

    klein = require_local(
        "black-forest-labs/FLUX.2-klein-base-9B",
        backend="UltraTex", route="--ultratex (models/huggingface)", size=WEIGHTS[0][1])
    ae = require_local(
        "black-forest-labs/FLUX.2-dev",
        backend="UltraTex", route="--ultratex (models/huggingface)", size=WEIGHTS[1][1])
    qwen = require_local(
        "Qwen/Qwen3-8B",
        backend="UltraTex", route="--ultratex (models/huggingface)", size=WEIGHTS[2][1])
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["KLEIN_9B_BASE_MODEL_PATH"] = str(Path(klein) / "flux-2-klein-base-9b.safetensors")
    env["AE_MODEL_PATH"] = str(Path(ae) / "ae.safetensors")
    env["QWEN3_8B_PATH"] = qwen
    return env


def generate_ultratex(
    views_json: Path,
    output_dir: Path,
    options: UltraTexOptions,
    log: Any | None = None,
    *,
    on_process: Callable[[subprocess.Popen | None], None] | None = None,
) -> UltraTexResult:
    """Run upstream's inference for `options.task` and return the result canvas.

    Raises RuntimeError with a reason when the platform, the checkout or the
    weights are not ready. `log(line)` receives the run's own output line by
    line, when the caller wants a live log; `on_process` hands the live process
    to the caller and then None when it ends, so a cancel button can kill it.
    """
    refusal = platform_refusal()
    if refusal:
        raise RuntimeError(refusal)
    repo = options.repo.expanduser().resolve()
    problems = verify_paths(repo)
    if problems:
        raise RuntimeError(
            f"UltraTex environment not found at {repo}: {', '.join(problems)}. "
            f"Run {setup_command()} first."
        )
    ensure_speed_patch(repo)
    if not views_json.is_file():
        raise ValueError(f"the views JSON does not exist: {views_json}")

    root = checkpoints_dir()
    missing = [relative for relative in LORA_FILES if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(
            f"UltraTex LoRA/decoder files are missing: {', '.join(missing)}. "
            f"Run {setup_command()} --weights-only."
        )
    if options.task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {options.task!r}")

    env = run_environment()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    project_dir = output_dir / f"task-{options.task}"
    project_dir.mkdir(parents=True, exist_ok=True)
    lora = lora_path(root, options.task)
    decoder = root / "flux2" / "decoder.pt"
    write_config(
        output_dir / f"ultratex_{options.task}.json", options,
        views_json=views_json,
        project_dir=project_dir,
        lora=lora,
        decoder=decoder,
    )

    command = build_command(
        venv_python(repo),
        repo / "UltraTex" / "inference_flux2.py",
        options,
        views_json=views_json,
        project_dir=project_dir,
        lora=lora,
        decoder=decoder,
    )
    if log is None:
        try:
            subprocess.run(command, cwd=repo / "UltraTex", env=env, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"UltraTex inference failed with exit code {exc.returncode}"
            ) from exc
    else:
        process = subprocess.Popen(
            command, cwd=repo / "UltraTex", env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True,
        )
        if on_process is not None:
            on_process(process)
        try:
            assert process.stdout is not None
            for raw in process.stdout:
                log(raw.rstrip("\n"))
            code = process.wait()
        finally:
            if on_process is not None:
                on_process(None)
        if code != 0:
            raise RuntimeError(f"UltraTex inference failed with exit code {code}")

    # The run writes `eval_<id>_<index>_result.png` for the single views JSON
    # entry; any *_result.png is that canvas. Upstream nests it one level down
    # (`checkpoint-<step>/`), so the glob must recurse.
    canvases = sorted(project_dir.glob("**/*_result.png"))
    if not canvases:
        raise RuntimeError(
            f"UltraTex reported success but wrote no result canvas under {project_dir}"
        )
    print(f"  [ultratex] {options.task}: {canvases[0]}", file=sys.stderr)
    return UltraTexResult(project_dir, canvases[0], options.task)
