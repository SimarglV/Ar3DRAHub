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
from typing import Any, Callable


@dataclass(frozen=True)
class Hi3DOptions:
    repo: Path
    resolution: int = 512
    steps: int = 25
    seed: int = 0
    cpu_offload: bool = True
    #: Refine 512 -> 1024 -> 1536, each rung feeding the next, instead of one
    #: refinement of the original mesh. The chain the PixelArtistry workflow uses.
    cascade: bool = False


@dataclass(frozen=True)
class Hi3DResult:
    """The refined GLB and the rung that produced it.

    `resolution` can be lower than the request: the 1536 rung reliably loses large
    parts of the subject on this repo's Windows SDPA port (measured ~63% of the
    input's surface rebuilt, against ~96% at 1024 and ~100% at 512).
    `generate_hi3d` measures that and steps down, so the record and the log say
    what the mesh actually is, not what was asked for.
    """

    path: Path
    resolution: int


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

#: How far a refined point may sit from the input surface, as a share of the model's
#: unit radius. Hi3D re-centres and scales its output, so both meshes are measured in
#: that frame; three percent is generous enough for the detail refinement adds, and
#: far wider than the distance between refined vertices.
COVERAGE_TOLERANCE = 0.03
#: Share of the input surface that must have refined geometry near it. Measured on the
#: teen character: 512 scored 1.000, 1024 ~0.96 (after the attention-port fix in
#: scripts/patch_hi3d_sdpa.py) and 1536 ~0.63. The gap is the missing slabs, not detail,
#: so the gate sits between a clean rung and a broken one with room for inputs that
#: legitimately move a little.
COVERAGE_THRESHOLD = 0.93
#: 30k area-weighted samples see a missing slab and still cost a fraction of a second.
COVERAGE_SAMPLES = 30_000


class RefineQualityError(RuntimeError):
    """The refined mesh is missing parts of the mesh it was refining."""


def _unit_frame(vertices):
    """Vertices centred on their mean and scaled to unit maximum radius, as Hi3D exports."""
    import numpy as np

    values = np.asarray(vertices, dtype=np.float64)
    centred = values - values.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(centred, axis=1).max())
    if not np.isfinite(radius) or radius <= 0.0:
        raise RefineQualityError("a mesh has zero extent; there is nothing to compare")
    return centred / radius


def _sample_surface(vertices, faces, count: int, seed: int = 0):
    """`count` points spread over the triangles by area, deterministically.

    Area weighting matters: the parts a refinement drops are often the big, flat
    ones -- a chest, a cape -- and sampling face-by-face would under-represent them.
    """
    import numpy as np

    tri = vertices[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                           axis=1) * 0.5
    total = float(areas.sum())
    if total <= 0.0:
        raise RefineQualityError("a mesh has no surface area; there is nothing to compare")
    rng = np.random.default_rng(seed)
    picks = rng.random(count) * total
    indices = np.clip(np.searchsorted(np.cumsum(areas), picks), 0, len(tri) - 1)
    r1, r2 = rng.random(count), rng.random(count)
    root = np.sqrt(r1)
    weights = np.stack([1.0 - root, root * (1.0 - r2), root * r2], axis=1)
    return (tri[indices] * weights[:, :, None]).sum(axis=1)


def surface_coverage(coarse_mesh: Path, refined: Path, *,
                     tolerance: float = COVERAGE_TOLERANCE,
                     samples: int = COVERAGE_SAMPLES) -> float:
    """How much of `coarse_mesh`'s surface has `refined` geometry near it, 0..1.

    The failure this catches is not a seam or a pinhole: it is a whole region of the
    subject that the refinement did not rebuild at all. A hairline seam moves no sample
    far from the surface; a missing chest leaves every sample taken there with nothing
    nearby. Both meshes are put in Hi3D's own frame first (centred, unit radius),
    because it does not preserve world scale.
    """
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree

    loaded = []
    for label, path in (("input", coarse_mesh), ("refined", refined)):
        try:
            mesh = trimesh.load(path, force="mesh", process=False)
        except Exception as exc:   # trimesh raises a zoo of types for a broken GLB
            raise RefineQualityError(f"cannot read the {label} mesh {path}: {exc}") from exc
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(vertices) == 0 or len(faces) == 0:
            raise RefineQualityError(f"the {label} mesh {path} has no triangles")
        loaded.append((_unit_frame(vertices), faces))

    (coarse_vertices, coarse_faces), (refined_vertices, refined_faces) = loaded
    points = _sample_surface(coarse_vertices, coarse_faces, int(samples))
    # Vertices alone are dense on a million-face refine but nearly empty on a coarse
    # test mesh, where a point in the middle of a face is far from every corner. The
    # surface samples close that gap without an exact point-to-triangle query, which
    # would need a spatial index over millions of triangles.
    surface = _sample_surface(refined_vertices, refined_faces, int(samples), seed=1)
    distance, _ = cKDTree(np.vstack([refined_vertices, surface])).query(points, workers=-1)
    return float((distance <= tolerance).mean())


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


#: The rungs a request can fall back through, heaviest first. A request starts at its
#: own rung; a lower one is tried only when the mesh fails the coverage gate above.
RESOLUTION_LADDER = (1536, 1024, 512)


#: The rungs a cascade refine climbs, lightest first. A cascade refines the coarse mesh
#: at 512, then that result at 1024, then that at 1536 -- coarse to fine, the chain the
#: PixelArtistry workflow uses -- instead of re-refining the original mesh each time.
CASCADE_RUNGS = (512, 1024, 1536)


def cascade_rungs(requested: int) -> list[int]:
    """The rungs a cascade climbs, from 512 up to `requested`."""
    requested = int(requested)
    if requested not in CASCADE_RUNGS:
        raise ValueError(
            f"cascade resolution must be one of {', '.join(map(str, CASCADE_RUNGS))}"
        )
    return [rung for rung in CASCADE_RUNGS if rung <= requested]


def _run_refinement(command: list[str], repo: Path, log: Any | None,
                    on_process: Callable[[subprocess.Popen | None], None] | None) -> None:
    """Run upstream's refine command, streaming its output when `log` is given.

    `on_process` hands the live process to the caller and then `None` when it ends,
    so a cancel button can kill it: the refinement prints nothing for ten minutes,
    and a run that only checks a flag between lines cannot be stopped.
    """
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    if log is None:
        try:
            subprocess.run(command, cwd=repo, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Hi3D refinement failed with exit code {exc.returncode}"
            ) from exc
        return
    process = subprocess.Popen(
        command, cwd=repo, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        text=True,
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
        raise RuntimeError(f"Hi3D refinement failed with exit code {code}")


def _run_cascade(prepared: Path, coarse_mesh: Path, output_dir: Path,
                 output_path: Path, python: Path, requested: int,
                 options: Hi3DOptions, model_dir: str, dino_dir: str, clip_dir: str,
                 repo: Path, log: Any | None,
                 on_process: Callable[[subprocess.Popen | None], None] | None,
                 check_quality: bool) -> Hi3DResult:
    """The cascade refine: each rung refines the previous rung's output, lightest first.

    The final rung lands on the usual `output_path`; every earlier rung keeps its own
    file (`*_hi3d_512.glb`), so the chain is inspectable. The coverage gate compares
    each rung against the mesh it refined; a failed rung keeps the previous one instead
    of shipping a mesh that lost part of the model.
    """
    rungs = cascade_rungs(requested)
    current = coarse_mesh
    for position, rung in enumerate(rungs):
        target = (output_path if position + 1 == len(rungs)
                  else output_dir / f"{output_path.stem}_{rung}{output_path.suffix}")
        command = build_refine_command(
            python,
            repo / "Hi3Dv1.0" / "demo" / "refine_mesh.py",
            prepared,
            current,
            target,
            Hi3DOptions(repo=options.repo, resolution=rung, steps=options.steps,
                        seed=options.seed, cpu_offload=options.cpu_offload),
            model_dir,
            dino_dir,
            clip_dir,
        )
        _run_refinement(command, repo, log, on_process)
        if not target.is_file():
            raise RuntimeError(f"Hi3D reported success but did not create {target}")
        if not check_quality:
            current = target
            continue
        score = surface_coverage(current, target)
        log and log(f"Hi3D {rung}: rebuilt {score:.1%} of the input surface")
        if score >= COVERAGE_THRESHOLD:
            current = target
            continue
        if position == 0:
            raise RefineQualityError(
                f"Hi3D rebuilt only {score:.0%} of the model's surface at {rung}: "
                "large parts of it are missing, not just seams. Finish without Hi3D, "
                "or report this run — the gate stops here rather than ship a broken mesh."
            )
        kept = rungs[position - 1]
        log and log(f"Hi3D {rung}: only {score:.0%} of the model's surface came back; "
                    f"keeping the {kept} rung instead")
        return Hi3DResult(output_dir / f"{output_path.stem}_{kept}{output_path.suffix}",
                          kept)
    return Hi3DResult(output_path, rungs[-1])


def generate_hi3d(
    image: Path,
    coarse_mesh: Path,
    output_dir: Path,
    options: Hi3DOptions,
    log: Any | None = None,
    *,
    on_process: Callable[[subprocess.Popen | None], None] | None = None,
    check_quality: bool = True,
) -> Hi3DResult:
    """Refine `coarse_mesh` against `image` and return the refined GLB and its rung.

    Raises RuntimeError with a reason when the platform, the checkout or the
    weights are not ready. The weights gate (require_local) refuses with sizes
    when any of the three repos is not already on disk. `log(line)` receives
    the refinement's own output line by line, when the caller wants a live log.

    After each attempt the result is measured against the input (`surface_coverage`).
    A rung that rebuilds less than `COVERAGE_THRESHOLD` of the input's surface is
    not returned as if it were fine: the next lower rung is tried, and the log says
    why. On the Windows SDPA port the 1536 rung loses large slabs of the subject;
    the gate steps down to 1024 (which the attention-port fix repairs) instead of
    presenting a holey mesh as finished.

    With `options.cascade` the refine climbs 512 -> 1024 -> 1536 (up to the requested
    rung), each rung refining the previous rung's output, so detail accumulates
    coarse to fine instead of one refinement of the original mesh doing all the work.
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
    requested = int(options.resolution)
    if options.cascade:
        return _run_cascade(
            prepared, coarse_mesh, output_dir, output_path, python, requested,
            options, str(model_dir), dino_dir, clip_dir, repo,
            log, on_process, check_quality,
        )
    rungs = [rung for rung in RESOLUTION_LADDER if rung <= requested]

    for position, rung in enumerate(rungs):
        command = build_refine_command(
            python,
            repo / "Hi3Dv1.0" / "demo" / "refine_mesh.py",
            prepared,
            coarse_mesh,
            output_path,
            Hi3DOptions(repo=options.repo, resolution=rung, steps=options.steps,
                        seed=options.seed, cpu_offload=options.cpu_offload),
            str(model_dir),
            dino_dir,
            clip_dir,
        )
        _run_refinement(command, repo, log, on_process)
        if not output_path.is_file():
            raise RuntimeError(f"Hi3D reported success but did not create {output_path}")
        if not check_quality:
            return Hi3DResult(output_path, rung)

        score = surface_coverage(coarse_mesh, output_path)
        log and log(f"Hi3D {rung}: rebuilt {score:.1%} of the input surface")
        if score >= COVERAGE_THRESHOLD:
            if rung != requested:
                log and log(f"Hi3D {rung}: kept this one; the requested {requested} "
                            "did not rebuild enough of the model")
            return Hi3DResult(output_path, rung)
        if position + 1 == len(rungs):
            raise RefineQualityError(
                f"Hi3D rebuilt only {score:.0%} of the model's surface at {rung}: "
                "large parts of it are missing, not just seams. Finish without Hi3D, "
                "or report this run — the gate stops here rather than ship a broken mesh."
            )
        log and log(f"Hi3D {rung}: only {score:.0%} of the model's surface came back; "
                    f"stepping down to {rungs[position + 1]}")
    raise RuntimeError("Hi3D produced no refinement to check")
