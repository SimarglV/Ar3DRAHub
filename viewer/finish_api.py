"""Queued finishing jobs for the browser: retopologise, repaint, compress an existing GLB.

Deliberately a sibling of `rig_api.py` rather than part of `generate_api.py`: this operates
on an asset that already exists, so it has no image-to-3D backend, no model readiness and no
setup. What it shares is the job shape — one at a time, SSE progress, artifacts fetched by
URL — so the browser can drive it with the same code it already uses.

Finishing an asset the viewer generated earlier is the point: the surface response
(`metallic`/`roughness`/`ior`) is tuned per asset by eye, and re-finishing with different
values must not mean regenerating the geometry underneath it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import usage_api

from Ar3DRAHub import blender as blender_lookup
from Ar3DRAHub import processes
from Ar3DRAHub.host import APPLE, NVIDIA, host_platform, venv_python

OUTPUT_ROOT = REPO / "output" / "finish"
# Where generation runs (and their records) live, for matching an uploaded model to the run
# that made it.
GENERATED_ROOT = REPO / "output"
WORKER = REPO / "scripts" / "retopo_repaint.py"
# The optional Marigold de-light + normal-map pass, run on the finished model afterwards.
MARIGOLD_FINISH = REPO / "scripts" / "marigold_finish.py"
# The optional Hi3D geometry refinement: replaces the retopo worker entirely and produces
# a geometry-only GLB (no textures, re-centred to unit radius). NVIDIA only.
HI3D_VENDOR = REPO / "vendor" / "hi3d"
HI3D_RESOLUTIONS = (512, 1024, 1536)
# The optional watertight rebuild: closes hairline gaps and merges every shell into one
# closed body. Runs on the CPU in both environments, so there is no platform gate -- only
# a time and memory cost that `across` sets.
WATERTIGHT_WORKER = REPO / "scripts" / "watertight.py"
WATERTIGHT_ACROSS = (64, 700)
# The external engine, when a machine has one. WTiVo lives in `vendor/wativo/` with its own
# `.venv`, the way marigold-v2 does: its native extension is built for CPython 3.12 and its
# torch pin is 2.8.0, neither of which the studio's 3.11 environment can host. Both folders
# are git-ignored -- the checkout is someone else's repo, cloned by the bootstrap script.
WTIVO_VENDOR = REPO / "vendor" / "wativo"
WTIVO_ENV_HOME = ("WTIVO_HOME", "WTIVO_DIR")
WTIVO_ENV_PYTHON = "WTIVO_PYTHON"
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
# The exact shape `create` builds — a resume takes its directory name from a URL, so
# the name is matched against the generator rather than merely sanitised.
RUN_DIRECTORY = re.compile(r"^[A-Za-z0-9_-]{1,80}__finish__\d{8}-\d{6}(?:-\d+)?$")
TERMINAL = {"done", "error", "cancelled"}

ARTIFACTS = {
    "result.glb": ("result_glb", "model/gltf-binary"),
    "record.json": ("record_path", "application/json"),
    # The watertight rebuild, when the run asked for one. Beside the result rather than
    # replacing it: it has no UVs, so it is a different deliverable, not a better one.
    "watertight.glb": ("watertight_glb", "model/gltf-binary"),
}

# Bounds, not preferences. Everything here reaches a subprocess argument, so each value is
# clamped to a range the underlying scripts actually accept — `blender_retopo_bake.py`
# rejects a face target outside 1k..200k, and a voxel fraction outside its own range melts
# the subject or exhausts memory.
SETTING_BOUNDS: dict[str, tuple[float, float]] = {
    "faces": (1000, 200000),
    "atlas": (1024, 4096),
    "angle": (1.0, 89.9),
    "voxel": (0.0, 0.05),
    "metallic": (0.0, 1.0),
    "roughness": (0.0, 1.0),
    "ior": (1.0, 3.0),
    "paint_seed": (0, 2**31 - 1),
    "paint_res": (256, 1024),
    "paint_steps": (1, 50),
    "paint_tex": (1024, 4096),
    "texture_size": (256, 4096),
    "hi3d_resolution": (512, 1536),
    "watertight_across": WATERTIGHT_ACROSS,
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "faces": 20000, "atlas": 2048, "angle": 89.0, "voxel": 0.004,
    "metallic": 0.25, "roughness": 0.65, "ior": 1.45,
    "paint_seed": 0, "paint_res": 512, "paint_steps": 15, "paint_tex": 4096,
    "texture_size": 2048, "skip_paint": True, "skip_compress": False, "skip_bake": False,
    "skip_photo": False, "keep_mesh": False, "skip_edge_fade": False, "skip_nudge": False,
    "high_detail": False, "hero": False, "marigold": False,
    "marigold_normals": False, "photo_fade_normals": False,
    "hi3d": False, "hi3d_resolution": 512, "hi3d_texture": False,
    "watertight": False, "watertight_across": 300, "watertight_tool": "own",
    "watertight_wativo_res": 512,
}

BOOL_SETTINGS = {"skip_paint", "skip_compress", "skip_bake", "skip_photo", "keep_mesh", "skip_edge_fade", "skip_nudge",
                 "high_detail", "hero", "marigold", "marigold_normals", "photo_fade_normals", "hi3d", "hi3d_texture",
                 "watertight"}

# Each string setting maps to the values it accepts, so a new enum validates against its
# own list instead of someone else's.
WATERTIGHT_TOOLS = ("own", "external")
STRING_SETTINGS = {"watertight_tool": WATERTIGHT_TOOLS}
# The external engine's two measured quality points, chosen on the "Faster rebuild"
# checkbox: 512 is fast and a lighter file, 1536 is the tool's full detail.
WATERTIGHT_TOOL_RES = (512, 1536)
CHOICE_SETTINGS = {"watertight_wativo_res": WATERTIGHT_TOOL_RES}

INTEGER_SETTINGS = {
    "faces", "atlas", "paint_seed", "paint_res", "paint_steps", "paint_tex", "texture_size",
    "hi3d_resolution", "watertight_across",
}

# Which artifact proves a stage already ran, for `--resume` and for describing a run that
# the browser lost track of: a key into the worker's `step_paths`, or None for the final
# model. Ordered as the worker runs them.
STAGE_ARTIFACTS: tuple[tuple[str, str | None], ...] = (
    ("retopologise", "retopo"),
    ("orient", "orient"),
    ("repaint", "painted"),
    ("photo", "photo"),
    ("bake", "baked"),
    ("compress", None),
)

# How the overall bar divides between stages. Grossly uneven on purpose: a measured run on
# 2026-09-21 spent 7.8s retopologising, 324.5s repainting and 0.6s compressing, so equal
# thirds would park the bar at 33% for five of its six minutes. The bake-detail stage took
# 4.3s on the orc (2026-09-27), about half a retopology.
STAGE_WEIGHTS: dict[str, float] = {
    "retopologise": 5.0, "orient": 0.5, "repaint": 90.0, "photo": 1.0, "bake": 2.0, "compress": 3.0,
}
# The hero route's own stages (scripts/hero_route.py): building the head is most of its run time.
HERO_WEIGHTS: dict[str, float] = {
    "body": 2.0, "measure": 5.0, "crop": 1.0, "head": 88.0, "graft": 1.0, "join": 1.0, "check": 2.0,
}
HERO_STAGES = ["body", "measure", "crop", "head", "graft", "join", "check"]
HERO_WORKER = REPO / "scripts" / "hero_route.py"

# Where the repaint's own log lines fall inside that stage. Same run: ~30s of setup, 165s
# of diffusion steps, 9s decoding views and 72s of super-res. The diffusion steps are
# barely half the stage, so finishing them must not park the bar at 100% -- that silent
# half-minute is exactly what reads as a hang.
PAINT_SETUP_FRACTION = 0.08
PAINT_STEPS_END_FRACTION = 0.65
PAINT_MARKS: tuple[tuple[str, float], ...] = (
    ("controls + dino ready", PAINT_SETUP_FRACTION),
    ("views decoded", 0.70),
    ("super-res", 0.78),
)
PAINT_STEP = re.compile(r"^\s*step (\d+)/(\d+)\b")


def normalise_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """Merge client settings over the defaults, rejecting anything out of range.

    Raises ValueError with the offending key. Unknown keys are ignored rather than passed
    through — a typo must not silently reach a subprocess argument.
    """
    settings = dict(DEFAULT_SETTINGS)
    for key, value in raw.items():
        if key not in DEFAULT_SETTINGS:
            continue
        if key in BOOL_SETTINGS:
            settings[key] = bool(value)
            continue
        if key in STRING_SETTINGS:
            allowed = STRING_SETTINGS[key]
            if value not in allowed:
                raise ValueError(f"{key} must be one of {', '.join(allowed)}, got {value!r}")
            settings[key] = value
            continue
        if key in CHOICE_SETTINGS:
            allowed = CHOICE_SETTINGS[key]
            try:
                number = int(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be one of {', '.join(map(str, allowed))}, "
                                 f"got {value!r}") from None
            if number not in allowed:
                raise ValueError(f"{key} must be one of {', '.join(map(str, allowed))}, "
                                 f"got {number}")
            settings[key] = number
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number, got {value!r}") from None
        low, high = SETTING_BOUNDS[key]
        if not low <= number <= high:
            raise ValueError(f"{key} must be within {low}..{high}, got {number}")
        settings[key] = int(number) if key in INTEGER_SETTINGS else number
    if settings["keep_mesh"]:
        # full detail: nothing is simplified, so there is nothing to repaint or bake onto
        settings.update(skip_paint=True, skip_bake=True)
    if settings["hi3d"] and settings["hi3d_resolution"] not in HI3D_RESOLUTIONS:
        raise ValueError(
            f"hi3d_resolution must be one of {', '.join(map(str, HI3D_RESOLUTIONS))}, "
            f"got {settings['hi3d_resolution']}"
        )
    return settings


REPAINT_NOTE = ("Repaint runs on Apple Silicon only for now. Here Pixel Match still keeps "
                "the front exact; the sides and back keep the generator's own paint.")


def capabilities(platform: str | None = None, find=blender_lookup.find_blender,
                 version=blender_lookup.blender_version,
                 installable=blender_lookup.can_install) -> dict[str, Any]:
    """What Finish can do on this machine, for the page to say before anyone clicks."""
    repaint = (platform or host_platform()) == APPLE
    found = find()
    found_version = version(found) if found else None
    hi3d, hi3d_reason = hi3d_readiness()
    watertight, watertight_reason = watertight_readiness()
    wativo, wativo_reason = wativo_readiness()
    return {
        "repaint": repaint,
        "repaint_note": None if repaint else REPAINT_NOTE,
        "blender": str(found) if found else None,
        "blender_version": ".".join(map(str, found_version)) if found_version else None,
        "blender_problem": (blender_lookup.missing_help() if found is None
                            else blender_lookup.version_problem(found_version)),
        "ready": found is not None,
        # Setup's Blender card offers Install Blender (scripts/bootstrap_blender.py).
        "blender_installable": found is None and installable(),
        "watertight": watertight,
        "watertight_note": None if watertight else watertight_reason,
        "wativo": wativo,
        "wativo_note": None if wativo else wativo_reason,
        "hi3d": hi3d,
        "hi3d_note": None if hi3d else hi3d_reason,
    }


def fit_to_machine(settings: dict[str, Any], platform: str | None = None) -> dict[str, Any]:
    """Settings this machine can actually run: no repaint off Apple Silicon (it is MLX),
    and no Hi3D refine off NVIDIA (there is no CPU path and no Metal backend)."""
    platform = platform or host_platform()
    if platform != APPLE:
        settings = {**settings, "skip_paint": True}
    if platform != NVIDIA:
        settings = {**settings, "hi3d": False, "hi3d_texture": False}
    if settings.get("hi3d_texture") and not settings.get("hi3d"):
        settings = {**settings, "hi3d_texture": False}
    return settings


def worker_module():
    """The worker, imported, so the stage plan has one definition rather than two.

    Its module level is constants and an argparse builder; the expensive imports live
    inside `main`, so this costs nothing and cannot drift from what actually runs.
    """
    spec = importlib.util.spec_from_file_location("retopo_repaint", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_command(
    job: FinishJob, settings: dict[str, Any], *, resume: bool = False,
    source: Path | None = None, source_record: Path | None = None,
) -> list[str]:
    """The worker invocation. Flags, not positions — the worker owns its own ordering.

    `source` overrides the mesh the worker finishes: a Hi3D-textured run hands it the
    refined mesh (with the refined sidecar as `source_record`) instead of the upload.
    """
    source = source or job.asset_path
    command = [
        sys.executable, "-u", str(WORKER),
        str(source), str(job.image_path), str(job.result_glb),
        *(["--steps-dir", str(job.steps_dir)] if job.steps_dir is not None else []),
        *(["--views", str(job.views_dir)] if uses_photo(job, settings) else []),
        *(["--native-dump", str(job.dump_path)]
          if job.dump_path.is_file() and not settings.get("keep_mesh")
          and source == job.asset_path else []),
        "--faces", str(settings["faces"]),
        "--atlas", str(settings["atlas"]),
        "--angle", str(settings["angle"]),
        "--voxel", str(settings["voxel"]),
        "--metallic", str(settings["metallic"]),
        "--roughness", str(settings["roughness"]),
        "--ior", str(settings["ior"]),
        "--paint-seed", str(settings["paint_seed"]),
        "--paint-res", str(settings["paint_res"]),
        "--paint-steps", str(settings["paint_steps"]),
        "--paint-tex", str(settings["paint_tex"]),
        "--texture-size", str(settings["texture_size"]),
    ]
    record = source_record or (job.asset_path.parent / "source.json")
    if record.is_file():
        command += ["--source-record", str(record)]
    if settings["skip_paint"]:
        command.append("--skip-paint")
    if settings["skip_compress"]:
        command.append("--skip-compress")
    if settings["skip_bake"]:
        command.append("--skip-bake")
    if settings.get("keep_mesh"):
        command.append("--keep-mesh")
    if settings.get("skip_edge_fade"):
        command.append("--no-edge-fade")
    if settings.get("skip_nudge"):
        command.append("--no-nudge")
    if settings.get("photo_fade_normals"):
        command.append("--fade-normals")
    if settings.get("high_detail"):
        command += ["--detail", "high"]
    if resume:
        command.append("--resume")
    return command


def hero_command(job: FinishJob, settings: dict[str, Any]) -> list[str]:
    """The hero route on this run's model: its body is the asset's own Pixal3D model, its head
    is built again from a close crop. Everything it makes stays under steps/hero."""
    record = job.input_dir / "source.json"
    return [
        sys.executable, "-u", str(HERO_WORKER), str(job.image_path),
        "--out", str((job.steps_dir or job.directory) / "hero"),
        "--body", str(job.asset_path), "--body-views", str(job.views_dir),
        *(["--body-dump", str(job.dump_path)] if job.dump_path.is_file() else []),
        *(["--body-record", str(record)] if record.is_file() else []),
        "--body-faces", str(settings["faces"]),
        "--result", str(job.result_glb),
    ]


def marigold_ready(job: FinishJob) -> bool:
    """Whether the Marigold pass can run: its venv is built and the front view is saved."""
    return (venv_python(REPO / "vendor" / "marigold-v2").is_file()
            and (job.views_dir / "view_000.png").is_file())


def find_wativo_home(explicit: str | Path | None = None) -> Path | None:
    """A WTiVo checkout, or None.

    `vendor/wativo` first -- that is where the bootstrap puts it, and where the studio
    expects to find it on a machine that ran the install. Then an explicit path, then
    `$WTIVO_HOME` / `$WTIVO_DIR`, so a developer can point at their own working copy
    without moving it into the repo. Only a checkout with its `wtivo.py` counts: the
    `inprocess.py` a checkout also ships is the ComfyUI bridge, not something this studio
    can run.
    """
    if explicit:
        candidate = Path(explicit).expanduser()
        if (candidate / "wtivo.py").is_file():
            return candidate
        if candidate.is_file():                       # gave the script itself
            return candidate.parent
        return None
    if (WTIVO_VENDOR / "wtivo.py").is_file():
        return WTIVO_VENDOR
    for name in WTIVO_ENV_HOME:
        value = os.environ.get(name)
        if value:
            candidate = Path(value).expanduser()
            if (candidate / "wtivo.py").is_file():
                return candidate
            if candidate.is_file() and candidate.name == "wtivo.py":
                return candidate.parent
    return None


def find_wativo_python(home: Path) -> Path | None:
    """An interpreter that can import WTiVo's native extension.

    `$WTIVO_PYTHON` first -- an explicit choice wins over a guess -- then the vendored
    `.venv`, then any Python dropped beside the checkout. The extension is built for
    CPython 3.12 and pins torch 2.8.0, so the studio's own 3.11 is never a fallback: it is
    a different error message. Whether the found interpreter can really load the extension
    is what `wativo_readiness` probes.
    """
    explicit = os.environ.get(WTIVO_ENV_PYTHON)
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate
    for relative in (".venv/Scripts/python.exe", ".venv/bin/python",
                     "venv/Scripts/python.exe", "venv/bin/python", "python.exe"):
        candidate = home / relative
        if candidate.is_file():
            return candidate
    return None


def wativo_readiness() -> tuple[bool, str | None]:
    """Whether the WTiVo engine can run here, or why not.

    Checked before a run starts, for the same reason Hi3D is: a missing interpreter should
    be a sentence on the card, not an exit code a minute in. The probe is the real thing,
    not a version print: it imports `cppmodules` through the found interpreter, with the
    checkout's `build/` on both the module and the DLL search path, because that import is
    exactly what a run needs and exactly what breaks on a wrong torch or a wrong Python.
    """
    home = find_wativo_home()
    if home is None:
        return False, ("WTiVo is not installed. Install it from Setup (the WTiVo card) or "
                       "run python scripts/bootstrap_wativo.py, or set WTIVO_HOME to a "
                       "checkout.")
    python = find_wativo_python(home)
    if python is None:
        return False, (f"no Python for WTiVo in {home.name}: it needs CPython 3.12 with "
                       "torch. Set WTIVO_PYTHON to that interpreter.")
    probe = (
        "import os, sys, torch\n"
        "build = os.path.join({root!r}, 'build')\n"
        "sys.path.insert(0, build)\n"
        "add = getattr(os, 'add_dll_directory', None)\n"
        "add(build) if add else None\n"
        "import cppmodules\n"
    ).format(root=str(home))
    completed = subprocess.run([str(python), "-c", probe],
                               capture_output=True, text=True, timeout=120)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        return False, (f"{python.name} cannot load WTiVo's native extension: "
                       f"{detail[-1] if detail else 'unknown'}. It needs CPython 3.12 "
                       "with torch 2.8.0.")
    return True, None


def watertight_readiness() -> tuple[bool, str | None]:
    """Whether the watertight rebuild can run here, or why not.

    It runs on the CPU and needs nothing beyond what the studio already imports, so the
    only real question is whether the script is there. Checked anyway, for the same reason
    Hi3D is: a missing worker should be a sentence on the card, not an exit code a minute
    into a run.
    """
    if not WATERTIGHT_WORKER.is_file():
        return False, f"the watertight script is missing: {WATERTIGHT_WORKER.name} in scripts/"
    try:
        import numpy  # noqa: F401  -- a fresh install without it fails here, not mid-run
        import scipy  # noqa: F401
        import trimesh  # noqa: F401
        from skimage import measure  # noqa: F401
    except ImportError as exc:
        return False, f"the watertight rebuild needs a package that is missing: {exc.name}"
    return True, None


def hi3d_readiness() -> tuple[bool, str | None]:
    """Whether the Hi3D refine stage can run here, or why not.

    Files only, no imports: the platform check, the checkout, and the materialised
    model folder. Checked before a run starts, because a Hi3D refusal mid-run would
    come after the picture upload and an expensive, doomed job.
    """
    from Ar3DRAHub import hi3d_backend

    refusal = hi3d_backend.platform_refusal()
    if refusal is not None:
        return False, refusal
    problems = hi3d_backend.verify_paths(HI3D_VENDOR)
    if problems:
        return False, ("Hi3D v1.0 is not installed. Set it up on Setup & Status "
                       "(scripts/bootstrap_hi3d.py, about 10 GB).")
    present, _, missing = hi3d_backend.model_dir_present()
    if not present:
        return False, ("Hi3D v1.0 weights are missing: "
                       f"{', '.join(missing)}. Set them up on Setup & Status.")
    return True, None


def run_hi3d(job: FinishJob) -> None:
    """The Hi3D refine stage: replaces the retopo worker and produces geometry-only GLBs.

    The refine conditions on the reference picture, so a run must have one. The result
    carries a derived sidecar chaining the source's licence with Hi3D's, so the chain
    survives the textureless GLB Hi3D exports.
    """
    refined = refine_stage(job, "Hi3D: refining the geometry at {resolution} "
                                 "(geometry only — no textures)")
    from Ar3DRAHub.provenance import set_glb_generator

    job.result_glb.parent.mkdir(parents=True, exist_ok=True)
    refined.replace(job.result_glb)
    set_glb_generator(job.result_glb)
    write_hi3d_sidecar(job, job.asset_path, job.result_glb)
    job.append_log(f"hi3d refine done -> {job.result_glb.name}")
    record = {
        "source": str(job.asset_path),
        "image": str(job.image_path),
        "output": str(job.result_glb),
        "stages": ["hi3d"],
        "hi3d": {
            "resolution": int(job.settings.get("hi3d_resolution", 512)),
            "geometry_only": True,
        },
        "size_bytes": job.result_glb.stat().st_size,
    }
    job.record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        (job.directory / PIXEL_MATCH_NAME).write_text(json.dumps({"pixel_match": "off"}),
                                                      encoding="utf-8")
    except OSError:
        pass


def refine_stage(job: FinishJob, message_template: str) -> Path:
    """Run the Hi3D refine itself and return the refined GLB (in the steps dir).

    Does not touch the result or the records: the caller decides whether this is the
    whole run (geometry-only) or the first half of a textured finish.
    """
    from Ar3DRAHub import hi3d_backend

    ready, reason = hi3d_readiness()
    if not ready:
        raise RuntimeError(reason or "Hi3D refine is not available here")
    if not job.image_path.is_file():
        raise RuntimeError("Hi3D refine needs the reference picture; none was uploaded")
    resolution = int(job.settings.get("hi3d_resolution", 512))
    job.emit({"phase": "hi3d", "overall_pct": 0,
              "message": message_template.format(resolution=resolution)})
    return hi3d_backend.generate_hi3d(
        image=job.image_path,
        coarse_mesh=job.asset_path,
        output_dir=(job.steps_dir or job.directory),
        options=hi3d_backend.Hi3DOptions(
            repo=HI3D_VENDOR, resolution=resolution, cpu_offload=True,
        ),
        log=job.append_log,
    )


def write_hi3d_sidecar(job: FinishJob, source: Path, output: Path) -> None:
    """The refined mesh's licence chain: the source's record plus Hi3D's terms."""
    from Ar3DRAHub.provenance import LICENSES, write_derived_sidecar

    profile = LICENSES["hi3d"]
    source_record = None
    record_file = job.input_dir / "source.json"
    if record_file.is_file():
        source_record = json.loads(record_file.read_text(encoding="utf-8"))
    write_derived_sidecar(
        source, output, operation="hi3d-refine",
        source_record=source_record,
        component_records=[{
            "component": "hi3d-official/Hi3Dv1.0",
            "purpose": "geometry refinement",
            "license": {"name": profile.license_name, "url": profile.license_url,
                        "conditions": list(profile.conditions)},
            "commercial_status": "commercial-conditional",
        }],
    )


def hi3d_transfer_command(job: FinishJob, refined: Path, textured: Path,
                          blender: Path | None = None) -> list[str]:
    """The headless Blender invocation that bakes the source's texture onto the refined mesh."""
    blender = blender or blender_lookup.find_blender()
    if blender is None:
        raise RuntimeError(blender_lookup.missing_help())
    return [
        str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
        "--python", str(REPO / "scripts" / "blender_hi3d_transfer.py"), "--",
        str(job.asset_path), str(refined), str(textured),
        str(job.settings["atlas"]), str(job.settings["metallic"]),
        str(job.settings["roughness"]), str(job.settings["ior"]),
    ]


def run_hi3d_textured(job: FinishJob) -> tuple[Path, Path]:
    """Hi3D refine + texture transfer. Returns (worker source, source record).

    The refined mesh is baked with the source's base colour and aligned back into the
    source's frame, so the normal Finish worker (Detail, Pixel Match, Marigold, compress)
    can run on it. The worker's source record is the refined mesh's sidecar, so the
    final record chains the original licence with Hi3D's.
    """
    refined = refine_stage(
        job, "Hi3D: refining the geometry at {resolution}, then transferring the texture")
    steps = job.steps_dir or job.directory
    textured = steps / "hi3d_textured.glb"
    job.emit({"phase": "hi3d-transfer", "overall_pct": 0,
              "message": "Transferring the original texture onto the refined geometry"})
    job.process = subprocess.Popen(
        hi3d_transfer_command(job, refined, textured), cwd=str(REPO),
        env=processes.child_env(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        **processes.TEXT_PIPE, **processes.group_popen_kwargs(),
    )
    assert job.process.stdout is not None
    for raw in job.process.stdout:
        line = raw.rstrip("\n")
        if line:
            job.append_log(line)
    code = job.process.wait()
    job.process = None
    if code != 0 or not textured.is_file():
        raise RuntimeError("the texture transfer failed; the refined mesh was not kept")
    write_hi3d_sidecar(job, job.asset_path, textured)
    return textured, textured.with_suffix(".provenance.json")


def watertight_command(source: Path, output: Path, across: int, tool: str = "own",
                       wativo_res: int = 512) -> list[str]:
    """The watertight rebuild as a subprocess, on the studio's own interpreter.

    Not a Blender invocation and not a separate venv: `scripts/watertight.py` imports only
    numpy, scipy, trimesh and scikit-image, all of which the studio already has.

    `tool` picks the engine: `own` is the MIT module, `external` is an installed WTiVo
    checkout, run on the interpreter its extension was built for, at `wativo_res` (512
    fast / 1536 detail). The choice is about speed, so the interface calls them "this
    machine" and "fast" -- the licence consequence lives in the sidecar, not in a
    checkbox label.
    """
    command = [sys.executable, "-u", str(WATERTIGHT_WORKER), str(source), str(output),
               "--across", str(int(across))]
    if tool == "external":
        home = find_wativo_home()
        if home is None:
            raise RuntimeError("WTiVo is not installed; install it from Setup or set "
                               "WTIVO_HOME to a checkout")
        # `wtivo.py` is a module, not an executable: run it with the interpreter that can
        # import its native extension. That is a different interpreter from ours on
        # purpose -- ours is 3.11 and the extension is built for 3.12.
        script = home / "wtivo.py"
        python = find_wativo_python(home) if script.is_file() else home / "wativo"
        command += ["--tool", "wativo", "--wativo-res", str(int(wativo_res))] + (
            ["--wativo-python", str(python), "--wativo-script", str(script)]
            if script.is_file() else ["--wativo", str(python)])
    return command


def run_watertight(job: FinishJob, source: Path) -> Path:
    """Rebuild `source` watertight, in the run's steps directory, and return it.

    Runs last, on whatever the rest of the pipeline produced: a shell that has been through
    Detail, Pixel Match and the bake has been cut up and re-welded, which is exactly where
    hairline gaps reappear. Doing it earlier would mean the later stages had to work on a
    mesh whose UVs are gone.

    The rebuild drops UVs, so this is offered as a separate result rather than folded into
    the finished model: `watertight.glb` keeps the geometry and nothing else.
    """
    ready, reason = watertight_readiness()
    if not ready:
        raise RuntimeError(reason or "the watertight rebuild is not available here")
    steps = job.steps_dir or job.directory
    target = steps / "watertight.glb"
    across = int(job.settings.get("watertight_across", 300))
    tool = "external" if job.settings.get("watertight_tool") == "external" else "own"
    if tool == "external":
        # The card promised a GPU engine, so the run checks the promise it made, not just
        # whether a folder exists: a wrong-torch venv fails the extension import, and the
        # error belongs before the run starts, next to every other refusal.
        ready, reason = wativo_readiness()
        if not ready:
            raise RuntimeError(reason or "the WTiVo engine is not available here")
        wativo_res = int(job.settings.get("watertight_wativo_res", 512))
        message = (f"Rebuilding watertight with WTiVo at {wativo_res} — this closes the "
                   "hairline gaps and merges every piece, and drops the texture")
    else:
        wativo_res = 512
        message = (f"Rebuilding watertight at {across} voxels across — this closes the "
                   "hairline gaps and merges every piece, and drops the texture")
    job.emit({"phase": "watertight", "overall_pct": 0, "message": message})
    job.process = subprocess.Popen(
        watertight_command(source, target, across, tool, wativo_res), cwd=str(REPO),
        env=processes.child_env(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        **processes.TEXT_PIPE, **processes.group_popen_kwargs(),
    )
    assert job.process.stdout is not None
    for raw in job.process.stdout:
        line = raw.rstrip("\n")
        if line:
            job.append_log(line)
    code = job.process.wait()
    job.process = None
    if job.cancel_requested:
        raise RuntimeError("watertight rebuild cancelled")
    if code != 0 or not target.is_file():
        raise RuntimeError("the watertight rebuild failed; the finished model was kept as it was")
    # The script writes its own sidecar when it can find the input's record. Copy it to
    # where the studio looks for a result's provenance, so the Library shows the chain.
    sidecar = target.with_suffix(".provenance.json")
    if sidecar.is_file():
        shutil.copy(sidecar, target.with_suffix(".retopo-repaint.json"))
    return target


def marigold_command(job: FinishJob) -> list[str]:
    """Run the Marigold de-light (+ optional normal bake) on the finished model, in place."""
    command = [sys.executable, "-u", str(MARIGOLD_FINISH), str(job.result_glb),
               "--views", str(job.views_dir), "--output", str(job.result_glb)]
    if not job.settings.get("marigold_normals"):
        command.append("--no-normals")
    return command


def run_marigold(job: FinishJob) -> tuple[bool, str | None]:
    """The optional Marigold pass after the worker. Returns (ran, note).

    `note` says why it was skipped or failed; the finished model is kept either way.
    """
    if not job.settings.get("marigold"):
        return False, None
    if not marigold_ready(job):
        return False, ("Marigold skipped: it needs Marigold V2 in Setup, and a Pixal3D "
                       "model with its saved camera.")
    message = "Marigold: de-lighting the front and baking surface normals" \
        if job.settings.get("marigold_normals") else "Marigold: de-lighting the front"
    job.emit({"phase": "marigold", "overall_pct": 98, "message": message})
    process = subprocess.Popen(
        marigold_command(job), cwd=str(REPO), env=processes.child_env(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        **processes.TEXT_PIPE, **processes.group_popen_kwargs(),
    )
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.rstrip("\n")
        if line:
            job.append_log(line)
    code = process.wait()
    job.append_log(f"marigold exited with code {code}")
    if code != 0:
        return False, "Marigold finish failed; the finished model was kept (see the log)"
    return True, None


def needed_inputs(job: FinishJob, settings: dict[str, Any]) -> list[Path]:
    """The input files a run cannot do without. The picture only feeds Pixel Match and the
    repaint, so a run with both off (one started without a picture) needs only the model.
    Hi3D refine conditions on the picture, so it is required there."""
    if settings.get("hi3d"):
        return [job.asset_path, job.image_path]
    if settings.get("skip_photo", False) and settings.get("skip_paint", True):
        return [job.asset_path]
    return [job.asset_path, job.image_path]


def uses_photo(job: FinishJob, settings: dict[str, Any]) -> bool:
    """Whether this run gets the photo stage: a camera was found and it was not turned off."""
    return not settings.get("skip_photo", False) and (job.views_dir / "transforms.json").is_file()


def photo_note(job: FinishJob, settings: dict[str, Any]) -> str | None:
    """Why Pixel Match did not run, when it was asked for and could not: None otherwise.

    Said at the start and again on the finished result, because a ticked box with no stage
    and a one-second start-up line read as Pixel Match having run."""
    if settings.get("skip_photo", False) or uses_photo(job, settings):
        return None
    return ("Pixel Match skipped: no saved camera for this model. It works on Pixal3D "
            "models generated in this lab, which keep their camera beside the .glb.")


# What a run prints when Pixel Match paints: the photo stage's own count, or the Hero route's stage line.
PHOTO_RAN = re.compile(r"^\[photo\] \d+ view\(s\)|^\[hero\s+\d+s\] .*Pixel Match")
PIXEL_MATCH_NAME = "pixel-match.json"   # beside the result: the Library reads it after the job is gone


def pixel_match_result(ran: bool, views_dir: Path, settings: dict[str, Any]) -> str:
    """Whether Pixel Match ran, in the words the Finish step shows: 'on (front + back)',
    'on (front)', 'off' or 'skipped: <why>'. Without it a skipped run and one where Pixel Match
    changed little look the same."""
    if ran:
        frames = (_read_json(views_dir / "transforms.json") or {}).get("frames") or []
        return "on (front + back)" if len(frames) >= 2 else "on (front)"
    if settings.get("skip_photo", False):
        return "off"
    if not (views_dir / "transforms.json").is_file():
        return "skipped: no saved camera (only Pixal3D models made here have one)"
    return "skipped"


def find_source_views(asset: bytes, root: Path = GENERATED_ROOT) -> Path | None:
    """The saved camera + view directory of the Pixal3D run that produced `asset`.

    The browser uploads bytes, not a path, so the model is matched by content: every
    Pixal3D run writes a record (`.provenance.json` now, `.json` on older runs) with the
    GLB's sha256, and its staged views sit beside the GLB as `<output>.svviews/`. Anything
    else (another generator, a model from elsewhere, an edited file) has no match and
    simply gets no photo stage.
    """
    for record_path, record in _matching_runs(asset, root):
        views = Path((record.get("output") or {}).get("path", "")).with_suffix(".svviews")
        if not (views / "transforms.json").is_file():
            views = record_path.with_name(_run_name(record_path) + ".svviews")
        if (views / "transforms.json").is_file():
            return views
    return None


def find_source_dump(asset: bytes, root: Path = GENERATED_ROOT) -> Path | None:
    """Pixal3D's saved clean-up input for the run that produced `asset`, or None.

    Matched by content like the camera, and found whether or not Pixel Match runs: the
    Pixal3D Finish route needs the dump, not the pictures. Runs from before the dump was
    saved, and builds without the patch, have none and take the classic route.
    """
    for record_path, record in _matching_runs(asset, root):
        name = record.get("post_dump")
        if not name:
            continue
        for folder in (Path((record.get("output") or {}).get("path", "")).parent,
                       record_path.parent):
            if (folder / name).is_file():
                return folder / name
    return None


def find_source_record(asset: bytes, root: Path = GENERATED_ROOT) -> Path | None:
    """Find a licence record by output hash, independently of camera/dump metadata."""
    digest = hashlib.sha256(asset).hexdigest()
    root = root.resolve()
    if not root.is_dir():
        return None
    for path in root.rglob("*.json"):
        if not path.resolve().is_relative_to(root):
            continue
        record = _read_json(path)
        if not isinstance(record, dict) or not record.get("license"):
            continue
        output = record.get("output")
        if isinstance(output, dict) and output.get("sha256") == digest:
            return path
    return None


def _run_name(record_path: Path) -> str:
    """`run.json` and `run.provenance.json` name the same run."""
    return record_path.name.removesuffix(".provenance.json").removesuffix(".json")


def _matching_runs(asset: bytes, root: Path):
    """(record path, record) for each Pixal3D run whose output has `asset`'s sha256.

    A run may have an old `.json` record, a new `.provenance.json`, or (during a
    transition) both. `rglob("*.json")` matches both, so a run is reported once.
    """
    digest = hashlib.sha256(asset).hexdigest()
    if not root.is_dir():
        return
    seen = set()
    for record_path in root.rglob("*.json"):
        if record_path.stat().st_size > 20000:
            continue
        key = (record_path.parent, _run_name(record_path))
        if key in seen:
            continue
        record = _read_json(record_path)
        if not isinstance(record, dict) or record.get("backend") != "pixal3d":
            continue
        if (record.get("output") or {}).get("sha256") == digest:
            seen.add(key)
            yield record_path, record


def link_or_copy(source: Path, target: Path) -> None:
    """A hard link where the filesystem allows one (a dump is 200-450 MB), else a copy."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def stage_bands(stages: list[str]) -> dict[str, tuple[float, float]]:
    """Slice 0..100 between the stages this run will actually execute."""
    weights = {**STAGE_WEIGHTS, **HERO_WEIGHTS}
    active = [stage for stage in stages if stage in weights]
    total = sum(weights[stage] for stage in active) or 1.0
    bands: dict[str, tuple[float, float]] = {}
    cursor = 0.0
    for stage in active:
        cursor_end = cursor + weights[stage] / total * 100.0
        bands[stage] = (cursor, cursor_end)
        cursor = cursor_end
    return bands


def remaining_seconds(elapsed: float, fraction: float) -> float | None:
    """Seconds still to go, extrapolated from how long a fraction took.

    Linear and self-correcting rather than a table of per-stage constants: the repaint
    scales with paint steps, paint resolution and face count, so any constant would be
    wrong for most runs. None until there is enough evidence to be worth showing.
    """
    if not 0.02 < fraction < 1.0 or elapsed <= 1.0:
        return None
    return round(elapsed * (1.0 - fraction) / fraction, 1)


class FinishProgress:
    """Turn one line of worker stdout into a progress event, or None.

    Two line shapes carry progress. `I2L_STAGE::phase::message` marks a stage boundary,
    and the repaint prints its own `step 7/15 107s` as it denoises. The second is the one
    that matters: the repaint is ~95% of the wall clock, and without it the panel shows a
    single row reading "estimating…" for six minutes, with nothing to separate a working
    run from a wedged one.

    The worker's final `I2L_STAGE::done` marker is deliberately swallowed. Only `run_job`
    knows the artifact URLs, so only `run_job` may emit the job's terminal event; letting
    the worker's marker through made the browser close its event stream one event early
    and render a download link pointing at `undefined` (found 2026-09-21, on a run whose
    finished GLB had been sitting complete on disk the whole time).
    """

    def __init__(self, stages: list[str], clock=time.monotonic):
        self.bands = stage_bands(stages)
        self.clock = clock
        self.started = clock()
        self.phase: str | None = None
        self.stage_started = self.started
        self.stage_fraction = 0.0

    def feed(self, line: str) -> dict[str, Any] | None:
        if line.startswith("I2L_STAGE::"):
            _, phase, message = line.split("::", 2)
            if phase == "done":
                return None
            self.phase = phase
            self.stage_started = self.clock()
            self.stage_fraction = 0.0
            return self._event(message)
        if self.phase != "repaint":
            return None
        match = PAINT_STEP.match(line)
        if match:
            step, total = int(match.group(1)), int(match.group(2))
            span = PAINT_STEPS_END_FRACTION - PAINT_SETUP_FRACTION
            fraction = PAINT_SETUP_FRACTION + span * step / max(total, 1)
            return self._advance(
                fraction, f"Denoising step {step}/{total}", step=step, total=total,
            )
        for needle, fraction in PAINT_MARKS:
            if needle in line:
                return self._advance(fraction, line.strip())
        return None

    def _advance(self, fraction: float, message: str, **extra: Any) -> dict[str, Any]:
        # Never backwards. A bar that retreats reads as a restart, and the paint stage's
        # marks do not arrive in a guaranteed order across resolutions.
        self.stage_fraction = max(self.stage_fraction, min(fraction, 1.0))
        return self._event(message, measured=True, **extra)

    def _event(self, message: str, *, measured: bool = False, **extra: Any) -> dict[str, Any]:
        low, high = self.bands.get(self.phase or "", (0.0, 100.0))
        overall = low + (high - low) * self.stage_fraction
        event: dict[str, Any] = {
            "phase": self.phase,
            "message": message,
            "overall_pct": round(overall, 1),
            "total_eta_seconds": remaining_seconds(
                self.clock() - self.started, overall / 100.0,
            ),
            **extra,
        }
        # A stage with no sub-progress of its own (retopology, compress) reports none:
        # a row frozen at "0%" is a worse lie than a row that simply says it is running.
        if measured:
            event["stage_pct"] = round(self.stage_fraction * 100)
            event["stage_eta_seconds"] = remaining_seconds(
                self.clock() - self.stage_started, self.stage_fraction,
            )
        return event


# Files only a pre-0.3.5 (flat) run has at its top level.
LEGACY_MARKERS = ("source.glb", "source.png", "settings.json", "result.glb")


def faces_label(faces: int) -> str:
    """A face count the way people say it: 5000 -> "5k", 12500 -> "12.5k", 800 -> "800"."""
    if faces < 1000:
        return str(faces)
    return f"{faces / 1000:g}k"


def final_name(directory_name: str, faces: int, full: bool = False, hero: bool = False,
               hi3d: bool = False) -> str:
    """The finished model's file name: the asset, then its face count (`vanguard_5k.glb`),
    `_full` for a full-detail finish that kept every triangle, `_hero` for one whose head
    was built again on its own (scripts/hero_route.py), or `_hi3d` for a geometry-only
    Hi3D refinement.

    The asset is the run folder's name up to its first `__`, which drops the generator
    and timestamp tags a generated file carries (`vanguard__pixal3d` -> `vanguard`).
    """
    tag = ("hi3d" if hi3d else "hero" if hero else "full" if full else faces_label(faces))
    return f"{directory_name.split('__')[0]}_{tag}.glb"


class FinishJob:
    """One run's folder. Laid out so the finished model is the only GLB at the top:

        vanguard_5k.glb                  the result
        vanguard_5k.retopo-repaint.json  what made it
        input/   source.glb, source.png, source.views/, settings.json
        steps/   1_retopo.glb, 2_painted.glb, 3_photo.glb, 4_baked.glb, logs, run.log

    Runs from before 0.3.5 kept everything flat (`result.glb` among `result_*.glb` and a
    `source.glb`); they are recognised by any of those top-level files and still read.
    """

    def __init__(self, job_id: str, directory: Path):
        self.id = job_id
        self.directory = directory
        self.legacy = any((directory / name).exists() for name in LEGACY_MARKERS)
        self.input_dir = directory if self.legacy else directory / "input"
        self.steps_dir = None if self.legacy else directory / "steps"
        self.asset_path = self.input_dir / "source.glb"
        self.image_path = self.input_dir / "source.png"
        # The source photo(s) and cameras for the photo stage, copied in from the Pixal3D
        # run so a resume or a re-run does not depend on that run still being on disk.
        self.views_dir = self.input_dir / "source.views"
        # Pixal3D's own clean-up input, linked in from the run for the Pixal3D route.
        self.dump_path = self.input_dir / "source.post.bin"
        self.settings_path = self.input_dir / "settings.json"
        self.log_path = (self.steps_dir or directory) / "run.log"
        self.settings: dict[str, Any] = dict(DEFAULT_SETTINGS)
        self.status = "queued"
        self.started = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self.condition = threading.Condition()
        self.process: subprocess.Popen[str] | None = None
        self.cancel_requested = False
        self.resume = False
        self.log_lines: deque[str] = deque(maxlen=300)

    @property
    def result_glb(self) -> Path:
        if self.legacy:
            return self.directory / "result.glb"
        # A Hi3D-textured run ends in the normal worker, so it takes the normal name
        # (a face count); only the geometry-only run carries the `_hi3d` tag.
        hi3d_only = bool(self.settings.get("hi3d")) and not self.settings.get("hi3d_texture")
        return self.directory / final_name(self.directory.name, int(self.settings["faces"]),
                                           bool(self.settings.get("keep_mesh")),
                                           bool(self.settings.get("hero")), hi3d_only)

    @property
    def record_path(self) -> Path:
        return self.result_glb.with_suffix(".retopo-repaint.json")

    @property
    def watertight_glb(self) -> Path:
        """The optional watertight rebuild, beside the result in `steps/`.

        A property rather than a field so `ARTIFACTS` can name it before the run has
        decided whether to make one; the file simply is not there when it did not.
        """
        return (self.steps_dir or self.directory) / "watertight.glb"

    def stage_artifact(self, key: str | None) -> Path:
        """The file that proves a stage ran (see STAGE_ARTIFACTS)."""
        if key is None:
            return self.result_glb
        return worker_module().step_paths(self.result_glb, self.steps_dir)[key]

    def emit(self, event: dict[str, Any]) -> None:
        payload = {"elapsed_seconds": round(time.monotonic() - self.started, 1), **event}
        with self.condition:
            self.events.append(payload)
            self.condition.notify_all()

    def append_log(self, line: str) -> None:
        self.log_lines.append(line)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class FinishJobManager:
    def __init__(self, output_root: Path = OUTPUT_ROOT, generated_root: Path = GENERATED_ROOT):
        self.output_root = output_root
        self.generated_root = generated_root
        self.jobs: dict[str, FinishJob] = {}
        self.active: str | None = None
        self.lock = threading.Lock()

    def create(
        self, asset_name: str, asset: bytes, image: bytes | None, settings: dict[str, Any],
    ) -> FinishJob:
        """A new run. `image` None finishes the mesh alone: no Pixel Match, no repaint."""
        with self.lock:
            active = self.jobs.get(self.active) if self.active else None
            if active is not None and active.status not in TERMINAL:
                raise RuntimeError("a finishing job is already running")
            stem = _slug(Path(asset_name).stem)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            directory = self.output_root / f"{stem}__finish__{stamp}"
            suffix = 2
            while directory.exists():
                directory = self.output_root / f"{stem}__finish__{stamp}-{suffix}"
                suffix += 1
            directory.mkdir(parents=True, exist_ok=False)
            job = FinishJob(uuid.uuid4().hex, directory)
            try:
                job.settings = normalise_settings(settings)
                if job.settings.get("hi3d_texture") and not job.settings.get("hi3d"):
                    raise ValueError("Keeping the texture needs Hi3D refine turned on.")
                job.settings = fit_to_machine(job.settings)
                if image is None:
                    job.settings.update(skip_photo=True, skip_paint=True)
                hero = bool(job.settings.get("hero"))
                hi3d = bool(job.settings.get("hi3d"))
                # Checked before any file is written: a Hi3D refusal after the upload and
                # the run directory would be an expensive way to say "set it up".
                if hi3d and hero:
                    raise ValueError("Hi3D refine and Hero face do not combine: pick one.")
                if hi3d and image is None:
                    raise ValueError("Hi3D refine needs the picture the model was made from.")
                if hi3d:
                    ready, reason = hi3d_readiness()
                    if not ready:
                        raise ValueError(reason or "Hi3D refine is not available here")
                job.input_dir.mkdir(parents=True, exist_ok=True)
                job.asset_path.write_bytes(asset)
                if image is not None:
                    job.image_path.write_bytes(image)
                if hero and image is None:
                    raise ValueError("Hero needs the picture the model was made from.")
                # The saved camera + views feed Pixel Match AND Marigold, so a Marigold-only
                # run still needs them even though Pixel Match is off.
                need_views = (not job.settings["skip_photo"]) or hero or job.settings.get("marigold")
                views = find_source_views(asset, self.generated_root) if need_views else None
                if hero and views is None:
                    raise ValueError("Hero works on Pixal3D models made from a picture: this model has no "
                                     "saved camera.")
                if views is not None:
                    shutil.copytree(views, job.views_dir)
                dump = find_source_dump(asset, self.generated_root)
                if dump is not None:
                    link_or_copy(dump, job.dump_path)
                record = find_source_record(asset, self.generated_root)
                if hero and record is None:
                    # Older Hero runs also use non-licence metadata from the source manifest.
                    record = next((path for path, _ in _matching_runs(asset, self.generated_root)), None)
                if record is not None:
                    shutil.copy(record, job.input_dir / "source.json")   # its licences, for provenance
                # Written up front, not with the record at the end: a run that dies
                # halfway is exactly the one worth resuming, and resuming it means
                # knowing the settings the surviving intermediates were made with.
                job.settings_path.write_text(json.dumps(job.settings, indent=2), encoding="utf-8")
            except Exception:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            self.jobs[job.id] = job
            self.active = job.id
            return job

    def adopt(self, name: str) -> FinishJob:
        """A job bound to an existing run directory, to be run with `--resume`.

        The settings come from the directory rather than the request. Re-tuning on resume
        would silently mix a new face target with an old retopology; a changed setting is
        a new run, and the browser offers it as one.
        """
        with self.lock:
            active = self.jobs.get(self.active) if self.active else None
            if active is not None and active.status not in TERMINAL:
                raise RuntimeError("a finishing job is already running")
            directory = _run_directory(self.output_root, name)
            job = FinishJob(uuid.uuid4().hex, directory)
            if not job.settings_path.is_file():
                raise RuntimeError(f"{name} predates resume support and has no settings.json")
            for required in needed_inputs(job, json.loads(job.settings_path.read_text(encoding="utf-8"))):
                if not required.is_file():
                    raise RuntimeError(f"{name} is missing {required.name}")
            job.settings = fit_to_machine(
                normalise_settings(json.loads(job.settings_path.read_text(encoding="utf-8"))))
            job.resume = True
            self.jobs[job.id] = job
            self.active = job.id
            return job

    def get(self, job_id: str) -> FinishJob | None:
        return self.jobs.get(job_id) if JOB_ID.fullmatch(job_id) else None

    def finish(self, job: FinishJob) -> None:
        with self.lock:
            if self.active == job.id:
                self.active = None


FINISH_JOBS = FinishJobManager()


def run_job(job: FinishJob, manager: FinishJobManager = FINISH_JOBS) -> None:  # Hail to the king, baby.
    if not WORKER.is_file():
        job.status = "error"
        job.emit({"phase": "error", "message": f"worker missing: {WORKER}"})
        manager.finish(job)
        return
    try:
        job.status = "running"
        worker_source = None
        worker_record = None
        if job.settings.get("hi3d"):
            if job.settings.get("hi3d_texture"):
                worker_source, worker_record = run_hi3d_textured(job)
            else:
                run_hi3d(job)
            if job.cancel_requested:
                job.status = "cancelled"
                job.emit({"phase": "error", "message": "Finishing cancelled"})
                manager.finish(job)
                return
            if worker_source is None:
                job.emit({
                    "phase": "done", "overall_pct": 100,
                    "message": (f"Hi3D refined at {job.result_glb.stat().st_size / 1048576:.1f} MB · "
                                "geometry only — no textures"),
                    "pixel_match": "off",
                    "result_url": f"/api/finish/{job.id}/result.glb",
                    "record_url": f"/api/finish/{job.id}/record.json",
                    "size_bytes": job.result_glb.stat().st_size,
                    "directory": job.directory.name,
                    "path": library_path(job.result_glb),
                    "source_url": served_url(job.asset_path),
                    "pixal3d": (job.views_dir / "transforms.json").is_file(),
                    "hi3d": True,
                })
                job.status = "done"
                usage_api.count_done("finish")   # nothing if the person turned sharing off
                manager.finish(job)
                return
            # Texture kept: fall through to the normal worker, which finishes the
            # refined mesh (Detail, Pixel Match, Marigold, compress) as usual.
            job.append_log("hi3d refine done; finishing the textured mesh")
        hero = bool(job.settings.get("hero"))
        stages = list(HERO_STAGES) if hero else worker_module().stage_plan(
            job.settings["skip_paint"], job.settings["skip_compress"],
            job.settings.get("skip_bake", False), photo=uses_photo(job, job.settings),
            keep_mesh=bool(job.settings.get("keep_mesh")),
        )
        if hero:
            message = ("Starting Hero: the body is finished, then the head is built again on its own "
                       "for a sharper face. This takes longer than a normal Finish.")
        elif job.resume:
            message = "Resuming from what is already on disk"
        elif "photo" in stages:
            message = "Starting: found the source camera, so Pixel Match will keep the photo's pixels"
        elif photo_note(job, job.settings):
            message = photo_note(job, job.settings)
        else:
            message = "Starting"
        job.emit({"phase": "queued", "overall_pct": 0, "stages": stages, "message": message})
        progress = FinishProgress(stages)
        job.process = subprocess.Popen(
            hero_command(job, job.settings) if hero else build_command(
                job, job.settings, resume=job.resume,
                source=worker_source, source_record=worker_record),
            cwd=str(REPO),
            env=processes.child_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, **processes.TEXT_PIPE,
            **processes.group_popen_kwargs(),
        )
        job.directory.joinpath("pid").write_text(processes.pid_record(job.process.pid), encoding="utf-8")
        assert job.process.stdout is not None
        photo_ran = False
        for raw in job.process.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            job.append_log(line)
            photo_ran = photo_ran or bool(PHOTO_RAN.search(line))
            event = progress.feed(line)
            if event:
                job.emit(event)
        return_code = job.process.wait()
        job.append_log(f"worker exited with code {return_code}")
        if job.cancel_requested:
            job.status = "cancelled"
            job.emit({"phase": "error", "message": "Finishing cancelled"})
        elif return_code != 0:
            job.status = "error"
            job.emit({
                "phase": "error", "message": failure_message(job.log_lines, return_code),
                "log_tail": "\n".join(job.log_lines)[-8000:],
            })
        elif job.result_glb.is_file():
            if hero and not job.record_path.is_file():
                provenance = job.result_glb.with_suffix(".provenance.json")
                if provenance.is_file():
                    shutil.copy(provenance, job.record_path)   # what record_url serves
            marigold_ran, marigold_note = run_marigold(job)
            watertight_target, watertight_error = None, None
            if job.settings.get("watertight"):
                # Last, and allowed to fail without losing the finish: the finished model is
                # the deliverable, the watertight rebuild is an extra one beside it.
                try:
                    watertight_target = run_watertight(job, job.result_glb)
                except RuntimeError as exc:
                    watertight_error = str(exc)
                    job.append_log(f"watertight: {watertight_error}")
            pixel_match = pixel_match_result(photo_ran, job.views_dir, job.settings)
            try:
                (job.directory / PIXEL_MATCH_NAME).write_text(json.dumps({"pixel_match": pixel_match}),
                                                             encoding="utf-8")
            except OSError:
                pass   # only the Library's line is lost; the result itself is fine
            # The done event (with the file's path) goes out BEFORE the status flips: a plan polls
            # the status and reads the last event, so the other order can lose the result.
            job.emit({
                "phase": "done", "overall_pct": 100,
                "message": (f"Finished at {job.result_glb.stat().st_size / 1048576:.1f} MB · "
                            f"Pixel Match: {pixel_match}"),
                "pixel_match": pixel_match,
                "result_url": f"/api/finish/{job.id}/result.glb",
                "record_url": f"/api/finish/{job.id}/record.json",
                "size_bytes": job.result_glb.stat().st_size,
                "directory": job.directory.name,
                "path": library_path(job.result_glb),
                "source_url": served_url(job.asset_path),
                "pixal3d": (job.views_dir / "transforms.json").is_file(),
                "marigold": marigold_ran,
                "marigold_note": marigold_note,
                "watertight": (watertight_target is not None
                               or bool(job.settings.get("watertight"))),
                "watertight_path": (library_path(watertight_target)
                                    if watertight_target is not None else None),
                "watertight_url": (f"/api/finish/{job.id}/watertight.glb"
                                   if watertight_target is not None else None),
                "watertight_error": watertight_error,
                "note": photo_note(job, job.settings),
            })
            job.status = "done"
            usage_api.count_done("finish")   # nothing if the person turned sharing off
        else:
            job.status = "error"
            job.emit({"phase": "error", "message": "worker exited without writing a GLB"})
    except Exception as exc:
        job.status = "error"
        job.emit({"phase": "error", "message": str(exc)})
    finally:
        job.directory.joinpath("pid").unlink(missing_ok=True)
        if job.process is not None and job.process.poll() is None:
            try:
                processes.terminate_group(job.process.pid)
                job.process.wait(timeout=30)
            except Exception as cleanup_exc:
                job.append_log(f"cleanup warning: {cleanup_exc}")
        manager.finish(job)


def failure_message(lines, return_code: int) -> str:
    """The headline for a failed run: the hero route's own reason when it gave one (it says what
    is missing, in plain words), else the exit code."""
    for line in reversed(list(lines)):
        if line.startswith("[hero] "):
            return line.removeprefix("[hero] ")
    return f"worker exited with code {return_code}"


def library_path(path: Path) -> str | None:
    """A file's path inside output/, as the studio's Library names it (the next step's input)."""
    try:
        return path.relative_to(OUTPUT_ROOT.parent).as_posix()
    except ValueError:
        return None


def cancel_job(job: FinishJob) -> None:
    if job.status in TERMINAL:
        raise RuntimeError(f"job is already {job.status}")
    job.cancel_requested = True
    job.status = "cancelling"
    if job.process is not None and job.process.poll() is None:
        processes.terminate_group(job.process.pid)


def status_payload(job: FinishJob) -> dict[str, Any]:
    return {
        "status": job.status,
        "settings": job.settings,
        "directory": job.directory.name,
        "log_tail": "\n".join(job.log_lines)[-4000:],
        "last_event": job.events[-1] if job.events else None,
    }


def _run_directory(root: Path, name: str) -> Path:
    """One run directory under `root`, by name, with no way out of it.

    The name reaches the filesystem straight from a URL, so it is matched against what
    `create` actually produces rather than merely scrubbed of "..".
    """
    if not RUN_DIRECTORY.fullmatch(name):
        raise RuntimeError(f"not a run directory name: {name!r}")
    directory = root / name
    if not directory.is_dir():
        raise RuntimeError(f"no such run: {name}")
    return directory


def describe_run(directory: Path) -> dict[str, Any]:
    """What survives of one run on disk, and whether it is worth resuming.

    Every stage leaves its artifact behind, so the directory alone says how far the run
    got — no server memory required, which is the point: jobs live in this process and a
    finished run must still be recoverable after a restart.
    """
    layout = FinishJob("", directory)
    settings = _read_json(layout.settings_path)
    if settings is not None:
        layout.settings.update(settings)
    complete = [
        stage for stage, key in STAGE_ARTIFACTS if layout.stage_artifact(key).is_file()
    ]
    result = layout.result_glb
    record = _read_json(layout.record_path)
    return {
        "directory": directory.name,
        "modified": directory.stat().st_mtime,
        "stages_complete": complete,
        "finished": result.is_file(),
        "size_bytes": result.stat().st_size if result.is_file() else None,
        "result_url": served_url(result) if result.is_file() else None,
        "source_url": served_url(layout.asset_path) if layout.asset_path.is_file() else None,
        # A copied-in camera means the source was a Pixal3D run (it faces the other way).
        "pixal3d": (layout.views_dir / "transforms.json").is_file(),
        "record_url": served_url(layout.record_path) if record else None,
        "seconds": (record or {}).get("seconds"),
        "settings": settings,
        # A finished run has nothing left to resume; one without its sources or its
        # settings cannot be resumed faithfully, so it is not offered.
        "resumable": (
            not result.is_file()
            and settings is not None
            and all(path.is_file() for path in needed_inputs(layout, settings))
        ),
    }


def list_runs(root: Path = OUTPUT_ROOT, limit: int = 25) -> list[dict[str, Any]]:
    """Every finishing run on disk, newest first."""
    if not root.is_dir():
        return []
    directories = sorted(
        (d for d in root.iterdir() if d.is_dir() and RUN_DIRECTORY.fullmatch(d.name)),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return [describe_run(d) for d in directories[:limit]]


def served_url(path: Path) -> str | None:
    """The URL the viewer's static handler serves this file at, if it serves it at all.

    The handler is rooted at the repository, so anything outside it has no URL — say so
    rather than emitting a link that would 404 in the browser.
    """
    try:
        return "/" + path.relative_to(REPO).as_posix()
    except ValueError:
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")[:80] or "asset"
