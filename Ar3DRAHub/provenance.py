from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from Ar3DRAHub import __version__
from Ar3DRAHub.matte import matte_model

# Who made the file. `made_with` sits beside the licence, never inside it: it says which
# app produced the asset, and must never read as if the studio licenses the model.
MADE_WITH_NAME = "Ar3DRA Hub"
MADE_WITH_URL = "https://github.com/SimarglV/Ar3DRAHub"

GLB_MAGIC = 0x46546C67
GLB_JSON_CHUNK = 0x4E4F534A


def made_with() -> dict[str, str]:
    """The app and version that produced a file, read from the package version."""
    return {"name": MADE_WITH_NAME, "version": __version__, "url": MADE_WITH_URL}


def stamp_provenance(record: dict[str, Any]) -> dict[str, Any]:
    """Add who made the file to a record. Licence fields are left exactly as they are."""
    record["made_with"] = made_with()
    return record


def write_provenance(path: Path, record: dict[str, Any], *, atomic: bool = False) -> Path:
    """The one way to write a `.provenance.json`.

    Stamps `made_with` first, so a writer cannot forget it, then writes the record. Old
    records without the field still load: readers parse plain JSON and none of them
    requires it. `atomic` writes beside the file and replaces it, so a crash never leaves
    half a record for a model that exists.
    """
    path = Path(path)
    stamp_provenance(record)
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if not atomic:
        path.write_text(text, encoding="utf-8")
        return path
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def glb_generator() -> str:
    return f"{MADE_WITH_NAME} {__version__}"


def set_glb_generator(path: Path, generator: str | None = None) -> Path:
    """Set `asset.generator` in a GLB and change nothing else.

    Done on the final file, after the last export (Blender writes its own value on the way
    through). The JSON chunk is re-padded to 4 bytes and the total length updated; the
    binary chunk is left untouched.
    """
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 12:
        raise ValueError(f"not a GLB version 2 file: {path}")
    magic, gl_version, _ = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or gl_version != 2:
        raise ValueError(f"not a GLB version 2 file: {path}")
    offset = 12
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        if chunk_type == GLB_JSON_CHUNK:
            start = offset + 8
            document = json.loads(data[start:start + chunk_length])
            asset = document.setdefault("asset", {})
            asset["generator"] = generator or glb_generator()
            payload = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode()
            payload += b" " * ((4 - len(payload) % 4) % 4)
            body = (data[:offset] + struct.pack("<II", len(payload), chunk_type) + payload
                    + data[start + chunk_length:])
            body = body[:8] + struct.pack("<I", len(body)) + body[12:]
            path.write_bytes(body)
            return path
        offset += 8 + chunk_length
    raise ValueError(f"no JSON chunk in {path}")


@dataclass(frozen=True)
class LicenseProfile:
    classification: str
    folder: str
    license_name: str
    license_url: str
    conditions: tuple[str, ...]
    models: tuple[str, ...]


# Qwen confirmed on 2026-09-21 that outputs are not part of the licensed Materials, so the
# pictures belong to whoever made them. The model runs on the user's own machine and its
# weights are never redistributed by us. The statement is cited so the record stands alone.
# The whole project ships under MIT, and every model record names it, whatever the
# upstream card said. The license_url points at the repo's own LICENSE file.
PROJECT_LICENSE_URL = "https://github.com/SimarglV/Ar3DRAHub/blob/main/LICENSE"

QWEN_OUTPUT_STATEMENT = "https://x.com/QwenDevs/status/2101917379785838660"
QWEN_OUTPUT_RIGHTS = (
    "The pictures you generate are yours: outputs are not part of the "
    f"licensed Materials ({QWEN_OUTPUT_STATEMENT}, 21 Sept 2026). The model runs on your own "
    "machine; its weights are not redistributed."
)
FLUX_OUTPUT_RIGHTS = (
    "The pictures you generate are yours, for any use including commercial: FLUX.2 [klein] 4B "
    "and its Qwen3-4B text encoder are both MIT."
)
# A model a person dropped into models/pictures/ themselves: its outputs are theirs, but its
# licence was not reviewed here, so the record says to verify it, never to assume it is clear.
CUSTOM_OUTPUT_RIGHTS = (
    "The pictures you generate are yours, but this model was added by you and its licence was "
    "not reviewed here: check the model's own licence before using the pictures commercially."
)


# Whichever remover this machine runs (Ar3DRAHub/matte.py picks it).
MATTE_LICENSES = {
    "birefnet-lite-lean": "MIT",
    "birefnet-general-lite": "MIT",
    "u2net": "MIT",
}

# Terms kept beside records that bundle the DINOv3 encoder.
DINOV3_TERMS = ()

LICENSES = {
    "sf3d": LicenseProfile(
        classification="commercial-clear",
        folder="commercial",
        license_name="MIT",
        license_url=PROJECT_LICENSE_URL,
        conditions=(),
        models=("stabilityai/stable-fast-3d",),
    ),
    "hunyuan-comfyui": LicenseProfile(
        classification="commercial-clear",
        folder="commercial",
        license_name="MIT",
        license_url=PROJECT_LICENSE_URL,
        conditions=(),
        models=("tencent/Hunyuan3D-2.1",),
    ),
    "trellis2": LicenseProfile(
        classification="commercial-clear",
        folder="commercial",
        license_name="MIT",
        license_url=PROJECT_LICENSE_URL,
        conditions=(
            "BRIA RMBG-2.0 is disabled and must not be loaded in commercial runs.",
            "No military, nuclear, espionage, gun or illegal-weapon use; no use that breaks "
            "sanctions or export rules.",
        ),
        models=("microsoft/TRELLIS.2-4B",),
    ),
    "pixal3d": LicenseProfile(
        classification="commercial-clear",
        folder="commercial",
        license_name="MIT",
        license_url=PROJECT_LICENSE_URL,
        conditions=(
            "No military, nuclear, espionage, gun or illegal-weapon use; no use that breaks "
            "sanctions or export rules.",
        ),
        models=("raven38/pixal3d-sv-q8_0-v1",),
    ),
    "hi3d": LicenseProfile(
        classification="commercial-clear",
        folder="commercial",
        license_name="MIT",
        license_url=PROJECT_LICENSE_URL,
        conditions=(
            "A related product or service above 200,000 monthly active users or "
            "USD 1,000,000 monthly revenue should notify the Licensor within 60 days.",
            "Commercial integrations display 'Powered by Hi3D 1.0'.",
        ),
        models=("hi3d-official/Hi3Dv1.0",),
    ),
    "ultratex": LicenseProfile(
        classification="commercial-clear",
        folder="commercial",
        license_name="MIT",
        license_url=PROJECT_LICENSE_URL,
        conditions=(
            "Do not host the model as a service and do not train a competing model.",
        ),
        models=("yiboz2001/UltraTex", "black-forest-labs/FLUX.2-klein-base-9B"),
    ),
    "qwen-image-2.1": LicenseProfile(
        classification="commercial-clear",
        folder="commercial",
        license_name="MIT",
        license_url=PROJECT_LICENSE_URL,
        conditions=(
            QWEN_OUTPUT_RIGHTS,
            "Attribution: say 'Built with Qwen'.",
        ),
        models=("Qwen/Qwen-Image-2.1",),
    ),
    "flux2-klein-4b": LicenseProfile(
        classification="commercial-clear",
        folder="commercial",
        license_name="MIT",
        license_url=PROJECT_LICENSE_URL,
        conditions=(FLUX_OUTPUT_RIGHTS,),
        models=("black-forest-labs/FLUX.2-klein-4B", "Qwen/Qwen3-4B"),
    ),
    "marigold-v2": LicenseProfile(
        classification="commercial-clear",
        folder="commercial",
        license_name="MIT",
        license_url=PROJECT_LICENSE_URL,
        conditions=(),
        models=("huawei-bayerlab/marigold-v2-0", "Qwen/Qwen-Image-Edit-2509"),
    ),
    # A picture model the person dropped into models/pictures/ themselves. There is nothing to
    # review: the licence is whatever that model declares, so it is conditional by default and
    # the record tells them to verify it. The manifest may name the licence, which overrides
    # the label and link on the picture's own record.
    "custom-model": LicenseProfile(
        classification="commercial-conditional",
        folder="conditional",
        license_name="A model you added — verify its licence",
        license_url="",
        conditions=(
            CUSTOM_OUTPUT_RIGHTS,
            "This model was added by you, not reviewed here. Its licence is whatever it "
            "declares; check it before using the pictures commercially.",
        ),
        models=(),
    ),
}


def load_run_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read run manifest {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("run manifest must be an object with schema_version 1")
    return data


def validate_run_policy(
    backend: str,
    use_case: str,
    distribution: str,
    allow_conditional: bool,
    model_identity: str | None = None,
    allow_unverified: bool = False,
    commercial_intent: bool = False,
    allow_research_only: bool = True,
) -> LicenseProfile:
    if use_case not in {"showcase", "game"}:
        raise ValueError("use_case must be 'showcase' or 'game'")
    if distribution not in {"private", "public", "worldwide"}:
        raise ValueError("distribution must be 'private', 'public', or 'worldwide'")
    try:
        profile = LICENSES[backend]
    except KeyError as exc:
        raise ValueError(
            f"backend {backend!r} has no reviewed license profile"
        ) from exc
    if profile.classification != "commercial-clear" and not allow_conditional:
        raise ValueError(
            f"{backend} is {profile.classification}; manifest disallows conditional models"
        )
    if commercial_intent and profile.classification != "commercial-clear":
        raise ValueError(
            f"{backend} is {profile.classification}; commercial_intent does not allow it"
        )
    # research-only is stricter than conditional and is not covered by allow_conditional.
    # Running a non-commercial model is refused for anything but a private showcase,
    # whatever the manifest says. (Its outputs are the user's; running it is the limit.)
    if profile.classification == "research-only" and not allow_research_only:
        raise ValueError(
            f"{backend} is research-only; license_policy.allow_research_only is false"
        )
    if profile.classification == "research-only" and not (
        use_case == "showcase" and distribution == "private"
    ):
        raise ValueError(
            f"{backend} is research-only (non-commercial); it is permitted only for a "
            f"private showcase, not for use_case={use_case!r} distribution={distribution!r}"
        )
    if not allow_unverified and (
        model_identity is None or model_identity not in profile.models
    ):
        if model_identity is None:
            raise ValueError(
                f"{backend} has no reviewed model identity; declare it in a run manifest "
                "with license_policy.allow_unverified: true if you have verified the "
                "licence yourself"
            )
        raise ValueError(
            f"model identity {model_identity!r} is not in the reviewed licence profile for "
            f"{backend}; declare it in a run manifest with license_policy.allow_unverified: true "
            f"if you have verified the licence yourself"
        )
    return profile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# The in-repo MLX port is MIT. The weights it loads are not.
HUNYUAN_MLX_CODE_COMPONENT = {
    "component": "ZimengXiong/hunyuan3d-mlx (in-repo shape+paint port)",
    "purpose": "MLX generation code tracked in this repository",
    "license": "MIT",
    "license_url": PROJECT_LICENSE_URL,
    "commercial_status": "commercial-clear",
}


def _hunyuan_weight_component(component: str, purpose: str, weights: str | None) -> dict[str, Any]:
    """One Tencent weight entry. `weights` is the file or repo this run actually loaded."""
    profile = LICENSES["hunyuan-comfyui"]
    record: dict[str, Any] = {
        "component": component,
        "purpose": purpose,
        "license": profile.license_name,
        "license_url": profile.license_url,
        "commercial_status": profile.classification,
        "conditions": list(profile.conditions),
    }
    if weights is not None:
        record["weights"] = weights
    return record


def write_hunyuan_sidecar(
    output: Path,
    image: Path,
    *,
    backend: str,
    parameters: dict[str, Any] | None = None,
    shape: bool = False,
    paint: bool = False,
    shape_weights: str | None = None,
    paint_weights: str | None = None,
    include_code: bool = False,
) -> Path:
    """Write a schema-1 licence sidecar beside a Hunyuan export.

    A shape-only export passes `shape` and leaves `paint` false, so the record
    does not claim a paint model. A paint export names the weights it loaded.
    """
    if not shape and not paint:
        raise ValueError("a Hunyuan export must name the shape or paint weights it loaded")
    output = Path(output)
    image = Path(image)
    profile = LICENSES["hunyuan-comfyui"]
    components: list[dict[str, Any]] = []
    if include_code:
        components.append(dict(HUNYUAN_MLX_CODE_COMPONENT))
    if shape:
        components.append(_hunyuan_weight_component(
            "Tencent Hunyuan3D-2/2.1 shape weights",
            "shape-stage diffusion model",
            shape_weights,
        ))
    if paint:
        components.append(_hunyuan_weight_component(
            "Tencent Hunyuan3D-2/2.1 paint weights (MLX-ready conversion)",
            "paint-stage texture generation model",
            paint_weights,
        ))
    record = {
        "schema_version": 1,
        "backend": backend,
        "input": {
            "path": str(image),
            "sha256": sha256_file(image),
            "source": "user-provided",
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "classification": profile.classification,
        },
        "model": {
            "backend": backend,
            "parameters": parameters or {},
        },
        "license": {
            "name": profile.license_name,
            "url": profile.license_url,
            "conditions": list(profile.conditions),
        },
        "components": components,
    }
    return write_provenance(output.with_suffix(".provenance.json"), record)


def hunyuan_paint_run_dir(outputs_root: Path, stem: str) -> Path:
    """A folder this process owns. Two runs do not share textured_mesh_pbr.glb."""
    safe = stem.replace("/", "_").replace("\\", "_") or "paint"
    return Path(outputs_root) / f"{safe}-{os.getpid()}"


# WTiVo (WatertightVoxel, SimarglV) is a fast, CUDA-based watertight rebuild, and its
# record names it as MIT like everything else. It is kept as an `external_tools` entry
# rather than merged into the output's licence, because it is a tool we ran, not a term
# on the file.
WTIVO_COMPONENT = {
    "component": "WTiVo (WatertightVoxel, SimarglV)",
    "purpose": "external watertight rebuild (voxel resample + tetrahedral cell cut)",
    "license": "MIT",
    "license_url": PROJECT_LICENSE_URL,
    "commercial_status": "gpl-in-chain",
    "copyleft": False,
    "note": ("The record names it because the chain that made the file includes it. "
             "Windows and an NVIDIA GPU only."),
}


def write_external_tool_sidecar(
    source: Path,
    output: Path,
    *,
    operation: str,
    tool: dict[str, Any],
    source_record: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
) -> Path:
    """A sidecar for a file made by running an outside program.

    `write_derived_sidecar` is for our own instruments, and it merges their licences into
    the output's licence name. An external tool is different: its licence is about *us
    running it*, not about the file, so the output keeps the input's licence and class
    exactly, and the tool is named under `external_tools` with its copyleft status.

    The two are kept apart on purpose. Merging the tool into a licence name would
    read as a licence on the output, which it is not -- but dropping it from the record
    altogether would leave a reader unable to tell what made the file. Both are wrong; the
    answer is a separate key that says what it is.
    """
    import copy

    source, output = Path(source), Path(output)
    source_record = source_record if source_record is not None else read_model_record(source)
    origin = source_record or {}
    unknown = {"name": "Unverified — source licence unavailable", "verified": False,
               "conditions": ["Verify the source model's licence before use."]}
    licence = copy.deepcopy(origin.get("license") or unknown)
    classification = (origin.get("output") or {}).get("classification") or origin.get("classification") or "unverified"
    if classification == "unverified":
        licence = {**licence, "verified": False} if isinstance(licence, dict) else {"name": licence, "verified": False}
    record = {
        "schema_version": 1,
        "operation": operation,
        "input": {"path": str(source), "sha256": sha256_file(source)},
        "output": {"path": str(output), "sha256": sha256_file(output), "classification": classification},
        "model": copy.deepcopy(origin.get("model", {})),
        # The input's licence, untouched: an external tool does not relicense its output.
        "license": licence,
        # What the source used, unchanged, plus nothing new -- the tool did not become part
        # of the source's component list.
        "components": [copy.deepcopy(c) for c in origin.get("components", [])],
        # The one new fact: which outside program touched the file.
        "external_tools": [copy.deepcopy(tool)],
        "parameters": dict(parameters or {}),
        "source_record": source_record,
    }
    return write_provenance(output.with_suffix(".provenance.json"), record, atomic=True)


def _git_revision(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _pipeline_revision() -> dict[str, Any] | None:
    """This repository's commit, and whether the tree was dirty when the run started.

    Distinct from `backend_revision`, which is the vendored backend checkout. The
    patches in `scripts/patch_trellis_*.py` change what the backend *does* while living
    here, so the backend SHA alone does not identify the code that produced an asset.

    Without this, runs are only comparable by parameters -- and parameters can lie. The
    pangolin generated on 2026-08-02 declares `bake_target_faces: 200000`, but the fix
    that made that value take effect on Metal landed five days later, so the number in
    its sidecar was never what ran.

    `dirty` matters as much as the commit: uncommitted changes mean the SHA does not
    pin the code, and a benchmark built on such a run is not reproducible.
    """
    repo = Path(__file__).resolve().parents[1]
    commit = _git_revision(repo)
    if commit is None:
        return None
    try:
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": commit, "dirty": None}
    return {"commit": commit, "dirty": bool(status)}


def _package_versions() -> dict[str, str]:
    result = {}
    for package in ("torch", "torchvision", "rembg", "transformers", "trimesh"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            pass
    return result


def finalize_output(
    generated: Path,
    image: Path,
    output_root: Path,
    backend: str,
    profile: LicenseProfile,
    intent: dict[str, Any],
    parameters: dict[str, Any],
    source_manifest: Path | None,
    backend_repo: Path | None = None,
    model_identity: str | None = None,
    coarse_mesh: Path | None = None,
) -> tuple[Path, Path]:
    run_id = uuid.uuid4().hex[:12]
    verified = model_identity is not None and model_identity in profile.models
    classification = profile.classification if verified else "unverified"
    destination_dir = output_root.resolve() / profile.folder
    destination_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{image.stem}__{backend}__{classification}__{run_id}"
    destination = destination_dir / f"{stem}{generated.suffix.lower()}"
    generated.replace(destination)
    if destination.suffix.lower() == ".glb":
        set_glb_generator(destination)   # after the move, so the file that travels carries it

    if verified:
        license_record: dict[str, Any] = {
            "name": profile.license_name,
            "url": profile.license_url,
            "conditions": list(profile.conditions),
        }
    else:
        license_record = {
            "name": "Unverified — user-declared model",
            "verified": False,
            "conditions": [
                "Verify this model's licence yourself; this run used an identity not in the reviewed profile."
            ],
        }

    model_record: dict[str, Any] = {"backend": backend, "parameters": parameters}
    if model_identity is not None:
        model_record["identity"] = model_identity

    record = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "intent": intent,
        "input": {
            "path": str(image),
            "sha256": sha256_file(image),
            "source": "user-provided",
            **(
                {
                    "coarse_mesh": {
                        "path": str(coarse_mesh),
                        "sha256": sha256_file(coarse_mesh),
                    }
                }
                if coarse_mesh is not None
                else {}
            ),
        },
        "output": {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "classification": classification,
        },
        "model": model_record,
        "license": license_record,
        "components": (
            [
                {
                    "component": f"rembg/{matte_model()}",
                    "purpose": "background removal",
                    "license": MATTE_LICENSES[matte_model()],
                    "commercial_status": "commercial-clear",
                }
            ]
            if backend in {"sf3d", "trellis2"}
            else []
        )
        + (
            [
                {
                    "component": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                    "purpose": "image conditioning",
                    "license": "MIT",
                    "commercial_status": "commercial-clear",
                    "license_url": "LICENSE",
                },
                {
                    "component": "BRIA RMBG-2.0",
                    "purpose": "background removal",
                    "loaded": False,
                    "commercial_status": "blocked",
                    "note": "Explicitly disabled by repository patch.",
                },
            ]
            if backend == "trellis2"
            else []
        )
        + (
            [
                {
                    "component": "facebook/dinov2-large",
                    "purpose": "image conditioning",
                    "license": "MIT",
                    "commercial_status": "commercial-clear",
                },
                {
                    "component": "openai/clip-vit-large-patch14",
                    "purpose": "image conditioning",
                    "license": "MIT",
                    "commercial_status": "commercial-clear",
                },
                {
                    "component": "coarse input mesh",
                    "purpose": "geometry refined by Hi3D",
                    "path": str(coarse_mesh),
                    "source_record": read_model_record(coarse_mesh),
                }
                if coarse_mesh is not None
                else {
                    "component": "coarse input mesh",
                    "purpose": "geometry refined by Hi3D",
                    "license": "unrecorded — no sidecar found beside the coarse mesh",
                },
            ]
            if backend == "hi3d"
            else []
        ),
        "software": {
            "packages": _package_versions(),
            "backend_revision": _git_revision(backend_repo) if backend_repo else None,
            "pipeline_revision": _pipeline_revision(),
        },
        "source_manifest": {
            "path": str(source_manifest.resolve()),
            "sha256": sha256_file(source_manifest),
        }
        if source_manifest
        else None,
    }
    sidecar = write_provenance(destination.with_suffix(".provenance.json"), record)
    return destination, sidecar


def read_run_record(glb: Path) -> dict[str, Any] | None:
    """The run's own record beside a model: `.provenance.json`, or `.json` on older runs."""
    for suffix in (".provenance.json", ".json"):
        try:
            record = json.loads(glb.with_suffix(suffix).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict):
            return record
    return None


def read_model_record(glb: Path) -> dict[str, Any] | None:
    """Read a licence-bearing record beside a model, including older run records."""
    for suffix in (".provenance.json", ".json", ".retopo-repaint.json"):
        try:
            record = json.loads(glb.with_suffix(suffix).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and record.get("license"):
            return record
    return None


def write_derived_sidecar(
    source: Path, output: Path, *, operation: str,
    source_record: dict[str, Any] | None = None,
    component_records: Sequence[dict[str, Any]] = (),
) -> Path:
    """Keep the full licence ancestry while recording this derived file's identity."""
    import copy

    source_record = source_record if source_record is not None else read_model_record(source)
    unknown = {"name": "Unverified — source licence unavailable", "verified": False,
               "conditions": ["Verify the source model's licence before use."]}
    origin = source_record or {}
    licence = copy.deepcopy(origin.get("license") or unknown)
    classification = (origin.get("output") or {}).get("classification") or origin.get("classification") or "unverified"
    records = [origin, *component_records]
    if component_records:
        licences = [r.get("license") or unknown for r in records]
        conditions = []
        for item in licences:
            for term in item.get("conditions", []) if isinstance(item, dict) else []:
                if term not in conditions:
                    conditions.append(term)
        names: list[str] = []
        for item in licences:
            name = str(item.get("name", "Unverified")) if isinstance(item, dict) else str(item)
            if name not in names:
                names.append(name)
        licence = {"name": " + ".join(names), "conditions": conditions}
        classes = {(r.get("output") or {}).get("classification") or r.get("classification") or "unverified"
                   for r in records}
        classification = classes.pop() if len(classes) == 1 else "unverified"
    if classification == "unverified":
        licence = {**licence, "verified": False} if isinstance(licence, dict) else {"name": licence, "verified": False}
    record = {
        "schema_version": 1, "operation": operation,
        "input": {"path": str(source), "sha256": sha256_file(source)},
        "output": {"path": str(output), "sha256": sha256_file(output), "classification": classification},
        "model": copy.deepcopy(origin.get("model", {})), "license": licence,
        "components": [copy.deepcopy(c) for r in records for c in r.get("components", [])],
        "source_record": source_record, "component_records": list(component_records),
    }
    return write_provenance(output.with_suffix(".provenance.json"), record, atomic=True)
