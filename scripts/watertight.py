#!/usr/bin/env python3
"""Rebuild a generated mesh as one closed solid, MIT and without CGAL.

A renderer is happy with triangle soup; a slicer, a physics engine and a Boolean are
not. This closes the hairline gaps between shells and merges the pieces into one
watertight body, using only numpy, scipy, trimesh and scikit-image.

The maths lives in `Ar3DRAHub/watertight.py` so it can be tested without a subprocess;
this file is the command-line wrapper the studio calls.

Two things it does not do, both worth knowing before you run it:

* **It resamples.** Detail below the voxel size is gone. `--across` is that knob: 300
  keeps a sword edge and closes a 1M-face mesh in about three minutes on a CPU, 700 is
  sharper and much slower. Too coarse melts fingers.
* **The output has no UVs.** Carry the texture across with `scripts/blender_hi3d_transfer.py`,
  or bake again afterwards.

A model that is a thin shell stays a thin shell: the wall comes back one voxel thick,
never as a filled block. The volume goes up because one voxel is thicker than the
sub-voxel skin it replaces, not because the inside was poured in.

Licensing: this tool is MIT and the result keeps its input's licence. Nothing is
fetched from the network.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# `sys.path[0]` is `scripts/`, not the repository root, so `Ar3DRAHub` is not importable
# until the root is added. Done before any repository import, and the same way every
# script in here does it.
REPO = Path(__file__).resolve().parents[1]
# Where `bootstrap_wativo.py` installs the optional external engine. The studio looks in
# the same place, so a command-line run and an interface run use one install.
WTIVO_VENDOR = REPO / "vendor" / "wativo"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild a GLB as one closed, watertight mesh (MIT, no CGAL).")
    parser.add_argument("mesh", type=Path, help="the input .glb")
    parser.add_argument("output", type=Path, help="destination .glb")
    parser.add_argument(
        "--across", type=int, default=300,
        help="voxels across the model's longest side. Higher keeps more detail and costs "
             "time and memory in proportion; lower is coarser and faster. Default 300 "
             "(about three minutes and 8 GB on a 1M-face input). Ignored with --tool "
             "wativo, which takes its resolution from --wativo-res.",
    )
    parser.add_argument(
        "--voxel", type=float, default=None,
        help="voxel size in the model's own units, overriding --across. Use this to match "
             "a wall thickness you care about.",
    )
    parser.add_argument(
        "--source-record", type=Path, default=None,
        help="the input's provenance sidecar, so the result's licence chains from it. "
             "Defaults to a sibling '.provenance.json', or the run's 'source.json'.",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="report the input's condition and exit without rebuilding.",
    )
    parser.add_argument(
        "--tool", choices=("own", "wativo"), default="own",
        help="which rebuild runs. 'own' is this repository's MIT module (numpy/scipy/"
             "trimesh/scikit-image, CPU). 'wativo' runs an installed WTiVo checkout on its "
             "own 3.12 environment -- much faster on a CUDA machine, and its geometry path "
             "runs in its own process. The result keeps the input's "
             "licence either way; the record names the tool that ran.",
    )
    parser.add_argument(
        "--wativo-res", type=int, choices=(512, 1536), default=512,
        help="resolution for --tool wativo: 512 is fast and a lighter file, 1536 is the "
             "tool's full detail and a much bigger file. Default 512.",
    )
    parser.add_argument(
        "--wativo", type=Path, default=None,
        help="path to a WTiVo checkout (the folder, or its wtivo.py), for --tool wativo. "
             "Defaults to $WTIVO_HOME / $WTIVO_DIR, then vendor/wativo, then 'wativo' on "
             "PATH.",
    )
    parser.add_argument(
        "--wativo-python", type=Path, default=None,
        help="the interpreter that runs WTiVo's own script, when that is how it is "
             "installed. Its native extension is built for CPython 3.12, which is not "
             "necessarily the interpreter running this file.",
    )
    parser.add_argument(
        "--wativo-script", type=Path, default=None,
        help="WTiVo's 'wtivo.py'. With --wativo-python, runs the script instead of a "
             "binary -- the usual install: a checkout plus a 3.12 environment.",
    )
    return parser


def find_wativo(explicit: Path | None) -> Path | None:
    """Where WTiVo is: the argument, then `vendor/wativo`, then `$WTIVO_HOME` / `$WTIVO_DIR`,
    then `wativo` on PATH.

    The same places the studio looks, in the same order, so a command-line run and an
    interface run use one install. `$WTIVO_HOME` points at a checkout -- the folder or its
    `wtivo.py` -- and is the escape hatch for one that lives somewhere else.
    """
    if explicit is not None:
        candidate = explicit.expanduser()
        return candidate if candidate.is_file() else None
    if (WTIVO_VENDOR / "wtivo.py").is_file():
        return WTIVO_VENDOR / "wtivo.py"
    for name in ("WTIVO_HOME", "WTIVO_DIR"):
        value = os.environ.get(name)
        if not value:
            continue
        candidate = Path(value).expanduser()
        if candidate.is_file():
            return candidate
        if (candidate / "wtivo.py").is_file():
            return candidate / "wtivo.py"
    found = shutil.which("wativo")
    return Path(found) if found else None


def find_wativo_python(home: Path) -> Path | None:
    """The interpreter in a WTiVo checkout that can import its native extension.

    `$WTIVO_PYTHON` first, then the vendored `.venv`. The extension is a CPython 3.12
    `.pyd` built against torch 2.8.0, so the studio's own 3.11 is not a fallback -- it is
    a different error message. Whether a venv is even present is what `None` means.
    """
    explicit = os.environ.get("WTIVO_PYTHON")
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


def wativo_command(program_argv: list[str], source: Path, output: Path,
                   res: int, voxel: float | None) -> list[str]:
    """The WTiVo invocation, using its real flags and the resolution the caller chose.

    Read from `wtivo.py` in `SimarglV-WatertightVoxel-ComfyuiNode` (v1.1). It takes
    `--input`/`--output`, not positionals, and its resolution flags are `--input-res` /
    `--final-res`. The CPU engine's `--across` knob is deliberately not mapped onto it:
    instead the caller picks one of two measured points on WTiVo's own scale -- 512 (fast,
    a lighter file) or 1536 (the tool's full detail, a much bigger file). It has no
    voxel-size flag either -- a voxel size is derived from the model's extent and the
    resolution -- so `voxel` is reported rather than silently dropped.

    Kept in one function because it is the only place that has to match someone else's
    command line: if upstream renames a flag, this is the whole fix.
    """
    command = [*program_argv, "--input", str(source), "--output", str(output),
               "--input-res", str(int(res)), "--final-res", str(int(res))]
    if voxel is not None:
        # No voxel-size flag exists. Its resolution is what a voxel size maps onto, so this
        # is reported rather than silently dropped -- the run uses `res`.
        print("[watertight] note: WTiVo has no --voxel-size; it sizes the grid from the "
              "model's extent and --wativo-res", flush=True)
    return command


def run_external(program_argv: list[str], source: Path, output: Path,
                 res: int, voxel: float | None) -> None:
    """Run the outside program, wait, report.

    `program_argv` is the command *up to the flags*: an interpreter plus WTiVo's script.
    Everything after that is `wativo_command`'s business.

    `WTIVO_SUBPROCESS=1` is set on the child on purpose: WTiVo redirects its own stdout to
    `wtivo_execution.log` when it is not a tty, which would leave this run's log with
    nothing but a wall-clock and a failure code. That env is WTiVo's own escape hatch.
    """
    command = wativo_command(program_argv, source, output, res, voxel)
    print(f"[watertight] running {Path(program_argv[0]).name} ({' '.join(command[1:])})", flush=True)
    started = time.time()
    completed = subprocess.run(command, cwd=str(source.parent), text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env={**os.environ, "WTIVO_SUBPROCESS": "1"})
    if completed.stdout:
        for line in completed.stdout.splitlines():
            print(f"[watertight] {line}", flush=True)
    if completed.returncode != 0:
        raise SystemExit(f"WTiVo failed with code {completed.returncode}")
    if not output.is_file():
        raise SystemExit(f"WTiVo exited cleanly but wrote no {output.name}")
    print(f"[watertight] WTiVo finished in {time.time() - started:.0f}s -> "
          f"{output.stat().st_size / 1048576:.1f} MB", flush=True)


def _drop_crumbs(path: Path) -> dict | None:
    """Drop whole shells the external tool left detached, in place, or None when clean.

    WTiVo at 512 leaves the body plus a few microscopic specks (measured on the knight:
    three 12-face shells). The checkbox promises one closed body, so those are dropped
    with the same edge-connectivity rule the CPU engine uses -- `keep_body` keeps every
    shell that is at least `KEEP_FRACTION` of the biggest one, which preserves a hollow
    body's inner wall and removes only crumbs. The tool's geometry is otherwise untouched,
    and a clean result is left byte for byte.
    """
    import trimesh

    from Ar3DRAHub import watertight as core

    vertices, faces = load_mesh(path)
    kept_vertices, kept_faces, sizes = core.keep_body(vertices, faces)
    floor = sizes[0] * core.KEEP_FRACTION
    kept_count = sum(size >= floor for size in sizes)
    if kept_count == len(sizes):
        return None
    cleaned = trimesh.Trimesh(vertices=kept_vertices, faces=kept_faces, process=False)
    cleaned.export(path)
    summary = {"dropped_shells": len(sizes) - kept_count,
               "dropped_faces": int(len(faces) - len(kept_faces)),
               "shells_after": len(core.shells(kept_vertices, kept_faces))}
    print(f"[watertight] dropped {summary['dropped_shells']} detached shell(s) "
          f"({summary['dropped_faces']:,} faces) the tool left beside the body; "
          f"{summary['shells_after']} shell(s) kept", flush=True)
    return summary


def _run_external(mesh_path: Path, output: Path, args, record: dict | None) -> int:
    """The `--tool wativo` path: run someone else's program, then record that we did.

    Two ways in, because WTiVo ships two ways. A `--wativo` binary is one. The usual case is
    a checkout plus its own 3.12 environment, which is `--wativo-script` (its `wtivo.py`)
    plus `--wativo-python` (the interpreter that can import the native extension). Ours is
    3.11 and cannot, so which Python runs it is the whole point.

    The output keeps the input's licence. The record gains one key the MIT path never
    writes -- `external_tools` -- naming WTiVo, because a file rebuilt by it should say
    so when someone asks what made it.

    Absent from the interface on purpose: the studio shows this as a choice between
    "fast" and "this machine", not as a licence negotiation. The fact lives in the JSON,
    which is where anyone who needs it will look.
    """
    from Ar3DRAHub.provenance import write_external_tool_sidecar

    if args.wativo_script is not None:
        if args.wativo_python is None:
            raise SystemExit("--wativo-script needs --wativo-python: WTiVo's own script "
                             "must run on the interpreter its extension was built for.")
        program = Path(args.wativo_python).expanduser()
        if not program.is_file():
            raise SystemExit(f"not found: {program}")
        target_argv = [str(program), str(Path(args.wativo_script).expanduser())]
    else:
        found = find_wativo(args.wativo)
        if found is None:
            hint = args.wativo if args.wativo is not None else "$WTIVO_HOME, vendor/wativo or PATH"
            raise SystemExit(
                f"WTiVo not found ({hint}). Run python scripts/bootstrap_wativo.py, pass "
                f"--wativo /path/to/wativo, or run with --tool own to use the built-in rebuild."
            )
        if args.wativo_python is not None:
            program = Path(args.wativo_python).expanduser()
            if not program.is_file():
                raise SystemExit(f"not found: {program}")
            target_argv = [str(program), str(found)]
        elif found.suffix == ".py":
            # The vendored install: a script plus the venv that can import its extension.
            # A script run by the studio's own 3.11 would fail on the .pyd, so the venv is
            # not a convenience here -- it is the only interpreter that works.
            python = find_wativo_python(found.parent)
            if python is None:
                raise SystemExit(
                    f"no Python for WTiVo in {found.parent}. Run python "
                    f"scripts/bootstrap_wativo.py, or pass --wativo-python."
                )
            target_argv = [str(python), str(found)]
        else:
            target_argv = [str(found)]
    run_external(target_argv, mesh_path, output, args.wativo_res, args.voxel)

    # The external tool can leave the body plus a few detached specks. The promise is one
    # closed body, so crumbs are dropped here -- after the tool, before the record, so the
    # record can say both things that happened.
    cleanup = _drop_crumbs(output)

    from Ar3DRAHub.provenance import set_glb_generator

    try:
        set_glb_generator(output)
    except ValueError:
        pass

    if record is not None:
        from Ar3DRAHub.provenance import WTIVO_COMPONENT, write_provenance

        # What ran, named honestly: an interpreter plus a script, or a bare binary. Calling
        # the interpreter a "binary" once misnamed the record's own evidence.
        program = Path(target_argv[0])
        tool = dict(WTIVO_COMPONENT)
        if len(target_argv) > 1:
            tool["interpreter"] = str(program)
            tool["script"] = str(Path(target_argv[1]))
        else:
            tool["binary"] = str(program)
        write_external_tool_sidecar(
            mesh_path, output, operation="watertight",
            tool=tool,
            source_record=record,
            parameters={"resolution": args.wativo_res, "voxel": args.voxel},
        )
        if cleanup is not None:
            # The cleanup is ours, not the outside tool's, so it gets its own key rather
            # than riding inside `external_tools`: the record says who made the file and
            # who tidied it, and that the tidying is MIT.
            sidecar = output.with_suffix(".provenance.json")
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            payload["postprocess"] = {
                "component": "Ar3DRAHub/watertight.py (keep_body)",
                "purpose": "dropped whole shells the external tool left detached",
                "license": "MIT",
                **cleanup,
            }
            write_provenance(sidecar, payload)
    else:
        print("[watertight] no source record found: the result carries no licence chain",
              flush=True)
    print(f"[watertight] out: {output} via {Path(target_argv[0]).name}", flush=True)
    return 0


def load_mesh(path: Path):
    """`(vertices, faces)` from a GLB, as plain arrays."""
    import numpy as np
    import trimesh

    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise SystemExit(f"{path} is not a single mesh")
    return np.asarray(loaded.vertices, dtype=np.float64), np.asarray(loaded.faces, dtype=np.int64)


def find_source_record(mesh: Path, explicit: Path | None) -> Path | None:
    """The sidecar that carries the input's licence, if there is one.

    An explicit path wins. Otherwise the convention of this repo: a derived file sits
    beside its `.provenance.json`, and a generation run keeps a `source.json` one
    directory up in the finish folder's `input/`.
    """
    if explicit is not None:
        return explicit
    for candidate in (mesh.with_suffix(".provenance.json"),
                      mesh.parent / "source.json",
                      mesh.parent.parent / "input" / "source.json"):
        if candidate.is_file():
            return candidate
    return None


def run_summary(result, faces_in: int, across: int | None) -> dict:
    """What this rebuild did, for the sidecar and for a person reading it later."""
    return {
        "voxel": float(result.voxel),
        "across": across,
        "faces_in": int(faces_in),
        "faces_out": int(len(result.faces)),
        "shells_before": int(result.shells_before),
        "shells_after": int(result.shells_after),
        "boundary_edges": int(result.boundary_edges),
        "notes": list(result.notes),
    }


def write_sidecar(source: Path, output: Path, record: dict, summary: dict) -> Path:
    """The result's licence chain, plus what this rebuild did to the mesh.

    The instrument is recorded as a component, never as a new licence term: MIT material
    adds no condition to the output, so the output keeps whatever the input's record
    said. Recording it anyway is the point -- six months later "what made this" is a
    question the file answers by itself.
    """
    import json

    from Ar3DRAHub.provenance import write_derived_sidecar, write_provenance

    sidecar = write_derived_sidecar(
        source, output, operation="watertight",
        source_record=record,
        component_records=[{
            "component": "Ar3DRAHub/watertight.py",
            "purpose": "watertight rebuild (voxel resample + marching cubes)",
            "license": "MIT",
            # `write_derived_sidecar` collapses the output's classification from this key
            # on every record, and treats a missing one as "unverified". Left out, an MIT
            # tool would silently downgrade a commercial-clear input to unverified, which
            # is the opposite of what an MIT licence does.
            "classification": "commercial-clear",
            "commercial_status": "commercial-clear",
            "note": ("Adds no condition to the output; the result keeps the input's licence. "
                     "Built on numpy, scipy, trimesh and scikit-image, all MIT."),
        }],
    )
    # The rebuild's own numbers sit beside the licence chain, not inside it: a later
    # reader wants to know the voxel size that produced these triangles, because that is
    # the only record of what detail was lost. Written back through `write_provenance`,
    # which stamps `made_with` -- a raw `write_text` here would drop it and every sidecar
    # in the repo is checked for that.
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["watertight"] = summary
    write_provenance(sidecar, payload)
    return sidecar


def main() -> int:
    args = build_parser().parse_args()
    from Ar3DRAHub import watertight as core

    mesh_path = args.mesh.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not mesh_path.is_file():
        raise SystemExit(f"not found: {mesh_path}")

    vertices, faces = load_mesh(mesh_path)
    welded_vertices, welded_faces = core.weld(vertices, faces)
    shells_before = len(core.shells(welded_vertices, welded_faces))
    open_edges = core.boundary_edges(welded_faces)
    print(f"[watertight] in: {len(faces):,} faces, {shells_before:,} shell(s), "
          f"{open_edges:,} open edge(s)", flush=True)
    if args.check_only:
        state = "watertight" if shells_before == 1 and open_edges == 0 else "NOT watertight"
        print(f"[watertight] {state}")
        return 0

    record_path = find_source_record(mesh_path, args.source_record)
    record = (json.loads(record_path.read_text(encoding="utf-8"))
              if record_path is not None else None)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.tool == "wativo":
        return _run_external(mesh_path, output, args, record)

    started = time.time()

    def log(line: str) -> None:
        print(f"[watertight] {time.time() - started:6.1f}s {line}", flush=True)

    result = core.rebuild(vertices, faces, voxel=args.voxel, across=args.across, log=log)

    import trimesh

    rebuilt = trimesh.Trimesh(vertices=result.vertices, faces=result.faces, process=False)
    if not rebuilt.is_watertight:
        # Written anyway, so the failure can be looked at rather than guessed at -- but
        # said out loud first, because a file on disk that is called "watertight" and is
        # not would be the worst of both worlds.
        print(f"[watertight] WARNING: still not watertight "
              f"({result.boundary_edges:,} open edge(s), {result.shells_after:,} shell(s))",
              flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    rebuilt.export(output)
    from Ar3DRAHub.provenance import set_glb_generator

    try:
        set_glb_generator(output)
    except ValueError:
        pass   # a non-GLB destination (an OBJ, an STL) simply has no generator field

    if record is not None:
        write_sidecar(mesh_path, output, record, run_summary(result, len(faces), args.across))
    else:
        print("[watertight] no source record found: the result carries no licence chain",
              flush=True)

    print(f"[watertight] out: {len(result.faces):,} faces, {result.shells_after:,} shell(s), "
          f"{result.boundary_edges:,} open edge(s), voxel {result.voxel:.6g} -> {output} "
          f"({output.stat().st_size / 1048576:.1f} MB, {time.time() - started:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
