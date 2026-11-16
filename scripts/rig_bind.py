"""Bind SkinTokens skin weights onto a mesh's own skeleton instead of generating a new one.

    <SkinTokens venv python> scripts/rig_bind.py --input rigged.glb --output bound.glb

SkinTokens can hold an existing skeleton fixed and predict only the weights onto it
(``demo.py --use_skeleton --use_transfer``), which the studio's auto-rig never used: it
always generates a new humanoid skeleton. This script is the "keep my skeleton" path. It
first covers every joint with at least one influenced vertex -- without that, the engine
prunes joints whose subtree carries no skin, mutilating the rig -- then runs the engine
with the skeleton held fixed, then drops far-bone weights (``clean_skin_weights.py``). The
geometry lives in ``Ar3DRAHub.rig_geometry``; only the Blender glue is here.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Ar3DRAHub import rig_geometry

SKINTOKENS = REPO / "vendor" / "SkinTokens"
CLEAN_WEIGHTS = REPO / "scripts" / "clean_skin_weights.py"


def demo_command(skintokens: Path, source: Path, result: Path) -> list[str]:
    """SkinTokens, holding the input skeleton fixed and regenerating its weights."""
    return [str(skintokens / "demo.py"), "--input", str(source), "--output", str(result),
            "--use_skeleton", "--use_transfer"]


def clean_command(clean: Path, source: Path, result: Path) -> list[str]:
    """The far-bone weight clean-up that follows the engine."""
    return [str(clean), "--in", str(source), "--out", str(result)]


def free_ports(count: int = 2) -> list[int]:
    """Ports the system hands out right now, all different, so SkinTokens' Blender helper
    never sits on a fixed port Windows reserves for Hyper-V/WSL."""
    probes = [socket.socket() for _ in range(count)]
    try:
        for probe in probes:
            probe.bind(("127.0.0.1", 0))
        return [probe.getsockname()[1] for probe in probes]
    finally:
        for probe in probes:
            probe.close()


def port_env() -> dict[str, str]:
    """The environment that hands SkinTokens two free ports (read by the patched spec.py)."""
    port, bpy_port = free_ports(2)
    return {"SKINTOKENS_PORT": str(port), "SKINTOKENS_BPY_PORT": str(bpy_port)}


def coverage_bind(source: Path, prepped: Path,
                  max_influences: int = rig_geometry.MAX_INFLUENCES) -> int:  # pragma: no cover - bpy
    """Give every joint at least one influenced vertex, and export a prepped GLB.

    The weight values are throwaway -- Mode B regenerates them -- but the coverage is
    load-bearing: without it the engine prunes joints whose subtree has no skin. Returns
    the number of joint/vertex-group assignments made.
    """
    import bpy
    import numpy as np

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))

    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        raise ValueError("the input has no armature; bind to a skeleton that already exists")
    all_meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    covered = 0
    for arm in arms:
        meshes = [o for o in all_meshes if o.find_armature() == arm]
        if not meshes:
            meshes = all_meshes
        # The engine resolves its armature by object-name lookup, so the object name and
        # the data-block name must agree; Blender makes no promise they do.
        if arm.name != arm.data.name:
            arm.name = arm.data.name
        names = [b.name for b in arm.data.bones]
        bone_pos = np.array([list(arm.matrix_world @ b.head_local) for b in arm.data.bones],
                            dtype=np.float64)
        for mesh in meshes:
            verts = np.array([list(mesh.matrix_world @ v.co) for v in mesh.data.vertices],
                             dtype=np.float64)
            for joint_index, vertex_indices in rig_geometry.joint_coverage(
                    bone_pos, verts, max_influences).items():
                group = (mesh.vertex_groups.get(names[joint_index])
                         or mesh.vertex_groups.new(name=names[joint_index]))
                group.add(vertex_indices, 1.0, "REPLACE")
                covered += 1
            if not any(m.type == "ARMATURE" for m in mesh.modifiers):
                mod = mesh.modifiers.new(name="Armature", type="ARMATURE")
                mod.object = arm

    prepped.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(prepped), export_format="GLB")
    return covered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path, help="rigged GLB with a skeleton")
    parser.add_argument("--output", required=True, type=Path, help="bound GLB to write")
    parser.add_argument("--skintokens", type=Path, default=SKINTOKENS,
                        help="SkinTokens checkout (default: vendor/SkinTokens)")
    parser.add_argument("--clean", type=Path, default=CLEAN_WEIGHTS,
                        help="clean_skin_weights.py location")
    args = parser.parse_args(argv)

    python = Path(sys.executable)
    env = dict(os.environ, **port_env())
    with tempfile.TemporaryDirectory(prefix="rig_bind_") as td:
        tmp = Path(td)
        prepped = tmp / "prepped.glb"
        skinned = tmp / "skinned.glb"
        joints = coverage_bind(args.input, prepped)
        print(f"covered {joints} joint/vertex group assignment(s)", flush=True)

        demo = subprocess.run(
            [str(python), *demo_command(args.skintokens, prepped, skinned)],
            cwd=str(args.skintokens), env=env, check=False)
        if demo.returncode != 0:
            print("SkinTokens failed (see the log above)", file=sys.stderr)
            return demo.returncode

        clean = subprocess.run(
            [str(python), *clean_command(args.clean, skinned, args.output)],
            cwd=str(REPO), check=False)
        if clean.returncode != 0:
            print("weight clean-up failed (see the log above)", file=sys.stderr)
            return clean.returncode
    print(f"bound: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
