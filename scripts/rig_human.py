"""Build a humanoid armature from placed joints and bind it with SkinTokens.

    <SkinTokens venv python> scripts/rig_human.py --input mesh.glb --output rigged.glb
                   [--joints joints.json]

The auto-rig invents a skeleton; this builds one from joint positions instead. Joints are
fractions of the mesh's bounding box (the space ``Ar3DRAHub.human_rig`` uses): a JSON
of ``{joint: [fx, fy, fz]}``, or the T-pose defaults when none is given — a hand-placement
tool nudges those onto the model and saves the result here. The armature is built in
Blender, exported beside the mesh, and ``scripts/rig_bind.py`` then holds that skeleton
fixed while SkinTokens predicts the weights. Geometry maths lives in
``Ar3DRAHub.human_rig``; only the Blender glue is here.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Ar3DRAHub import human_rig

RIG_BIND = REPO / "scripts" / "rig_bind.py"


def load_joints(path: Path | None) -> dict[str, tuple[float, float, float]]:
    """Placed fractions, over the T-pose defaults: a partial JSON fills in what it names."""
    joints = human_rig.defaults()
    if path is not None:
        for name, fraction in json.loads(path.read_text(encoding="utf-8")).items():
            if name not in joints:
                raise ValueError(f"unknown joint {name!r}; the humanoid has {len(joints)}")
            joints[name] = tuple(float(axis) for axis in fraction)
    missing = [name for name in human_rig.joint_order() if name not in joints]
    if missing:
        raise ValueError(f"joints missing from the template: {missing}")
    return joints


def build_armature(source: Path, prepped: Path,
                   joints: dict[str, tuple[float, float, float]]) -> int:  # pragma: no cover - bpy
    """Import the mesh, build the armature from placed fractions, export mesh + armature.

    Returns the number of bones built.
    """
    import bpy
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise ValueError("the input has no mesh")

    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for obj in meshes:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            for axis in range(3):
                lo[axis] = min(lo[axis], world[axis])
                hi[axis] = max(hi[axis], world[axis])
    size = hi - lo

    world = {name: Vector(human_rig.world_positions(lo, size, {name: joints[name]})[name])
             for name in joints}

    arm_data = bpy.data.armatures.new("Armature")
    arm = bpy.data.objects.new("Armature", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones
    for name, head, tail, _parent in human_rig.bone_list(joints):
        bone = eb.new(name)
        bone.head = world[name]
        bone.tail = human_rig.world_positions(lo, size, {name: tail})[name]
        if (bone.tail - bone.head).length < 1e-5:
            bone.tail = bone.head + Vector((0, 0, 1e-3))
    for name, _head, _tail, parent in human_rig.bone_list(joints):
        if parent:
            eb[name].parent = eb[parent]
    bpy.ops.object.mode_set(mode="OBJECT")

    prepped.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(prepped), export_format="GLB")
    return len(arm_data.bones)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path, help="mesh GLB with no skeleton")
    parser.add_argument("--output", required=True, type=Path, help="rigged GLB to write")
    parser.add_argument("--joints", type=Path, help="JSON of {joint: [fx, fy, fz]} fractions")
    args = parser.parse_args(argv)

    joints = load_joints(args.joints)
    env = dict(os.environ)
    with tempfile.TemporaryDirectory(prefix="rig_human_") as td:
        prepped = Path(td) / "prepped.glb"
        bones = build_armature(args.input, prepped, joints)
        print(f"built {bones} bones", flush=True)

        bind = subprocess.run(
            [sys.executable, str(RIG_BIND), "--input", str(prepped), "--output", str(args.output)],
            cwd=str(REPO), env=env, check=False)
        if bind.returncode != 0:
            print("bind failed (see the log above)", file=sys.stderr)
            return bind.returncode
    print(f"rigged: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
