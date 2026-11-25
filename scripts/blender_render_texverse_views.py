"""Render the six canonical UltraTex views of a mesh: world normals and silhouette.

Run (headless): blender --background --python-exit-code 1 --python scripts/blender_render_texverse_views.py -- MESH.glb OUT_DIR [resolution]

**Why.** UltraTex conditions on an asset's six canonical views as world-space
normal maps: front, left, back, right, top and bottom, framed exactly the way
G-buffer TexVerse frames its training data. The camera poses (in
Ar3DRAHub/ultratex_backend.py) are the dataset's own pose files (decoded from
the demo objects shipped in the UltraTex checkout), so the renders match what
the model saw during training: perspective
cameras 1.67 units from the origin looking straight at it, lens 56.25 mm (fx 6400
at 4096 px on a 36 mm sensor), and the object normalised so its largest dimension
is 0.6 units (measured 84% frame fill on the demo's side view). Each tile's alpha
channel is the silhouette; the RGB is the world-space normal encoded as (N+1)/2.

The tiles land as bump_normal_world/000.webp .. 005.webp in the TexVerse layout
the UltraTex dataset reader consumes (its load_and_concat tiles [0,1,3] over
[2,4,5] of a 2x3 canvas).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Ar3DRAHub.ultratex_backend import GUIDED_POSES, LENS, POSES, TARGET_SIZE  # noqa: E402


def parse_args(argv: list[str]) -> tuple[str, str, int, bool]:
    """Arguments after Blender's ``--`` separator: mesh, output dir, resolution, guided.

    ``guided`` selects the 45-degree offset view set (the 12-view guided pass)
    instead of the six canonical views. Kept free of ``bpy`` so it is testable.
    """
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    if len(argv) < 2:
        raise SystemExit(
            "usage: blender --background --python-exit-code 1 --python "
            "scripts/blender_render_texverse_views.py -- MESH.glb OUT_DIR [resolution] [guided]"
        )
    mesh, output = argv[0], argv[1]
    resolution = int(argv[2]) if len(argv) > 2 else 1024
    if resolution <= 0:
        raise SystemExit(f"resolution must be positive, got {resolution}")
    guided = len(argv) > 3 and argv[3] not in ("0", "", "false", "no")
    return mesh, output, resolution, guided


def pose_to_blender(pose: tuple[tuple[float, float, float, float], ...]) -> tuple:
    """A TexVerse pose as a Blender 4x4 world matrix (row-major).

    The pose is already a camera-to-world transform in the exact layout Blender
    stores: row i holds the i-th world component of the camera's axes (columns
    are right, up, backward) and the i-th component of the position. The only
    change is the affine fourth row.
    """
    x, y, z = pose
    return (x, y, z, (0.0, 0.0, 0.0, 1.0))


def main() -> int:
    import bpy  # noqa: PLC0415  -- Blender-only, kept out of the testable part

    mesh_path, output_dir, resolution, guided = parse_args(sys.argv)
    output = Path(output_dir)
    target = output / "bump_normal_world"
    target.mkdir(parents=True, exist_ok=True)
    poses = GUIDED_POSES if guided else POSES

    # --- scene, object, normalisation -------------------------------------
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "WEBP"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.quality = 90
    cycles = scene.cycles
    cycles.samples = 8
    try:
        cycles.device = "GPU"
    except TypeError:
        pass
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.import_scene.gltf(filepath=str(Path(mesh_path).resolve()))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise SystemExit(f"no meshes imported from {mesh_path}")
    # One object, so the normal material and the normalisation apply once.
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.ops.object.join()
    obj = bpy.context.object

    # Normalise: centre at the origin, largest dimension to TARGET_SIZE.
    extent = max(dimension for dimension in obj.dimensions)
    if extent <= 0.0:
        raise SystemExit("the imported mesh has zero extent")
    scale = TARGET_SIZE / extent
    obj.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0.0, 0.0, 0.0)

    # --- world-normal material ---------------------------------------------
    material = bpy.data.materials.new(name="texverse_world_normal")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    geometry = nodes.new("ShaderNodeNewGeometry")
    mix = nodes.new("ShaderNodeVectorMath")
    mix.operation = "MULTIPLY_ADD"
    mix.inputs[1].default_value = (0.5, 0.5, 0.5)
    mix.inputs[2].default_value = (0.5, 0.5, 0.5)
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(geometry.outputs["Normal"], mix.inputs[0])
    links.new(mix.outputs["Vector"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(material)

    # --- camera and the six renders ----------------------------------------
    camera = bpy.data.cameras.new("texverse_camera")
    camera.type = "PERSP"
    camera.lens = LENS
    camera.sensor_width = 36.0
    camera_obj = bpy.data.objects.new("texverse_camera", camera)
    bpy.context.scene.collection.objects.link(camera_obj)
    scene.camera = camera_obj
    import mathutils  # noqa: PLC0415

    for index, pose in enumerate(poses):
        camera_obj.matrix_world = mathutils.Matrix(pose_to_blender(pose))
        scene.render.filepath = str(target / f"{index:03d}.webp")
        bpy.ops.render.render(write_still=True)
        print(f"view {index:03d} rendered", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
