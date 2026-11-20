"""Transfer the old mesh's texture onto Hi3D-refined geometry by ray-cast baking.

Run (headless): blender --background --python-exit-code 1 --python scripts/blender_hi3d_transfer.py -- SOURCE.glb REFINED.glb OUT.glb [size] [metallic] [roughness] [ior]

**Why.** Hi3D refine produces geometry only: no textures, no UVs, and the mesh is
re-centred and normalised to unit radius. To let the rest of the Finish pipeline
(Detail, Pixel Match, Marigold) run on the refined geometry, the old mesh's base
colour must be carried across. This script:

1. aligns the refined mesh back into the source's frame, by bounding boxes -- a
   uniform scale and a translation, since Hi3D keeps the orientation;
2. unwraps the refined mesh (the same island-unwrap the retopo bake uses);
3. bakes the source's base colour onto it, selected-to-active ray casting, and
   ships the refined mesh with the baked albedo.

The alignment is approximate (bounding boxes, not surface matching), so Pixel Match
afterwards can drift slightly on front-facing details. Geometry is not modified:
the refined mesh keeps every triangle.

Run with:
    blender --background --python scripts/blender_hi3d_transfer.py -- SOURCE.glb REFINED.glb OUT.glb
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


def parse_args(argv: list[str]) -> tuple[str, str, str, int, float, float, float]:
    """Arguments after Blender's ``--`` separator. Kept free of ``bpy`` so it is testable."""
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    if len(argv) < 3:
        raise SystemExit(
            "usage: blender --background --python scripts/blender_hi3d_transfer.py "
            "-- SOURCE.glb REFINED.glb OUT.glb [size] [metallic] [roughness] [ior]"
        )
    size = int(argv[3]) if len(argv) > 3 else 2048
    metallic = float(argv[4]) if len(argv) > 4 else 0.25
    roughness = float(argv[5]) if len(argv) > 5 else 0.65
    ior = float(argv[6]) if len(argv) > 6 else 1.45
    if size not in (256, 512, 1024, 2048, 4096):
        raise SystemExit(f"atlas size must be 256, 512, 1024, 2048 or 4096, got {size}")
    for name, value in (("metallic", metallic), ("roughness", roughness)):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be within 0..1, got {value}")
    if not 1.0 <= ior <= 3.0:
        raise SystemExit(f"ior must be within 1.0..3.0, got {ior}")
    return (argv[0], argv[1], argv[2], size, metallic, roughness, ior)


def align_transform(
    source_min: tuple[float, float, float],
    source_max: tuple[float, float, float],
    refined_min: tuple[float, float, float],
    refined_max: tuple[float, float, float],
) -> tuple[float, tuple[float, float, float]]:
    """(scale, offset) mapping the refined mesh into the source's frame, by bounding boxes.

    Hi3D normalises the refined mesh to unit maximum radius around the origin, keeping
    the source's orientation. So the map is a uniform scale -- source extent over
    refined extent -- plus the translation that lands the refined centre on the
    source centre. `vertex' = vertex * scale + offset`.
    """
    def centre(lo: tuple, hi: tuple) -> tuple[float, float, float]:
        return tuple((a + b) / 2.0 for a, b in zip(lo, hi))

    def extent(lo: tuple, hi: tuple) -> float:
        return max(b - a for a, b in zip(lo, hi))

    source_span = extent(source_min, source_max)
    refined_span = extent(refined_min, refined_max)
    if refined_span <= 0.0 or source_span <= 0.0:
        raise ValueError("a mesh has zero extent; cannot align by bounding boxes")
    scale = source_span / refined_span
    sc, rc = centre(source_min, source_max), centre(refined_min, refined_max)
    offset = tuple(s - scale * r for s, r in zip(sc, rc))
    return scale, offset


def align_matrix(
    scale: float, offset: tuple[float, float, float]
) -> tuple[tuple[float, ...], ...]:
    """4x4 matrix (row-major) mapping a point ``p`` to ``p * scale + offset``.

    ``align_transform`` returns a uniform scale and a translation; this assembles
    them into the homogeneous matrix that applies the scale first, then the
    translation (``Translation(offset) @ Scale(scale)``). Pure and mathutils-free
    so the order is testable: composing them the other way would scale the offset
    too, silently misplacing the mesh.
    """
    ox, oy, oz = offset
    return (
        (scale, 0.0, 0.0, ox),
        (0.0, scale, 0.0, oy),
        (0.0, 0.0, scale, oz),
        (0.0, 0.0, 0.0, 1.0),
    )


def main() -> int:
    import bpy

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import blender_retopo_bake as retopo

    (source, refined, destination, size, metallic, roughness, ior) = parse_args(list(sys.argv))

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.import_scene.gltf(filepath=source)
    sources = [o for o in bpy.data.objects if o.type == "MESH"]
    if not sources:
        raise SystemExit(f"no mesh found in {source}")
    original = sources[0]
    image = retopo.base_colour_image(
        original.data.materials[0] if original.data.materials else None)
    if image is None:
        raise SystemExit("could not find the base colour texture to transfer")

    bpy.ops.import_scene.gltf(filepath=refined)
    refined_meshes = [o for o in bpy.data.objects
                      if o.type == "MESH" and o is not original]
    if not refined_meshes:
        raise SystemExit(f"no mesh found in {refined}")
    target = refined_meshes[0]
    target.name = "REFINED"

    def bounds(obj) -> tuple[tuple, tuple]:
        local = [obj.matrix_world @ v.co for v in obj.data.vertices]
        xs = [v.x for v in local]
        ys = [v.y for v in local]
        zs = [v.z for v in local]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    scale, offset = align_transform(bounds(original)[0], bounds(original)[1],
                                    bounds(target)[0], bounds(target)[1])
    import mathutils

    matrix = mathutils.Matrix(align_matrix(scale, offset)) @ target.matrix_world
    target.matrix_world = matrix
    print(f"TRANSFER:: aligned refined -> source frame (scale {scale:.4f})")

    # Bake the alignment into the vertices and reset the transform to identity:
    # Pixel Match (photo_paint.read_glb) only accepts GLBs whose node transforms
    # are all identity, so the bbox alignment must not ship as a node transform.
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    target.data.transform(matrix)
    target.matrix_world = mathutils.Matrix.Identity(4)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    # Weld first: glTF arrives split along every UV and normal seam.
    bpy.ops.mesh.remove_doubles(threshold=1e-6)
    left = retopo.unwrap_finish(bpy, target.data, math.radians(89.0))
    print(f"TRANSFER:: uv faces still overlapping={left:,}")
    bpy.ops.object.mode_set(mode="OBJECT")

    # --- emission material on the SOURCE, so the bake reads its albedo -----------------
    emit = bpy.data.materials.new("EMIT_SRC")
    emit.use_nodes = True
    nodes, links = emit.node_tree.nodes, emit.node_tree.links
    for node in list(nodes):
        if node.type != "OUTPUT_MATERIAL":
            nodes.remove(node)
    output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.interpolation = "Closest"
    emission = nodes.new("ShaderNodeEmission")
    links.new(tex.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    original.data.materials.clear()
    original.data.materials.append(emit)

    # --- bake target on the REFINED -----------------------------------------------------
    target_image = bpy.data.images.new("transferred_albedo", width=size, height=size,
                                       alpha=False)
    dst = bpy.data.materials.new("REFINED_MAT")
    dst.use_nodes = True
    dnodes, dlinks = dst.node_tree.nodes, dst.node_tree.links
    dst_tex = dnodes.new("ShaderNodeTexImage")
    dst_tex.image = target_image
    target.data.materials.append(dst)
    target.active_material_index = 0
    for node in dnodes:
        node.select = False
    dst_tex.select = True
    dnodes.active = dst_tex

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    try:
        scene.cycles.device = "CPU"
        scene.cycles.samples = 1
    except AttributeError:
        pass
    bake = scene.render.bake
    bake.target = "IMAGE_TEXTURES"
    bake.use_selected_to_active = True
    bake.margin = 16
    bake.use_clear = True
    bake.cage_extrusion = retopo.ray_distance(target.dimensions)
    bake.max_ray_distance = retopo.ray_distance(target.dimensions) * 2.0

    bpy.ops.object.select_all(action="DESELECT")
    original.select_set(True)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    print("TRANSFER:: baking source -> refined ...")
    bpy.ops.object.bake(type="EMIT")

    # --- ship the refined mesh with the baked albedo ------------------------------------
    principled = dnodes.new("ShaderNodeBsdfPrincipled")
    doutput = next(n for n in dnodes if n.type == "OUTPUT_MATERIAL")
    dlinks.new(dst_tex.outputs["Color"], principled.inputs["Base Color"])
    dlinks.new(principled.outputs["BSDF"], doutput.inputs["Surface"])
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    if "IOR" in principled.inputs:
        principled.inputs["IOR"].default_value = ior

    bpy.data.objects.remove(original, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.ops.export_scene.gltf(filepath=destination, export_format="GLB", use_selection=True)
    print(f"TRANSFER:: wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
