#!/usr/bin/env python3
"""De-light a generated GLB and bake front+back normal maps with Marigold V2.

A single-image 3D generator paints from a *lit* photo, so the photo's own shadows and
highlights bake into the base-colour texture, and the asset then gets lit a second time
in the engine. Marigold V2 predicts the same surface *without* light (albedo) and its
surface orientation (camera-space normals). This projects both back onto the mesh and
replaces the base colour where the photo painted, baking the normals into a tangent-space
`normalTexture`.

The projection is the *same* one Pixel Match uses (Ar3DRAHub/photo_paint.py): each texel's
3D position is sent through the saved camera from the run's `transforms.json`, so the
albedo and normal maps -- which Marigold predicts from that very view, at that very size --
land on exactly the pixels Pixel Match painted from. The sides the two views cannot see
keep the generator's own paint and the worker's own normal map.

The normal map is baked against the mesh's *own* `NORMAL` and `TANGENT` attributes -- the
exact smooth frame the renderer interpolates -- rather than a frame re-derived from the
UVs, and the GLB is rewritten at the byte level, so the geometry (positions, normals,
tangents) survives untouched. The previous round-trip through trimesh dropped those two
attributes, which is what left the model flat-shaded in facets.

    python scripts/marigold_finish.py model.glb --views views_dir --output out.glb

`--views` is a Pixal3D run's `.svviews` directory (a `transforms.json` plus `view_000.png`
and, for a front+back build, `view_180.png`). Marigold runs on each view through its own
venv (scripts/marigold_v2_predict.py), or pass `--albedo` / `--normals` / `--back-albedo` /
`--back-normals` to already-generated maps.

Licence: the maps come from huawei-bayerlab/marigold-v2-0 (Apache-2.0) over the
Qwen-Image-Edit-2509 backbone (Apache-2.0); both are recorded in the derived sidecar.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Ar3DRAHub import host, photo_paint
from Ar3DRAHub.provenance import LICENSES, write_derived_sidecar
from scripts.attach_normal_map import soften
from scripts.project_markings import dilate_into_gutter


def delight(base: np.ndarray, projected: np.ndarray, seen: np.ndarray,
            amount: float) -> np.ndarray:
    """De-light the texture: replace the baked photo colour with Marigold's albedo.

    Replacing outright bakes Marigold's imperfect hue predictions in as grey or washed
    patches (eyes, mouth, metal). So the full albedo colour is used only where the albedo
    is confidently coloured (saturated); where it is grey or washed out, the generator's
    own hue is kept and only the luminance is transferred. ``amount`` 0 leaves the texture
    untouched, 1 is the full de-light. ``seen`` is a 0..1 trust weight per texel (the
    blend Pixel Match would apply), so a surface two cameras both see is de-lit once.
    """
    luma = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    base_lum = (base @ luma)[..., None]
    alb_lum = (projected @ luma)[..., None]
    ratio = np.clip(alb_lum / np.maximum(base_lum, 16.0), 0.0, 4.0)
    hue_kept = base * ratio
    alb_max = projected.max(-1)
    alb_sat = (alb_max - projected.min(-1)) / np.maximum(alb_max, 1.0)
    confident = alb_sat > 0.15
    lit = np.where(confident[..., None], projected, hue_kept)
    weight = np.clip(float(amount), 0.0, 1.0) * seen[..., None].astype(np.float32)
    return np.clip(base * (1.0 - weight) + lit * weight, 0.0, 255.0).astype(np.uint8)


def camera_to_object_normals(n_cam: np.ndarray, view: photo_paint.View) -> np.ndarray:
    """Camera-space normals (x right, y up, z toward the viewer) to object space.

    The camera's rotation (`cam_to_world`) carries a camera-space direction into view
    space, then `from_view_space` returns it to the GLB's axes. The rotation is transposed
    because `project` works in row-vector form (`cam = points @ w2c[:3,:3].T`).
    """
    return photo_paint.from_view_space(n_cam @ view.cam_to_world[:3, :3].T)


def texel_faces(uvs: np.ndarray, faces: np.ndarray, width: int, height: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Which UV triangle covers each atlas texel, and the barycentric weights there.

    Returns (covered, face_of, bary) with covered/face_of of shape (H, W) and bary
    (H, W, 3). Rasterised exactly the way `photo_paint.paint_texture` does (glTF UVs scaled
    by the atlas size), so the per-texel surface points match Pixel Match's.
    """
    texel_xy = np.stack([uvs[:, 0] * width, uvs[:, 1] * height], axis=1)
    face_of, bary = photo_paint.rasterize(
        texel_xy[faces], np.zeros((len(faces), 3)), (height, width))
    return face_of >= 0, face_of, bary


def texel_surface(uvs: np.ndarray, faces: np.ndarray, texture_size: tuple[int, int]
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The UV raster shared by the de-light and the normal bake.

    Returns (covered, flat, texel_face, texel_bary): `covered` is the (H, W) atlas mask,
    `flat` its flattened index, and the last two index the covered texels' faces and
    barycentric weights.
    """
    covered, face_of, bary = texel_faces(uvs, faces, texture_size[0], texture_size[1])
    flat = covered.ravel()
    return covered, flat, face_of.ravel()[flat], bary.reshape(-1, 3)[flat]


def view_projection_texels(positions: np.ndarray, faces: np.ndarray, view: photo_paint.View,
                           settings: photo_paint.Settings, outward: float,
                           mesh_scale: float, texture_size: tuple[int, int],
                           covered: np.ndarray, flat: np.ndarray, texel_face: np.ndarray,
                           texel_bary: np.ndarray, dilate: int
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-atlas-texel (weight, x, y) for one view, with the gutter dilated.

    `weight` is 0..1 trust (0 where the view cannot see the surface); `x`, `y` are float
    pixel coordinates in the view's image -- and therefore in any Marigold map generated
    from it, which shares its size. Returns full-atlas (H, W) arrays.
    """
    width, height = texture_size
    corners = faces[texel_face]

    view_pos = photo_paint.to_view_space(positions, photo_paint.PIXAL3D_FRAME, mesh_scale)
    normals = photo_paint.vertex_normals(view_pos, faces)
    points = (texel_bary[:, :, None] * view_pos[corners]).sum(axis=1)
    texel_n = (texel_bary[:, :, None] * normals[corners]).sum(axis=1)
    texel_n /= np.maximum(np.linalg.norm(texel_n, axis=1, keepdims=True), 1e-12)

    weight, x, y = photo_paint.view_projection(points, texel_n, view, view_pos, faces,
                                               settings, outward)

    packed = np.zeros((height, width, 3), dtype=np.float32)
    packed.reshape(-1, 3)[flat] = np.stack([weight, x, y], axis=1)
    packed, _ = dilate_into_gutter(packed, covered, dilate)
    return packed[..., 0], packed[..., 1], packed[..., 2]


def sample_at(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear-sample `image` at full-atlas pixel coordinates (x, y) of shape (H, W)."""
    flat = photo_paint.sample_rgb(image, x.ravel(), y.ravel())
    return flat.reshape(x.shape[0], x.shape[1], 3)


def de_light(base: np.ndarray, views_alb: list[tuple[np.ndarray, np.ndarray]],
             amount: float) -> np.ndarray:
    """Blend Marigold's albedo over the base colour where the views see the surface.

    `views_alb` is a list of (sampled albedo, weight) full-atlas arrays. Weights from the
    front and back views are summed (clipped to 1) and the albedo averaged by them, so a
    surface two cameras both see is de-lit once, not twice.
    """
    base = base[..., :3].astype(np.float32)
    total = np.zeros(base.shape[:2], dtype=np.float32)
    acc = np.zeros_like(base)
    for alb, weight in views_alb:
        total += weight
        acc += alb * weight[..., None]
    projected = acc / np.maximum(total, 1e-6)[..., None]
    return delight(base, projected, np.clip(total, 0.0, 1.0), amount)


def bake_normal_map(positions: np.ndarray, faces: np.ndarray, normals: np.ndarray,
                    tangents: np.ndarray, view: photo_paint.View, normals_img: Image.Image,
                    settings: photo_paint.Settings, outward: float, mesh_scale: float,
                    texture_size: tuple[int, int], flat: np.ndarray, texel_face: np.ndarray,
                    texel_bary: np.ndarray, covered: np.ndarray, dilate: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Sample camera-space normals at each vertex's pixels and bake a tangent-space map.

    The object-space normal is sampled per *vertex* through the real camera, then carried
    to tangent space with the mesh's *own* `NORMAL` and `TANGENT` attributes -- the exact
    smooth frame the renderer interpolates -- so the low-poly triangulation cannot bake in
    as facets. The frame is interpolated across each UV triangle with the same barycentric
    weights the rasteriser uses.

    Returns (n_xyz, weight): the full-atlas tangent-space normals in [-1, 1] (flat where
    this view carries no weight) and the per-texel 0..1 trust weight. The weight is kept
    continuous, so the caller can blend this view's map over the worker's without the hard
    seam a boolean mask would leave.
    """
    height, width = texture_size[1], texture_size[0]

    view_pos = photo_paint.to_view_space(positions, photo_paint.PIXAL3D_FRAME, mesh_scale)
    vn = photo_paint.vertex_normals(view_pos, faces)
    weight, x, y = photo_paint.view_projection(view_pos, vn, view, view_pos, faces,
                                               settings, outward)

    raw = np.asarray(normals_img.convert("RGB"), dtype=np.float64)
    n_cam = photo_paint.sample_rgb(raw, x, y) / 255.0 * 2.0 - 1.0   # (V, 3)
    n_obj = camera_to_object_normals(n_cam, view)

    # The mesh's own frame: smooth vertex normals + Mikktspace tangent (w = handedness).
    tangent = tangents[:, :3]
    bitangent = np.cross(normals, tangent) * tangents[:, 3, None]
    n_tan = np.stack([(n_obj * tangent).sum(-1), (n_obj * bitangent).sum(-1),
                      (n_obj * normals).sum(-1)], axis=-1)
    n_tan /= np.maximum(np.linalg.norm(n_tan, axis=1, keepdims=True), 1e-12)

    n_tan[weight <= 0.0] = 0.0
    weight_v = weight.astype(np.float32)

    corners = faces[texel_face]
    n_tan_texel = (texel_bary[:, :, None] * n_tan[corners]).sum(axis=1)
    weight_texel = (texel_bary * weight_v[corners]).sum(axis=1)

    atlas = np.zeros((height, width, 3), dtype=np.float32)
    atlas.reshape(-1, 3)[flat] = n_tan_texel.astype(np.float32)
    weight_full = np.zeros((height, width), dtype=np.float32)
    weight_full.reshape(-1)[flat] = weight_texel
    packed = np.concatenate([atlas, weight_full[..., None]], axis=-1)
    packed, _ = dilate_into_gutter(packed, covered, dilate)

    norm = np.linalg.norm(packed[..., :3], axis=-1, keepdims=True)
    n_xyz = np.where(norm > 1e-6, packed[..., :3] / np.maximum(norm, 1e-6), 0.0)
    weight_map = np.clip(packed[..., 3], 0.0, 1.0)
    flat = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    n_xyz = np.where(weight_map[..., None] > 0.0, n_xyz, flat)
    return n_xyz, weight_map


def marigold_python() -> Path:
    return host.venv_python(REPO / "vendor" / "marigold-v2")


def generate_maps(view: Path, out_dir: Path, need_normals: bool) -> dict[str, Path]:
    """Run Marigold V2 on `view` through its own venv. Returns {modality: png path}."""
    if not marigold_python().is_file():
        raise SystemExit("Marigold V2 is not installed. Run scripts/bootstrap_marigold_v2.py, "
                         "or pass --albedo / --normals to already-generated maps.")
    out_dir.mkdir(parents=True, exist_ok=True)
    modality = "all" if need_normals else "albedo"
    subprocess.run(
        [str(marigold_python()), str(REPO / "scripts" / "marigold_v2_predict.py"),
         str(view), "--modality", modality, "--out", str(out_dir)],
        check=True,
    )
    maps = {"albedo": out_dir / f"{view.stem}_albedo.png"}
    if need_normals:
        maps["normals"] = out_dir / f"{view.stem}_normals.png"
    return maps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", type=Path, help="generated .glb to finish")
    parser.add_argument("--views", type=Path, required=True,
                        help="the run's saved-camera directory (transforms.json + view_*.png)")
    parser.add_argument("--albedo", type=Path, default=None,
                        help="Marigold V2 albedo PNG for the front view, same size as it")
    parser.add_argument("--normals", type=Path, default=None,
                        help="Marigold V2 normals PNG for the front view")
    parser.add_argument("--back-albedo", type=Path, default=None,
                        help="Marigold V2 albedo PNG for the back view")
    parser.add_argument("--back-normals", type=Path, default=None,
                        help="Marigold V2 normals PNG for the back view")
    parser.add_argument("--no-normals", action="store_true",
                        help="de-light only; do not bake a normal map")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--amount", type=float, default=1.0,
                        help="1 replaces the seen surface with albedo (default); less blends")
    parser.add_argument("--normal-strength", type=float, default=1.0,
                        help="dose of the normal map; 1 is the raw bake, lower blends toward flat")
    parser.add_argument("--dilate", type=int, default=6)
    args = parser.parse_args(argv)

    if not args.mesh.is_file():
        raise SystemExit(f"not found: {args.mesh}")
    if not args.views.is_dir() or not (args.views / "transforms.json").is_file():
        raise SystemExit(f"--views has no transforms.json: {args.views}")
    front_png = args.views / "view_000.png"
    if not front_png.is_file():
        raise SystemExit(f"not found: {front_png}")
    back_png = args.views / "view_180.png"
    back = back_png if back_png.is_file() else None
    output = args.output or args.mesh.with_name(f"{args.mesh.stem}__finish.glb")

    do_normals = not args.no_normals

    def ensure(view: Path, albedo: Path | None, normals: Path | None) -> tuple[Path, Path | None]:
        """Generate any missing map for `view`, returning (albedo, normals or None)."""
        need_albedo = albedo is None
        need_normals = do_normals and normals is None
        if need_albedo or need_normals:
            maps = generate_maps(view, output.parent, need_normals=need_normals)
            albedo = maps["albedo"] if need_albedo else albedo
            normals = maps["normals"] if need_normals else normals
        if not albedo.is_file():
            raise SystemExit(f"not found: {albedo}")
        if do_normals and normals is not None and not normals.is_file():
            raise SystemExit(f"not found: {normals}")
        return albedo, normals if do_normals else None

    front_albedo, front_normals = ensure(front_png, args.albedo, args.normals)
    back_albedo = back_normals = None
    if back is not None:
        back_albedo, back_normals = ensure(back, args.back_albedo, args.back_normals)

    positions, uvs, faces, base_texture, normals, tangents = photo_paint.read_surface(args.mesh)
    texture_size = (base_texture.shape[1], base_texture.shape[0])  # (W, H)

    views, mesh_scale = photo_paint.load_views(args.views)
    front = views[0]
    back_view = views[1] if len(views) >= 2 and back is not None else None
    settings = photo_paint.DEFAULT_SETTINGS

    outward = photo_paint.outward_sign(
        photo_paint.to_view_space(positions, photo_paint.PIXAL3D_FRAME, mesh_scale), faces)

    covered, flat, texel_face, texel_bary = texel_surface(uvs, faces, texture_size)

    def project_view(view: photo_paint.View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return view_projection_texels(positions, faces, view, settings, outward, mesh_scale,
                                      texture_size, covered, flat, texel_face, texel_bary,
                                      args.dilate)

    front_weight, front_x, front_y = project_view(front)
    alb = sample_at(
        np.asarray(Image.open(front_albedo).convert("RGB")), front_x, front_y).astype(np.float32)
    views_alb: list[tuple[np.ndarray, np.ndarray]] = [(alb, front_weight)]

    normal_maps: list[tuple[np.ndarray, np.ndarray]] = []
    if do_normals:
        normal_maps.append(bake_normal_map(
            positions, faces, normals, tangents, front, Image.open(front_normals),
            settings, outward, mesh_scale, texture_size, flat, texel_face, texel_bary,
            covered, args.dilate))

    seen_total = int((front_weight > 0.5).sum())
    if back_view is not None:
        back_weight, back_x, back_y = project_view(back_view)
        back_alb = sample_at(
            np.asarray(Image.open(back_albedo).convert("RGB")), back_x, back_y).astype(np.float32)
        views_alb.append((back_alb, back_weight))
        seen_total += int((back_weight > 0.5).sum())
        if do_normals:
            normal_maps.append(bake_normal_map(
                positions, faces, normals, tangents, back_view, Image.open(back_normals),
                settings, outward, mesh_scale, texture_size, flat, texel_face, texel_bary,
                covered, args.dilate))

    new_base = de_light(base_texture, views_alb, args.amount)

    # Merge the Marigold normal maps over the worker's own, blending by the views' trust
    # so a surface seen only at a glancing angle (the arms' tops and bottoms) fades into
    # the worker's relief instead of switching to it along a hard seam.
    new_normal: Image.Image | None = None
    if do_normals:
        doc, binary = photo_paint._split_glb(args.mesh.read_bytes())
        try:
            existing = photo_paint.read_image(
                doc, binary, photo_paint.normal_texture_image_index(doc))
        except ValueError:
            existing = None

        if len(normal_maps) == 1:
            marigold = normal_maps[0][0]
            total_w = normal_maps[0][1]
        else:
            total_w = np.clip(normal_maps[0][1] + normal_maps[1][1], 0.0, 1.0)
            marigold = (normal_maps[0][0] * normal_maps[0][1][..., None]
                        + normal_maps[1][0] * normal_maps[1][1][..., None])
            marigold = marigold / np.maximum(total_w, 1e-6)[..., None]
            norm = np.linalg.norm(marigold, axis=-1, keepdims=True)
            marigold = np.where(norm > 1e-6, marigold / np.maximum(norm, 1e-6), 0.0)

        if existing is not None:
            existing_img = Image.fromarray(existing).convert("RGB")
            if existing_img.size != (texture_size[0], texture_size[1]):
                existing_img = existing_img.resize(texture_size, Image.LANCZOS)
            worker = np.asarray(existing_img, dtype=np.float32) / 255.0 * 2.0 - 1.0
            blended = marigold * total_w[..., None] + worker * (1.0 - total_w[..., None])
            norm = np.linalg.norm(blended, axis=-1, keepdims=True)
            blended = np.where(norm > 1e-6, blended / np.maximum(norm, 1e-6), 0.0)
        else:
            blended = marigold
        pixels = np.clip((blended * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        combined = Image.fromarray(pixels, mode="RGB")
        new_normal = soften(combined, args.normal_strength)

    # Write at the byte level: only the two textures change, so NORMAL/TANGENT survive.
    data = args.mesh.read_bytes()
    data = photo_paint.replace_base_colour(data, photo_paint.encode_png(new_base))
    if new_normal is not None:
        normal_png = photo_paint.encode_png(np.asarray(new_normal.convert("RGB")))
        try:
            index = photo_paint.normal_texture_image_index(photo_paint._split_glb(data)[0])
            data = photo_paint.replace_image(data, normal_png, index)
        except ValueError:
            data = photo_paint.add_normal_texture(data, normal_png)

    components = [{
        "component": "huawei-bayerlab/marigold-v2-0 (albedo)",
        "purpose": "de-lighting the texture",
        "license": LICENSES["marigold-v2"].license_name,
        "license_url": LICENSES["marigold-v2"].license_url,
        "commercial_status": "commercial-clear",
    }]
    if do_normals:
        components.append({
            "component": "huawei-bayerlab/marigold-v2-0 (normals)",
            "purpose": "surface-normal detail",
            "license": LICENSES["marigold-v2"].license_name,
            "license_url": LICENSES["marigold-v2"].license_url,
            "commercial_status": "commercial-clear",
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    write_derived_sidecar(args.mesh, output, operation="marigold-finish",
                          component_records=tuple(components))
    share = 100.0 * seen_total / max(int(covered.sum()), 1)
    print(f"de-lighted {seen_total:,} texels ({share:.0f}% of covered)"
          f"{'; normal map baked' if do_normals else ''} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
