"""The UltraTex view baker: canvas splitting, the canonical view cameras, and the
normalisation that keeps the bake in the same frame as the render. Pure parts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ultratex_bake_views.py"


def _load():
    spec = importlib.util.spec_from_file_location("ultratex_bake_views", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bake = _load()


def test_split_canvas_places_the_tiles_like_load_and_concat():
    # Each tile carries its dataset index, so the placement is readable.
    canvas = np.zeros((20, 30, 3), dtype=np.uint8)
    canvas[:10, :10] = 0        # 000 -> top-left
    canvas[:10, 10:20] = 10     # 001 -> top-middle
    canvas[:10, 20:30] = 30     # 003 -> top-right
    canvas[10:20, :10] = 20     # 002 -> bottom-left
    canvas[10:20, 10:20] = 40   # 004 -> bottom-middle
    canvas[10:20, 20:30] = 50   # 005 -> bottom-right
    tiles = bake.split_canvas(canvas)
    assert [int(tile[0, 0, 0]) for tile in tiles] == [0, 10, 20, 30, 40, 50]


def test_normalize_centres_and_scales_to_the_target_size():
    points = np.array([[2.0, -1.0, 0.0], [4.0, 1.0, 0.0], [3.0, 0.0, 2.0]])
    normalised, (centre, scale) = bake.normalize(points)
    extent = np.ptp(points, axis=0).max()          # 2.0
    assert centre == pytest.approx((3.0, 0.0, 1.0))
    assert scale == pytest.approx(bake.TARGET_SIZE / extent)
    assert np.ptp(normalised, axis=0).max() == pytest.approx(bake.TARGET_SIZE)
    centre_of_normalised = (normalised.min(axis=0) + normalised.max(axis=0)) / 2
    assert centre_of_normalised == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_normalize_refuses_a_flat_mesh():
    with pytest.raises(ValueError, match="zero extent"):
        bake.normalize(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))


def test_view_fov_matches_the_dataset_focal():
    # fx 6400 at 4096 px scales with the resolution; the horizontal half-angle
    # is atan((W/2) / fx).
    fov = bake.view_fov(1024)
    expected = 2.0 * np.arctan(512.0 / 1600.0)
    assert fov == pytest.approx(expected)


def test_build_views_uses_the_canonical_poses_in_tile_order():
    views = bake.build_views([np.zeros((8, 8, 4), dtype=np.uint8)] * 6, 1024)
    assert len(views) == 6
    for index, view in enumerate(views):
        assert view.cam_to_world.shape == (4, 4)
        assert view.cam_to_world[0, 3] == pytest.approx(bake.POSES[index][0][3])
        assert view.cam_to_world[1, 3] == pytest.approx(bake.POSES[index][1][3])
        assert view.cam_to_world[2, 3] == pytest.approx(bake.POSES[index][2][3])
        # The camera looks down its own -Z: forward is -column 2, toward the
        # origin from 1.67 units away.
        forward = -view.cam_to_world[:3, 2]
        position = view.cam_to_world[:3, 3]
        assert np.dot(forward, -position) > 0.99 * np.linalg.norm(position)


def test_bake_settings_do_not_align_or_recolour():
    assert bake.BAKE_SETTINGS.align is False
    assert bake.BAKE_SETTINGS.match_colour is False
    assert bake.BAKE_SETTINGS.local_colour is False


def test_pack_metallic_roughness_makes_an_rgba_map_from_the_rgb_paint():
    # The baked texture is RGB; the glTF map needs RGBA. This regression is
    # for the run that died writing channel 3 into a 3-channel array.
    painted = np.zeros((4, 5, 3), dtype=np.uint8)
    painted[..., 0] = 77        # roughness (dataset red)
    painted[..., 1] = 33        # ignored (dataset green duplicates red)
    painted[..., 2] = 199       # metallic (dataset blue)
    packed = bake.pack_metallic_roughness(painted)
    assert packed.shape == (4, 5, 4)
    assert packed.dtype == np.uint8
    assert (packed[..., 1] == 77).all()     # green = roughness
    assert (packed[..., 2] == 199).all()    # blue = metallic
    assert (packed[..., 3] == 255).all()    # alpha opaque
    assert (packed[..., 0] == 0).all()      # red unused


def _canvas(tmp_path: Path, name: str) -> Path:
    from PIL import Image
    path = tmp_path / name
    Image.new("RGBA", (30, 20), (0, 0, 0, 0)).save(path)
    return path


def test_build_views_pairs_guided_tiles_with_the_guided_poses():
    from Ar3DRAHub import ultratex_backend
    tiles = [np.zeros((8, 8, 4), dtype=np.uint8)] * 6
    views = bake.build_views(tiles, 1024, poses=ultratex_backend.GUIDED_POSES)
    assert len(views) == 6
    for index, view in enumerate(views):
        pose = ultratex_backend.GUIDED_POSES[index]
        assert view.cam_to_world[0, 3] == pytest.approx(pose[0][3])
        assert view.cam_to_world[1, 3] == pytest.approx(pose[1][3])
        assert view.cam_to_world[2, 3] == pytest.approx(pose[2][3])


def test_albedo_views_merge_the_canonical_and_guided_six_into_twelve(tmp_path):
    from Ar3DRAHub import ultratex_backend
    canonical = _canvas(tmp_path, "canonical.png")
    guided = _canvas(tmp_path, "guided.png")
    views = bake.albedo_views(canonical, 1024)
    assert len(views) == 6
    views = bake.albedo_views(canonical, 1024, guided)
    assert len(views) == 12
    # the first six keep the canonical poses, the last six the guided ones
    for index, view in enumerate(views[6:]):
        pose = ultratex_backend.GUIDED_POSES[index]
        assert view.cam_to_world[0, 3] == pytest.approx(pose[0][3])
        assert view.cam_to_world[1, 3] == pytest.approx(pose[1][3])
        assert view.cam_to_world[2, 3] == pytest.approx(pose[2][3])
