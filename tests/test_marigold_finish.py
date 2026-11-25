"""Tests for scripts/marigold_finish.py: the de-light blend and its pre-flight checks.

The projection reuses Ar3DRAHub/photo_paint.py (Pixel Match's own camera), which is tested
there; what is new here is the de-light blend (`delight`), the camera-to-object normal
transform, the UV raster that finds each texel's surface, and the CLI's refusal to run on
missing files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.marigold_finish import (
    camera_to_object_normals,
    delight,
    texel_faces,
    view_named,
)


def test_full_amount_replaces_with_the_projected_albedo():
    base = np.full((2, 2, 3), 100.0, dtype=np.float32)
    projected = np.full((2, 2, 3), 200.0, dtype=np.float32)
    seen = np.ones((2, 2), dtype=bool)
    assert np.allclose(delight(base, projected, seen, 1.0), 200.0)


def test_de_light_keeps_the_original_hue_where_the_albedo_is_grey():
    """The fix for grey patches: a grey albedo changes brightness, never the colour."""
    base = np.array([[[200.0, 60.0, 60.0]]], dtype=np.float32)   # a red
    projected = np.array([[[100.0, 100.0, 100.0]]], dtype=np.float32)  # grey albedo
    out = delight(base, np.broadcast_to(projected, base.shape), np.ones((1, 1), bool), 1.0)[0, 0]
    assert out[0] > out[1] + 40 and out[0] > out[2] + 40   # still red, not grey


def test_zero_amount_leaves_the_base_untouched():
    base = np.full((2, 2, 3), 100.0, dtype=np.float32)
    projected = np.full((2, 2, 3), 200.0, dtype=np.float32)
    seen = np.ones((2, 2), dtype=bool)
    assert np.allclose(delight(base, projected, seen, 0.0), 100.0)


def test_unseen_texels_keep_the_base_even_at_full_amount():
    base = np.full((2, 2, 3), 100.0, dtype=np.float32)
    projected = np.full((2, 2, 3), 200.0, dtype=np.float32)
    seen = np.zeros((2, 2), dtype=bool)
    assert np.allclose(delight(base, projected, seen, 1.0), 100.0)


def test_half_amount_is_a_midpoint_blend():
    base = np.full((1, 1, 3), 100.0, dtype=np.float32)
    projected = np.full((1, 1, 3), 200.0, dtype=np.float32)
    seen = np.ones((1, 1), dtype=bool)
    assert np.allclose(delight(base, projected, seen, 0.5), 150.0)


def test_a_missing_mesh_is_refused(monkeypatch, capsys):
    from scripts import marigold_finish as fin

    with pytest.raises(SystemExit, match="not found"):
        fin.main(["no-such.glb", "--views", "nowhere"])


def test_a_views_dir_without_transforms_is_refused(tmp_path, monkeypatch):
    from scripts import marigold_finish as fin

    mesh = tmp_path / "m.glb"
    mesh.write_bytes(b"")
    views = tmp_path / "views"
    views.mkdir()
    with pytest.raises(SystemExit, match="no transforms.json"):
        fin.main([str(mesh), "--views", str(views)])


def test_texel_faces_covers_a_full_atlas_square():
    """Two triangles covering the whole atlas together cover every texel of it."""
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    covered, face_of, bary = texel_faces(uvs, faces, 8, 8)
    assert covered.all()
    assert set(np.unique(face_of)) == {0, 1}
    # Barycentric weights are non-negative and sum to one at every texel.
    assert np.allclose(bary.sum(axis=-1), 1.0)


def test_a_front_and_back_view_surface_normal_points_outward():
    """A normal toward the viewer (+z camera) lands on the outward side of the mesh.

    The front camera sees the +Z surface, the back camera (a half turn about the vertical)
    the -Z surface. If the rotation is applied untransposed, the back normal points inward
    -- the green-channel inversion bug.
    """
    from Ar3DRAHub import photo_paint as pp

    def view(rotation):
        c2w = np.eye(4)
        c2w[:3, :3] = rotation
        return pp.View(np.zeros((8, 8, 4), np.uint8), c2w, 0.35, "v")

    front = view([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    back = view([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    toward_viewer = np.array([[0.0, 0.0, 1.0]])
    assert np.allclose(camera_to_object_normals(toward_viewer, front), [[0, 0, 1]])
    assert np.allclose(camera_to_object_normals(toward_viewer, back), [[0, 0, -1]])


def test_view_named_finds_the_back_by_name_not_position():
    """A four-view build puts the right view at index 1; the back is still `view_180.png`."""
    from Ar3DRAHub import photo_paint as pp

    def named(name):
        return pp.View(np.zeros((8, 8, 4), np.uint8), np.eye(4), 0.35, name)

    views = [named("view_000.png"), named("view_090.png"),
             named("view_180.png"), named("view_270.png")]
    assert view_named(views, "view_180.png") is views[2]
    assert view_named(views, "view_090.png") is views[1]
    assert view_named(views, "view_999.png") is None
