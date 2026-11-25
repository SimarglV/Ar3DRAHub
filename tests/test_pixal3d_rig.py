"""The four-view Pixal3D turntable rig: the cameras must form one consistent pose.

Pixal3D conditions through camera-aware projection, so a wrong pose lands features in the
wrong voxels and the model returns a generic mesh. These tests pin the cheap properties
that decide that: four rigid cameras, all looking at the origin from the same distance, a
quarter turn apart, frame 0 the front. They run the real ``turntable_rig`` and the real
``photo_paint.project``, never a re-derived copy of either.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from Ar3DRAHub import photo_paint as pp
from scripts import pixal3d_generate as pg


def _rig():
    rig = pg.turntable_rig()
    matrices = [np.asarray(frame["transform_matrix"], dtype=float)
                for frame in rig["frames"]]
    return rig, matrices


def _view(frame):
    return pp.View(np.zeros((768, 768, 4), dtype=np.uint8),
                   np.asarray(frame["transform_matrix"], dtype=float),
                   frame["camera_angle_x"], frame["file_path"])


def test_four_frames_in_turntable_order():
    rig, _ = _rig()
    names = [frame["file_path"] for frame in rig["frames"]]
    assert names == ["view_000.png", "view_090.png", "view_180.png", "view_270.png"]
    assert rig["mesh_scale"] == 1
    assert rig["camera_angle_x"] == pg.DEFAULT_FOV
    assert all(frame["camera_angle_x"] == pg.DEFAULT_FOV for frame in rig["frames"])


def test_every_camera_is_a_rigid_proper_transform():
    _, matrices = _rig()
    for matrix in matrices:
        assert matrix.shape == (4, 4)
        assert np.allclose(matrix[3], [0, 0, 0, 1])
        rotation = matrix[:3, :3]
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9)


def test_cameras_look_at_the_origin_from_the_same_distance():
    rig, _ = _rig()
    origin = np.array([[0.0, 0.0, 0.0]])
    for frame in rig["frames"]:
        x, y, depth = pp.project(origin, _view(frame))
        # The optical axis passes through the origin: it lands on the principal point.
        assert np.allclose(x, 768 / 2.0, atol=1e-6)
        assert np.allclose(y, 768 / 2.0, atol=1e-6)
        # Same orbit radius for all four, not a camera that drifts in or out.
        assert np.isclose(depth, pg.CANONICAL_DISTANCE, atol=1e-6)


def test_views_are_a_quarter_turn_apart():
    _, matrices = _rig()
    # The rotation taking camera i to camera i+1 (wrapping) is the same quarter turn each
    # time, and four of them return to where they started.
    steps = [matrices[(i + 1) % 4][:3, :3] @ matrices[i][:3, :3].T for i in range(4)]
    for step in steps:
        assert np.allclose(step, steps[0], atol=1e-9)
    assert np.allclose(steps[0] @ steps[0] @ steps[0] @ steps[0], np.eye(3), atol=1e-9)
    # A true quarter turn: not identity, and not a half turn in disguise.
    assert not np.allclose(steps[0], np.eye(3), atol=1e-9)
    assert not np.allclose(steps[0] @ steps[0], np.eye(3), atol=1e-9)


def _cuts():
    return [(name, Image.new("RGBA", (40, 40), (10, 20, 30, 255)))
            for name in ("view_000.png", "view_090.png", "view_180.png", "view_270.png")]


def test_write_views_stages_every_view_and_the_turntable_rig(tmp_path):
    cuts = _cuts()
    model_dir, paint_dir = tmp_path / "mv", tmp_path / "sv"
    pg.write_views(cuts, model_dir, paint_dir)
    for name, _ in cuts:
        assert (model_dir / name).is_file()
        assert (paint_dir / name).is_file()
    assert json.loads((model_dir / "transforms.json").read_text()) == pg.turntable_rig()
    assert json.loads((paint_dir / "transforms.json").read_text()) == pg.turntable_rig()
    # Model views are 768 px; the paint views are never below 768.
    assert Image.open(model_dir / "view_000.png").size == (768, 768)
    assert Image.open(paint_dir / "view_000.png").size == (768, 768)


def test_write_views_paint_matches_the_largest_picture(tmp_path):
    cuts = [("view_000.png", Image.new("RGBA", (40, 40))),
            ("view_090.png", Image.new("RGBA", (1000, 50))),
            ("view_180.png", Image.new("RGBA", (40, 40))),
            ("view_270.png", Image.new("RGBA", (40, 40)))]
    pg.write_views(cuts, tmp_path / "mv", tmp_path / "sv")
    # The 1000 px wide side keeps its detail in the paint, so paint_size is 1000.
    assert Image.open(tmp_path / "sv" / "view_000.png").size == (1000, 1000)
    # The model views stay at 768.
    assert Image.open(tmp_path / "mv" / "view_000.png").size == (768, 768)


def test_write_views_refuses_names_that_do_not_match_the_rig(tmp_path):
    cuts = [("view_000.png", Image.new("RGBA", (8, 8))),
            ("view_999.png", Image.new("RGBA", (8, 8)))]
    with pytest.raises(ValueError, match="do not match the rig"):
        pg.write_views(cuts, tmp_path / "mv", tmp_path / "sv")


def test_cut_view_mattes_a_picture_without_alpha(tmp_path, monkeypatch):
    src = tmp_path / "side.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(src)
    monkeypatch.setattr(pg, "has_alpha", lambda p: False)
    matted = tmp_path / "matted.png"
    Image.new("RGBA", (8, 8), (1, 2, 3, 128)).save(matted)
    monkeypatch.setattr(pg, "matte", lambda p, d=None: (matted, "u2net"))
    out = pg.cut_view(src, tmp_path / "out.glb", "left")
    assert np.asarray(out).shape == (8, 8, 4)
    assert tuple(np.asarray(out)[0, 0]) == (1, 2, 3, 128)


def test_cut_view_keeps_an_already_matted_picture(tmp_path, monkeypatch):
    src = tmp_path / "side.png"
    Image.new("RGBA", (8, 8), (5, 6, 7, 200)).save(src)
    monkeypatch.setattr(pg, "has_alpha", lambda p: True)
    monkeypatch.setattr(pg, "matte", lambda p, d=None: (_ for _ in ()).throw(
        AssertionError("matted again")))
    out = pg.cut_view(src, tmp_path / "out.glb", "right")
    assert tuple(np.asarray(out)[0, 0]) == (5, 6, 7, 200)
