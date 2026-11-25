"""The TexVerse view renderer: argument handling and the pose conversion that turns
the dataset's camera matrices into Blender world matrices. Pure parts only; no Blender."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_render_texverse_views.py"


def _load():
    spec = importlib.util.spec_from_file_location("blender_render_texverse_views", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


renderer = _load()


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def _mv(matrix, vector):
    """A direction in the camera's local frame, expressed in the world frame.

    Row-major world matrix: world_i = sum over j of matrix[i][j] * vector[j].
    """
    return tuple(_dot(row[:3], vector) for row in matrix[:3])


def test_parse_args_defaults():
    mesh, output, resolution, guided = renderer.parse_args(["--", "a.glb", "out"])
    assert (mesh, output, resolution, guided) == ("a.glb", "out", 1024, False)


def test_parse_args_reads_the_resolution():
    assert renderer.parse_args(["--", "a.glb", "out", "512"])[2] == 512


def test_parse_args_reads_the_guided_flag():
    assert renderer.parse_args(["--", "a.glb", "out", "512", "guided"])[3] is True
    for off in ("0", "", "false", "no"):
        assert renderer.parse_args(["--", "a.glb", "out", "512", off])[3] is False


@pytest.mark.parametrize("argv", [["--", "a.glb"], ["--", "a.glb", "out", "0"]])
def test_bad_arguments_are_refused(argv):
    with pytest.raises(SystemExit):
        renderer.parse_args(argv)


def test_the_six_poses_are_at_the_dataset_distance_and_orthonormal():
    for pose in renderer.POSES:
        rotation = [row[:3] for row in pose]
        # Rows are orthonormal: R @ R^T = I.
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert _dot(rotation[i], rotation[j]) == pytest.approx(expected, abs=1e-6)
        # Every camera sits 1.67 units from the origin.
        position = [row[3] for row in pose]
        assert (position[0] ** 2 + position[1] ** 2 + position[2] ** 2) \
            == pytest.approx(1.67**2, abs=1e-6)


def test_pose_to_blender_keeps_position_and_looks_forward():
    for pose in renderer.POSES:
        matrix = renderer.pose_to_blender(pose)
        # The translation column survives: the camera position is the pose's
        # fourth column.
        assert (matrix[0][3], matrix[1][3], matrix[2][3]) == \
            (pose[0][3], pose[1][3], pose[2][3])
        # Blender's local -Z (the view direction) maps to the negated third
        # column: the camera looks along -Z = forward, toward the origin.
        view = _mv(matrix, (0.0, 0.0, -1.0))
        assert view == pytest.approx(tuple(-row[2] for row in pose), abs=1e-6)
        # Blender's local +Y maps to the second column: the image up.
        up = _mv(matrix, (0.0, 1.0, 0.0))
        assert up == pytest.approx(tuple(row[1] for row in pose), abs=1e-6)


def test_front_view_is_at_minus_y():
    """View 000 is the front: camera on -Y looking +Y, matching render_ref_0."""
    matrix = renderer.pose_to_blender(renderer.POSES[0])
    assert (matrix[0][3], matrix[1][3], matrix[2][3]) == pytest.approx((0.0, -1.67, 0.0))
    assert _mv(matrix, (0.0, 0.0, -1.0)) == pytest.approx((0.0, 1.0, 0.0))
