"""The Hi3D texture-transfer script: argument handling and the bounding-box alignment
that lands the refined mesh back in the source's frame. Pure parts only; no Blender."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_hi3d_transfer.py"


def _load():
    spec = importlib.util.spec_from_file_location("blender_hi3d_transfer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transfer = _load()


def test_parse_args_defaults():
    args = transfer.parse_args(["--", "a.glb", "b.glb", "c.glb"])
    assert args == ("a.glb", "b.glb", "c.glb", 2048, 0.25, 0.65, 1.45)


def test_parse_args_reads_all_knobs():
    args = transfer.parse_args(["--", "a.glb", "b.glb", "c.glb", "1024", "0.1", "0.9", "1.6"])
    assert args == ("a.glb", "b.glb", "c.glb", 1024, 0.1, 0.9, 1.6)


@pytest.mark.parametrize("argv", [
    ["--", "a.glb"], ["--", "a.glb", "b.glb", "c.glb", "500"], ["--", "a.glb", "b.glb", "c.glb", "2048", "2.0"],
])
def test_bad_arguments_are_refused(argv):
    with pytest.raises(SystemExit):
        transfer.parse_args(argv)


def test_alignment_maps_the_refined_bbox_onto_the_source_bbox():
    # Source: 2..4 on X, -1..1 on Y, 0..2 on Z (extent 2, centre (3, 0, 1)).
    # Refined: unit sphere around the origin (extent 2, centre (0, 0, 0)).
    scale, offset = transfer.align_transform((2, -1, 0), (4, 1, 2), (-1, -1, -1), (1, 1, 1))
    assert scale == pytest.approx(1.0)
    assert offset == pytest.approx((3.0, 0.0, 1.0))
    # A refined corner lands on the source corner.
    corner = (1 * scale + offset[0], 1 * scale + offset[1], 1 * scale + offset[2])
    assert corner == pytest.approx((4.0, 1.0, 2.0))


def test_alignment_scales_when_the_source_is_larger():
    # Source extent 4, refined extent 2 -> scale 2.
    scale, offset = transfer.align_transform((0, 0, 0), (4, 4, 4), (-1, -1, -1), (1, 1, 1))
    assert scale == pytest.approx(2.0)
    assert offset == pytest.approx((2.0, 2.0, 2.0))


def test_a_degenerate_mesh_cannot_be_aligned():
    with pytest.raises(ValueError, match="zero extent"):
        transfer.align_transform((0, 0, 0), (1, 1, 1), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def _apply(m, point):
    x, y, z = point
    return (m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
            m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
            m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3])


def test_align_matrix_scales_then_translates():
    # The order matters: the offset must not be scaled. A point maps to
    # p * scale + offset, not p * scale + offset * scale.
    m = transfer.align_matrix(2.0, (10.0, 20.0, 30.0))
    assert _apply(m, (1.0, 2.0, 3.0)) == pytest.approx((12.0, 24.0, 36.0))
    assert _apply(m, (0.0, 0.0, 0.0)) == pytest.approx((10.0, 20.0, 30.0))
