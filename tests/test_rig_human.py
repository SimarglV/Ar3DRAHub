"""Hand-rigging: the humanoid build script reads placed joints over the T-pose defaults."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rig_human as rh

from Ar3DRAHub import human_rig


def test_load_joints_defaults_to_the_tpose_when_no_file():
    joints = rh.load_joints(None)
    assert joints == human_rig.defaults()
    assert len(joints) == 52


def test_load_joints_fills_a_partial_json_over_the_defaults(tmp_path):
    placed = tmp_path / "joints.json"
    placed.write_text('{"Head": [0.50, 0.50, 0.93], "Hips": [0.50, 0.50, 0.51]}')
    joints = rh.load_joints(placed)
    assert joints["Head"] == (0.50, 0.50, 0.93)
    assert joints["Hips"] == (0.50, 0.50, 0.51)
    assert joints["LeftArm"] == human_rig.defaults()["LeftArm"]  # untouched


def test_load_joints_refuses_an_unknown_joint(tmp_path):
    placed = tmp_path / "joints.json"
    placed.write_text('{"Tail": [0.5, 0.5, 0.5]}')
    with pytest.raises(ValueError, match="unknown joint"):
        rh.load_joints(placed)
