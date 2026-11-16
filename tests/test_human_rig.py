"""The 52-joint humanoid template: counts, hierarchy, left/right, and the bone list."""

from __future__ import annotations

import numpy as np
import pytest

from Ar3DRAHub import human_rig as hr


def test_the_template_is_52_joints_22_body_plus_30_fingers():
    assert len(hr.defaults()) == 52
    assert len(hr.joint_order()) == 52
    assert len(hr.parents()) == 52
    assert len(hr.BODY_ORDER) == 22


def test_every_joint_has_a_parent_except_the_hips():
    tree = hr.parents()
    assert tree["Hips"] is None
    for name, parent in tree.items():
        if name != "Hips":
            assert parent in tree


def test_fingers_hang_off_the_hand_three_deep():
    tree = hr.parents()
    assert tree["LeftHandThumb1"] == "LeftHand"
    assert tree["LeftHandThumb2"] == "LeftHandThumb1"
    assert tree["LeftHandThumb3"] == "LeftHandThumb2"
    assert tree["RightHandPinky3"] == "RightHandPinky2"


def test_left_is_plus_x_and_right_is_minus_x():
    d = hr.defaults()
    assert d["LeftArm"][0] > 0.5 > d["RightArm"][0]
    assert d["LeftUpLeg"][0] > 0.5 > d["RightUpLeg"][0]
    assert d["LeftHand"][0] > d["RightHand"][0]


def test_hand_joints_fan_outward_from_the_hand():
    left = hr.hand_joints(hr.BODY["LeftHand"], away=+1.0, side="Left")
    assert len(left) == 15
    hand_x = hr.BODY["LeftHand"][0]
    for (x, _, _) in left.values():
        assert x > hand_x  # fingers reach further from the midline than the hand
    assert "LeftHandThumb1" in left and "LeftHandPinky3" in left


def test_bone_list_is_parents_before_children():
    rows = hr.bone_list(hr.defaults())
    assert len(rows) == 52
    seen = {name for name, *_ in rows}
    for name, _head, _tail, parent in rows:
        if parent is not None:
            assert parent in seen  # its parent row came earlier in the list


def test_bone_list_tail_is_the_child_head_for_chains():
    d = hr.defaults()
    rows = {name: (head, tail, parent) for name, head, tail, parent in hr.bone_list(d)}
    tree = hr.parents()
    for name in hr.joint_order():
        children = [c for c in hr.joint_order() if tree.get(c) == name]
        if children:
            head, tail, _ = rows[name]
            assert head == pytest.approx(d[name])          # head is the joint itself
            assert tail == pytest.approx(d[children[0]])   # tail continues into the child


def test_leaf_bones_get_a_direction_not_a_degenerate_tail():
    rows = {name: (head, tail) for name, head, tail, _ in hr.bone_list(hr.defaults())}
    for name in ("Head", "LeftToeBase", "RightHandThumb3", "LeftHandPinky3"):
        head, tail = rows[name]
        assert tail != head


def test_world_positions_scale_fractions_by_the_bounding_box():
    lo, size = (1.0, -2.0, 0.0), (10.0, 20.0, 5.0)
    world = hr.world_positions(lo, size, {"Hips": (0.5, 0.5, 0.52)})
    x, y, z = world["Hips"]
    assert x == pytest.approx(1.0 + 0.5 * 10.0)
    assert y == pytest.approx(-2.0 + 0.5 * 20.0)
    assert z == pytest.approx(0.0 + 0.52 * 5.0)


def test_joint_group_paints_each_body_part():
    assert hr.joint_group("Head") == "head"
    assert hr.joint_group("Neck") == "head"
    assert hr.joint_group("Hips") == "spine"
    assert hr.joint_group("Spine2") == "spine"
    assert hr.joint_group("LeftShoulder") == "arm"
    assert hr.joint_group("LeftForeArm") == "arm"
    assert hr.joint_group("LeftHand") == "hand"
    assert hr.joint_group("RightHandPinky3") == "hand"
    assert hr.joint_group("LeftUpLeg") == "leg"
    assert hr.joint_group("RightToeBase") == "leg"


def test_every_joint_has_a_known_group():
    groups = {"head", "spine", "arm", "hand", "leg"}
    for name in hr.joint_order():
        assert hr.joint_group(name) in groups


def test_mirror_maps_left_to_right_and_back():
    assert hr.mirror("LeftArm") == "RightArm"
    assert hr.mirror("RightHandThumb2") == "LeftHandThumb2"
    assert hr.mirror("LeftHandPinky3") == "RightHandPinky3"


def test_mirror_returns_none_on_the_midline():
    for name in ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"):
        assert hr.mirror(name) is None


def test_mirror_is_an_involution_on_every_joint():
    for name in hr.joint_order():
        twin = hr.mirror(name)
        if twin is not None:
            assert hr.mirror(twin) == name


def _tpose_humanoid(n: int = 400) -> np.ndarray:
    """A dense T-pose humanoid in glTF space (x right, y up, z front).

    Roughly unit-proportioned: head and neck above a torso, arms straight out to the
    sides at shoulder height, two legs, feet on the ground. Used to check the fit lands
    on the landmarks the mesh states plainly.
    """
    rng = np.random.default_rng(0)

    def box(lo, hi):
        return lo + rng.random((n, 3)) * (np.array(hi) - np.array(lo))

    parts = [
        box((-0.20, 0.70, -0.14), (0.20, 1.30, 0.14)),   # torso
        box((-0.10, 1.30, -0.14), (0.10, 1.75, 0.14)),   # head
        box((-0.07, 1.22, -0.07), (0.07, 1.42, 0.07)),   # neck, narrower than the head
        box((0.20, 1.12, -0.07), (0.90, 1.24, 0.07)),    # left arm (+x)
        box((-0.90, 1.12, -0.07), (-0.20, 1.24, 0.07)),  # right arm (-x)
        box((0.06, 0.12, -0.12), (0.20, 0.72, 0.12)),    # left leg
        box((-0.20, 0.12, -0.12), (-0.06, 0.72, 0.12)),  # right leg
        box((0.05, 0.00, -0.16), (0.25, 0.10, 0.20)),    # left foot
        box((-0.25, 0.00, -0.16), (-0.05, 0.10, 0.20)),  # right foot
    ]
    return np.vstack(parts)


def test_fit_body_lands_on_a_t_pose():
    body = hr.fit_body(_tpose_humanoid())
    # Head near the crown, neck between hips and head, hands at the arm span.
    assert body["Head"][2] > 0.9
    assert 0.5 < body["Neck"][2] < body["Head"][2]
    assert body["LeftHand"][0] > 0.9 and body["RightHand"][0] < 0.1
    assert body["LeftHand"][0] + body["RightHand"][0] == pytest.approx(1.0, abs=0.03)
    # Hips at the pelvis, feet near the ground, toes forward of the ankles.
    assert 0.3 < body["Hips"][2] < 0.6
    assert body["LeftFoot"][2] < 0.15 and body["RightFoot"][2] < 0.15
    assert body["LeftToeBase"][1] < body["LeftFoot"][1]  # toe forward of the foot


def test_fit_body_mirrors_left_and_right():
    body = hr.fit_body(_tpose_humanoid())
    for left, right in (("LeftArm", "RightArm"), ("LeftHand", "RightHand"),
                        ("LeftUpLeg", "RightUpLeg"), ("LeftFoot", "RightFoot")):
        assert body[left][0] + body[right][0] == pytest.approx(1.0, abs=0.03)
        # Front/back and up/down match across the body; the arm depth inherits a little
        # sampling noise from the hand median, so depth gets a looser tolerance.
        assert body[left][1] == pytest.approx(body[right][1], abs=0.08)
        assert body[left][2] == pytest.approx(body[right][2], abs=0.02)


def test_fit_defaults_has_all_52_joints():
    joints = hr.fit_defaults(_tpose_humanoid())
    assert sorted(joints) == sorted(hr.joint_order())


def test_fit_body_refuses_a_mesh_it_cannot_measure():
    with pytest.raises(ValueError):
        hr.fit_body([[0.0, 0.0, 0.0]] * 4)          # too few vertices
    with pytest.raises(ValueError):
        hr.fit_body([[x, 0.0, 0.0] for x in range(20)])   # flat: no height or width
