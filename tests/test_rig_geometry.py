"""Rigging geometry: similarity alignment, skeleton adoption, weight harvest, coverage.

The techniques here are standard (Umeyama, ICP, k-d tree transfer) and are reimplemented
rather than copied; the engine they serve is SkinTokens (MIT).
"""

from __future__ import annotations

import numpy as np
import pytest

from Ar3DRAHub import rig_geometry as rg


def _rotation(index: int) -> np.ndarray:
    """One of the 24 proper axis rotations, chosen deterministically."""
    return list(rg.axis_rotations())[index]


# ----------------------------------------------------------------------------- umeyama


def test_umeyama_recovers_a_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    r = _rotation(3)
    s0, t0 = 1.7, np.array([5.0, -2.0, 3.0])
    dst = rg.apply_similarity(src, s0, r, t0)

    s, rhat, that = rg.umeyama(src, dst)
    assert s == pytest.approx(s0, abs=1e-9)
    assert np.allclose(rhat, r, atol=1e-9)
    assert np.allclose(that, t0, atol=1e-9)
    assert np.allclose(rg.apply_similarity(src, s, rhat, that), dst, atol=1e-8)


def test_umeyama_returns_a_proper_rotation_for_a_reflected_cloud():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(30, 3))
    dst = src * np.array([-1.0, 1.0, 1.0])  # a mirror, which no rotation can reproduce
    s, r, _ = rg.umeyama(src, dst)
    assert np.linalg.det(r) > 0
    assert s > 0


def test_umeyama_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        rg.umeyama(np.zeros((4, 3)), np.zeros((5, 3)))


# ----------------------------------------------------------------------- axis rotations


def test_axis_rotations_are_24_distinct_proper_rotations():
    rotations = list(rg.axis_rotations())
    assert len(rotations) == 24
    assert len({tuple(r.flatten()) for r in rotations}) == 24
    for r in rotations:
        assert np.linalg.det(r) > 0
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)


# ---------------------------------------------------------------------- adopt skeleton


def test_adopt_skeleton_recovers_a_known_transform():
    rng = np.random.default_rng(1)
    user = rng.normal(size=(80, 3)) * np.array([2.0, 0.5, 1.0])
    r0 = _rotation(5)
    s0, t0 = 1.4, np.array([3.0, -1.0, 0.5])
    pred = rg.apply_similarity(user, s0, r0, t0)

    s, r, t = rg.adopt_skeleton(pred, user)
    recovered = rg.apply_similarity(pred, s, r, t)
    spread = float(np.linalg.norm(user - user.mean(axis=0), axis=1).max())
    dist = np.linalg.norm(recovered - user, axis=1)
    assert dist.max() < 0.02 * spread


def test_adopt_skeleton_refuses_an_unrelated_cloud():
    rng = np.random.default_rng(3)
    user = rng.normal(size=(80, 3))
    pred = rng.normal(size=(80, 3))
    with pytest.raises(ValueError):
        rg.adopt_skeleton(pred, user)


# ------------------------------------------------------------------- map by position


def test_map_bones_by_position_pairs_by_location():
    pred_names = ["b0", "b1", "b2"]
    pred_pos = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    user_names = ["hip", "knee", "ankle"]
    user_pos = [[10, 20, 30], [11, 20, 30], [10, 21, 30]]  # same shape, moved and scaled
    assert rg.map_bones_by_position(pred_names, pred_pos, user_names, user_pos) == {
        "b0": "hip", "b1": "knee", "b2": "ankle",
    }


# ------------------------------------------------------------------- weight harvest


def test_harvest_weights_renames_and_renormalizes():
    pred_verts = [[0, 0, 0]]
    pred_weights = [{"a": 0.5, "b": 0.5}]  # sums to 1.0 but in two parts
    bone_map = {"a": "arm", "b": "chest"}
    out = rg.harvest_weights(pred_verts, pred_weights, [[0, 0, 0]], bone_map)
    assert set(out) == {"arm", "chest"}
    assert out["arm"][0] == pytest.approx(0.5)
    assert out["chest"][0] == pytest.approx(0.5)


def test_harvest_weights_keeps_only_the_top_influences():
    pred_weights = [{"a": 0.5, "b": 0.2, "c": 0.15, "d": 0.1, "e": 0.05}]
    bone_map = {name: name.upper() for name in "abcde"}
    out = rg.harvest_weights([[0, 0, 0]], pred_weights, [[0, 0, 0]], bone_map)
    assert set(out) == {"A", "B", "C", "D"}  # 'e' is trimmed
    assert sum(w[0] for w in out.values()) == pytest.approx(1.0)


def test_harvest_weights_drops_bones_without_a_user_joint():
    # A predicted bone with no user joint (a collision) is dropped, and its weight is
    # not redistributed -- the engine reports it as a warning, not a silent renormalise.
    pred_weights = [{"a": 0.7, "gone": 0.3}]
    bone_map = {"a": "arm"}  # 'gone' has no counterpart
    out = rg.harvest_weights([[0, 0, 0]], pred_weights, [[0, 0, 0]], bone_map)
    assert set(out) == {"arm"}
    assert out["arm"][0] == pytest.approx(0.7)


# ---------------------------------------------------------------- vertex order check


def test_verify_vertex_order_passes_when_intact():
    rng = np.random.default_rng(4)
    src = rng.normal(size=(50, 3))
    probe = rg.apply_similarity(src, 2.0, _rotation(2), np.array([1.0, 2.0, 3.0]))
    residual, _ = rg.verify_vertex_order(src, probe)
    assert residual < 1e-6


def test_verify_vertex_order_fails_when_scrambled():
    rng = np.random.default_rng(5)
    src = rng.normal(size=(50, 3))
    probe = rng.permutation(src)
    with pytest.raises(ValueError):
        rg.verify_vertex_order(src, probe)


def test_verify_vertex_order_rejects_a_vertex_count_mismatch():
    with pytest.raises(ValueError):
        rg.verify_vertex_order(np.zeros((6, 3)), np.zeros((7, 3)))


# ----------------------------------------------------------------------- joint coverage


def test_joint_coverage_gives_every_joint_a_vertex():
    rng = np.random.default_rng(6)
    joints = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    verts = rng.normal(scale=0.05, size=(30, 3))  # all clustered near the first joint
    coverage = rg.joint_coverage(joints, verts)
    assert set(coverage) == {0, 1, 2}
    assert all(indices for indices in coverage.values())
    # No vertex is assigned to more joints than the engine can export.
    per_vertex: dict[int, int] = {}
    for indices in coverage.values():
        for vi in indices:
            per_vertex[vi] = per_vertex.get(vi, 0) + 1
    assert all(count <= rg.MAX_INFLUENCES for count in per_vertex.values())


def test_joint_coverage_spread_verts_cover_naturally():
    rng = np.random.default_rng(7)
    joints = [[0, 0, 0], [5, 0, 0]]
    verts = rng.normal(size=(40, 3)) * np.array([5.0, 0.5, 0.5])
    coverage = rg.joint_coverage(joints, verts)
    assert set(coverage) == {0, 1}
