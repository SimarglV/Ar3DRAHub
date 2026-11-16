"""Pure rigging geometry, testable without Blender or a GPU.

The maths behind binding a skeleton to a mesh and transferring skin weights onto it:
similarity alignment (Umeyama), the 24-rotation enumeration that places a predicted
skeleton back into model space, position-keyed weight harvest, joint-coverage assignment,
and the vertex-order fidelity check that refuses to write a skin it cannot vouch for.

Every function here is numpy/scipy only. The Blender orchestration that calls these lives
in ``scripts/rig_bind.py``; this module is imported by it and by the tests. The techniques
are standard (Umeyama 1991, iterative closest point, k-d tree weight transfer) and are
reimplemented here rather than copied from any closed source; the engine they drive is
SkinTokens, MIT-licensed.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree

# What game engines take; matches SkinTokens' own export and the eval.
MAX_INFLUENCES = 4

#: A vertex-order check fails past this residual, as a fraction of the model's radius.
FIDELITY_TOL = 1e-3

#: Skeleton adoption refuses past this residual, as a fraction of the model's radius.
ADOPT_RESIDUAL = 0.02


def umeyama(src: Sequence[Sequence[float]], dst: Sequence[Sequence[float]]
            ) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity mapping ``src`` onto ``dst``.

    Returns ``(s, R, t)`` with ``dst ~= s * (R @ src.T).T + t``. A unit scale and an
    axis conversion are the only transformations Blender's importers are allowed to apply,
    so a residual far from zero means the two point sets no longer correspond.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[1] != 3:
        raise ValueError("src and dst must be Nx3 arrays of the same shape")
    if len(src) == 0:
        raise ValueError("cannot align empty point sets")
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    cs = src - mu_s
    cd = dst - mu_d
    cov = cd.T @ cs / len(src)
    u, d, vt = np.linalg.svd(cov)
    s_ = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_[2, 2] = -1.0
    r = u @ s_ @ vt
    var_s = float((cs ** 2).sum() / len(src))
    scale = float((d * np.diag(s_)).sum() / var_s) if var_s > 0 else 1.0
    t = mu_d - scale * (r @ mu_s)
    return scale, r, t


def apply_similarity(points: Sequence[Sequence[float]], s: float,
                     r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply a similarity transform: ``s * (R @ points.T).T + t``."""
    pts = np.asarray(points, dtype=np.float64)
    return (s * (r @ pts.T)).T + t


def axis_rotations() -> Iterable[np.ndarray]:
    """The 24 proper rotations of a cube, as 3x3 matrices.

    Every axis permutation crossed with every sign pattern, keeping determinant +1.
    A reflection would mirror the character, so those are excluded. Enumerating the
    right answer is cheaper and safer than searching for it: an iterative fit can slide
    the scale toward zero and still report convergence.
    """
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            r = np.zeros((3, 3))
            for row, col in enumerate(perm):
                r[row, col] = signs[row]
            if np.linalg.det(r) > 0:
                yield r


def _bbox(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The axis-aligned bounding box of a point cloud: (centre, extent)."""
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return (lo + hi) / 2.0, hi - lo


def _radius(points: np.ndarray) -> float:
    """The greatest distance from the centroid, or 1.0 when it is degenerate."""
    return float(np.linalg.norm(points - points.mean(axis=0), axis=1).max()) or 1.0


def adopt_skeleton(pred_verts: Sequence[Sequence[float]],
                   user_verts: Sequence[Sequence[float]],
                   max_residual: float = ADOPT_RESIDUAL
                   ) -> tuple[float, np.ndarray, np.ndarray]:
    """Place a predicted point cloud back into a user mesh's frame.

    The engine returns its work in a canonical box, so a generated skeleton must be
    moved back before it is any use. The transform is a uniform scale, a translation and
    one of the 24 axis rotations. Rather than fit iteratively from a guess, every
    rotation is tried: scale and translation come from bounding boxes (which cannot
    collapse), the best candidate wins, then a rigid ICP with the scale held fixed
    refines it. If nothing lands on the mesh, raise rather than invent a skeleton.

    Returns ``(s, R, t)`` mapping ``pred_verts`` onto ``user_verts``.
    """
    pred = np.asarray(pred_verts, dtype=np.float64)
    user = np.asarray(user_verts, dtype=np.float64)
    c_u, e_u = _bbox(user)
    radius = _radius(user)
    tree = cKDTree(user)

    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for r in axis_rotations():
        rotated = pred @ r.T
        c_p, e_p = _bbox(rotated)
        live = (e_p > 1e-9) & (e_u > 1e-9)
        if not live.any():
            continue
        ratios = e_u[live] / e_p[live]
        # A candidate whose axes disagree about the scale is the wrong candidate; the
        # right one has a uniform scale by construction.
        if float(ratios.max() / ratios.min()) > 1.10:
            continue
        s = float(np.median(ratios))
        t = c_u - s * c_p
        dist, _ = tree.query((s * rotated) + t, k=1)
        resid = float(dist.mean())
        if best is None or resid < best[0]:
            best = (resid, s, r, t)

    if best is None:
        raise ValueError(
            "the predicted mesh does not match the user mesh at any orientation; "
            "refusing to guess a skeleton")

    resid, s, r, t = best
    # Refine rigidly: scale held fixed at the value the bounding boxes gave, so there is
    # no scale for the ICP to slide toward zero.
    for _ in range(6):
        moved = (s * (pred @ r.T)) + t
        dist, nearest = tree.query(moved, k=1)
        _, r2, t2 = umeyama(moved, user[nearest])
        r, t = r2 @ r, r2 @ t + t2

    dist, _ = tree.query((s * (pred @ r.T)) + t, k=1)
    resid = float(dist.mean())
    if resid > max_residual * radius:
        raise ValueError(
            f"the predicted mesh does not sit on the user mesh after alignment "
            f"(off by {100 * resid / radius:.1f}% of the model); refusing to place a "
            f"skeleton it cannot line up")
    return s, r, t


def map_bones_by_position(pred_names: Sequence[str],
                          pred_pos: Sequence[Sequence[float]],
                          user_names: Sequence[str],
                          user_pos: Sequence[Sequence[float]]) -> dict[str, str]:
    """Map predicted bone names onto user joint names by position.

    Both skeletons are normalized to unit scale first, so the engine's own scale cannot
    skew the pairing. Returns ``{predicted_bone_name: user_joint_name}``.
    """
    pred = np.asarray(pred_pos, dtype=np.float64)
    user = np.asarray(user_pos, dtype=np.float64)
    if len(pred) != len(pred_names) or len(user) != len(user_names):
        raise ValueError("names and positions must line up")

    def unit(points: np.ndarray) -> np.ndarray:
        centre = points.mean(axis=0)
        spread = np.linalg.norm(points - centre, axis=1).max() or 1.0
        return (points - centre) / spread

    tree = cKDTree(unit(user))
    _, idx = tree.query(unit(pred), k=1)
    return {pred_names[i]: user_names[int(j)] for i, j in enumerate(idx)}


def harvest_weights(pred_verts: Sequence[Sequence[float]],
                    pred_weights: Sequence[dict[str, float]],
                    user_verts: Sequence[Sequence[float]],
                    bone_map: dict[str, str],
                    max_influences: int = MAX_INFLUENCES) -> dict[str, dict[int, float]]:
    """Transfer per-vertex weights onto a user mesh by position.

    ``pred_weights[i]`` is the ``{bone: weight}`` on predicted vertex ``i``. Each user
    vertex takes the weights of its nearest predicted vertex, keeps the top
    ``max_influences`` by weight, renormalizes, and renames bones through ``bone_map``.
    Returns ``{user_joint_name: {user_vertex_index: weight}}``.
    """
    pred = np.asarray(pred_verts, dtype=np.float64)
    user = np.asarray(user_verts, dtype=np.float64)
    if len(pred) != len(pred_weights):
        raise ValueError("pred_weights must have one entry per predicted vertex")
    tree = cKDTree(pred)
    _, nearest = tree.query(user, k=1)

    out: dict[str, dict[int, float]] = {}
    for user_vi, pred_vi in enumerate(nearest):
        weights = pred_weights[int(pred_vi)]
        if not weights:
            continue
        top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:max_influences]
        total = sum(weight for _, weight in top) or 1.0
        for pred_bone, weight in top:
            joint = bone_map.get(pred_bone)
            if joint is None:
                continue
            out.setdefault(joint, {})[user_vi] = weight / total
    return out


def verify_vertex_order(source: Sequence[Sequence[float]],
                        probe: Sequence[Sequence[float]],
                        tolerance: float = FIDELITY_TOL) -> tuple[float, float]:
    """Whether two point sets are the same vertices in the same order.

    If an export round-trip preserved the vertex order, the best-fit similarity through
    the index pairing lands on machine zero; if it scrambled them, no similarity can fit
    it and the residual blows up to the size of the model. Returns
    ``(residual, radius)`` where ``residual`` is the worst per-vertex error as a fraction
    of the model's radius. Raises when the residual exceeds ``tolerance``.
    """
    source = np.asarray(source, dtype=np.float64)
    probe = np.asarray(probe, dtype=np.float64)
    if source.shape != probe.shape:
        raise ValueError(
            f"source has {len(source)} vertices but the probe has {len(probe)}; "
            f"weights travel by vertex index and will not be guessed")
    if len(source) < 4:
        raise ValueError(f"{len(source)} vertices is too few to verify")
    s, r, t = umeyama(source, probe)
    err = np.linalg.norm(apply_similarity(source, s, r, t) - probe, axis=1)
    radius = _radius(probe)
    residual = float(err.max() / radius)
    if residual > tolerance:
        raise ValueError(
            f"the round-trip did not preserve this mesh's vertex order "
            f"(worst vertex is {100 * residual:.2f}% of the model away from where it "
            f"should be); refusing to write a skin that cannot be vouched for")
    return residual, radius


def joint_coverage(joint_pos: Sequence[Sequence[float]],
                   verts: Sequence[Sequence[float]],
                   max_influences: int = MAX_INFLUENCES) -> dict[int, list[int]]:
    """Assign every joint at least one influenced vertex.

    Nearest-joint assignment first; a joint no vertex is nearest to is force-covered with
    the closest vertex that still has a free influence slot. Returns
    ``{joint_index: [vertex_index, ...]}``.
    """
    joints = np.asarray(joint_pos, dtype=np.float64)
    verts = np.asarray(verts, dtype=np.float64)
    if len(joints) == 0:
        return {}
    joint_tree = cKDTree(joints)
    _, nearest_joint = joint_tree.query(verts, k=1)

    # A vertex may carry at most max_influences joints: glTF keeps only the top 4, so a
    # fifth influence would vanish on export and the joint it served would be pruned.
    coverage: dict[int, list[int]] = {}
    used: dict[int, int] = {}
    for vi, ji in enumerate(nearest_joint):
        ji = int(ji)
        coverage.setdefault(ji, []).append(int(vi))
        used[int(vi)] = used.get(int(vi), 0) + 1

    # Force-cover joints no vertex was nearest to, reusing vertices that still have a
    # free influence slot.
    vert_tree = cKDTree(verts)
    for ji in range(len(joints)):
        if coverage.get(ji):
            continue
        _, candidates = vert_tree.query(joints[ji], k=min(64, len(verts)))
        candidates = np.atleast_1d(candidates)
        vi = next((int(c) for c in candidates if used.get(int(c), 0) < max_influences),
                  int(candidates[0]))
        coverage.setdefault(ji, []).append(vi)
        used[vi] = used.get(vi, 0) + 1
    return coverage
