"""The 52-joint humanoid skeleton, as a testable template for hand-rigging.

Joint positions are fractions of the mesh's bounding box in the same space
``scripts/blender_joint_markers.py`` uses: x=0.5 is the midline (x>0.5 is the character's
LEFT), y=0 is the front, and z=0 is the ground. A hand-rigging tool pre-places these
fractions and the user nudges each marker onto the model; ``bone_list`` then turns the
placed joints into the (bone, head, tail, parent) rows an armature builder reads.

The names and hierarchy mirror SkinTokens' ``mixamo.yaml`` (Hips, Spine..., Head, shoulders,
arms, forearms, hands, upLegs, legs, feet, toeBases, and 3 joints per finger), so a
manually-placed skeleton skins with SkinTokens and maps to the preset moves the same way
the auto-rig's does.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# The 22 body joints, in build order (a bone's parent always comes before it).
BODY_ORDER: tuple[str, ...] = (
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
)

# Default fractions for the body, a T-pose: arms out to the sides, feet on the ground.
BODY: dict[str, tuple[float, float, float]] = {
    "Hips": (0.50, 0.50, 0.52),
    "Spine": (0.50, 0.50, 0.60),
    "Spine1": (0.50, 0.50, 0.66),
    "Spine2": (0.50, 0.50, 0.72),
    "Neck": (0.50, 0.50, 0.80),
    "Head": (0.50, 0.50, 0.90),
    "LeftShoulder": (0.58, 0.50, 0.72),
    "LeftArm": (0.70, 0.50, 0.70),
    "LeftForeArm": (0.84, 0.50, 0.68),
    "LeftHand": (0.96, 0.50, 0.66),
    "RightShoulder": (0.42, 0.50, 0.72),
    "RightArm": (0.30, 0.50, 0.70),
    "RightForeArm": (0.16, 0.50, 0.68),
    "RightHand": (0.04, 0.50, 0.66),
    "LeftUpLeg": (0.56, 0.50, 0.40),
    "LeftLeg": (0.56, 0.50, 0.24),
    "LeftFoot": (0.56, 0.48, 0.06),
    "LeftToeBase": (0.56, 0.45, 0.01),
    "RightUpLeg": (0.44, 0.50, 0.40),
    "RightLeg": (0.44, 0.50, 0.24),
    "RightFoot": (0.44, 0.48, 0.06),
    "RightToeBase": (0.44, 0.45, 0.01),
}

BODY_PARENTS: dict[str, str | None] = {
    "Hips": None,
    "Spine": "Hips", "Spine1": "Spine", "Spine2": "Spine1",
    "Neck": "Spine2", "Head": "Neck",
    "LeftShoulder": "Spine2", "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
    "RightShoulder": "Spine2", "RightArm": "RightShoulder",
    "RightForeArm": "RightArm", "RightHand": "RightForeArm",
    "LeftUpLeg": "Hips", "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg", "LeftToeBase": "LeftFoot",
    "RightUpLeg": "Hips", "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg", "RightToeBase": "RightFoot",
}

# (finger, front-back offset, up-down offset): the fan from the hand. Thumb hangs down and
# back; index through pinky step across the palm. All approximate -- the user nudges them.
FINGERS: dict[str, tuple[float, float]] = {
    "Thumb": (-0.010, -0.035),
    "Index": (-0.005, 0.010),
    "Middle": (0.000, 0.015),
    "Ring": (0.005, 0.010),
    "Pinky": (0.010, 0.005),
}

#: How far a leaf bone's tail extends past its head, as a fraction of the bounding box.
LEAF_LENGTH = 0.02


def hand_joints(hand: Sequence[float], away: float, side: str
                ) -> dict[str, tuple[float, float, float]]:
    """15 finger joints (3 per finger) fanned out from a hand fraction.

    ``away`` is +1 when the fingers point toward +X (the left hand) and -1 for the right;
    ``side`` is ``"Left"`` or ``"Right"`` for the joint names.
    """
    hx, hy, hz = hand
    joints: dict[str, tuple[float, float, float]] = {}
    for finger, (dy, dz) in FINGERS.items():
        for i in (1, 2, 3):
            joints[f"{side}Hand{finger}{i}"] = (
                round(hx + away * 0.035 * i, 3),
                round(hy + dy * (1.0 + 0.3 * (i - 1)), 3),
                round(hz + dz * (1.0 + 0.3 * (i - 1)), 3),
            )
    return joints


def joint_order() -> tuple[str, ...]:
    """Every joint, in build order: body first, then each hand's fingers."""
    order = list(BODY_ORDER)
    for side in ("Left", "Right"):
        for finger in FINGERS:
            order.extend(f"{side}Hand{finger}{i}" for i in (1, 2, 3))
    return tuple(order)


def defaults() -> dict[str, tuple[float, float, float]]:
    """Every joint's default fraction, body plus both hands' fingers."""
    joints = dict(BODY)
    joints.update(hand_joints(BODY["LeftHand"], away=+1.0, side="Left"))
    joints.update(hand_joints(BODY["RightHand"], away=-1.0, side="Right"))
    return joints


def parents() -> dict[str, str | None]:
    """Every joint's parent, body plus both hands' fingers (each finger-1 hangs off the hand)."""
    tree = dict(BODY_PARENTS)
    for side, hand_name in (("Left", "LeftHand"), ("Right", "RightHand")):
        for finger in FINGERS:
            base = f"{side}Hand{finger}"
            tree[f"{base}1"] = hand_name
            tree[f"{base}2"] = f"{base}1"
            tree[f"{base}3"] = f"{base}2"
    return tree


def joint_group(name: str) -> str:
    """The bone group a joint belongs to, so a hand-rig skeleton can colour each part:
    head, spine, arm (shoulders to forearms), hand (palms and fingers), or leg (to toes)."""
    if name in ("Head", "Neck"):
        return "head"
    if name in ("Hips", "Spine", "Spine1", "Spine2"):
        return "spine"
    if "Hand" in name:
        return "hand"
    if "Shoulder" in name or "Arm" in name:
        return "arm"
    if "UpLeg" in name or "Leg" in name or "Foot" in name or "ToeBase" in name:
        return "leg"
    return "other"


def mirror(name: str) -> str | None:
    """The same joint on the other side of the body, or None for a midline joint."""
    if name.startswith("Left"):
        return "Right" + name[len("Left"):]
    if name.startswith("Right"):
        return "Left" + name[len("Right"):]
    return None


def bone_list(positions: dict[str, tuple[float, float, float]]
              ) -> list[tuple[str, tuple, tuple, str | None]]:
    """``(bone, head, tail, parent)`` rows from placed joint fractions.

    A bone's tail is its child's head (the chain continues); a leaf bone's tail extends a
    little past its head along the parent->head direction, so Head, fingertips and toes
    get a direction. All values stay in fraction space; the builder scales them to the
    mesh's real bounding box.
    """
    tree = parents()
    rows: list[tuple[str, tuple, tuple, str | None]] = []
    for name in joint_order():
        head = tuple(positions[name])
        children = [c for c in joint_order() if tree.get(c) == name]
        parent = tree[name]
        if children:
            tail = tuple(positions[children[0]])
        elif parent is not None:
            ph = positions[parent]
            dx, dy, dz = head[0] - ph[0], head[1] - ph[1], head[2] - ph[2]
            length = (dx * dx + dy * dy + dz * dz) ** 0.5
            if length > 1e-9:
                tail = (head[0] + dx / length * LEAF_LENGTH,
                        head[1] + dy / length * LEAF_LENGTH,
                        head[2] + dz / length * LEAF_LENGTH)
            else:
                tail = (head[0], head[1], head[2] + LEAF_LENGTH)
        else:
            tail = (head[0], head[1], head[2] + LEAF_LENGTH)
        rows.append((name, head, tail, parent))
    return rows


def world_positions(bbox_min: Sequence[float], bbox_size: Sequence[float],
                    fractions: dict[str, tuple[float, float, float]]
                    ) -> dict[str, tuple[float, float, float]]:
    """Fractions of a bounding box to world positions: ``min + fraction * size`` per axis."""
    lo, size = tuple(bbox_min), tuple(bbox_size)
    return {name: (lo[0] + f[0] * size[0],
                   lo[1] + f[1] * size[1],
                   lo[2] + f[2] * size[2]) for name, f in fractions.items()}


def _lerp(a: Sequence[float], b: Sequence[float], t: float) -> tuple[float, float, float]:
    """Linear interpolation between two fraction triples."""
    return tuple(round(ai + (bi - ai) * t, 4) for ai, bi in zip(a, b))


def fit_body(vertices: Sequence[Sequence[float]]) -> dict[str, tuple[float, float, float]]:
    """Place the 22 body joints onto a mesh, in this module's fraction space.

    ``vertices`` are the mesh's own coordinates straight from the GLB, in glTF space
    (x right, y up, z front). The fit assumes a standing humanoid that is roughly
    left-right symmetric with arms out to the sides (a T-pose or an A-pose): it measures
    the landmarks the mesh states plainly -- the crown, the neck pinch, the arm span, the
    crotch where the legs meet, the feet -- and spaces the rest along those, so the
    skeleton lands on the model's own proportions instead of a fixed T-pose. Fingers are
    not fitted; they fan from the hands as ever. Raises ValueError on a mesh too small or
    degenerate to measure.
    """
    pts = np.asarray(vertices, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 16:
        raise ValueError("need at least 16 vertices to fit a skeleton")
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    size = hi - lo
    if size[1] <= 1e-9 or size[0] <= 1e-9:
        raise ValueError("the mesh has no height or width to fit against")

    gx = pts[:, 0]
    gy = pts[:, 1]
    mid_x = 0.5 * (lo[0] + hi[0])
    mid_z = 0.5 * (lo[2] + hi[2])
    height = float(size[1])

    def frac(x: float, y: float, z: float) -> tuple[float, float, float]:
        # fx left/right, fy front/back (front = high z), fz up.
        return ((x - lo[0]) / size[0], (hi[2] - z) / size[2], (y - lo[1]) / size[1])

    def width_at(y: float, band: float = 0.06) -> float:
        mask = np.abs(gy - y) <= band * height
        return float(gx[mask].max() - gx[mask].min()) if mask.any() else 0.0

    # Per-slice width profile, lightly smoothed so a thin arm does not throw it off.
    n_bins = 48
    edges = np.linspace(lo[1], hi[1], n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (gy >= edges[i]) & (gy < edges[i + 1])
        if mask.sum() >= 3:
            widths[i] = float(gx[mask].max() - gx[mask].min())
    smooth = np.convolve(widths, np.ones(3) / 3, mode="same")

    # Hands: the vertices nearest each x-extreme (the arm span in a T-pose).
    k = max(8, len(pts) // 200)
    order = np.argsort(gx)
    left_hand = np.median(pts[order[-k:]], axis=0)    # +x side, fx > 0.5
    right_hand = np.median(pts[order[:k]], axis=0)    # -x side, fx < 0.5
    hand_y = 0.5 * (left_hand[1] + right_hand[1])

    # Neck: the narrowest slice of the upper body (below the crown, above the arms).
    upper = (centres >= lo[1] + 0.55 * height) & (centres <= lo[1] + 0.86 * height)
    neck_y = (float(centres[upper][int(np.argmin(smooth[upper]))])
              if upper.any() else lo[1] + 0.80 * height)

    # Shoulders: the top of the wide arm-span region (a T-pose has a thin plateau, an
    # A-pose a taller one that runs down to the hands).
    arm_span = float(smooth.max())
    wide = smooth >= 0.7 * arm_span
    shoulder_y = float(centres[wide].max()) if wide.any() else hand_y

    # Crotch: the highest slice below the middle where the legs leave a central x-gap.
    gap_half = 0.02 * size[0]
    gap = np.zeros(n_bins, dtype=bool)
    for i in range(n_bins):
        mask = (gy >= edges[i]) & (gy < edges[i + 1])
        if mask.sum() < 3:
            continue
        xs = gx[mask]
        gap[i] = not ((xs >= mid_x - gap_half) & (xs <= mid_x + gap_half)).any()
    lower = np.flatnonzero(gap[: n_bins // 2])
    crotch_y = float(edges[int(lower.max()) + 1]) if lower.size else lo[1] + 0.46 * height

    hips_y = crotch_y + 0.03 * height
    head_y = neck_y + 0.9 * (hi[1] - neck_y)
    torso_half = 0.5 * width_at(hips_y) or 0.20 * size[0]

    # Feet: the left/right clusters at ground level.
    ground_band = (gy - lo[1]) <= 0.10 * height
    feet = pts[ground_band]
    if feet.size and (feet[:, 0] >= mid_x).any() and (feet[:, 0] < mid_x).any():
        foot_left = float(np.median(feet[feet[:, 0] >= mid_x], axis=0)[0])
        foot_right = float(np.median(feet[feet[:, 0] < mid_x], axis=0)[0])
    else:
        foot_left = mid_x + 0.10 * size[0]
        foot_right = mid_x - 0.10 * size[0]

    body = {
        "Hips": frac(mid_x, hips_y, mid_z),
        "Spine": frac(mid_x, hips_y + 0.33 * (shoulder_y - hips_y), mid_z),
        "Spine1": frac(mid_x, hips_y + 0.66 * (shoulder_y - hips_y), mid_z),
        "Spine2": frac(mid_x, shoulder_y, mid_z),
        "Neck": frac(mid_x, neck_y, mid_z),
        "Head": frac(mid_x, head_y, mid_z),
    }
    l_sh = frac(mid_x + torso_half, shoulder_y, mid_z)
    r_sh = frac(mid_x - torso_half, shoulder_y, mid_z)
    l_hand = frac(left_hand[0], left_hand[1], left_hand[2])
    r_hand = frac(right_hand[0], right_hand[1], right_hand[2])
    body.update({
        "LeftShoulder": l_sh, "RightShoulder": r_sh,
        "LeftArm": _lerp(l_sh, l_hand, 0.5), "LeftForeArm": _lerp(l_sh, l_hand, 0.75),
        "LeftHand": l_hand,
        "RightArm": _lerp(r_sh, r_hand, 0.5), "RightForeArm": _lerp(r_sh, r_hand, 0.75),
        "RightHand": r_hand,
    })

    # Legs: a mostly-vertical chain from the hip to the foot, knees and ankles spaced
    # along the way, the foot planted at its own x.
    for side, foot_x in (("Left", foot_left), ("Right", foot_right)):
        sign = 1.0 if side == "Left" else -1.0
        knee_y = hips_y - 0.55 * (hips_y - lo[1])
        ankle_y = lo[1] + 0.06 * height
        toe_y = lo[1] + 0.015 * height
        knee_x = mid_x + 0.55 * (foot_x - mid_x)
        body[f"{side}UpLeg"] = frac(mid_x + sign * 0.02 * size[0],
                                     hips_y - 0.20 * (hips_y - lo[1]), mid_z)
        body[f"{side}Leg"] = frac(knee_x, knee_y, mid_z)
        body[f"{side}Foot"] = frac(foot_x, ankle_y, mid_z)
        body[f"{side}ToeBase"] = frac(foot_x, toe_y, mid_z + 0.10 * size[2])
    return body


def fit_defaults(vertices: Sequence[Sequence[float]]) -> dict[str, tuple[float, float, float]]:
    """All 52 joints: the body fitted to ``vertices``, fingers fanned from the hands."""
    body = fit_body(vertices)
    joints = dict(body)
    joints.update(hand_joints(body["LeftHand"], away=+1.0, side="Left"))
    joints.update(hand_joints(body["RightHand"], away=-1.0, side="Right"))
    return joints
