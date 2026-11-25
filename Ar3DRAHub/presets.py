"""Machine presets: ready-made sets of settings, tuned to how much VRAM and RAM a machine has.

One source of truth shared by the agent (mcp_server) and the studio UI (the catalogue the
Create sheet reads), so the two never drift. Each preset sets every knob that costs memory
or time across the whole pipeline:

- the picture: which engine draws, whether a back view is drawn;
- the finish: how many triangles stay, the texture size, and the quality extras --
  Marigold de-lighting, Hi3D geometry refinement, the UltraTex texture repaint and the
  watertight rebuild;
- the motion: how many samples to pick the best from, and how hard the prompt is followed.

The ladder: minimal runs everywhere, normal is the current default, high adds the
picture-guided quality passes a 24 GB card can host, ultra turns everything on -- the
Hi3D refine at full cascade, the UltraTex repaint with the 12-view guided pass, Marigold
normals and the watertight extra -- for a 32 GB / 128 GB machine. Any knob a user or an
agent sets explicitly always wins over the preset, and the server's own guards still
apply (UltraTex and Hi3D run only on NVIDIA, UltraTex only when its 36 GB setup exists).
"""

from __future__ import annotations

from typing import Any

PRESETS: dict[str, dict[str, Any]] = {
    "minimal": {
        "label": "Minimal", "for": "VRAM 8 GB or less, RAM 32 GB or less",
        "picture_engine": "flux-light", "front_and_back": False, "finish_detail": "light",
        "texture_size": 1024, "motion_repetitions": 1, "motion_cfg_scale": 3.0,
        "marigold": False, "marigold_normals": False,
        "hi3d": False, "hi3d_resolution": 512, "hi3d_texture": False,
        "texture": "keep", "ultratex_steps": 4, "ultratex_guided": False,
        "watertight": False,
    },
    "normal": {
        "label": "Normal", "for": "VRAM 12 GB, RAM 64 GB",
        "picture_engine": "flux", "front_and_back": True, "finish_detail": "game",
        "texture_size": 2048, "motion_repetitions": 2, "motion_cfg_scale": 3.0,
        "marigold": False, "marigold_normals": False,
        "hi3d": False, "hi3d_resolution": 512, "hi3d_texture": False,
        "texture": "keep", "ultratex_steps": 4, "ultratex_guided": False,
        "watertight": False,
    },
    "high": {
        "label": "High", "for": "VRAM 24 GB, RAM 128 GB",
        "picture_engine": "flux", "front_and_back": True, "finish_detail": "more",
        "texture_size": 4096, "motion_repetitions": 3, "motion_cfg_scale": 4.0,
        "marigold": True, "marigold_normals": False,
        "hi3d": True, "hi3d_resolution": 512, "hi3d_texture": True,
        "texture": "keep", "ultratex_steps": 4, "ultratex_guided": False,
        "watertight": False,
    },
    "ultra": {
        "label": "Ultra", "for": "VRAM 32 GB, RAM 128 GB",
        "picture_engine": "flux", "front_and_back": True, "finish_detail": "high",
        "texture_size": 4096, "motion_repetitions": 4, "motion_cfg_scale": 5.0,
        "marigold": True, "marigold_normals": True,
        "hi3d": True, "hi3d_resolution": 1024, "hi3d_texture": True,
        "texture": "ultratex", "ultratex_steps": 8, "ultratex_guided": True,
        "watertight": True,
    },
}

# The same detail names in the studio's Finish control's vocabulary: the Steps panel uses a
# number for the first three ("light", "game", "more"), a name for the rest. The agent's
# finish_detail keeps the names above.
STUDIO_DETAIL = {"light": "10000", "game": "20000", "more": "40000", "high": "high"}

#: The finish-stage knobs a preset carries, in the studio's finish-settings vocabulary.
FINISH_KEYS = ("marigold", "marigold_normals", "hi3d", "hi3d_resolution", "hi3d_texture",
               "texture", "ultratex_steps", "ultratex_guided", "watertight")


def catalog_presets() -> list[dict[str, Any]]:
    """The presets as the studio's Create sheet reads them (finish detail in the studio's terms)."""
    out = []
    for name, spec in PRESETS.items():
        out.append({
            "id": name, "label": spec["label"], "for": spec["for"],
            "picture_engine": spec["picture_engine"], "front_and_back": spec["front_and_back"],
            "finish_detail": STUDIO_DETAIL.get(spec["finish_detail"], spec["finish_detail"]),
            "texture_size": spec["texture_size"],
            "motion_repetitions": spec["motion_repetitions"], "motion_cfg_scale": spec["motion_cfg_scale"],
            **{key: spec[key] for key in FINISH_KEYS},
        })
    return out
