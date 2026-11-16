"""Machine presets: ready-made sets of settings, tuned to how much VRAM and RAM a machine has.

One source of truth shared by the agent (mcp_server) and the studio UI (the catalogue the
Create sheet reads), so the two never drift. Each preset sets the knobs that cost memory the
most: which picture engine draws, whether a back view is drawn, how many triangles the finish
keeps, the texture size, and how hard the motion step works.
"""

from __future__ import annotations

from typing import Any

PRESETS: dict[str, dict[str, Any]] = {
    "minimal": {
        "label": "Minimal", "for": "VRAM 8 GB or less, RAM 32 GB or less",
        "picture_engine": "flux-light", "front_and_back": False, "finish_detail": "light",
        "texture_size": 1024, "motion_repetitions": 1, "motion_cfg_scale": 3.0,
    },
    "normal": {
        "label": "Normal", "for": "VRAM 12 GB, RAM 64 GB",
        "picture_engine": "flux", "front_and_back": True, "finish_detail": "game",
        "texture_size": 2048, "motion_repetitions": 2, "motion_cfg_scale": 3.0,
    },
    "high": {
        "label": "High", "for": "VRAM 24 GB, RAM 128 GB",
        "picture_engine": "flux", "front_and_back": True, "finish_detail": "more",
        "texture_size": 4096, "motion_repetitions": 3, "motion_cfg_scale": 4.0,
    },
    "ultra": {
        "label": "Ultra", "for": "VRAM 32 GB, RAM 128 GB",
        "picture_engine": "flux", "front_and_back": True, "finish_detail": "high",
        "texture_size": 4096, "motion_repetitions": 4, "motion_cfg_scale": 5.0,
    },
}

# The same detail names in the studio's Finish control's vocabulary: the Steps panel uses a
# number for the first three ("light", "game", "more"), a name for the rest. The agent's
# finish_detail keeps the names above.
STUDIO_DETAIL = {"light": "10000", "game": "20000", "more": "40000", "high": "high"}


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
        })
    return out
