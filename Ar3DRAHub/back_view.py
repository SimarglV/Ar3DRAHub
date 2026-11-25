"""The back and side pictures for a multi-view build: what the model is asked, and shown.

Two rules, found the hard way (2026-10-10 to 10-12, eight characters; 2026-11-22; 2026-11-24):

- **Name the view, not the turn.** "Turn to back view" and "Change the perspective to the
  left side view" still read to the model as "keep it roughly the same", so it mirrors
  instead of turning: the back shows the palms, the side shows the front. The models are
  trained on the plain camera phrase itself, so the prompt states the view directly —
  "Back view", "left side view", "right side view" — plus a "keep the same pose,
  proportions, size and framing" clause. A ComfyUI Flux multi-view rig with exactly this
  phrasing proved it 2026-11-24 where the turn wording had failed.

- **Name only what you want.** Naming a thing summons it, so the prompt names no body parts
  (it works for any subject, not just a character). The reference is the front's cut-out
  laid on plain light grey, so the model redraws the subject and not a backdrop.

Same pose, size and place in the frame matter as much as the drawing: Pixal3D reads the
views through cameras exactly a quarter turn apart.
"""

from __future__ import annotations

from PIL import Image

BACK_PROMPT = (
    "Back view, seen from directly behind. Keep the same pose, same proportions, "
    "same size and same framing as the reference, and the same outline, colours and "
    "materials. Flat even lighting, plain light grey background."
)


def side_prompt(side: str, with_back: bool = False, flux: bool = False) -> str:
    """The prompt for one side view: name the subject's `side` directly.

    `side` is ``"left"`` or ``"right"`` — the subject's own side, not the viewer's, so the
    picture lands on the matching turntable camera (see `scripts/pixal3d_generate.py`). The
    prompt states the view as the plain camera phrase the models are trained on ("left side
    view"), not a turn instruction ("change the perspective") that reads as "keep it roughly
    the same" and mirrors instead of turning. With `with_back`, two reference images are
    passed and named by the engine's own tags — Flux reads ``image 1`` / ``image 2``, Qwen
    reads ``<image1>`` / ``<image2>``.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', not {side!r}")
    front_ref = "image 1" if flux else "<image1>"
    back_ref = "image 2" if flux else "<image2>"
    if with_back:
        return (
            f"{front_ref} is the subject from the front, {back_ref} from behind. Draw the "
            f"{side} side view. Keep the same pose, same proportions, same size and same "
            f"framing as the references, and the same outline, colours and materials. "
            f"Flat even lighting, plain light grey background."
        )
    return (
        f"{side} side view. Keep the same pose, same proportions, same size and same "
        f"framing as the reference, and the same outline, colours and materials. "
        f"Flat even lighting, plain light grey background."
    )


# The backdrop the reference is laid on: light grey, like the backgrounds our prompts ask for.
GREY = 236


def reference_on_grey(cut_out: Image.Image) -> Image.Image:
    """The cut-out front composited on plain grey, as an RGB picture for the engine to read."""
    rgba = cut_out.convert("RGBA")
    backdrop = Image.new("RGBA", rgba.size, (GREY, GREY, GREY, 255))
    return Image.alpha_composite(backdrop, rgba).convert("RGB")
