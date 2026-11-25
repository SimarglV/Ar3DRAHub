"""Tests for Ar3DRAHub.back_view: the prompt and reference for a back or side picture."""

from __future__ import annotations

import re

import numpy as np
import pytest
from PIL import Image

from Ar3DRAHub import back_view


def test_the_back_prompt_names_only_what_it_wants():
    # Naming a thing summons it: "no face" draws a face. The prompt says what to show.
    words = set(re.findall(r"[a-z']+", back_view.BACK_PROMPT.lower()))
    assert not words & {"no", "not", "never", "without", "don't", "avoid", "hidden"}


def test_the_back_prompt_names_the_view_and_keeps_the_pose():
    prompt = back_view.BACK_PROMPT.lower()
    for phrase in ("back view", "same pose", "same size", "same framing"):
        assert phrase in prompt
    assert "turn to back view" not in prompt


def test_the_back_prompt_names_no_body_parts():
    # Naming a part summons it, even on a subject that has no such part.
    words = set(re.findall(r"[a-z']+", back_view.BACK_PROMPT.lower()))
    assert not words & {"head", "body", "neck", "shoulders", "clothing", "arms", "legs", "ears"}


def test_the_back_prompt_is_a_view_not_a_flip():
    # "Seen from behind" alone reads as a mirror; the direct view word is the fix.
    prompt = back_view.BACK_PROMPT.lower()
    assert "back view" in prompt and "directly behind" in prompt
    assert "turn to" not in prompt


def test_the_back_prompt_holds_the_front_outline():
    # Pixal3D builds whatever the back adds, so the back keeps the front's outline.
    assert "same outline" in back_view.BACK_PROMPT.lower()


def test_the_reference_is_the_cut_out_on_plain_grey():
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[:2, :, :3] = (200, 10, 10)
    rgba[:2, :, 3] = 255  # top half is the subject, bottom half see-through
    out = np.asarray(back_view.reference_on_grey(Image.fromarray(rgba, "RGBA")))
    assert out.shape == (4, 4, 3)
    assert tuple(out[0, 0]) == (200, 10, 10)
    assert tuple(out[3, 3]) == (back_view.GREY,) * 3


def test_the_side_prompt_names_only_what_it_wants():
    words = set(re.findall(r"[a-z']+", back_view.side_prompt("left").lower()))
    assert not words & {"no", "not", "never", "without", "don't", "avoid", "hidden"}


def test_the_side_prompt_names_the_view_and_keeps_the_pose():
    prompt = back_view.side_prompt("right").lower()
    for phrase in ("right side view", "same pose", "same size"):
        assert phrase in prompt
    assert "change the perspective" not in prompt


def test_the_side_prompt_names_no_body_parts():
    words = set(re.findall(r"[a-z']+", back_view.side_prompt("right").lower()))
    assert not words & {"head", "body", "neck", "shoulders", "clothing", "arms", "legs", "ears"}


def test_the_side_prompt_names_the_view_not_a_turn():
    # A turn instruction reads as "keep it the same"; the direct view name is the fix.
    prompt = back_view.side_prompt("left").lower()
    assert "left side view" in prompt
    assert "change the perspective" not in prompt


def test_the_side_prompt_holds_the_front_outline():
    assert "same outline" in back_view.side_prompt("left").lower()


def test_the_two_sides_differ_only_in_which_side():
    left, right = back_view.side_prompt("left"), back_view.side_prompt("right")
    assert left != right
    assert left.replace("left", "X") == right.replace("right", "X")


def test_the_side_prompt_refuses_anything_but_left_or_right():
    with pytest.raises(ValueError):
        back_view.side_prompt("behind")


def test_the_side_prompt_with_back_names_the_two_references():
    prompt = back_view.side_prompt("left", with_back=True)
    assert "<image1>" in prompt and "<image2>" in prompt
    assert "draw the left side view" in prompt.lower()


def test_flux_names_its_references_in_natural_language():
    prompt = back_view.side_prompt("left", with_back=True, flux=True)
    assert "image 1" in prompt and "image 2" in prompt
    assert "<image1>" not in prompt and "<image2>" not in prompt


def test_the_side_prompt_without_back_keeps_the_single_reference():
    prompt = back_view.side_prompt("left").lower()
    assert "<image1>" not in prompt and "<image2>" not in prompt
    assert "image 1" not in prompt
