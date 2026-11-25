"""Contracts for the UltraTex empty-prompt fast path patch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_ultratex_speed.py"
SPEC = importlib.util.spec_from_file_location("patch_ultratex_speed", SCRIPT)
patcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patcher)


def test_replacement_fails_closed_when_upstream_anchor_changes():
    try:
        patcher.replace_exact("upstream changed", "old", "new", count=1, label="gate")
    except RuntimeError as error:
        assert "expected 1 source anchors, found 0" in str(error)
    else:
        raise AssertionError("an unknown upstream shape must not be patched silently")


def test_replacement_is_idempotent():
    source, changed = patcher.replace_exact("old", "old", "new", count=1, label="gate")
    assert (source, changed) == ("new", True)
    source, changed = patcher.replace_exact(source, "old", "new", count=1, label="gate")
    assert (source, changed) == ("new", False)


TRAINER = (
    patcher.GET_MODELS + "\n\n"
    + patcher.ENCODE_SIGNATURE + "\n"
    + patcher.ENCODE_CTX + "\n\n"
    + patcher.INFERENCE_SIGNATURE + "\n"
    + patcher.ENCODE_CALL + "\n"
    + patcher.CFG_BLOCK + "\n"
    + patcher.CFG_LOOP + "\n"
)

RUNNER = (
    patcher.ARGS_FLAG + "\n\n"
    + patcher.MODEL_LOAD + "\n"
    + patcher.ENCODER_EVAL + "\n\n"
    + patcher.MAIN_ANCHOR + "\n"
    + patcher.MAIN_LOAD + "\n"
    + patcher.INFERENCE_CALL + "\n"
    + patcher.MANIFEST_LINE + "\n"
)


def _write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return path


def test_trainer_patch_is_exact_and_idempotent(tmp_path):
    path = _write(tmp_path, "train_flux2.py", TRAINER)
    assert patcher.patch_trainer(tmp_path, "train_flux2.py") is True
    text = path.read_text(encoding="utf-8")
    assert "use_cfg: bool = True" in text
    assert "ctx_override: torch.Tensor | None = None" in text
    assert "ctx_override=ctx_override" in text
    assert "if use_cfg:" in text
    assert "x=x_in," in text                 # the single-stream forward
    assert "load_te: bool = True" in text
    assert "ctx_override is required when text_encoder is None" in text
    assert patcher.patch_trainer(tmp_path, "train_flux2.py") is False
    assert path.read_text(encoding="utf-8") == text


def test_mr_trainer_patch_matches_the_same_anchors(tmp_path):
    path = _write(tmp_path, "train_flux2_mr.py", TRAINER)
    assert patcher.patch_trainer(tmp_path, "train_flux2_mr.py") is True
    text = path.read_text(encoding="utf-8")
    assert "if use_cfg:" in text
    assert patcher.patch_trainer(tmp_path, "train_flux2_mr.py") is False


def test_runner_patch_is_exact_and_idempotent(tmp_path):
    path = _write(tmp_path, "inference_flux2.py", RUNNER)
    assert patcher.patch_runner(tmp_path) is True
    text = path.read_text(encoding="utf-8")
    assert "fast_empty_prompt: bool = False" in text
    assert "def encode_empty_prompt_once(" in text
    assert "load_te=not args.fast_empty_prompt" in text
    assert "use_cfg=not args.fast_empty_prompt" in text
    assert '"empty_prompt_fast": args.fast_empty_prompt' in text
    # The helper must release the encoder unconditionally.
    assert "del text_encoder" in text
    assert "gc.collect()" in text
    assert patcher.patch_runner(tmp_path) is False
    assert path.read_text(encoding="utf-8") == text


def test_patch_preserves_crlf_line_endings(tmp_path):
    path = _write(tmp_path, "train_flux2.py", TRAINER.replace("\n", "\r\n"))
    assert patcher.patch_trainer(tmp_path, "train_flux2.py") is True
    raw = path.read_bytes()
    assert b"\r\n" in raw
    assert b"\r\r\n" not in raw      # Windows newline translation must not double the CR
    assert b"use_cfg" in raw


def test_patch_preserves_lf_line_endings(tmp_path):
    path = _write(tmp_path, "train_flux2.py", TRAINER)
    assert patcher.patch_trainer(tmp_path, "train_flux2.py") is True
    assert b"\r" not in path.read_bytes()


def test_the_training_loop_call_is_never_touched(tmp_path):
    """The 12-space-indented encode_conditions call in the training loop must
    not receive ctx_override; only the inference() call site (8 spaces) does."""
    training_call = patcher.ENCODE_CALL.replace("        latent_hw,", "            latent_hw,") \
        .replace("    ) = encode_conditions", "            ) = encode_conditions")
    path = _write(tmp_path, "train_flux2.py", TRAINER + "\n" + training_call + "\n")
    assert patcher.patch_trainer(tmp_path, "train_flux2.py") is True
    text = path.read_text(encoding="utf-8")
    assert text.count("ctx_override=ctx_override") == 1
    assert "            ) = encode_conditions(batch, ae, text_encoder, accelerator.device)" in text


def test_is_patched_needs_both_trainers_and_the_runner(tmp_path):
    assert patcher.is_patched(tmp_path) is False
    _write(tmp_path, "train_flux2.py", TRAINER)
    patcher.patch_trainer(tmp_path, "train_flux2.py")
    assert patcher.is_patched(tmp_path) is False     # mr + runner still missing
    _write(tmp_path, "train_flux2_mr.py", TRAINER)
    patcher.patch_trainer(tmp_path, "train_flux2_mr.py")
    assert patcher.is_patched(tmp_path) is False     # runner still missing
    _write(tmp_path, "inference_flux2.py", RUNNER)
    patcher.patch_runner(tmp_path)
    assert patcher.is_patched(tmp_path) is True


def test_apply_refuses_a_non_ultratex_tree(tmp_path):
    try:
        patcher.apply(tmp_path)
    except RuntimeError as error:
        assert "not an UltraTex source tree" in str(error)
    else:
        raise AssertionError("a random directory must not be patched")


def test_the_patch_ships_the_empty_prompt_guard():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "--fast_empty_prompt requires every prompt to be empty" in source
    # The single-stream branch must stay the exact same equation as CFG
    # at guidance with identical branches: one plain model forward.
    assert "pred = model(" in source
    assert "pred_uncond + guidance * (pred_cond - pred_uncond)" in source
