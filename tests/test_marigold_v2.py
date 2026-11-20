"""Tests for Ar3DRAHub/marigold_v2.py: the pure logic the predictor is built on.

The heavy path (torch, diffusers, peft) is deferred inside ``Predictor`` and is
exercised manually, like every other model backend; what is tested here is the
filesystem layout, the modality table and the resolution rounding a run depends on
before it spends minutes loading the backbone.
"""

from __future__ import annotations

import pytest

from Ar3DRAHub import marigold_v2 as m


def test_get_spec_depth_variant():
    spec = m.get_spec("depth", "Log-stage2")
    assert spec.modality == "depth"
    assert spec.subfolder == "depth/Log-stage2"
    assert spec.embed_prefix == "qwen_edit_2509_qwen_depth_realimg512"
    assert spec.far_is_high is True


def test_disparity_grows_down_not_up():
    assert m.get_spec("depth", "Disparity-base").far_is_high is False


def test_get_spec_normals_and_albedo():
    assert m.get_spec("normals").subfolder == "normals"
    assert m.get_spec("normals").embed_prefix == "qwen_edit_2509_qwen_normals_dummy512"
    assert m.get_spec("albedo").subfolder == "albedo"


def test_get_spec_rejects_unknown():
    with pytest.raises(ValueError):
        m.get_spec("segmentation")
    with pytest.raises(ValueError):
        m.get_spec("depth", "nope")


def test_round_up_and_target_size():
    assert m._round_up(768) == 768
    assert m._round_up(100) == 112
    assert m.target_size(768, 768) == (768, 768)
    assert m.target_size(2000, 2000, 1024) == (1024, 1024)


def test_models_dir_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv(m.MODELS_DIR_ENV, str(tmp_path / "weights"))
    assert m.models_dir() == tmp_path / "weights"


def test_resolve_paths_layout(tmp_path):
    spec, qwen, gguf, checkpoint, embeds, mask = m.resolve_paths(tmp_path, "depth")
    assert qwen == tmp_path / "Qwen-Image-Edit-2509"
    assert gguf == tmp_path / "gguf" / m.GGUF_NAME
    assert checkpoint == tmp_path / "Marigold-V2" / "depth" / "Log-stage2" / "trainables.safetensors"
    assert embeds.name == "qwen_edit_2509_qwen_depth_realimg512_prompt_embeds.pt"
    assert spec.modality == "depth"


def test_missing_files_lists_then_clears(tmp_path):
    missing = m.missing_files(tmp_path, "depth")
    assert "Qwen-Image-Edit-2509/vae/config.json" in missing
    assert "gguf/Qwen-Image-Edit-2509-Q6_K.gguf" in missing
    for rel in missing:
        path = tmp_path.joinpath(*rel.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    assert m.missing_files(tmp_path, "depth") == []


def test_save_prediction_writes_npy_and_png(tmp_path):
    import numpy as np

    depth = np.full((16, 16), 0.5, dtype=np.float32)
    npy, png = m.save_prediction(depth, tmp_path, "stem", "depth")
    assert npy.is_file() and png.is_file()
    assert np.load(npy).shape == (16, 16)
