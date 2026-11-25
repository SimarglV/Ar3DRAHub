"""Tests for dropped picture models: a folder in models/pictures/ becomes a choosable engine.

The whole point is "drop three files and a model.json, and the model appears in the picker".
No GPU: these test the scanner, the manifest, and the direct-file weight resolution.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

VIEWER = Path(__file__).resolve().parents[1] / "viewer"
sys.path.insert(0, str(VIEWER.parent))


def _load():
    sys.path.insert(0, str(VIEWER))
    spec = importlib.util.spec_from_file_location("image_api", VIEWER / "image_api.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


api = _load()


def _model_folder(root: Path, name: str = "my-flux", label: str | None = None,
                  license: dict | None = None) -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    for filename in ("model.gguf", "clip.gguf", "vae.safetensors"):
        (folder / filename).write_bytes(b"x")
    spec = {"diffusion": "model.gguf", "text_encoder": "clip.gguf", "vae": "vae.safetensors"}
    if label is not None:
        spec["label"] = label
    if license is not None:
        spec["license"] = license
    (folder / "model.json").write_text(json.dumps(spec), encoding="utf-8")
    return folder


def test_a_dropped_folder_becomes_an_engine(tmp_path):
    folder = _model_folder(tmp_path, label="My FLUX 8B",
                           license={"name": "Apache-2.0", "url": "https://example.org/lic"})
    engine = api.engine_from_folder(folder)
    assert engine.id == "custom-my-flux"
    assert engine.label == "My FLUX 8B"
    assert engine.setup_id is None and engine.licence_key == "custom-model"
    assert engine.paths == {"diffusion_model": folder / "model.gguf",
                            "llm": folder / "clip.gguf", "vae": folder / "vae.safetensors"}
    assert engine.licence_name == "Apache-2.0" and engine.licence_url == "https://example.org/lic"


def test_a_folder_without_a_manifest_is_skipped(tmp_path):
    (tmp_path / "not-a-model").mkdir()
    assert api.engine_from_folder(tmp_path / "not-a-model") is None


def test_a_manifest_missing_a_role_is_skipped(tmp_path):
    folder = _model_folder(tmp_path)
    spec = {"diffusion": "model.gguf", "vae": "vae.safetensors"}   # no text_encoder
    (folder / "model.json").write_text(json.dumps(spec), encoding="utf-8")
    assert api.engine_from_folder(folder) is None


def test_custom_engines_scans_the_pictures_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(api.models_home, "chosen", lambda *a, **k: tmp_path)
    _model_folder(tmp_path / api.CUSTOM_PICTURE_DIR, "a-model")
    _model_folder(tmp_path / api.CUSTOM_PICTURE_DIR, "b-model")
    engines = api.custom_engines()
    assert [e.id for e in engines] == ["custom-a-model", "custom-b-model"]
    assert [e.id for e in api.all_engines()][-2:] == ["custom-a-model", "custom-b-model"]


def test_pick_engine_resolves_a_dropped_model_by_id(tmp_path, monkeypatch):
    monkeypatch.setattr(api.models_home, "chosen", lambda *a, **k: tmp_path)
    _model_folder(tmp_path / api.CUSTOM_PICTURE_DIR, "my-flux")
    assert api.pick_engine("custom-my-flux").label == "my-flux"


def test_resolve_weights_reads_the_dropped_files(tmp_path):
    folder = _model_folder(tmp_path)
    engine = api.engine_from_folder(folder)
    assert api.resolve_weights(engine)["diffusion_model"] == folder / "model.gguf"


def test_resolve_weights_reports_a_missing_dropped_file(tmp_path):
    folder = _model_folder(tmp_path)
    (folder / "vae.safetensors").unlink()
    engine = api.engine_from_folder(folder)
    with pytest.raises(api.MissingWeights) as exc:
        api.resolve_weights(engine)
    assert "vae.safetensors" in str(exc.value)


def test_weight_manifest_names_the_folder_and_file(tmp_path):
    engine = api.engine_from_folder(_model_folder(tmp_path))
    manifest = api.weight_manifest(engine)
    assert manifest["diffusion_model"]["file"] == "model.gguf"
    assert manifest["diffusion_model"]["folder"] == str(tmp_path / "my-flux")


def test_a_dropped_model_without_a_declared_license_asks_to_verify(tmp_path):
    engine = api.engine_from_folder(_model_folder(tmp_path))
    assert "verify" in engine.licence_name.lower()
    assert engine.output_rights == api.CUSTOM_OUTPUT_RIGHTS
