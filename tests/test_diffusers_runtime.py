"""Tests for the diffusers picture runtime: raw checkpoints dropped into models/pictures/.

No GPU: these test the manifest's runtime detection, the command the runner is given, and the
bootstrap's torch selection — the pure parts. The generation itself needs torch + a model.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

VIEWER = Path(__file__).resolve().parents[1] / "viewer"
REPO = VIEWER.parent
sys.path.insert(0, str(REPO))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


api = _load("image_api", VIEWER / "image_api.py")
boot = _load("bootstrap_diffusers", REPO / "scripts" / "bootstrap_diffusers.py")
runner = _load("diffusers_generate", REPO / "scripts" / "diffusers_generate.py")


def _model(root: Path, diffusion: str, **extra) -> Path:
    folder = root / "my-model"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.json").write_text(json.dumps({"diffusion": diffusion, **extra}),
                                       encoding="utf-8")
    return folder


def test_a_single_file_checkpoint_is_a_diffusers_engine(tmp_path):
    folder = _model(tmp_path, "model.safetensors")
    (folder / "model.safetensors").write_bytes(b"x")
    engine = api.engine_from_folder(folder)
    assert engine.runtime == "diffusers"
    assert engine.paths == {"diffusion_model": folder / "model.safetensors"}


def test_a_ckpt_checkpoint_is_a_diffusers_engine(tmp_path):
    folder = _model(tmp_path, "model.ckpt")
    (folder / "model.ckpt").write_bytes(b"x")
    assert api.engine_from_folder(folder).runtime == "diffusers"


def test_a_diffusers_folder_is_a_diffusers_engine(tmp_path):
    (tmp_path / "my-model" / "checkpoint").mkdir(parents=True)
    (tmp_path / "my-model" / "checkpoint" / "model_index.json").write_text("{}")
    engine = api.engine_from_folder(_model(tmp_path, "checkpoint"))
    assert engine.runtime == "diffusers"
    assert engine.paths["diffusion_model"] == tmp_path / "my-model" / "checkpoint"


def test_a_gguf_model_stays_sdcpp(tmp_path):
    folder = _model(tmp_path, "model.gguf", text_encoder="clip.gguf", vae="vae.safetensors")
    for name in ("model.gguf", "clip.gguf", "vae.safetensors"):
        (folder / name).write_bytes(b"x")
    engine = api.engine_from_folder(folder)
    assert engine.runtime == "sdcpp"
    assert set(engine.paths) == {"diffusion_model", "llm", "vae"}


def test_a_manifest_can_swap_the_vae_and_text_encoder(tmp_path):
    folder = _model(tmp_path, "model.safetensors", text_encoder="te/", vae="vae/")
    (folder / "te").mkdir()
    (folder / "vae").mkdir()
    engine = api.engine_from_folder(folder)
    assert engine.runtime == "diffusers"
    assert engine.paths["text_encoder"] == folder / "te"
    assert engine.paths["vae"] == folder / "vae"


def test_a_manifest_can_name_a_second_text_encoder(tmp_path):
    folder = _model(tmp_path, "model.safetensors", text_encoder="te1/",
                    text_encoder_2="te2/", vae="vae/")
    engine = api.engine_from_folder(folder)
    assert engine.runtime == "diffusers"
    assert engine.paths["text_encoder"] == folder / "te1"
    assert engine.paths["text_encoder_2"] == folder / "te2"
    assert engine.paths["vae"] == folder / "vae"


def test_diffusers_command_passes_the_swapped_components(tmp_path):
    engine = api.engine_from_folder(_model(tmp_path, "model.safetensors",
                                           text_encoder="te/", vae="vae/"))
    weights = {"diffusion_model": tmp_path / "my-model" / "model.safetensors",
               "text_encoder": tmp_path / "my-model" / "te",
               "vae": tmp_path / "my-model" / "vae"}
    command = api.build_command("a fox", _settings(), Path("/out/x.png"), weights, engine=engine)
    assert command[command.index("--text-encoder") + 1] == str(tmp_path / "my-model" / "te")
    assert command[command.index("--vae") + 1] == str(tmp_path / "my-model" / "vae")
    assert "--text-encoder-2" not in command


def test_diffusers_command_passes_a_second_text_encoder(tmp_path):
    engine = api.engine_from_folder(_model(tmp_path, "model.safetensors", text_encoder_2="te2/"))
    weights = {"diffusion_model": tmp_path / "my-model" / "model.safetensors",
               "text_encoder_2": tmp_path / "my-model" / "te2"}
    command = api.build_command("a fox", _settings(), Path("/out/x.png"), weights, engine=engine)
    assert command[command.index("--text-encoder-2") + 1] == str(tmp_path / "my-model" / "te2")


def _settings():
    return {"width": 512, "height": 512, "steps": 20, "cfg_scale": 4.5, "seed": 7,
            "negative_prompt": ""}


def test_diffusers_command_runs_the_runner_with_the_checkpoint(tmp_path):
    engine = api.engine_from_folder(_model(tmp_path, "model.safetensors"))
    command = api.build_command("a fox", _settings(), Path("/out/x.png"),
                                {"diffusion_model": tmp_path / "my-model" / "model.safetensors"},
                                engine=engine)
    assert command[0].split(".")[-1] in ("python", "exe")   # the diffusers venv's interpreter
    assert "diffusers_generate.py" in command[1]
    assert command[command.index("--model") + 1] == str(tmp_path / "my-model" / "model.safetensors")
    assert command[command.index("-p") + 1] == "a fox"
    assert command[command.index("--steps") + 1] == "20"
    assert "--llm_vision" not in command and "--diffusion-model" not in command


def test_diffusers_command_edits_from_a_reference(tmp_path):
    engine = api.engine_from_folder(_model(tmp_path, "model.safetensors"))
    command = api.build_command("its back", _settings(), Path("/out/x.png"),
                                {"diffusion_model": tmp_path / "my-model" / "model.safetensors"},
                                engine=engine, references=[Path("/in/front.png")])
    assert command[command.index("-r") + 1] == str(Path("/in/front.png"))
    assert "--strength" in command
    assert "--mask" not in command


def test_diffusers_command_inpaints_when_given_a_mask(tmp_path):
    engine = api.engine_from_folder(_model(tmp_path, "model.safetensors"))
    command = api.build_command("a red satchel", _settings(), Path("/out/x.png"),
                                {"diffusion_model": tmp_path / "my-model" / "model.safetensors"},
                                engine=engine, init=Path("/in/pic.png"), mask=Path("/in/mask.png"))
    assert command[command.index("-r") + 1] == str(Path("/in/pic.png"))
    assert command[command.index("--mask") + 1] == str(Path("/in/mask.png"))
    assert "--strength" in command


def test_runtime_present_checks_the_right_runtime(tmp_path, monkeypatch):
    engine = api.engine_from_folder(_model(tmp_path, "model.safetensors"))
    interpreter = tmp_path / "python.exe"
    monkeypatch.setattr(api, "diffusers_python", lambda: interpreter)
    interpreter.write_bytes(b"x")
    assert api.runtime_present(engine) is True
    interpreter.unlink()
    assert api.runtime_present(engine) is False
    assert api.runtime_present(api.FLUX) is api.BINARY.exists()


def test_nvidia_gets_a_cuda_torch_and_apple_gets_pypi():
    from Ar3DRAHub.host import APPLE, NVIDIA
    index, torch, tv = boot.torch_install(NVIDIA)
    assert index and "+cu" in torch and "+cu" in tv
    index, torch, tv = boot.torch_install(APPLE)
    assert index is None and torch == "torch" and tv == "torchvision"


def test_pip_commands_put_torch_on_its_own_index():
    from Ar3DRAHub.host import NVIDIA
    commands = boot.pip_commands("uv", Path("/venv/python"), NVIDIA)
    assert "--index-url" in commands[0]
    assert any("diffusers" in c for c in commands[1])


def test_the_runner_parses_a_text_to_image_call():
    args = runner.build_parser().parse_args(
        ["--model", "/m.safetensors", "-p", "hi", "-o", "/o.png"])
    assert args.model.name == "m.safetensors"
    assert args.reference is None and args.output.name == "o.png"


def test_the_runner_parses_an_inpainting_call():
    args = runner.build_parser().parse_args(
        ["--model", "/m.safetensors", "-p", "hi", "-o", "/o.png",
         "-r", "/in/pic.png", "--mask", "/in/mask.png", "--strength", "0.9"])
    assert args.reference.name == "pic.png" and args.mask.name == "mask.png"
    assert args.strength == 0.9


def test_the_runner_parses_a_vae_and_text_encoder():
    args = runner.build_parser().parse_args(
        ["--model", "/m.safetensors", "-p", "hi", "-o", "/o.png",
         "--text-encoder", "/te", "--text-encoder-2", "/te2", "--vae", "/vae"])
    assert args.text_encoder.name == "te" and args.vae.name == "vae"
    assert args.text_encoder_2.name == "te2"
