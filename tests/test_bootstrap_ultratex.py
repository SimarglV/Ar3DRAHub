"""The UltraTex installer: picks a route from the machine, announces before it
fetches, and never downloads without a yes. Nothing here touches the network."""

from __future__ import annotations

import io

import bootstrap_ultratex as boot
import pytest


def test_pinned_commit_is_a_full_sha():
    assert len(boot.PINNED_COMMIT) == 40


def test_weights_total_about_36_gigabytes():
    assert 35.0 <= boot.total_gb() <= 37.0


def test_weights_are_public_repositories():
    repos = {repo for repo, _, _ in boot.WEIGHTS}
    assert repos == {
        "black-forest-labs/FLUX.2-klein-base-9B",
        "black-forest-labs/FLUX.2-dev",
        "Qwen/Qwen3-8B",
    }


def test_the_official_lora_layout_is_expected_under_checkpoints():
    assert boot.LORA_FILES == (
        "flux2/lora/dit_lora.safetensors",
        "flux2_mr/lora/dit_lora.safetensors",
        "flux2/decoder.pt",
    )
    # One source of truth: the bootstrap reads the layout from the backend.
    from Ar3DRAHub.ultratex_backend import LORA_FILES as backend_files
    assert boot.LORA_FILES == backend_files


@pytest.mark.parametrize("family,platform_id", [
    ("windows", "nvidia"),
    ("linux", "nvidia"),
])
def test_route_choice(family, platform_id):
    route, why = boot.choose_route(platform_id, family)
    assert route == "cuda" and why


@pytest.mark.parametrize("family,platform_id", [
    ("macos", "apple-silicon"),
    ("windows", "amd"),
])
def test_no_route_says_why(family, platform_id):
    route, why = boot.choose_route(platform_id, family)
    assert route is None and "NVIDIA" in why


def test_the_plan_uses_cu128_and_pins_the_sparse_kernel():
    commands = boot.pip_commands("uv", boot.venv_python())
    torch_at = next(i for i, c in enumerate(commands) if any(a.startswith("torch==") for a in c))
    assert f"torch=={boot.TORCH}" in commands[torch_at]
    assert commands[torch_at][-1].endswith("/cu128")
    flat = [" ".join(c) for c in commands]
    assert not any(name in c for name in ("deepspeed", "tensorboard") for c in flat)
    assert any("triton-windows" in c for c in flat)      # the paper's sparse kernel
    assert any("modelscope" in c for c in flat)          # the ModelScope downloader
    assert any("huggingface-hub" in c for c in flat)


def test_announcement_names_backend_route_sizes_and_licence():
    text = boot.announcement("cuda", "torch cu128, dense")
    assert "UltraTex" in text
    assert "yiboz2001/UltraTex" in text
    assert "cuda" in text
    assert "36.4 GB" in text
    assert "FLUX.2-klein-base-9B" in text
    assert "Qwen3-8B" in text
    assert "Yibo-Zhang/UltraTex" in text
    assert "ModelScope" in text
    assert "MIT" in text
    assert "MIT" in text
    assert "host" in text


def test_lora_files_present_lists_what_is_still_missing(monkeypatch, tmp_path):
    from Ar3DRAHub import models_home

    root = tmp_path / "models" / "ultratex" / "checkpoints"
    monkeypatch.setattr(models_home, "vendor_models", lambda route: root.parent)
    # Nothing on disk: all three are missing, so the download must run.
    assert set(boot.lora_files_present()) == set(boot.LORA_FILES)
    # Everything on disk: nothing missing, so the download must be skipped.
    for relative in boot.LORA_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"lora")
    assert boot.lora_files_present() == []


def test_locate_modelscope_files_finds_the_official_layout(tmp_path):
    for relative, marker in (
        ("checkpoints/flux2/lora/dit_lora.safetensors", b"albedo"),
        ("checkpoints/flux2_mr/lora/dit_lora.safetensors", b"mr"),
        ("checkpoints/flux2/decoder.pt", b"decoder"),
        ("README.md", b"readme"),          # ignored, not searched
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(marker)
    found = boot.locate_modelscope_files(tmp_path)
    assert found["albedo"].read_bytes() == b"albedo"
    assert found["mr"].read_bytes() == b"mr"
    assert found["decoder"].read_bytes() == b"decoder"


def test_locate_modelscope_files_ignores_the_flux1_backbone(tmp_path):
    """The ModelScope repo ships both backbones; only the FLUX.2 files are ours."""
    for folder in ("flux1", "flux2"):
        (tmp_path / folder / "lora").mkdir(parents=True)
        (tmp_path / folder / "lora" / "dit_lora.safetensors").write_bytes(b"x")
        (tmp_path / folder / "decoder.pt").write_bytes(b"d")
    (tmp_path / "flux2_mr" / "lora").mkdir(parents=True)
    (tmp_path / "flux2_mr" / "lora" / "dit_lora.safetensors").write_bytes(b"m")
    found = boot.locate_modelscope_files(tmp_path)
    assert found["decoder"].parent.name == "flux2"
    assert "flux2_mr" in found["mr"].parts
    assert "flux1" not in found["albedo"].parts


def test_locate_modelscope_files_demands_exactly_one_decoder(tmp_path):
    (tmp_path / "checkpoints" / "flux2" / "lora").mkdir(parents=True)
    (tmp_path / "checkpoints" / "flux2" / "lora" / "dit_lora.safetensors").write_bytes(b"a")
    (tmp_path / "checkpoints" / "flux2_mr" / "lora").mkdir(parents=True)
    (tmp_path / "checkpoints" / "flux2_mr" / "lora" / "dit_lora.safetensors").write_bytes(b"b")
    with pytest.raises(RuntimeError, match="decoder.pt"):
        boot.locate_modelscope_files(tmp_path)


def test_locate_modelscope_files_needs_both_loras(tmp_path):
    (tmp_path / "checkpoints" / "flux2").mkdir(parents=True)
    (tmp_path / "checkpoints" / "flux2" / "decoder.pt").write_bytes(b"d")
    (tmp_path / "checkpoints" / "flux2" / "lora").mkdir(parents=True)
    (tmp_path / "checkpoints" / "flux2" / "lora" / "dit_lora.safetensors").write_bytes(b"a")
    with pytest.raises(RuntimeError, match="dit_lora.safetensors"):
        boot.locate_modelscope_files(tmp_path)


def test_main_refuses_when_no_route(monkeypatch, capsys):
    monkeypatch.setattr(boot, "choose_route", lambda: (None, "test reason"))
    monkeypatch.setattr(boot.models_home, "apply", lambda *a, **k: None)
    assert boot.main(["--yes"]) == 1
    assert "test reason" in capsys.readouterr().out


def test_main_refuses_download_without_a_yes(monkeypatch):
    monkeypatch.setattr(boot, "choose_route", lambda: ("cuda", "test"))
    monkeypatch.setattr(boot.models_home, "apply", lambda *a, **k: None)
    monkeypatch.setattr(boot, "install_code", lambda: (_ for _ in ()).throw(
        AssertionError("install_code must not run")))
    monkeypatch.setattr(boot, "install_weights", lambda: (_ for _ in ()).throw(
        AssertionError("install_weights must not run")))
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    assert boot.main([]) == 1
