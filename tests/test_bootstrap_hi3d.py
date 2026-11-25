"""The Hi3D v1.0 installer: picks a route from the machine, announces before it
fetches, and never downloads without a yes. Nothing here touches the network."""

from __future__ import annotations

import io

import bootstrap_hi3d as boot
import pytest


def test_pinned_commit_is_a_full_sha():
    assert len(boot.PINNED_COMMIT) == 40


def test_weights_total_about_ten_gigabytes():
    assert 9.0 <= boot.total_gb() <= 11.0


def test_weights_are_public_repositories():
    repos = {repo for repo, _, _ in boot.WEIGHTS}
    assert repos == {
        "hi3d-official/Hi3Dv1.0",
        "facebook/dinov2-large",
        "openai/clip-vit-large-patch14",
    }


def test_the_model_repo_fetches_only_the_loader_allowlist():
    from Ar3DRAHub.hi3d_backend import REQUIRED_FILES

    model = next(repo for repo, _, _ in boot.WEIGHTS
                 if repo == "hi3d-official/Hi3Dv1.0")
    patterns = next(p for repo, p, _ in boot.WEIGHTS if repo == model)
    assert sorted(patterns) == sorted(REQUIRED_FILES)


def test_materialize_copies_real_files_and_is_idempotent(tmp_path):
    snapshot = tmp_path / "snapshot"
    target = tmp_path / "models" / "hi3d"
    for relative in boot.REQUIRED_FILES:
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"payload-" + relative.encode())
    assert boot.materialize(snapshot, target) == len(boot.REQUIRED_FILES)
    for relative in boot.REQUIRED_FILES:
        assert (target / relative).is_file()
        assert (target / relative).read_bytes() == b"payload-" + relative.encode()
    # Second run: nothing to write.
    assert boot.materialize(snapshot, target) == 0


def test_materialize_rewrites_a_file_whose_size_changed(tmp_path):
    snapshot = tmp_path / "snapshot"
    target = tmp_path / "models" / "hi3d"
    for relative in boot.REQUIRED_FILES:
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"new-" + relative.encode())
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"old")
    written = boot.materialize(snapshot, target)
    assert written == len(boot.REQUIRED_FILES)
    for relative in boot.REQUIRED_FILES:
        assert (target / relative).read_bytes() == b"new-" + relative.encode()


@pytest.mark.parametrize("family,platform_id,driver,capability,expected", [
    ("linux", None, (13, 0), "90", "linux-official"),
    ("windows", None, (13, 0), "120", "windows-sdpa"),
    ("windows", None, (12, 8), "89", "windows-sdpa"),
])
def test_route_choice(family, platform_id, driver, capability, expected):
    from Ar3DRAHub import host

    if platform_id is None:
        platform_id = host.NVIDIA
    route, why = boot.choose_route(family, capability, driver, platform_id)
    assert route == expected and why


@pytest.mark.parametrize("family,platform_id,driver,capability,needle", [
    ("macos", "apple-silicon", (13, 0), None, "NVIDIA"),
    ("windows", "nvidia", (13, 0), "75", "8.0"),
    ("windows", "nvidia", (13, 0), None, "8.0"),
    ("linux", "nvidia", (12, 0), "89", "CUDA 12.4"),
    ("linux", "amd", None, None, "NVIDIA"),
])
def test_no_route_says_why(monkeypatch, family, platform_id, driver, capability, needle):
    # choose_route falls back to the real card when a probe is omitted; pin those
    # probes to known values so the test does not depend on this machine's GPU.
    from Ar3DRAHub import host

    monkeypatch.setattr(host, "compute_capability", lambda: capability)
    monkeypatch.setattr(host, "driver_cuda_version", lambda: driver)
    route, why = boot.choose_route(family, capability, driver, platform_id)
    assert route is None and needle in why


def test_windows_plan_has_no_flash_attn_and_uses_the_cu128_index():
    commands = boot.pip_commands("windows-sdpa", "uv", boot.venv_python())
    torch_at = next(i for i, c in enumerate(commands) if any(a.startswith("torch==") for a in c))
    assert commands[torch_at][-1].endswith("/cu128")
    assert f"torch=={boot.TORCH_WIN}" in commands[torch_at]
    flat = [" ".join(c) for c in commands]
    assert not any("flash-attn" in c for c in flat)
    assert any("spconv-cu126" in c for c in flat)
    assert any("triton-windows" in c for c in flat)
    # The editable install comes last and stays dependency-free: the pins above
    # are the ones this repo chose, not whatever pip resolves.
    assert commands[-1][-2:] == ["-e", str(boot.CHECKOUT)]
    assert "--no-deps" in commands[-1]


def test_linux_plan_is_the_measured_stack():
    commands = boot.pip_commands("linux-official", "uv", boot.venv_python())
    torch_at = next(i for i, c in enumerate(commands) if any(a.startswith("torch==") for a in c))
    assert f"torch=={boot.TORCH_LINUX}" in commands[torch_at]
    assert commands[torch_at][-1].endswith("/cu124")
    flat = [" ".join(c) for c in commands]
    assert any("flash-attn==2.7.4.post1" in c for c in flat)
    assert any(str(boot.CHECKOUT / "docker" / "constraints.txt") in c for c in flat)
    assert any("[inference]" in c for c in flat)


def test_unknown_route_has_no_plan():
    with pytest.raises(ValueError, match="no install plan"):
        boot.pip_commands("bogus", "uv", boot.venv_python())


def test_announcement_names_backend_route_sizes_and_licence():
    text = boot.announcement("windows-sdpa", "the SDPA port")
    assert "Hi3D v1.0" in text
    assert "windows-sdpa" in text
    assert "SDPA" in text
    assert "10.0 GB" in text
    assert "hi3d-official/Hi3Dv1.0" in text
    assert "MIT" in text
    assert "Powered by Hi3D" in text
    assert "96 GB H20" in text


def test_main_refuses_when_no_route(monkeypatch, capsys):
    monkeypatch.setattr(boot, "choose_route", lambda: (None, "test reason"))
    monkeypatch.setattr(boot.models_home, "apply", lambda *a, **k: None)
    assert boot.main(["--yes"]) == 1
    assert "test reason" in capsys.readouterr().out


def test_main_refuses_download_without_a_yes(monkeypatch):
    monkeypatch.setattr(boot, "choose_route",
                        lambda: ("windows-sdpa", "test"))
    monkeypatch.setattr(boot.models_home, "apply", lambda *a, **k: None)
    monkeypatch.setattr(boot, "install_code", lambda route: (_ for _ in ()).throw(
        AssertionError("install_code must not run")))
    monkeypatch.setattr(boot, "install_weights", lambda: (_ for _ in ()).throw(
        AssertionError("install_weights must not run")))
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    assert boot.main([]) == 1
