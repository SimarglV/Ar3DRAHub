"""The Hi3D v1.0 refinement backend: builds the official refine command, gates the
weights, and refuses machines it cannot run on. Nothing here touches the GPU."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from Ar3DRAHub import hi3d_backend
from Ar3DRAHub.hi3d_backend import Hi3DOptions, build_refine_command, generate_hi3d


def test_refine_command_passes_every_flag():
    options = Hi3DOptions(repo=Path("vendor/hi3d"), resolution=1024, steps=12, seed=7)
    command = build_refine_command(
        Path("venv/python"), Path("demo/refine_mesh.py"),
        Path("in.png"), Path("coarse.glb"), Path("out.glb"),
        options, "/models/hi3d", "/models/dino", "/models/clip",
    )
    assert [str(c) for c in command[:2]] == [
        str(Path("venv/python")), str(Path("demo/refine_mesh.py"))
    ]
    assert command[command.index("--resolution") + 1] == "1024"
    assert command[command.index("--num-inference-steps") + 1] == "12"
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--model") + 1] == "/models/hi3d"
    assert command[command.index("--dino-model") + 1] == "/models/dino"
    assert command[command.index("--clip-model") + 1] == "/models/clip"
    assert "--local-files-only" in command
    assert "--sequential-cpu-offload" in command


def test_refine_command_omits_offload_when_disabled():
    options = Hi3DOptions(repo=Path("vendor/hi3d"), cpu_offload=False)
    command = build_refine_command(
        Path("p"), Path("d"), Path("i"), Path("c"), Path("o"),
        options, "m", "d", "c",
    )
    assert "--sequential-cpu-offload" not in command


def test_platform_refusal_on_macos(monkeypatch):
    from Ar3DRAHub import host

    monkeypatch.setattr(host, "host_platform", lambda: host.APPLE)
    monkeypatch.setattr(host, "os_family", lambda: "macos")
    refusal = hi3d_backend.platform_refusal()
    assert refusal is not None and "NVIDIA" in refusal


def test_platform_allows_windows_nvidia(monkeypatch):
    from Ar3DRAHub import host

    monkeypatch.setattr(host, "host_platform", lambda: host.NVIDIA)
    monkeypatch.setattr(host, "os_family", lambda: "windows")
    assert hi3d_backend.platform_refusal() is None


def test_verify_paths_on_windows_requires_the_sdpa_port(monkeypatch, tmp_path):
    from Ar3DRAHub import host

    monkeypatch.setattr(host, "os_family", lambda: "windows")
    problems = hi3d_backend.verify_paths(tmp_path)
    assert any("SDPA port" in p for p in problems)


def test_generate_refuses_missing_checkout(monkeypatch, tmp_path):
    from Ar3DRAHub import host

    monkeypatch.setattr(host, "host_platform", lambda: host.NVIDIA)
    monkeypatch.setattr(host, "os_family", lambda: "windows")
    with pytest.raises(RuntimeError, match="bootstrap_hi3d"):
        generate_hi3d(
            Path("in.png"), Path("coarse.glb"), tmp_path,
            Hi3DOptions(repo=tmp_path),
        )


def test_generate_refuses_missing_coarse_mesh(monkeypatch, tmp_path):
    from Ar3DRAHub import host

    monkeypatch.setattr(host, "host_platform", lambda: host.NVIDIA)
    monkeypatch.setattr(host, "os_family", lambda: "linux")
    # Build a full fake checkout so the refusal is specifically the coarse mesh.
    for name in ("Hi3Dv1.0/demo/refine_mesh.py", "Hi3Dv1.0/sparc3d/_sdpa_varlen.py",
                 ".hi3d-build-complete", ".venv/bin/python"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(host, "venv_python", lambda repo: tmp_path / ".venv" / "bin" / "python")
    with pytest.raises(ValueError, match="coarse mesh"):
        generate_hi3d(
            Path("in.png"), tmp_path / "coarse.glb", tmp_path,
            Hi3DOptions(repo=tmp_path),
        )


def test_generate_gates_all_three_weight_repos(monkeypatch, tmp_path):
    from Ar3DRAHub import host, models_home, weight_gate

    monkeypatch.setattr(host, "host_platform", lambda: host.NVIDIA)
    monkeypatch.setattr(host, "os_family", lambda: "linux")
    monkeypatch.setattr(host, "venv_python", lambda repo: tmp_path / ".venv" / "bin" / "python")
    for name in ("Hi3Dv1.0/demo/refine_mesh.py", ".hi3d-build-complete", ".venv/bin/python"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    # The materialized model folder the loader accepts: real files, no symlinks.
    model_root = tmp_path / "models" / "hi3d"
    for relative in hi3d_backend.REQUIRED_FILES:
        path = model_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    monkeypatch.setattr(models_home, "vendor_models", lambda route: model_root)
    coarse = tmp_path / "coarse.glb"
    coarse.touch()

    seen = []

    def fake_require(repo_id, **kwargs):
        seen.append((repo_id, kwargs))
        return f"/cache/{repo_id}"

    monkeypatch.setattr(weight_gate, "require_local", fake_require)
    import PIL.Image

    image = PIL.Image.new("RGBA", (8, 8), (255, 0, 0, 0))
    image.save(tmp_path / "in.png")

    def fake_run(command, cwd, env, check):
        out = tmp_path / "in_hi3d.glb"
        out.write_bytes(b"glb")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(hi3d_backend.subprocess, "run", fake_run)
    result = generate_hi3d(
        tmp_path / "in.png", coarse, tmp_path, Hi3DOptions(repo=tmp_path),
        check_quality=False,
    )
    repos = {repo for repo, _ in seen}
    assert repos == {
        "facebook/dinov2-large",
        "openai/clip-vit-large-patch14",
    }
    assert result.path.name == "in_hi3d.glb"
    assert result.resolution == 512


def test_generate_refuses_when_the_model_folder_is_missing_artifacts(monkeypatch, tmp_path):
    from Ar3DRAHub import host, models_home

    monkeypatch.setattr(host, "host_platform", lambda: host.NVIDIA)
    monkeypatch.setattr(host, "os_family", lambda: "linux")
    monkeypatch.setattr(host, "venv_python", lambda repo: tmp_path / ".venv" / "bin" / "python")
    for name in ("Hi3Dv1.0/demo/refine_mesh.py", ".hi3d-build-complete", ".venv/bin/python"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(models_home, "vendor_models",
                        lambda route: tmp_path / "models" / "hi3d")
    coarse = tmp_path / "coarse.glb"
    coarse.touch()
    with pytest.raises(SystemExit) as excinfo:
        generate_hi3d(tmp_path / "in.png", coarse, tmp_path, Hi3DOptions(repo=tmp_path))
    message = str(excinfo.value)
    assert "7.2 GB" in message and "bootstrap_hi3d" in message


def test_weight_sizes_are_stated():
    sizes = dict(hi3d_backend.WEIGHTS)
    assert "7.2" in sizes["hi3d-official/Hi3Dv1.0"]
    assert "1.2" in sizes["facebook/dinov2-large"]
    assert "1.6" in sizes["openai/clip-vit-large-patch14"]


def test_required_files_cover_the_loader_allowlist():
    from pathlib import PurePosixPath

    expected = {
        "model_index.json",
        "scheduler/scheduler_config.json",
        "vae/config.json",
        "vae/model.safetensors",
        "transformer_512/config.json",
        "transformer_512/model.safetensors",
        "transformer_1024/config.json",
        "transformer_1024/model.safetensors",
    }
    assert set(hi3d_backend.REQUIRED_FILES) == expected
    # The loader consumes files under these folders only; a nested path that is
    # not in the allowlist would trip its unexpected-symlink refusal.
    for relative in hi3d_backend.REQUIRED_FILES:
        assert PurePosixPath(relative).suffix in {".json", ".safetensors", ""}


def test_generate_surfaces_the_subprocess_failure(monkeypatch, tmp_path):
    from Ar3DRAHub import host, models_home, weight_gate

    monkeypatch.setattr(host, "host_platform", lambda: host.NVIDIA)
    monkeypatch.setattr(host, "os_family", lambda: "linux")
    monkeypatch.setattr(host, "venv_python", lambda repo: tmp_path / ".venv" / "bin" / "python")
    for name in ("Hi3Dv1.0/demo/refine_mesh.py", ".hi3d-build-complete", ".venv/bin/python"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    model_root = tmp_path / "models" / "hi3d"
    for relative in hi3d_backend.REQUIRED_FILES:
        path = model_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    monkeypatch.setattr(models_home, "vendor_models", lambda route: model_root)
    coarse = tmp_path / "coarse.glb"
    coarse.touch()
    monkeypatch.setattr(weight_gate, "require_local", lambda repo_id, **kw: f"/cache/{repo_id}")

    import PIL.Image

    image = PIL.Image.new("RGBA", (8, 8), (255, 0, 0, 0))
    image.save(tmp_path / "in.png")

    def fake_run(command, cwd, env, check):
        raise subprocess.CalledProcessError(3, command)

    monkeypatch.setattr(hi3d_backend.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="exit code 3"):
        generate_hi3d(tmp_path / "in.png", coarse, tmp_path, Hi3DOptions(repo=tmp_path))


def _ready_install(monkeypatch, tmp_path):
    """The gates a run passes before the model work: platform, checkout, weights, image."""
    from Ar3DRAHub import host, models_home, weight_gate

    monkeypatch.setattr(host, "host_platform", lambda: host.NVIDIA)
    monkeypatch.setattr(host, "os_family", lambda: "linux")
    monkeypatch.setattr(host, "venv_python", lambda repo: tmp_path / ".venv" / "bin" / "python")
    for name in ("Hi3Dv1.0/demo/refine_mesh.py", ".hi3d-build-complete", ".venv/bin/python"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    model_root = tmp_path / "models" / "hi3d"
    for relative in hi3d_backend.REQUIRED_FILES:
        path = model_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    monkeypatch.setattr(models_home, "vendor_models", lambda route: model_root)
    monkeypatch.setattr(weight_gate, "require_local", lambda repo_id, **kw: f"/cache/{repo_id}")
    import PIL.Image

    PIL.Image.new("RGBA", (8, 8), (255, 0, 0, 0)).save(tmp_path / "in.png")
    coarse = tmp_path / "coarse.glb"
    coarse.write_bytes(b"glb")
    return coarse


def test_surface_coverage_is_one_for_the_same_mesh(tmp_path):
    import trimesh

    mesh = tmp_path / "box.glb"
    trimesh.creation.box().export(mesh)
    assert hi3d_backend.surface_coverage(mesh, mesh) > 0.999


def test_surface_coverage_falls_when_regions_are_missing(tmp_path):
    import trimesh

    box = tmp_path / "box.glb"
    trimesh.creation.box().export(box)
    plane = trimesh.Trimesh(
        vertices=[[-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]],
        faces=[[0, 1, 2], [0, 2, 3]], process=False,
    )
    plane.export(tmp_path / "plane.glb")
    assert hi3d_backend.surface_coverage(box, tmp_path / "plane.glb") < 0.5


def test_a_rung_that_loses_the_model_is_replaced_by_the_next_lower_one(monkeypatch, tmp_path):
    import trimesh

    coarse = _ready_install(monkeypatch, tmp_path)
    attempts = []
    scores = {1536: 0.52, 1024: 0.99, 512: 0.99}

    def fake_run(command, repo, log, on_process):
        attempts.append(int(command[command.index("--resolution") + 1]))
        trimesh.creation.box().export(Path(command[command.index("--output") + 1]))

    monkeypatch.setattr(hi3d_backend, "_run_refinement", fake_run)
    monkeypatch.setattr(hi3d_backend, "surface_coverage",
                        lambda coarse_mesh, refined: scores[attempts[-1]])
    lines = []
    result = generate_hi3d(
        tmp_path / "in.png", coarse, tmp_path,
        Hi3DOptions(repo=tmp_path, resolution=1536), log=lines.append,
    )
    assert attempts == [1536, 1024]
    assert result.resolution == 1024
    assert any("stepping down to 1024" in line for line in lines)


def test_the_floor_fails_rather_than_shipping_a_broken_mesh(monkeypatch, tmp_path):
    import trimesh

    coarse = _ready_install(monkeypatch, tmp_path)
    attempts = []

    def fake_run(command, repo, log, on_process):
        attempts.append(int(command[command.index("--resolution") + 1]))
        trimesh.creation.box().export(Path(command[command.index("--output") + 1]))

    monkeypatch.setattr(hi3d_backend, "_run_refinement", fake_run)
    monkeypatch.setattr(hi3d_backend, "surface_coverage", lambda *args: 0.4)
    with pytest.raises(hi3d_backend.RefineQualityError, match="40%"):
        generate_hi3d(
            tmp_path / "in.png", coarse, tmp_path,
            Hi3DOptions(repo=tmp_path, resolution=1024), log=lambda line: None,
        )
    assert attempts == [1024, 512]


def test_cascade_rungs_climb_lightest_first():
    assert hi3d_backend.cascade_rungs(512) == [512]
    assert hi3d_backend.cascade_rungs(1024) == [512, 1024]
    assert hi3d_backend.cascade_rungs(1536) == [512, 1024, 1536]
    with pytest.raises(ValueError, match="cascade resolution"):
        hi3d_backend.cascade_rungs(768)


def test_a_cascade_chains_each_rung_into_the_next(monkeypatch, tmp_path):
    import trimesh

    coarse = _ready_install(monkeypatch, tmp_path)
    attempts = []

    def fake_run(command, repo, log, on_process):
        attempts.append({
            "resolution": int(command[command.index("--resolution") + 1]),
            "coarse": Path(command[command.index("--coarse-mesh") + 1]),
            "output": Path(command[command.index("--output") + 1]),
        })
        trimesh.creation.box().export(attempts[-1]["output"])

    monkeypatch.setattr(hi3d_backend, "_run_refinement", fake_run)
    result = generate_hi3d(
        tmp_path / "in.png", coarse, tmp_path,
        Hi3DOptions(repo=tmp_path, resolution=1024, cascade=True),
        check_quality=False,
    )
    assert [a["resolution"] for a in attempts] == [512, 1024]
    assert attempts[0]["coarse"] == coarse
    assert attempts[1]["coarse"] == tmp_path / "in_hi3d_512.glb"
    assert result.path == tmp_path / "in_hi3d.glb"
    assert result.resolution == 1024


def test_a_cascade_keeps_the_previous_rung_when_the_next_fails_coverage(monkeypatch, tmp_path):
    import trimesh

    coarse = _ready_install(monkeypatch, tmp_path)
    attempts = []
    scores = {512: 0.99, 1024: 0.5}

    def fake_run(command, repo, log, on_process):
        attempts.append(int(command[command.index("--resolution") + 1]))
        trimesh.creation.box().export(Path(command[command.index("--output") + 1]))

    monkeypatch.setattr(hi3d_backend, "_run_refinement", fake_run)
    monkeypatch.setattr(hi3d_backend, "surface_coverage",
                        lambda current, target: scores[attempts[-1]])
    lines = []
    result = generate_hi3d(
        tmp_path / "in.png", coarse, tmp_path,
        Hi3DOptions(repo=tmp_path, resolution=1024, cascade=True), log=lines.append,
    )
    assert attempts == [512, 1024]
    assert result.path == tmp_path / "in_hi3d_512.glb"
    assert result.resolution == 512
    assert any("keeping the 512 rung" in line for line in lines)


def test_a_cascade_refuses_when_its_first_rung_loses_the_model(monkeypatch, tmp_path):
    import trimesh

    coarse = _ready_install(monkeypatch, tmp_path)
    attempts = []

    def fake_run(command, repo, log, on_process):
        attempts.append(int(command[command.index("--resolution") + 1]))
        trimesh.creation.box().export(Path(command[command.index("--output") + 1]))

    monkeypatch.setattr(hi3d_backend, "_run_refinement", fake_run)
    monkeypatch.setattr(hi3d_backend, "surface_coverage", lambda *args: 0.4)
    with pytest.raises(hi3d_backend.RefineQualityError, match="40%"):
        generate_hi3d(
            tmp_path / "in.png", coarse, tmp_path,
            Hi3DOptions(repo=tmp_path, resolution=1024, cascade=True),
            log=lambda line: None,
        )
    assert attempts == [512]
