"""The UltraTex texturing backend: builds the official inference command, gates
the weights, and refuses machines it cannot run on. Nothing here touches the GPU."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

from Ar3DRAHub import ultratex_backend
from Ar3DRAHub.ultratex_backend import (
    UltraTexOptions,
    build_command,
    generate_ultratex,
    lora_path,
    write_config,
)


def test_platform_refusal_on_macos(monkeypatch):
    from Ar3DRAHub import host

    monkeypatch.setattr(host, "host_platform", lambda: host.APPLE)
    monkeypatch.setattr(host, "os_family", lambda: "macos")
    refusal = ultratex_backend.platform_refusal()
    assert refusal is not None and "NVIDIA" in refusal


def test_platform_allows_windows_nvidia(monkeypatch):
    from Ar3DRAHub import host

    monkeypatch.setattr(host, "host_platform", lambda: host.NVIDIA)
    monkeypatch.setattr(host, "os_family", lambda: "windows")
    assert ultratex_backend.platform_refusal() is None


def test_verify_paths_requires_the_build_marker(monkeypatch, tmp_path):
    from Ar3DRAHub import host

    monkeypatch.setattr(host, "os_family", lambda: "windows")
    monkeypatch.setattr(host, "venv_python", lambda repo: tmp_path / ".venv" / "python.exe")
    problems = ultratex_backend.verify_paths(tmp_path)
    assert any("completed build" in p for p in problems)


def test_write_config_uses_the_papers_speed_modes_and_names_every_path(tmp_path):
    config_path = tmp_path / "cfg.json"
    config = write_config(
        config_path,
        UltraTexOptions(repo=Path("vendor/ultratex"), resolution=1024, steps=12,
                        seed=7, task="albedo"),
        views_json=Path("views.json"),
        project_dir=Path("out/albedo"),
        lora=Path("models/checkpoints/flux2/lora"),
        decoder=Path("models/checkpoints/flux2/decoder.pt"),
    )
    assert config["sparse_attention"] is False          # the vendored kernel hangs on sm_120
    assert config["inference_drop_background_tokens"] is True   # BTD prunes the sequence
    assert config["fast_empty_prompt"] is True          # embed once, free Qwen, single CFG-free forward
    assert config["resolution"] == 1024
    assert config["inference_num_steps"] == 12
    assert config["seed"] == 7
    assert config["task"] == "albedo"
    assert config["max_samples"] == 1        # one canvas per task, not four clones
    assert config["decoder_ckpt"].endswith("decoder.pt")
    lora = Path(config["resume_from_checkpoint"])
    assert (lora.parent.name, lora.name) == ("flux2", "lora")
    # JSON is what is written, so the studio needs no yaml dependency; the
    # upstream parser accepts it (JSON is valid YAML).
    assert json.loads(config_path.read_text(encoding="utf-8")) == config


def test_lora_path_routes_the_task_to_its_lora():
    root = Path("/models/checkpoints")
    assert lora_path(root, "albedo") == root / "flux2" / "lora"
    assert lora_path(root, "mr") == root / "flux2_mr" / "lora"
    with pytest.raises(ValueError, match="task must be one of"):
        lora_path(root, "normal")


def test_build_command_carries_every_flag():
    options = UltraTexOptions(repo=Path("vendor/ultratex"), resolution=1024, steps=12,
                              seed=7, task="mr")
    command = build_command(
        Path("venv/python"), Path("UltraTex/inference_flux2.py"), options,
        views_json=Path("views.json"),
        project_dir=Path("out/mr"),
        lora=Path("models/checkpoints/flux2_mr/lora"),
        decoder=Path("models/checkpoints/flux2/decoder.pt"),
    )
    assert [str(c) for c in command[:2]] == [
        str(Path("venv/python")), str(Path("UltraTex/inference_flux2.py"))
    ]
    assert command[command.index("--task") + 1] == "mr"
    assert command[command.index("--resolution") + 1] == "1024"
    assert command[command.index("--inference_num_steps") + 1] == "12"
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--guidance") + 1] == "4.0"
    assert command[command.index("--eval_data_json") + 1] == str(Path("views.json"))
    assert command[command.index("--sparse_attention") + 1] == "false"
    assert command[command.index("--inference_drop_background_tokens") + 1] == "true"
    assert command[command.index("--fast_empty_prompt") + 1] == "true"
    assert command[command.index("--max_samples") + 1] == "1"
    lora_arg = Path(command[command.index("--resume_from_checkpoint") + 1])
    assert lora_arg.parts[-2:] == ("flux2_mr", "lora")


def test_weight_sizes_are_stated():
    sizes = dict(ultratex_backend.WEIGHTS)
    assert "18.2" in sizes["black-forest-labs/FLUX.2-klein-base-9B"]
    assert "0.4" in sizes["black-forest-labs/FLUX.2-dev"]
    assert "16.4" in sizes["Qwen/Qwen3-8B"]


def _write_checkout(tree: Path) -> None:
    """Write a minimal UltraTex checkout whose files carry the patch anchors,
    so the real `ensure_speed_patch` path runs in these tests too."""
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "patch_ultratex_speed.py"
    spec = importlib.util.spec_from_file_location("patch_ultratex_speed", script)
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)
    trainer = (
        patcher.GET_MODELS + "\n\n"
        + patcher.ENCODE_SIGNATURE + "\n"
        + patcher.ENCODE_CTX + "\n\n"
        + patcher.INFERENCE_SIGNATURE + "\n"
        + patcher.ENCODE_CALL + "\n"
        + patcher.CFG_BLOCK + "\n"
        + patcher.CFG_LOOP + "\n"
    )
    runner = (
        patcher.ARGS_FLAG + "\n\n"
        + patcher.MODEL_LOAD + "\n"
        + patcher.ENCODER_EVAL + "\n\n"
        + patcher.MAIN_ANCHOR + "\n"
        + patcher.MAIN_LOAD + "\n"
        + patcher.INFERENCE_CALL + "\n"
        + patcher.MANIFEST_LINE + "\n"
    )
    for name, text in (("train_flux2.py", trainer),
                       ("train_flux2_mr.py", trainer),
                       ("inference_flux2.py", runner)):
        path = tree / "UltraTex" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _ready_install(monkeypatch, tmp_path):
    """The gates a run passes before the model work: platform, checkout, loras."""
    from Ar3DRAHub import host, models_home

    monkeypatch.setattr(host, "host_platform", lambda: host.NVIDIA)
    monkeypatch.setattr(host, "os_family", lambda: "linux")
    monkeypatch.setattr(host, "venv_python", lambda repo: tmp_path / ".venv" / "bin" / "python")
    for name in ("UltraTex/inference_flux2.py", ".ultratex-build-complete",
                 ".venv/bin/python"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    _write_checkout(tmp_path)
    root = tmp_path / "models" / "ultratex" / "checkpoints"
    for relative in ultratex_backend.LORA_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"lora")
    monkeypatch.setattr(models_home, "vendor_models", lambda route: root.parent)
    (tmp_path / "views.json").write_text("[]")
    return tmp_path / "views.json"


def test_generate_requires_all_three_base_model_repos(monkeypatch, tmp_path):
    from Ar3DRAHub import weight_gate

    views = _ready_install(monkeypatch, tmp_path)
    seen = []

    def fake_require(repo_id, **kwargs):
        seen.append((repo_id, kwargs))
        return f"/cache/{repo_id}"

    monkeypatch.setattr(weight_gate, "require_local", fake_require)

    def fake_run(command, cwd, env, check):
        target = Path(command[command.index("--project_dir") + 1])
        target.mkdir(parents=True, exist_ok=True)
        (target / "eval_test_0_result.png").write_bytes(b"png")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(ultratex_backend.subprocess, "run", fake_run)
    result = generate_ultratex(
        views, tmp_path / "out", UltraTexOptions(repo=tmp_path),
    )
    repos = {repo for repo, _ in seen}
    assert repos == {
        "black-forest-labs/FLUX.2-klein-base-9B",
        "black-forest-labs/FLUX.2-dev",
        "Qwen/Qwen3-8B",
    }
    assert result.task == "albedo"
    assert result.canvas.is_file() and result.canvas.name.endswith("_result.png")


def test_generate_pins_the_model_paths_into_the_environment(monkeypatch, tmp_path):
    from Ar3DRAHub import weight_gate

    views = _ready_install(monkeypatch, tmp_path)
    monkeypatch.setattr(weight_gate, "require_local", lambda repo_id, **kw: f"/cache/{repo_id}")

    seen = {}

    def fake_run(command, cwd, env, check):
        seen["env"] = env
        target = Path(command[command.index("--project_dir") + 1])
        target.mkdir(parents=True, exist_ok=True)
        (target / "eval_x_0_result.png").write_bytes(b"png")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(ultratex_backend.subprocess, "run", fake_run)
    generate_ultratex(views, tmp_path / "out", UltraTexOptions(repo=tmp_path))
    env = seen["env"]
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["KLEIN_9B_BASE_MODEL_PATH"].endswith("flux-2-klein-base-9b.safetensors")
    assert env["AE_MODEL_PATH"].endswith("ae.safetensors")
    assert env["QWEN3_8B_PATH"] == "/cache/Qwen/Qwen3-8B"


def test_generate_refuses_missing_lora_files(monkeypatch, tmp_path):
    from Ar3DRAHub import host, models_home

    monkeypatch.setattr(host, "host_platform", lambda: host.NVIDIA)
    monkeypatch.setattr(host, "os_family", lambda: "linux")
    monkeypatch.setattr(host, "venv_python", lambda repo: tmp_path / "python")
    for name in ("UltraTex/inference_flux2.py", ".ultratex-build-complete", "python"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    _write_checkout(tmp_path)
    monkeypatch.setattr(models_home, "vendor_models", lambda route: tmp_path / "empty")
    (tmp_path / "views.json").write_text("[]")
    with pytest.raises(RuntimeError, match="LoRA/decoder"):
        generate_ultratex(tmp_path / "views.json", tmp_path / "out",
                          UltraTexOptions(repo=tmp_path))


def test_generate_refuses_missing_checkout(monkeypatch, tmp_path):
    from Ar3DRAHub import host

    monkeypatch.setattr(host, "host_platform", lambda: host.NVIDIA)
    monkeypatch.setattr(host, "os_family", lambda: "windows")
    with pytest.raises(RuntimeError, match="bootstrap_ultratex"):
        generate_ultratex(
            tmp_path / "views.json", tmp_path, UltraTexOptions(repo=tmp_path))


def test_generate_applies_the_speed_patch_to_the_checkout(monkeypatch, tmp_path):
    from Ar3DRAHub import weight_gate

    views = _ready_install(monkeypatch, tmp_path)
    monkeypatch.setattr(weight_gate, "require_local", lambda repo_id, **kw: f"/cache/{repo_id}")
    seen = {}

    def fake_run(command, cwd, env, check):
        seen["command"] = command
        target = Path(command[command.index("--project_dir") + 1])
        target.mkdir(parents=True, exist_ok=True)
        (target / "eval_x_0_result.png").write_bytes(b"png")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(ultratex_backend.subprocess, "run", fake_run)
    generate_ultratex(views, tmp_path / "out", UltraTexOptions(repo=tmp_path))

    trainer = (tmp_path / "UltraTex" / "train_flux2.py").read_text(encoding="utf-8")
    runner = (tmp_path / "UltraTex" / "inference_flux2.py").read_text(encoding="utf-8")
    assert "use_cfg: bool = True" in trainer
    assert "def encode_empty_prompt_once(" in runner
    command = [str(part) for part in seen["command"]]
    assert command[command.index("--fast_empty_prompt") + 1] == "true"


def test_generate_finds_the_canvas_nested_under_checkpoint(monkeypatch, tmp_path):
    """Upstream nests the canvas in checkpoint-<step>/; the backend must look
    there, not only in the project root (the flat-only glob lost a finished
    run's result once the real runner wrote its first nested output)."""
    from Ar3DRAHub import weight_gate

    views = _ready_install(monkeypatch, tmp_path)
    monkeypatch.setattr(weight_gate, "require_local", lambda repo_id, **kw: f"/cache/{repo_id}")

    def fake_run(command, cwd, env, check):
        target = Path(command[command.index("--project_dir") + 1]) / "checkpoint-0"
        target.mkdir(parents=True, exist_ok=True)
        (target / "eval_x_0_result.png").write_bytes(b"png")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(ultratex_backend.subprocess, "run", fake_run)
    result = generate_ultratex(views, tmp_path / "out", UltraTexOptions(repo=tmp_path))
    assert result.canvas.name == "eval_x_0_result.png"
    assert "checkpoint-0" in result.canvas.parts


def test_generate_surfaces_the_subprocess_failure(monkeypatch, tmp_path):
    from Ar3DRAHub import weight_gate

    views = _ready_install(monkeypatch, tmp_path)
    monkeypatch.setattr(weight_gate, "require_local", lambda repo_id, **kw: f"/cache/{repo_id}")

    def fake_run(command, cwd, env, check):
        raise subprocess.CalledProcessError(3, command)

    monkeypatch.setattr(ultratex_backend.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="exit code 3"):
        generate_ultratex(views, tmp_path / "out", UltraTexOptions(repo=tmp_path))


def _pose_close(expected, actual):
    for expected_row, actual_row in zip(expected, actual):
        assert all(abs(e - a) < 1e-9 for e, a in zip(expected_row, actual_row))


def test_ring_pose_reproduces_the_four_canonical_side_poses():
    """The ring family IS the dataset's own side-view family: the diagonals
    interpolate it, they do not invent a new one."""
    for index, azimuth in enumerate((0, 90, 180, 270)):
        _pose_close(ultratex_backend.POSES[index], ultratex_backend.ring_pose(azimuth))


def test_elevated_pose_reproduces_top_and_bottom_at_ninety():
    """At 90 degrees of elevation the tilted pose is the canonical top/bottom."""
    for azimuth, canonical_index in ((0, 4), (180, 5)):
        _pose_close(ultratex_backend.POSES[canonical_index],
                    ultratex_backend.elevated_pose(azimuth, 90))


def test_the_twelve_poses_are_distinct_viewpoints_at_the_camera_distance():
    viewpoints = set()
    for pose in (*ultratex_backend.POSES, *ultratex_backend.GUIDED_POSES):
        position = (pose[0][3], pose[1][3], pose[2][3])
        assert abs(math.hypot(*position) - ultratex_backend._CAMERA_DISTANCE) < 1e-9
        viewpoints.add(tuple(round(component, 6) for component in position))
    assert len(viewpoints) == 12


def test_every_guided_pose_is_right_handed_and_looks_at_the_origin():
    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    for pose in ultratex_backend.GUIDED_POSES:
        right = (pose[0][0], pose[1][0], pose[2][0])
        up = (pose[0][1], pose[1][1], pose[2][1])
        backward = (pose[0][2], pose[1][2], pose[2][2])
        position = (pose[0][3], pose[1][3], pose[2][3])
        for a, b in zip(cross(right, up), backward):
            assert abs(a - b) < 1e-9          # right-handed frame
        assert dot(backward, position) > 0    # backward points at the origin
