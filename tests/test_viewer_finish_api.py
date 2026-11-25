"""Tests for the browser finishing job.

Every setting here ends up as a subprocess argument, so the tests that matter are the ones
covering what a browser can send: out-of-range numbers, unknown keys, and wrong types. The
underlying scripts reject bad values with a SystemExit deep inside a Blender run, which
would surface to the user as "worker exited with code 1" several minutes later.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from glb_fixtures import valid_glb

MODULE = Path(__file__).resolve().parents[1] / "viewer" / "finish_api.py"


def _load():
    spec = importlib.util.spec_from_file_location("finish_api", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


finish = _load()


def test_defaults_are_used_when_nothing_is_sent():
    settings = finish.normalise_settings({})
    assert settings == finish.DEFAULT_SETTINGS
    assert settings is not finish.DEFAULT_SETTINGS  # a copy, not the shared dict


def test_an_unset_choice_keeps_its_default():
    """An agent's Advanced table leaves string choices unset, which reaches here as None;
    None must mean "the default stands", not a range error."""
    settings = finish.normalise_settings({"watertight": True, "watertight_tool": None})
    assert settings["watertight"] is True
    assert settings["watertight_tool"] == "own"


def test_client_values_override_defaults():
    settings = finish.normalise_settings({"metallic": 0.648, "roughness": 0.686, "ior": 1.4})
    assert settings["metallic"] == pytest.approx(0.648)
    assert settings["roughness"] == pytest.approx(0.686)
    assert settings["ior"] == pytest.approx(1.4)
    assert settings["faces"] == finish.DEFAULT_SETTINGS["faces"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("faces", 500), ("faces", 500000),      # blender_retopo_bake rejects both
        ("metallic", 1.5), ("roughness", -0.1),
        ("ior", 0.5), ("ior", 4.0),
        ("voxel", 0.5),                          # coarse enough to melt the subject
        ("texture_size", 8192), ("paint_steps", 0),
    ],
)
def test_out_of_range_values_are_rejected_before_a_subprocess_sees_them(key, value):
    with pytest.raises(ValueError) as caught:
        finish.normalise_settings({key: value})
    assert key in str(caught.value)


def test_a_non_numeric_value_is_rejected_with_its_key():
    with pytest.raises(ValueError) as caught:
        finish.normalise_settings({"faces": "lots"})
    assert "faces" in str(caught.value)


def test_the_watertight_tool_is_validated_against_its_own_choices():
    assert finish.normalise_settings({})["watertight_tool"] == "own"
    assert finish.normalise_settings({"watertight_tool": "external"})["watertight_tool"] == "external"
    with pytest.raises(ValueError) as caught:
        finish.normalise_settings({"watertight_tool": "wativo"})
    assert "watertight_tool" in str(caught.value)
    assert "own" in str(caught.value) and "external" in str(caught.value)


def test_the_wativo_resolution_is_one_of_the_two_measured_points():
    assert finish.normalise_settings({})["watertight_wativo_res"] == 512
    assert finish.normalise_settings({"watertight_wativo_res": 1536})["watertight_wativo_res"] == 1536
    with pytest.raises(ValueError) as caught:
        finish.normalise_settings({"watertight_wativo_res": 700})
    assert "watertight_wativo_res" in str(caught.value)
    assert "512" in str(caught.value) and "1536" in str(caught.value)


def test_each_string_setting_carries_its_own_allowed_values():
    """The enum lives beside its key, not bolted to `watertight_tool`: a second string
    setting must never be validated against WTiVo's engine names."""
    assert finish.STRING_SETTINGS == {"watertight_tool": ("own", "external"),
                                      "texture": ("keep", "ultratex")}
    assert finish.CHOICE_SETTINGS == {"watertight_wativo_res": (512, 1536)}


def test_unknown_keys_are_ignored_rather_than_forwarded():
    """A typo must not reach the command line as a stray flag."""
    settings = finish.normalise_settings({"facees": 1234, "metallic": 0.4})
    assert "facees" not in settings
    assert settings["faces"] == finish.DEFAULT_SETTINGS["faces"]
    assert settings["metallic"] == pytest.approx(0.4)


def test_integer_settings_stay_integers():
    """`--faces 40000.0` is not accepted by argparse's int type."""
    settings = finish.normalise_settings({"faces": 39999.6, "paint_res": 512.0})
    assert isinstance(settings["faces"], int)
    assert isinstance(settings["paint_res"], int)


def test_zero_voxel_is_allowed_because_it_means_skip_the_remesh():
    assert finish.normalise_settings({"voxel": 0})["voxel"] == 0


def test_hi3d_settings_are_accepted():
    settings = finish.normalise_settings({"hi3d": True, "hi3d_resolution": 1024, "hi3d_texture": True})
    assert settings["hi3d"] is True
    assert settings["hi3d_resolution"] == 1024
    assert settings["hi3d_texture"] is True


def test_hi3d_chain_settings_are_accepted():
    settings = finish.normalise_settings({
        "hi3d": True, "hi3d_cascade": False, "hi3d_remesh": False,
        "hi3d_remesh_voxel": 0.001,
    })
    assert settings["hi3d_cascade"] is False
    assert settings["hi3d_remesh"] is False
    assert settings["hi3d_remesh_voxel"] == 0.001


def test_ultratex_settings_are_accepted():
    settings = finish.normalise_settings({
        "texture": "ultratex", "ultratex_steps": 8, "ultratex_seed": 7,
    })
    assert settings["texture"] == "ultratex"
    assert settings["ultratex_steps"] == 8
    assert settings["ultratex_seed"] == 7


def test_an_unknown_texture_choice_is_refused():
    with pytest.raises(ValueError, match="texture"):
        finish.normalise_settings({"texture": "photoshop"})


def test_ultratex_is_dropped_off_nvidia():
    kept = finish.fit_to_machine({"texture": "ultratex"}, platform=finish.NVIDIA)
    assert kept["texture"] == "ultratex"
    dropped = finish.fit_to_machine({"texture": "ultratex"}, platform=finish.APPLE)
    assert dropped["texture"] == "keep"


def test_an_ultratex_run_needs_the_picture():
    job = finish.FinishJob("0" * 32, Path("/tmp/finish-job"))
    assert finish.needed_inputs(job, {"texture": "ultratex"}) == \
        [job.asset_path, job.image_path]


def test_ultratex_skips_when_not_requested():
    import finish_api

    job = finish.FinishJob("0" * 32, Path("/tmp/finish-job"))
    assert finish_api.run_ultratex(job) == (False, None)


def test_ultratex_skips_with_a_note_when_not_ready(monkeypatch):
    import finish_api

    job = finish.FinishJob("0" * 32, Path("/tmp/finish-job"))
    job.settings["texture"] = "ultratex"
    monkeypatch.setattr(finish_api, "ultratex_readiness", lambda: (False, "no card"))
    ran, note = finish_api.run_ultratex(job)
    assert ran is False and "no card" in note


def test_the_ultratex_stage_renders_denoises_and_bakes(monkeypatch, tmp_path):
    """The stage runs the view render, both denoise tasks, the bake, and replaces the
    worker's result with the textured GLB and its licence chain."""
    import finish_api
    from Ar3DRAHub import ultratex_backend

    monkeypatch.setattr(finish_api, "ultratex_readiness", lambda: (True, None))
    monkeypatch.setattr(finish_api, "host_platform", lambda: finish_api.NVIDIA)
    manager = finish_api.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    job = manager.create("knight.glb", b"glb", b"png", {"texture": "ultratex"})
    job.result_glb.parent.mkdir(parents=True, exist_ok=True)
    job.result_glb.write_bytes(valid_glb(generator="the worker"))

    commands = []
    tasks = []

    def fake_worker(job_, command, phase, message, failure):
        commands.append(command)
        if Path(command[command.index("-u") + 1]).name == "ultratex_bake_views.py":
            Path(command[5]).write_bytes(valid_glb(generator="the ultratex bake"))

    monkeypatch.setattr(finish_api, "_run_worker_process", fake_worker)

    def fake_generate(views_json, out_dir, options, log=None, on_process=None):
        tasks.append(options.task)
        canvas = Path(out_dir) / f"task-{options.task}" / f"c_{options.task}.png"
        canvas.parent.mkdir(parents=True, exist_ok=True)
        canvas.write_bytes(b"png")
        return ultratex_backend.UltraTexResult(canvas.parent, canvas, options.task)

    monkeypatch.setattr(ultratex_backend, "generate_ultratex", fake_generate)
    ran, note = finish_api.run_ultratex(job)
    assert (ran, note) == (True, None)
    names = [Path(command[command.index("-u") + 1]).name for command in commands]
    assert names == ["ultratex_views.py", "ultratex_bake_views.py"]
    assert tasks == ["albedo", "mr"]
    from glb_fixtures import read_glb

    gltf = read_glb(job.result_glb.read_bytes())[0]
    assert gltf["asset"]["generator"] == "the ultratex bake"   # the worker's GLB was replaced
    sidecar = json.loads(job.result_glb.with_suffix(".provenance.json").read_text(
        encoding="utf-8"))
    assert sidecar["operation"] == "ultratex-texture"
    name = sidecar["license"]["name"]
    assert "MIT" in name


def test_the_guided_stage_renders_twelve_views_and_bakes_both_canvases(monkeypatch, tmp_path):
    """ultratex_guided adds the 45-degree pass: a second views render, a second
    albedo denoise, and the bake receives both canvases."""
    import finish_api
    from Ar3DRAHub import ultratex_backend

    monkeypatch.setattr(finish_api, "ultratex_readiness", lambda: (True, None))
    monkeypatch.setattr(finish_api, "host_platform", lambda: finish_api.NVIDIA)
    manager = finish_api.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    job = manager.create("knight.glb", b"glb", b"png",
                         {"texture": "ultratex", "ultratex_guided": True})
    job.result_glb.parent.mkdir(parents=True, exist_ok=True)
    job.result_glb.write_bytes(valid_glb(generator="the worker"))

    commands = []
    tasks = []

    def fake_worker(job_, command, phase, message, failure):
        commands.append(command)
        if Path(command[command.index("-u") + 1]).name == "ultratex_bake_views.py":
            Path(command[5]).write_bytes(valid_glb(generator="the ultratex bake"))

    monkeypatch.setattr(finish_api, "_run_worker_process", fake_worker)

    def fake_generate(views_json, out_dir, options, log=None, on_process=None):
        tasks.append(options.task)
        canvas = Path(out_dir) / f"task-{options.task}" / f"c_{options.task}.png"
        canvas.parent.mkdir(parents=True, exist_ok=True)
        canvas.write_bytes(b"png")
        return ultratex_backend.UltraTexResult(canvas.parent, canvas, options.task)

    monkeypatch.setattr(ultratex_backend, "generate_ultratex", fake_generate)
    ran, note = finish_api.run_ultratex(job)
    assert (ran, note) == (True, None)

    views_command = commands[0]
    assert views_command[-1] == "--guided"          # the render ran in guided mode
    assert tasks == ["albedo", "albedo", "mr"]      # the guided pass reuses the albedo task
    bake_command = commands[-1]
    assert "--guided" in bake_command
    guided_canvas = Path(bake_command[bake_command.index("--guided") + 1])
    assert guided_canvas.name == "c_albedo.png"     # the second albedo canvas
    sidecar = json.loads(job.result_glb.with_suffix(".provenance.json").read_text(
        encoding="utf-8"))
    assert "12-view guided" in sidecar["component_records"][0]["purpose"]


def test_the_coherence_guards_drop_sub_knobs_without_their_parent(monkeypatch):
    monkeypatch.setattr(finish, "host_platform", lambda: finish.NVIDIA)
    # Marigold normals ride with Marigold
    assert finish.fit_to_machine({"marigold_normals": True})["marigold_normals"] is False
    assert finish.fit_to_machine({"marigold": True, "marigold_normals": True})["marigold_normals"] is True
    # fading normals where the photo painted needs the photo's paint
    assert finish.fit_to_machine({"skip_photo": True, "photo_fade_normals": True})["photo_fade_normals"] is False
    # a geometry-only Hi3D refine has no texture to repaint onto
    assert finish.fit_to_machine({"hi3d": True, "texture": "ultratex"})["texture"] == "keep"
    assert finish.fit_to_machine({"hi3d": True, "hi3d_texture": True,
                                  "texture": "ultratex"})["texture"] == "ultratex"


def test_marigold_runs_before_the_ultratex_repaint(tmp_path, monkeypatch):
    """Marigold de-lights the photo paint first; the UltraTex repaint then paints on a
    de-lit base and is never de-lit itself."""
    import finish_api

    manager = finish_api.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    job = manager.create("knight.glb", b"glb", b"png", {})
    job.image_path.parent.mkdir(parents=True, exist_ok=True)
    job.image_path.write_bytes(b"png")
    job.asset_path.parent.mkdir(parents=True, exist_ok=True)
    job.asset_path.write_bytes(b"glb")

    class Worker:
        pid = 3
        returncode = 0
        stdout = iter(["baking\n"])

        def __init__(self, *args, **kwargs):
            job.result_glb.parent.mkdir(parents=True, exist_ok=True)
            job.result_glb.write_bytes(valid_glb(generator="the finish worker"))

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(finish_api.subprocess, "Popen", Worker)
    order = []
    monkeypatch.setattr(finish_api, "run_marigold",
                        lambda job_: order.append("marigold") or (True, None))
    monkeypatch.setattr(finish_api, "run_ultratex",
                        lambda job_: order.append("ultratex") or (True, None))

    finish_api.run_job(job, manager)
    assert order == ["marigold", "ultratex"]
    assert job.status == "done"


def test_the_ultratex_stage_keeps_the_result_when_the_render_fails(monkeypatch, tmp_path):
    import finish_api

    monkeypatch.setattr(finish_api, "ultratex_readiness", lambda: (True, None))
    monkeypatch.setattr(finish_api, "host_platform", lambda: finish_api.NVIDIA)
    manager = finish_api.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    job = manager.create("knight.glb", b"glb", b"png", {"texture": "ultratex"})
    job.result_glb.parent.mkdir(parents=True, exist_ok=True)
    job.result_glb.write_bytes(valid_glb(generator="the worker"))

    def broken(job_, command, phase, message, failure):
        raise RuntimeError("the render blew up")

    monkeypatch.setattr(finish_api, "_run_worker_process", broken)
    ran, note = finish_api.run_ultratex(job)
    assert ran is False and "render blew up" in note
    from glb_fixtures import read_glb

    gltf = read_glb(job.result_glb.read_bytes())[0]
    assert gltf["asset"]["generator"] == "the worker"          # kept as it was


def test_hi3d_remesh_voxel_out_of_bounds_is_rejected():
    with pytest.raises(ValueError, match="hi3d_remesh_voxel"):
        finish.normalise_settings({"hi3d_remesh_voxel": 1.0})


def test_hi3d_texture_without_hi3d_is_refused():
    manager = finish.FinishJobManager(Path("/tmp/finish"), Path("/tmp/none"))
    with pytest.raises(ValueError, match="needs Hi3D refine"):
        manager.create("knight.glb", b"glb", b"png", {"hi3d_texture": True})


@pytest.mark.parametrize("resolution", [256, 512 + 128, 2048])
def test_a_hi3d_resolution_outside_the_three_passes_is_rejected(resolution):
    with pytest.raises(ValueError) as caught:
        finish.normalise_settings({"hi3d": True, "hi3d_resolution": resolution})
    assert "hi3d_resolution" in str(caught.value)


def test_hi3d_is_dropped_off_nvidia():
    settings = finish.fit_to_machine({"hi3d": True, "hi3d_texture": True}, platform=finish.APPLE)
    assert settings["hi3d"] is False and settings["hi3d_texture"] is False
    kept = finish.fit_to_machine({"hi3d": True, "hi3d_texture": True}, platform=finish.NVIDIA)
    assert kept["hi3d"] is True and kept["hi3d_texture"] is True


def test_hi3d_texture_without_hi3d_is_dropped_by_fit_to_machine():
    settings = finish.fit_to_machine({"hi3d_texture": True}, platform=finish.NVIDIA)
    assert settings["hi3d_texture"] is False


def test_a_hi3d_result_is_named_hi3d():
    name = finish.final_name("knight__pixal3d__20260101-010101", 20000, hi3d=True)
    assert name == "knight_hi3d.glb"


def test_a_hi3d_run_needs_the_picture():
    job = finish.FinishJob("0" * 32, Path("/tmp/finish-job"))
    assert finish.needed_inputs(job, {"hi3d": True}) == [job.asset_path, job.image_path]
    assert finish.needed_inputs(job, {"skip_photo": True, "skip_paint": True}) == [job.asset_path]


def test_the_refine_stage_remeshes_and_passes_the_cascade(monkeypatch, tmp_path):
    import finish_api
    from Ar3DRAHub import hi3d_backend

    manager = finish_api.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    job = manager.create("knight.glb", b"glb", b"png", {"hi3d": True, "hi3d_resolution": 1024})
    job.image_path.parent.mkdir(parents=True, exist_ok=True)
    job.image_path.write_bytes(b"png")
    job.asset_path.parent.mkdir(parents=True, exist_ok=True)
    job.asset_path.write_bytes(b"glb")

    remeshed = tmp_path / "remeshed.glb"
    seen = {}

    def fake_generate(image, coarse_mesh, output_dir, options, log=None, on_process=None):
        seen["coarse"] = Path(coarse_mesh)
        seen["cascade"] = options.cascade
        seen["resolution"] = options.resolution
        return hi3d_backend.Hi3DResult(remeshed, options.resolution)

    monkeypatch.setattr(hi3d_backend, "generate_hi3d", fake_generate)
    monkeypatch.setattr(finish_api, "hi3d_readiness", lambda: (True, None))
    monkeypatch.setattr(finish_api, "hi3d_remesh_stage", lambda job_: remeshed)
    result = finish_api.refine_stage(job, "Hi3D: refining the geometry at {resolution}")
    assert seen["coarse"] == remeshed
    assert seen["cascade"] is True
    assert seen["resolution"] == 1024
    assert result.resolution == 1024


def test_the_refine_stage_skips_the_remesh_when_asked(monkeypatch, tmp_path):
    import finish_api
    from Ar3DRAHub import hi3d_backend

    manager = finish_api.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    job = manager.create("knight.glb", b"glb", b"png", {"hi3d": True, "hi3d_remesh": False})
    job.image_path.parent.mkdir(parents=True, exist_ok=True)
    job.image_path.write_bytes(b"png")
    job.asset_path.parent.mkdir(parents=True, exist_ok=True)
    job.asset_path.write_bytes(b"glb")

    seen = {}

    def fake_generate(image, coarse_mesh, output_dir, options, log=None, on_process=None):
        seen["coarse"] = Path(coarse_mesh)
        return hi3d_backend.Hi3DResult(seen["coarse"], 512)

    monkeypatch.setattr(hi3d_backend, "generate_hi3d", fake_generate)
    monkeypatch.setattr(finish_api, "hi3d_readiness", lambda: (True, None))

    def no_remesh(job_):
        raise AssertionError("the remesh stage must not run when hi3d_remesh is false")

    monkeypatch.setattr(finish_api, "hi3d_remesh_stage", no_remesh)
    finish_api.refine_stage(job, "Hi3D: refining the geometry at {resolution}")
    assert seen["coarse"] == job.asset_path


def test_the_hi3d_remesh_stage_runs_blender_twice(monkeypatch, tmp_path):
    import finish_api

    manager = finish_api.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    job = manager.create("knight.glb", b"glb", b"png",
                         {"hi3d": True, "hi3d_remesh_voxel": 0.001})
    job.asset_path.parent.mkdir(parents=True, exist_ok=True)
    job.asset_path.write_bytes(b"glb")
    monkeypatch.setattr(finish_api.blender_lookup, "find_blender",
                        lambda: Path("/fake/blender.exe"))

    commands = []

    class FakeBlender:
        pid = 1
        returncode = 0
        stdout = iter([])

        def __init__(self, command, **kwargs):
            commands.append(command)
            target = Path(command[-2])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(valid_glb(generator="remesh"))

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(finish_api.subprocess, "Popen", FakeBlender)
    result = finish_api.hi3d_remesh_stage(job)
    assert len(commands) == 2
    for command in commands:
        assert command[command.index("--python") + 1].endswith("blender_voxel_remesh.py")
        assert command[-1] == "0.001"
    assert commands[1][-3] == commands[0][-2]      # the second pass eats the first's output
    assert result == job.steps_dir / "hi3d_remesh2.glb"
    assert job.events[-1]["phase"] == "hi3d-remesh"


def test_the_command_carries_every_setting():
    job = finish.FinishJob("0" * 32, Path("/tmp/finish-job"))
    settings = finish.normalise_settings({"metallic": 0.648, "faces": 39935})
    command = finish.build_command(job, settings)

    assert command[2].endswith("retopo_repaint.py")
    assert str(job.asset_path) in command
    assert str(job.image_path) in command
    assert str(job.result_glb) in command
    assert command[command.index("--metallic") + 1] == "0.648"
    assert command[command.index("--faces") + 1] == "39935"


def test_skip_flags_appear_only_when_asked():
    job = finish.FinishJob("0" * 32, Path("/tmp/finish-job"))
    plain = finish.build_command(job, finish.normalise_settings({"skip_paint": False}))
    assert "--skip-paint" not in plain
    assert "--skip-compress" not in plain

    skipped = finish.build_command(job, finish.normalise_settings({"skip_paint": True}))
    assert "--skip-paint" in skipped
    assert "--skip-bake" not in plain
    no_bake = finish.build_command(job, finish.normalise_settings({"skip_bake": True}))
    assert "--skip-bake" in no_bake


def test_the_bake_has_a_band_between_repaint_and_compress():
    bands = finish.stage_bands(["retopologise", "repaint", "bake", "compress"])
    assert bands["repaint"][1] == pytest.approx(bands["bake"][0])
    assert bands["bake"][1] == pytest.approx(bands["compress"][0])


def test_the_worker_and_the_viewer_agree_on_the_stage_order():
    """One stage plan, imported from the worker, and the resume table must follow it."""
    plan = finish.worker_module().stage_plan(False, False, photo=True)
    assert [stage for stage, _ in finish.STAGE_ARTIFACTS] == plan
    assert set(plan) == set(finish.STAGE_WEIGHTS)


def _progress(stages=("retopologise", "repaint", "bake", "compress")):
    """A tracker on a clock we drive, so the ETAs are assertable rather than wall-clock."""
    now = {"t": 0.0}
    tracker = finish.FinishProgress(list(stages), clock=lambda: now["t"])
    return tracker, now


def test_stage_lines_become_progress_events():
    tracker, _ = _progress()
    event = tracker.feed("I2L_STAGE::repaint::Repainting from fox.png at 512px")
    assert event["phase"] == "repaint"
    assert event["message"] == "Repainting from fox.png at 512px"
    # The band's floor: entering a stage is 0% of it, not some fraction of the way in.
    assert event["overall_pct"] == pytest.approx(5.0)


def test_the_bands_cover_exactly_the_stages_that_will_run():
    full = finish.stage_bands(["retopologise", "repaint", "compress"])
    assert full["retopologise"][0] == 0.0
    assert full["compress"][1] == pytest.approx(100.0)
    # The repaint is ~95% of the wall clock, so it must own most of the bar.
    low, high = full["repaint"]
    assert high - low > 80

    without_paint = finish.stage_bands(["retopologise", "compress"])
    assert "repaint" not in without_paint
    assert without_paint["compress"][1] == pytest.approx(100.0)


def test_progress_never_goes_backwards_through_a_real_run():
    tracker, now = _progress()
    lines = [
        "I2L_STAGE::retopologise::Retopologising to 40,000 faces",
        "I2L_STAGE::repaint::Repainting from source.png at 512px",
        "controls + dino ready (5s)",
        "  step 1/15 37s",
        "  step 15/15 202s",
        "views decoded (211s)",
        "super-res x4 (283s, views -> 2048px)",
        "I2L_STAGE::compress::Compressing textures to 2048px",
    ]
    percentages = []
    for index, line in enumerate(lines):
        now["t"] = float(index * 20)
        event = tracker.feed(line)
        assert event is not None, line
        percentages.append(event["overall_pct"])
    assert percentages == sorted(percentages)
    assert percentages[0] == 0.0


def test_a_late_paint_mark_cannot_drag_the_bar_back():
    tracker, now = _progress()
    tracker.feed("I2L_STAGE::repaint::Repainting")
    now["t"] = 100.0
    ahead = tracker.feed("super-res x4 (283s)")["overall_pct"]
    behind = tracker.feed("  step 1/15 37s")["overall_pct"]
    assert behind == ahead


def test_denoising_steps_are_the_only_sub_stage_progress_there_is():
    tracker, now = _progress()
    tracker.feed("I2L_STAGE::repaint::Repainting")
    now["t"] = 60.0
    event = tracker.feed("  step 6/15 95s")
    assert (event["step"], event["total"]) == (6, 15)
    assert 0 < event["stage_pct"] < 100
    assert event["stage_eta_seconds"] > 0
    assert event["total_eta_seconds"] > 0


def test_a_stage_with_no_sub_progress_claims_none():
    # A row frozen at "0%" reads as a stall; retopology and compression report no
    # percentage at all rather than a fake one.
    tracker, _ = _progress()
    event = tracker.feed("I2L_STAGE::retopologise::Retopologising")
    assert "stage_pct" not in event
    assert "stage_eta_seconds" not in event


def test_the_workers_own_done_marker_is_not_the_jobs_done():
    # The bug this replaces: the worker's terminal marker reached the browser as a
    # phase-"done" event with no result_url, the browser closed its stream on it, and the
    # real completion event -- the only one carrying the artifact URLs -- never arrived.
    tracker, _ = _progress()
    tracker.feed("I2L_STAGE::compress::Compressing")
    assert tracker.feed("I2L_STAGE::done::Finished at 4.8 MB") is None
    assert tracker.feed("I2L_STAGE::done::Finished at 4.8 MB :: all good") is None


def test_an_eta_is_withheld_until_there_is_evidence_for_one():
    assert finish.remaining_seconds(elapsed=0.0, fraction=0.5) is None
    assert finish.remaining_seconds(elapsed=50.0, fraction=0.0) is None
    assert finish.remaining_seconds(elapsed=50.0, fraction=1.0) is None
    assert finish.remaining_seconds(elapsed=50.0, fraction=0.5) == 50.0


def test_ordinary_output_is_not_mistaken_for_a_stage():
    tracker, _ = _progress()
    assert tracker.feed("[compress] 32.0 -> 4.8 MB") is None
    assert tracker.feed("Blender quit") is None
    # A paint-shaped line outside the paint stage is still not progress.
    assert tracker.feed("  step 3/15 62s") is None


def test_a_message_containing_the_separator_survives():
    tracker, _ = _progress()
    event = tracker.feed("I2L_STAGE::compress::Compressing to 2048px :: fast path")
    assert event["message"] == "Compressing to 2048px :: fast path"


def test_resume_is_asked_for_explicitly():
    job = finish.FinishJob("0" * 32, Path("/tmp/finish-job"))
    settings = finish.normalise_settings({})
    assert "--resume" not in finish.build_command(job, settings)
    assert "--resume" in finish.build_command(job, settings, resume=True)


def _run_dir(root: Path, name: str, files: dict[str, str]) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    for filename, contents in files.items():
        (directory / filename).write_text(contents)
    return directory


def test_a_finished_run_is_described_from_disk_alone(tmp_path):
    # The job registry lives in the server's memory and dies with it; the directory is
    # what makes a finished run recoverable afterwards.
    directory = _run_dir(tmp_path, "fox__finish__20260921-122518", {
        "source.glb": "g", "source.png": "p", "settings.json": "{}",
        "result_retopo.glb": "r", "result_painted.glb": "a", "result.glb": "done",
        "result.retopo-repaint.json": '{"seconds": {"repaint": 324.5}}',
    })
    run = finish.describe_run(directory)
    assert run["finished"] is True
    assert run["stages_complete"] == ["retopologise", "repaint", "compress"]
    assert run["seconds"] == {"repaint": 324.5}
    assert run["resumable"] is False  # nothing left to resume


def test_a_run_that_died_after_the_repaint_is_resumable(tmp_path):
    directory = _run_dir(tmp_path, "fox__finish__20260921-122519", {
        "source.glb": "g", "source.png": "p", "settings.json": "{}",
        "result_retopo.glb": "r", "result_painted.glb": "a",
    })
    run = finish.describe_run(directory)
    assert run["stages_complete"] == ["retopologise", "repaint"]
    assert run["resumable"] is True
    assert run["result_url"] is None


def test_a_run_url_is_one_the_static_handler_can_actually_serve():
    # describe_run's links go straight into an <a href>, so they have to be repo-relative
    # paths the viewer's own file handler resolves -- not absolute filesystem paths.
    inside = finish.OUTPUT_ROOT / "fox__finish__20260921-122518" / "result.glb"
    assert finish.served_url(inside) == (
        "/output/finish/fox__finish__20260921-122518/result.glb"
    )
    assert finish.served_url(Path("/etc/passwd")) is None


def test_a_run_without_its_settings_is_not_offered_for_resume(tmp_path):
    # Resuming means reusing intermediates, which is only faithful with the settings that
    # produced them -- a directory from before settings.json cannot promise that.
    directory = _run_dir(tmp_path, "fox__finish__20260921-122520", {
        "source.glb": "g", "source.png": "p", "result_retopo.glb": "r",
    })
    assert finish.describe_run(directory)["resumable"] is False


def test_runs_are_listed_newest_first_and_strangers_are_ignored(tmp_path):
    import os
    for index, name in enumerate(["a__finish__20260921-120000", "b__finish__20260921-130000"]):
        directory = _run_dir(tmp_path, name, {"source.glb": "g"})
        os.utime(directory, (1000 + index, 1000 + index))
    (tmp_path / "not-a-run").mkdir()
    (tmp_path / "scratch__notfinish__20260921-120000").mkdir()
    listed = [run["directory"] for run in finish.list_runs(tmp_path)]
    assert listed == ["b__finish__20260921-130000", "a__finish__20260921-120000"]


@pytest.mark.parametrize("name", [
    "../../etc", "fox__finish__20260921-122518/../..", "nope", "", "fox__finish__2026",
])
def test_a_resume_name_that_is_not_a_run_directory_is_refused(tmp_path, name):
    with pytest.raises(RuntimeError):
        finish._run_directory(tmp_path, name)


def test_adopting_a_directory_reuses_its_recorded_settings(tmp_path):
    _run_dir(tmp_path, "fox__finish__20260921-122518", {
        "source.glb": "g", "source.png": "p",
        "settings.json": '{"faces": 12000, "paint_steps": 8}',
        "result_retopo.glb": "r",
    })
    manager = finish.FinishJobManager(tmp_path)
    job = manager.adopt("fox__finish__20260921-122518")
    assert job.resume is True
    assert job.settings["faces"] == 12000
    assert job.settings["paint_steps"] == 8
    assert "--resume" in finish.build_command(job, job.settings, resume=job.resume)


def test_adopting_a_directory_that_lost_its_source_is_refused(tmp_path):
    _run_dir(tmp_path, "fox__finish__20260921-122518", {"settings.json": "{}"})
    with pytest.raises(RuntimeError, match="source.glb"):
        finish.FinishJobManager(tmp_path).adopt("fox__finish__20260921-122518")


def test_job_ids_are_validated_before_lookup():
    manager = finish.FinishJobManager(Path("/tmp/finish-root"))
    assert manager.get("../../etc/passwd") is None
    assert manager.get("not-a-job-id") is None
    assert manager.get("0" * 32) is None  # well-formed but unknown


# --- The photo stage: match the upload to its Pixal3D run, copy its camera in -------------


def _pixal3d_run(root, glb_bytes=b"glb-bytes", record_name="robot__pixal3d__1.json"):
    import hashlib
    run = root / "seb" / "robot__pixal3d__1"
    views = run / "robot__pixal3d__1.svviews"
    views.mkdir(parents=True)
    (views / "transforms.json").write_text('{"frames": []}')
    glb = run / "robot__pixal3d__1.glb"
    glb.write_bytes(glb_bytes)
    (run / record_name).write_text(json.dumps({
        "backend": "pixal3d",
        "output": {"path": str(glb), "sha256": hashlib.sha256(glb_bytes).hexdigest()},
    }))
    return views


@pytest.mark.parametrize("record_name", ["robot__pixal3d__1.json", "robot__pixal3d__1.provenance.json"])
def test_an_uploaded_pixal3d_model_finds_its_camera(tmp_path, record_name):
    views = _pixal3d_run(tmp_path, record_name=record_name)
    assert finish.find_source_views(b"glb-bytes", tmp_path) == views


def test_a_run_with_both_record_names_is_found_once(tmp_path):
    """rglob("*.json") matches both; a run must not be offered twice."""
    views = _pixal3d_run(tmp_path)
    run = views.parent
    (run / "robot__pixal3d__1.provenance.json").write_text(
        (run / "robot__pixal3d__1.json").read_text())
    assert len(list(finish._matching_runs(b"glb-bytes", tmp_path))) == 1
    assert finish.find_source_views(b"glb-bytes", tmp_path) == views


def test_a_model_with_no_matching_run_gets_no_camera(tmp_path):
    _pixal3d_run(tmp_path)
    assert finish.find_source_views(b"edited-or-foreign", tmp_path) is None


def test_the_camera_is_copied_into_the_run_and_passed_to_the_worker(tmp_path):
    _pixal3d_run(tmp_path / "generated")
    manager = finish.FinishJobManager(tmp_path / "finish", tmp_path / "generated")
    job = manager.create("robot.glb", b"glb-bytes", b"png", {})
    assert (job.views_dir / "transforms.json").is_file()
    command = finish.build_command(job, job.settings)
    assert command[command.index("--views") + 1] == str(job.views_dir)


def test_skipping_the_photo_leaves_the_camera_out(tmp_path):
    _pixal3d_run(tmp_path / "generated")
    manager = finish.FinishJobManager(tmp_path / "finish", tmp_path / "generated")
    job = manager.create("robot.glb", b"glb-bytes", b"png", {"skip_photo": True})
    assert not job.views_dir.exists()
    assert "--views" not in finish.build_command(job, job.settings)


def test_a_finished_run_offers_its_source_for_comparison(tmp_path):
    run = tmp_path / "robot__finish__20260928-120000"
    run.mkdir()
    for name in ("source.glb", "source.png", "result.glb"):
        (run / name).write_bytes(b"x")
    described = finish.describe_run(run)
    assert described["source_url"] is None or described["source_url"].endswith("source.glb")


# --- Run folder layout (0.3.5): the finished model is the only GLB at the top ----------

@pytest.mark.parametrize("faces, label", [
    (5000, "5k"), (40000, "40k"), (12500, "12.5k"), (800, "800"), (1000, "1k"),
])
def test_face_counts_are_named_the_way_people_say_them(faces, label):
    assert finish.faces_label(faces) == label


def test_the_final_model_is_named_after_the_asset_and_its_faces():
    assert finish.final_name("vanguard__pixal3d__finish__20260929-101010", 5000) == "vanguard_5k.glb"
    assert finish.final_name("fox__finish__20260921-122518", 40000) == "fox_40k.glb"
    assert finish.final_name("fox__finish__20260921-122518", 20000, full=True) == "fox_full.glb"


def test_full_detail_skips_the_repaint_and_bake_and_tells_the_worker():
    settings = finish.normalise_settings({"keep_mesh": True, "skip_paint": False})
    assert settings["skip_paint"] and settings["skip_bake"]
    job = finish.FinishJob("0" * 32, Path("/tmp/fox__finish__20260921-122518"))
    job.settings = settings
    command = finish.build_command(job, settings)
    assert "--keep-mesh" in command and job.result_glb.name == "fox_full.glb"
    assert "--keep-mesh" not in finish.build_command(job, finish.normalise_settings({}))


def test_a_new_run_keeps_inputs_and_steps_out_of_the_top_folder(tmp_path):
    job = finish.FinishJob("0" * 32, tmp_path / "vanguard__pixal3d__finish__20260929-101010")
    job.settings = finish.normalise_settings({"faces": 5000})

    assert not job.legacy
    assert job.result_glb.name == "vanguard_5k.glb"
    assert job.result_glb.parent == job.directory
    assert job.record_path.name == "vanguard_5k.retopo-repaint.json"
    assert job.asset_path == job.directory / "input" / "source.glb"
    assert job.image_path == job.directory / "input" / "source.png"
    assert job.views_dir == job.directory / "input" / "source.views"
    assert job.settings_path == job.directory / "input" / "settings.json"
    assert job.log_path == job.directory / "steps" / "run.log"
    assert job.stage_artifact("retopo") == job.directory / "steps" / "1_retopo.glb"
    assert job.stage_artifact("baked") == job.directory / "steps" / "4_baked.glb"
    assert job.stage_artifact(None) == job.result_glb


def test_a_new_run_tells_the_worker_where_the_steps_go(tmp_path):
    job = finish.FinishJob("0" * 32, tmp_path / "fox__finish__20260929-101010")
    command = finish.build_command(job, finish.normalise_settings({}))
    assert command[command.index("--steps-dir") + 1] == str(job.directory / "steps")


def test_an_old_flat_run_keeps_its_old_names_and_no_steps_folder(tmp_path):
    directory = _run_dir(tmp_path, "fox__finish__20260921-122518", {"source.glb": "g"})
    job = finish.FinishJob("0" * 32, directory)
    assert job.legacy
    assert job.result_glb == directory / "result.glb"
    assert job.record_path == directory / "result.retopo-repaint.json"
    assert job.stage_artifact("painted") == directory / "result_painted.glb"
    assert "--steps-dir" not in finish.build_command(job, finish.normalise_settings({}))


def test_a_new_run_is_described_and_resumable_from_disk(tmp_path):
    name = "vanguard__pixal3d__finish__20260929-101010"
    directory = tmp_path / name
    for relative, contents in {
        "input/source.glb": "g", "input/source.png": "p",
        "input/settings.json": json.dumps({"faces": 5000}),
        "steps/1_retopo.glb": "r", "steps/2_painted.glb": "a",
    }.items():
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        (directory / relative).write_text(contents)

    described = finish.describe_run(directory)
    assert described["stages_complete"] == ["retopologise", "repaint"]
    assert described["resumable"] is True
    assert described["finished"] is False

    (directory / "vanguard_5k.glb").write_text("done")
    described = finish.describe_run(directory)
    assert described["finished"] is True
    assert described["resumable"] is False
    assert described["result_url"] is None or described["result_url"].endswith("/vanguard_5k.glb")


def test_create_writes_inputs_into_the_input_folder(tmp_path):
    manager = finish.FinishJobManager(output_root=tmp_path, generated_root=tmp_path / "none")
    job = manager.create("vanguard__pixal3d.glb", b"glb", b"png", {"faces": 5000})
    assert (job.directory / "input" / "source.glb").read_bytes() == b"glb"
    assert (job.directory / "input" / "source.png").read_bytes() == b"png"
    assert (job.directory / "input" / "settings.json").is_file()
    assert job.result_glb.name == "vanguard_5k.glb"
    assert [p.name for p in job.directory.iterdir()] == ["input"]
    # And it resumes: the adopted job finds the same files.
    manager.finish(job)
    adopted = manager.adopt(job.directory.name)
    assert adopted.asset_path == job.asset_path
    assert adopted.result_glb == job.result_glb


# --- Finish on machines without MLX or without Blender --------------------------------

def test_the_repaint_is_forced_off_away_from_apple_silicon():
    settings = finish.normalise_settings({"skip_paint": False})
    assert finish.fit_to_machine(settings, platform="nvidia")["skip_paint"] is True
    assert finish.fit_to_machine(settings, platform="apple-silicon")["skip_paint"] is False


def test_capabilities_say_what_this_machine_can_do(tmp_path):
    exe = tmp_path / "blender"
    exe.write_text("")
    mac = finish.capabilities("apple-silicon", find=lambda: exe, version=lambda _: (5, 2))
    assert mac["repaint"] and mac["ready"] and mac["repaint_note"] is None
    assert mac["blender_version"] == "5.2" and mac["blender_problem"] is None

    nvidia = finish.capabilities("nvidia", find=lambda: exe, version=lambda _: (5, 2))
    assert not nvidia["repaint"] and nvidia["ready"]
    assert "front exact" in nvidia["repaint_note"]


def test_capabilities_report_the_ultratex_availability(tmp_path, monkeypatch):
    exe = tmp_path / "blender"
    exe.write_text("")
    monkeypatch.setattr(finish, "ultratex_readiness", lambda: (True, None))
    ready = finish.capabilities("nvidia", find=lambda: exe, version=lambda _: (5, 2))
    assert ready["ultratex"] is True and ready["ultratex_note"] is None
    monkeypatch.setattr(finish, "ultratex_readiness", lambda: (False, "needs the setup"))
    missing = finish.capabilities("nvidia", find=lambda: exe, version=lambda _: (5, 2))
    assert missing["ultratex"] is False and missing["ultratex_note"] == "needs the setup"


def test_capabilities_explain_a_missing_or_old_blender(tmp_path):
    missing = finish.capabilities("nvidia", find=lambda: None, version=lambda _: None)
    assert not missing["ready"] and "blender.org" in missing["blender_problem"]
    exe = tmp_path / "blender"
    exe.write_text("")
    old = finish.capabilities("nvidia", find=lambda: exe, version=lambda _: (3, 6))
    assert old["ready"] and "older than" in old["blender_problem"]


def test_an_nvidia_run_never_asks_the_worker_for_a_repaint(tmp_path, monkeypatch):
    monkeypatch.setattr(finish, "host_platform", lambda: "nvidia")
    manager = finish.FinishJobManager(output_root=tmp_path, generated_root=tmp_path / "none")
    job = manager.create("fox.glb", b"glb", b"png", {"skip_paint": False})
    assert job.settings["skip_paint"] is True
    assert "--skip-paint" in finish.build_command(job, job.settings)


def test_repaint_is_opt_in_so_finish_never_needs_the_hunyuan_paint_weights():
    """Quick finish by default (0.3.5): Pixal3D models arrive painted, Pixel Match keeps the
    front exact, and nobody should download 8.7 GB of Hunyuan paint to finish one."""
    assert finish.DEFAULT_SETTINGS["skip_paint"] is True
    assert finish.normalise_settings({})["skip_paint"] is True
    assert finish.normalise_settings({"skip_paint": False})["skip_paint"] is False


def test_a_skipped_pixel_match_is_said_plainly_and_stays_on_screen(tmp_path):
    # Seen on a real NVIDIA pod: Pixel Match stayed ticked, no stage ran, and the only word
    # about it was a start-up line replaced a second later. The note rides on the finished
    # result, and says what Pixel Match needs.
    manager = finish.FinishJobManager(tmp_path / "finish", tmp_path / "generated")
    job = manager.create("robot.glb", b"some-other-model", b"png", {})
    note = finish.photo_note(job, job.settings)
    assert note and "Pixel Match" in note and "skipped" in note and "Pixal3D" in note
    assert finish.photo_note(job, {**job.settings, "skip_photo": True}) is None


def test_no_note_when_the_camera_is_found(tmp_path):
    _pixal3d_run(tmp_path / "generated")
    manager = finish.FinishJobManager(tmp_path / "finish", tmp_path / "generated")
    job = manager.create("robot.glb", b"glb-bytes", b"png", {})
    assert finish.photo_note(job, job.settings) is None


def test_capabilities_offer_a_blender_install_only_where_one_can_run():
    # Linux x86_64 with no Blender: the Setup card gets an Install Blender button.
    linux = finish.capabilities("nvidia", find=lambda: None, version=lambda _: None,
                                installable=lambda: True)
    assert linux["blender_installable"] is True
    mac = finish.capabilities("apple-silicon", find=lambda: None, version=lambda _: None,
                              installable=lambda: False)
    assert mac["blender_installable"] is False


def test_no_install_offered_once_blender_is_found(tmp_path):
    exe = tmp_path / "blender"
    exe.write_text("")
    found = finish.capabilities("nvidia", find=lambda: exe, version=lambda _: (4, 2),
                                installable=lambda: True)
    assert found["blender_installable"] is False


def test_a_finished_model_is_named_by_its_path_in_output(tmp_path, monkeypatch):
    # Create hands this to the next step; the done event's result_url is a download link, not a path
    monkeypatch.setattr(finish, "OUTPUT_ROOT", tmp_path / "output" / "finish")
    glb = tmp_path / "output" / "finish" / "knight__finish__20261006-220000" / "knight_40k.glb"
    assert finish.library_path(glb) == "finish/knight__finish__20261006-220000/knight_40k.glb"
    assert finish.library_path(tmp_path / "elsewhere.glb") is None


def test_a_run_without_a_picture_finishes_the_mesh_alone(tmp_path):
    manager = finish.FinishJobManager(output_root=tmp_path, generated_root=tmp_path / "none")
    job = manager.create("upload.glb", b"glb", None, {"faces": 10000, "skip_photo": False})
    assert job.settings["skip_photo"] is True and job.settings["skip_paint"] is True
    assert not job.image_path.exists()
    command = finish.build_command(job, job.settings)
    assert "--skip-paint" in command and "--views" not in command
    # and a run that died halfway is still offered for resume, and adopts cleanly
    assert finish.describe_run(job.directory)["resumable"] is True
    manager.finish(job)
    assert manager.adopt(job.directory.name).asset_path == job.asset_path


def test_a_run_with_pixel_match_still_needs_its_picture(tmp_path):
    job = finish.FinishJob("", tmp_path)
    assert finish.needed_inputs(job, {"skip_photo": False}) == [job.asset_path, job.image_path]
    assert finish.needed_inputs(job, {"skip_photo": True, "skip_paint": False}) == [job.asset_path, job.image_path]
    assert finish.needed_inputs(job, {"skip_photo": True, "skip_paint": True}) == [job.asset_path]


def test_turning_edge_fade_off_tells_the_worker():
    job = finish.FinishJob("0" * 32, Path("/tmp/fox__finish__20260921-122518"))
    assert finish.normalise_settings({})["skip_edge_fade"] is False
    assert "--no-edge-fade" not in finish.build_command(job, finish.normalise_settings({}))
    assert "--no-edge-fade" in finish.build_command(job, finish.normalise_settings({"skip_edge_fade": True}))


def test_turning_nudge_off_tells_the_worker():
    job = finish.FinishJob("0" * 32, Path("/tmp/fox__finish__20260921-122518"))
    assert finish.normalise_settings({})["skip_nudge"] is False
    assert "--no-nudge" not in finish.build_command(job, finish.normalise_settings({}))
    assert "--no-nudge" in finish.build_command(job, finish.normalise_settings({"skip_nudge": True}))


def test_fading_normals_where_pixel_match_painted_is_opt_in():
    job = finish.FinishJob("0" * 32, Path("/tmp/fox__finish__20260921-122518"))
    assert finish.normalise_settings({})["photo_fade_normals"] is False
    assert "--fade-normals" not in finish.build_command(job, finish.normalise_settings({}))
    assert "--fade-normals" in finish.build_command(
        job, finish.normalise_settings({"photo_fade_normals": True}))


def test_marigold_normals_are_opt_in_too():
    job = finish.FinishJob("0" * 32, Path("/tmp/fox__finish__20260921-122518"))
    assert finish.normalise_settings({})["marigold_normals"] is False
    assert "--no-normals" in finish.marigold_command(job)                    # default: de-light only
    job.settings["marigold_normals"] = True
    assert "--no-normals" not in finish.marigold_command(job)


# --- The Pixal3D route: link the run's clean-up dump in and hand it to the worker -------------


def _with_dump(views):
    """Give a `_pixal3d_run` the dump and record line pixal3d_generate.py writes."""
    run = views.parent
    dump = run / "robot__pixal3d__1.post.bin"
    dump.write_bytes(b"dump")
    record_path = next(path for path in (run / "robot__pixal3d__1.provenance.json",
                                         run / "robot__pixal3d__1.json") if path.is_file())
    record = json.loads(record_path.read_text())
    record["post_dump"] = dump.name
    record_path.write_text(json.dumps(record))
    return dump


@pytest.mark.parametrize("record_name", ["robot__pixal3d__1.json", "robot__pixal3d__1.provenance.json"])
def test_the_dump_is_linked_in_and_passed_to_the_worker(tmp_path, record_name):
    dump = _with_dump(_pixal3d_run(tmp_path / "generated", record_name=record_name))
    manager = finish.FinishJobManager(tmp_path / "finish", tmp_path / "generated")
    job = manager.create("robot.glb", b"glb-bytes", b"png", {})
    assert job.dump_path.read_bytes() == b"dump"
    assert job.dump_path.stat().st_ino == dump.stat().st_ino   # a link, not a 400 MB copy
    command = finish.build_command(job, job.settings)
    assert command[command.index("--native-dump") + 1] == str(job.dump_path)


def test_the_dump_is_found_even_with_pixel_match_off(tmp_path):
    """The route needs the dump, not the pictures."""
    _with_dump(_pixal3d_run(tmp_path / "generated"))
    manager = finish.FinishJobManager(tmp_path / "finish", tmp_path / "generated")
    job = manager.create("robot.glb", b"glb-bytes", b"png", {"skip_photo": True})
    assert "--native-dump" in finish.build_command(job, job.settings)


def test_full_detail_never_takes_the_pixal3d_route(tmp_path):
    _with_dump(_pixal3d_run(tmp_path / "generated"))
    manager = finish.FinishJobManager(tmp_path / "finish", tmp_path / "generated")
    job = manager.create("robot.glb", b"glb-bytes", b"png", {"keep_mesh": True})
    assert "--native-dump" not in finish.build_command(job, job.settings)


def test_a_run_from_before_the_dump_takes_the_classic_route(tmp_path):
    _pixal3d_run(tmp_path / "generated")
    manager = finish.FinishJobManager(tmp_path / "finish", tmp_path / "generated")
    job = manager.create("robot.glb", b"glb-bytes", b"png", {})
    assert not job.dump_path.exists()
    assert "--native-dump" not in finish.build_command(job, job.settings)


def test_a_dump_is_copied_when_it_cannot_be_linked(tmp_path, monkeypatch):
    source = tmp_path / "a.post.bin"
    source.write_bytes(b"dump")

    def no_links(*args):
        raise OSError("cross-device link")

    monkeypatch.setattr(finish.os, "link", no_links)
    finish.link_or_copy(source, tmp_path / "b.post.bin")
    assert (tmp_path / "b.post.bin").read_bytes() == b"dump"


def test_high_detail_reaches_the_worker():
    job = finish.FinishJob("0" * 32, Path("/tmp/finish-job"))
    assert "--detail" not in finish.build_command(job, finish.normalise_settings({}))
    settings = finish.normalise_settings({"faces": 50000, "high_detail": True})
    command = finish.build_command(job, settings)
    assert command[command.index("--detail") + 1] == "high"
    assert command[command.index("--faces") + 1] == "50000"


# --- Hero: the body finished, the head built again from a close crop (scripts/hero_route.py)

def test_hero_is_a_setting_and_names_its_result(tmp_path):
    assert finish.normalise_settings({"hero": 1})["hero"] is True
    assert finish.final_name("elf__pixal3d__1", 32000, hero=True) == "elf_hero.glb"


def test_hero_stages_match_the_route_it_runs():
    import sys
    sys.path.insert(0, str(MODULE.parents[1]))
    from scripts import hero_route
    assert finish.HERO_STAGES == list(hero_route.STAGES)
    assert set(finish.HERO_STAGES) == set(finish.HERO_WEIGHTS)


def test_a_hi3d_run_lands_the_refined_mesh_as_the_result(tmp_path, monkeypatch):
    """The refine replaces the retopo worker: result is the refined GLB, with a record and
    a licence sidecar chaining the source's licence with Hi3D's."""
    import finish_api
    from Ar3DRAHub import hi3d_backend

    manager = finish_api.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    job = manager.create("knight.glb", b"glb", b"png", {"hi3d": True})
    job.image_path.parent.mkdir(parents=True, exist_ok=True)
    job.image_path.write_bytes(b"png")
    job.asset_path.parent.mkdir(parents=True, exist_ok=True)
    job.asset_path.write_bytes(b"glb")
    job.input_dir.mkdir(parents=True, exist_ok=True)

    refined = job.directory / "steps" / "source_hi3d.glb"
    refined.parent.mkdir(parents=True, exist_ok=True)
    refined.write_bytes(valid_glb(generator="the hi3d refine"))

    seen = {}

    def fake_generate(image, coarse_mesh, output_dir, options, log=None, on_process=None):
        seen["image"] = Path(image)
        seen["coarse"] = Path(coarse_mesh)
        seen["resolution"] = options.resolution
        return hi3d_backend.Hi3DResult(refined, 512)

    monkeypatch.setattr(hi3d_backend, "generate_hi3d", fake_generate)
    monkeypatch.setattr(finish_api, "hi3d_readiness", lambda: (True, None))
    monkeypatch.setattr(finish_api, "hi3d_remesh_stage", lambda job_: job_.asset_path)
    finish_api.run_job(job, manager)
    assert seen["image"] == job.image_path and seen["coarse"] == job.asset_path
    assert seen["resolution"] == 512
    assert job.status == "done"
    assert job.result_glb.name == "knight_hi3d.glb"
    from glb_fixtures import read_glb

    gltf = read_glb(job.result_glb.read_bytes())[0]
    assert gltf["asset"]["generator"].startswith("Ar3DRA Hub")
    assert job.events[-1]["phase"] == "done" and job.events[-1].get("hi3d") is True
    assert "geometry only" in job.events[-1]["message"]
    assert job.events[-1]["hi3d_resolution"] == 512
    record = json.loads(job.record_path.read_text(encoding="utf-8"))
    assert record["stages"] == ["hi3d"] and record["hi3d"]["geometry_only"] is True
    assert record["hi3d"]["resolution"] == 512
    assert record["hi3d"]["requested_resolution"] == 512
    sidecar = json.loads(job.result_glb.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert sidecar["operation"] == "hi3d-refine"
    assert "MIT" in sidecar["license"]["name"]


def test_a_hi3d_geometry_only_run_still_builds_the_watertight_extra(tmp_path, monkeypatch):
    """A Hi3D run without a texture ends before the worker, so it used to drop the watertight
    rebuild the person ticked: the run said done and no watertight.glb existed."""
    import finish_api
    from Ar3DRAHub import hi3d_backend

    manager = finish_api.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    job = manager.create("knight.glb", b"glb", b"png",
                         {"hi3d": True, "watertight": True, "watertight_across": 450})
    job.image_path.parent.mkdir(parents=True, exist_ok=True)
    job.image_path.write_bytes(b"png")
    job.asset_path.parent.mkdir(parents=True, exist_ok=True)
    job.asset_path.write_bytes(b"glb")
    job.input_dir.mkdir(parents=True, exist_ok=True)
    refined = job.directory / "steps" / "source_hi3d.glb"
    refined.parent.mkdir(parents=True, exist_ok=True)
    refined.write_bytes(valid_glb(generator="the hi3d refine"))

    monkeypatch.setattr(hi3d_backend, "generate_hi3d",
                        lambda *a, **k: hi3d_backend.Hi3DResult(refined, 512))
    monkeypatch.setattr(finish_api, "hi3d_readiness", lambda: (True, None))
    monkeypatch.setattr(finish_api, "hi3d_remesh_stage", lambda job_: job_.asset_path)

    rebuilt = job.directory / "steps" / "watertight.glb"
    seen = {}

    def fake_watertight(job_, source):
        seen["source"] = Path(source)
        seen["across"] = job_.settings["watertight_across"]
        rebuilt.write_bytes(valid_glb(generator="the watertight rebuild"))
        return rebuilt

    monkeypatch.setattr(finish_api, "run_watertight", fake_watertight)
    finish_api.run_job(job, manager)

    assert seen["source"] == job.result_glb     # rebuilt from what Hi3D made
    assert seen["across"] == 450
    done = job.events[-1]
    assert done["phase"] == "done" and done["hi3d"] is True
    assert done["watertight"] is True and done["watertight_error"] is None
    assert done["watertight_url"] == f"/api/finish/{job.id}/watertight.glb"


def test_a_hi3d_geometry_only_run_keeps_the_result_when_the_rebuild_fails(tmp_path, monkeypatch):
    """The rebuild is an extra file beside the finish, not a condition of it."""
    import finish_api
    from Ar3DRAHub import hi3d_backend

    manager = finish_api.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    job = manager.create("knight.glb", b"glb", b"png", {"hi3d": True, "watertight": True})
    job.image_path.parent.mkdir(parents=True, exist_ok=True)
    job.image_path.write_bytes(b"png")
    job.asset_path.parent.mkdir(parents=True, exist_ok=True)
    job.asset_path.write_bytes(b"glb")
    job.input_dir.mkdir(parents=True, exist_ok=True)
    refined = job.directory / "steps" / "source_hi3d.glb"
    refined.parent.mkdir(parents=True, exist_ok=True)
    refined.write_bytes(valid_glb(generator="the hi3d refine"))

    monkeypatch.setattr(hi3d_backend, "generate_hi3d",
                        lambda *a, **k: hi3d_backend.Hi3DResult(refined, 512))
    monkeypatch.setattr(finish_api, "hi3d_readiness", lambda: (True, None))
    monkeypatch.setattr(finish_api, "hi3d_remesh_stage", lambda job_: job_.asset_path)

    def broken(job_, source):
        raise RuntimeError("the watertight rebuild is not available here")

    monkeypatch.setattr(finish_api, "run_watertight", broken)
    finish_api.run_job(job, manager)

    assert job.status == "done"
    done = job.events[-1]
    assert done["watertight"] is True and done["watertight_url"] is None
    assert "not available" in done["watertight_error"]
    assert any("watertight: " in line for line in job.log_lines)


def test_a_hi3d_run_records_the_rung_that_actually_made_the_mesh(tmp_path, monkeypatch):
    """Asking for 1536 and getting 512 must not be written down as 1536."""
    import finish_api
    from Ar3DRAHub import hi3d_backend

    manager = finish_api.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    job = manager.create("knight.glb", b"glb", b"png", {"hi3d": True,
                                                       "hi3d_resolution": 1536})
    job.image_path.parent.mkdir(parents=True, exist_ok=True)
    job.image_path.write_bytes(b"png")
    job.asset_path.parent.mkdir(parents=True, exist_ok=True)
    job.asset_path.write_bytes(b"glb")
    job.input_dir.mkdir(parents=True, exist_ok=True)
    refined = job.directory / "steps" / "source_hi3d.glb"
    refined.parent.mkdir(parents=True, exist_ok=True)
    refined.write_bytes(valid_glb(generator="the hi3d refine"))
    monkeypatch.setattr(hi3d_backend, "generate_hi3d",
                        lambda *a, **k: hi3d_backend.Hi3DResult(refined, 512))
    monkeypatch.setattr(finish_api, "hi3d_readiness", lambda: (True, None))
    monkeypatch.setattr(finish_api, "hi3d_remesh_stage", lambda job_: job_.asset_path)
    finish_api.run_job(job, manager)
    record = json.loads(job.record_path.read_text(encoding="utf-8"))
    assert record["hi3d"]["resolution"] == 512
    assert record["hi3d"]["requested_resolution"] == 1536
    assert job.events[-1]["hi3d_resolution"] == 512


def test_a_hi3d_run_still_honours_cancel_after_the_refine(tmp_path, monkeypatch):
    import finish_api
    from Ar3DRAHub import hi3d_backend

    manager = finish_api.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    job = manager.create("knight.glb", b"glb", b"png", {"hi3d": True})
    job.image_path.parent.mkdir(parents=True, exist_ok=True)
    job.image_path.write_bytes(b"png")
    job.asset_path.parent.mkdir(parents=True, exist_ok=True)
    job.asset_path.write_bytes(b"glb")
    refined = job.directory / "steps" / "source_hi3d.glb"
    refined.parent.mkdir(parents=True, exist_ok=True)
    refined.write_bytes(valid_glb(generator="the hi3d refine"))
    job.cancel_requested = True
    monkeypatch.setattr(hi3d_backend, "generate_hi3d",
                        lambda *a, **k: hi3d_backend.Hi3DResult(refined, 512))
    monkeypatch.setattr(finish_api, "hi3d_readiness", lambda: (True, None))
    monkeypatch.setattr(finish_api, "hi3d_remesh_stage", lambda job_: job_.asset_path)
    finish_api.run_job(job, manager)
    assert job.status == "cancelled"


def test_a_hi3d_textured_run_falls_through_to_the_normal_worker(tmp_path, monkeypatch):
    """Keep the texture: refine, transfer, then the ordinary Finish worker runs on the
    refined mesh with its sidecar as the source record."""
    import finish_api
    from Ar3DRAHub import hi3d_backend

    manager = finish_api.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    job = manager.create("knight.glb", b"glb", b"png", {"hi3d": True, "hi3d_texture": True})
    job.image_path.parent.mkdir(parents=True, exist_ok=True)
    job.image_path.write_bytes(b"png")
    job.asset_path.parent.mkdir(parents=True, exist_ok=True)
    job.asset_path.write_bytes(b"glb")

    refined = job.directory / "steps" / "source_hi3d.glb"
    refined.parent.mkdir(parents=True, exist_ok=True)
    refined.write_bytes(b"refined")
    textured = job.directory / "steps" / "hi3d_textured.glb"
    textured.write_bytes(valid_glb(generator="the hi3d refine"))
    monkeypatch.setattr(hi3d_backend, "generate_hi3d",
                        lambda *a, **k: hi3d_backend.Hi3DResult(refined, 512))
    monkeypatch.setattr(finish_api, "hi3d_readiness", lambda: (True, None))
    monkeypatch.setattr(finish_api, "hi3d_remesh_stage", lambda job_: job_.asset_path)

    class Transfer:
        pid = 2
        returncode = 0
        stdout = iter(["TRANSFER:: wrote textured.glb\n"])

        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(finish_api.subprocess, "Popen", Transfer)
    seen = {}

    def fake_build(job_arg, settings, resume=False, source=None, source_record=None):
        seen["source"] = source
        seen["source_record"] = source_record
        return ["worker"]

    monkeypatch.setattr(finish_api, "build_command", fake_build)

    class Worker:
        pid = 3
        returncode = 0
        stdout = iter(["baking\n"])

        def __init__(self, *args, **kwargs):
            job.result_glb.parent.mkdir(parents=True, exist_ok=True)
            job.result_glb.write_bytes(valid_glb(generator="the finish worker"))

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(finish_api.subprocess, "Popen", Worker)
    finish_api.run_job(job, manager)
    assert seen["source"] == textured
    assert seen["source_record"] == textured.with_suffix(".provenance.json")
    assert job.status == "done"
    assert job.result_glb.name == "knight_20k.glb"          # the normal name, not _hi3d
    assert job.events[-1]["phase"] == "done" and job.events[-1].get("hi3d") is not True
    assert job.result_glb.is_file()


def test_hi3d_transfer_command_runs_blender_headless(tmp_path, monkeypatch):
    import finish_api

    manager = finish_api.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    job = manager.create("knight.glb", b"glb", b"png", {"hi3d": True, "hi3d_texture": True})
    exe = tmp_path / "blender"
    exe.write_text("")
    command = finish_api.hi3d_transfer_command(job, tmp_path / "refined.glb",
                                               tmp_path / "textured.glb", blender=exe)
    assert command[:6] == [str(exe), "--background", "--factory-startup", "--python-exit-code", "1", "--python"]
    assert command[6].endswith("blender_hi3d_transfer.py")
    assert command[8:11] == [str(job.asset_path), str(tmp_path / "refined.glb"),
                             str(tmp_path / "textured.glb")]
    assert command[11] == str(job.settings["atlas"])


def test_hi3d_transfer_needs_blender(tmp_path, monkeypatch):
    import finish_api

    manager = finish_api.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    job = manager.create("knight.glb", b"glb", b"png", {"hi3d": True})
    monkeypatch.setattr(finish_api.blender_lookup, "find_blender", lambda: None)
    with pytest.raises(RuntimeError, match="Blender"):
        finish_api.hi3d_transfer_command(job, tmp_path / "r.glb", tmp_path / "t.glb",
                                         blender=None)


def test_create_refuses_hi3d_without_the_picture_or_the_setup(tmp_path, monkeypatch):
    manager = finish.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    with pytest.raises(ValueError, match="picture"):
        manager.create("knight.glb", b"glb", None, {"hi3d": True})
    monkeypatch.setattr(finish, "hi3d_readiness", lambda: (False, "not installed: test"))
    manager2 = finish.FinishJobManager(tmp_path / "finish2", tmp_path / "none")
    with pytest.raises(ValueError, match="not installed: test"):
        manager2.create("knight.glb", b"glb", b"png", {"hi3d": True})


def test_create_refuses_hi3d_with_hero(tmp_path, monkeypatch):
    manager = finish.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    with pytest.raises(ValueError, match="do not combine"):
        manager.create("knight.glb", b"glb", b"png", {"hi3d": True, "hero": True})


def test_hi3d_readiness_says_why_when_the_checkout_or_weights_are_gone(tmp_path, monkeypatch):
    import finish_api
    from Ar3DRAHub import hi3d_backend

    monkeypatch.setattr(hi3d_backend, "platform_refusal", lambda: None)
    monkeypatch.setattr(hi3d_backend, "verify_paths",
                        lambda repo: [f"missing venv python: {repo}/.venv"])
    monkeypatch.setattr(finish_api, "HI3D_VENDOR", tmp_path)
    ready, reason = finish_api.hi3d_readiness()
    assert ready is False and "not installed" in reason
    monkeypatch.setattr(hi3d_backend, "verify_paths", lambda repo: [])
    monkeypatch.setattr(hi3d_backend, "model_dir_present",
                        lambda: (False, tmp_path, ["model_index.json"]))
    ready, reason = finish_api.hi3d_readiness()
    assert ready is False and "weights are missing" in reason


def test_a_hero_run_starts_from_the_asset_s_own_model(tmp_path):
    _pixal3d_run(tmp_path / "generated")
    run = tmp_path / "generated" / "seb" / "robot__pixal3d__1"
    (run / "robot__pixal3d__1.post.bin").write_bytes(b"dump")
    record = json.loads((run / "robot__pixal3d__1.json").read_text())
    (run / "robot__pixal3d__1.json").write_text(json.dumps({**record, "post_dump": "robot__pixal3d__1.post.bin"}))
    manager = finish.FinishJobManager(tmp_path / "finish", tmp_path / "generated")
    job = manager.create("robot.glb", b"glb-bytes", b"png", {"hero": True, "faces": 32000})
    command = finish.hero_command(job, job.settings)
    assert command[2].endswith("hero_route.py") and command[3] == str(job.image_path)
    assert command[command.index("--body") + 1] == str(job.asset_path)
    assert command[command.index("--body-views") + 1] == str(job.views_dir)
    assert command[command.index("--body-dump") + 1] == str(job.dump_path)
    assert command[command.index("--body-record") + 1] == str(job.input_dir / "source.json")
    assert command[command.index("--body-faces") + 1] == "32000"
    assert command[command.index("--result") + 1] == str(job.result_glb)
    assert job.result_glb.name.endswith("_hero.glb")


def test_hero_needs_the_picture_and_the_model_s_camera(tmp_path):
    manager = finish.FinishJobManager(tmp_path / "finish", tmp_path / "none")
    with pytest.raises(ValueError, match="picture"):
        manager.create("robot.glb", b"glb-bytes", None, {"hero": True})
    manager = finish.FinishJobManager(tmp_path / "finish2", tmp_path / "none")
    with pytest.raises(ValueError, match="saved camera"):
        manager.create("robot.glb", b"glb-bytes", b"png", {"hero": True})


def test_a_failed_hero_run_leads_with_its_own_reason():
    lines = ["I2L_STAGE::body::Finishing the body",
             "[hero] cannot start, not installed: Hero face upscaler (Real-ESRGAN, 67 MB). Install them from the Setup page."]
    assert finish.failure_message(lines, 2).startswith("cannot start, not installed: Hero face upscaler")
    assert finish.failure_message(["Traceback ...", "boom"], 1) == "worker exited with code 1"


def test_done_is_said_only_after_the_done_event_names_the_file(tmp_path, monkeypatch):
    """A plan polls the status and reads the last event: "done" before the event lost the file.
    Same race that lost a rig mid-plan on 2026-10-25 (see test_autorig_api)."""
    import finish_api

    job = finish_api.FinishJob("0" * 32, tmp_path / "out" / "fox__finish__20261025-010101")
    job.settings = {**job.settings, "hero": True}

    class Worker:
        pid = 1
        returncode = 0
        stdout = iter(["[hero] done\n"])

        def __init__(self, *args, **kwargs):
            job.result_glb.parent.mkdir(parents=True, exist_ok=True)
            job.result_glb.write_bytes(b"glb")

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(finish_api, "hero_command", lambda job, settings: ["worker"])
    monkeypatch.setattr(finish_api.subprocess, "Popen", Worker)
    seen = []
    real_emit = job.emit

    def emit(event):
        if event.get("phase") == "done":
            seen.append(job.status)
        real_emit(event)

    job.emit = emit
    finish_api.run_job(job, finish_api.FinishJobManager(tmp_path / "out", tmp_path / "out"))
    assert seen and "done" not in seen
    assert job.status == "done" and job.events[-1]["phase"] == "done"


def _views(tmp_path: Path, files: list[str]) -> Path:
    views = tmp_path / "views"
    views.mkdir()
    (views / "transforms.json").write_text(json.dumps({"frames": [{"file_path": f} for f in files]}))
    return views


def test_pixel_match_says_front_or_front_and_back_from_the_saved_views(tmp_path):
    two = _views(tmp_path, ["view_000.png", "view_180.png"])
    assert finish.pixel_match_result(True, two, {}) == "on (front + back)"
    four = tmp_path / "four"
    four.mkdir()
    (four / "transforms.json").write_text(json.dumps({"frames": [
        {"file_path": "view_000.png"}, {"file_path": "view_090.png"},
        {"file_path": "view_180.png"}, {"file_path": "view_270.png"}]}))
    assert finish.pixel_match_result(True, four, {}) == "on (4 views)"
    one = tmp_path / "one"
    one.mkdir()
    (one / "transforms.json").write_text(json.dumps({"frames": [{"file_path": "input.png"}]}))
    assert finish.pixel_match_result(True, one, {}) == "on (front)"


def test_pixel_match_that_did_not_run_says_why(tmp_path):
    views = _views(tmp_path, ["view_000.png"])
    assert finish.pixel_match_result(False, views, {"skip_photo": True}) == "off"
    assert finish.pixel_match_result(False, tmp_path / "nowhere", {}).startswith("skipped: no saved camera")
    assert finish.pixel_match_result(False, views, {}) == "skipped"


def test_both_kinds_of_run_are_recognised_by_their_real_log_lines():
    assert finish.PHOTO_RAN.search("[photo] 2 view(s); 16% of the texture from the photo")
    assert finish.PHOTO_RAN.search("[hero      5s] body: Finish 32000 + Pixel Match ...")
    assert not finish.PHOTO_RAN.search("Starting: found the source camera, so Pixel Match will keep the photo's pixels")
    assert not finish.PHOTO_RAN.search("[hero      5s] body: Finish 32000 ...")


def test_a_finished_run_says_and_keeps_whether_pixel_match_ran(tmp_path, monkeypatch):
    import finish_api

    job = finish_api.FinishJob("1" * 32, tmp_path / "out" / "fox__finish__20261027-010101")
    job.views_dir.mkdir(parents=True)
    (job.views_dir / "transforms.json").write_text(json.dumps({"frames": [{}, {}]}))

    class Worker:
        pid = 1
        returncode = 0
        stdout = iter(["[photo] 2 view(s); 16% of the texture from the photo\n", "baking\n"])

        def __init__(self, *args, **kwargs):
            job.result_glb.parent.mkdir(parents=True, exist_ok=True)
            job.result_glb.write_bytes(b"glb")

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(finish_api, "build_command",
                        lambda job, settings, resume=False, source=None, source_record=None: ["worker"])
    monkeypatch.setattr(finish_api.subprocess, "Popen", Worker)
    finish_api.run_job(job, finish_api.FinishJobManager(tmp_path / "out", tmp_path / "out"))
    done = job.events[-1]
    assert done["phase"] == "done" and done["pixel_match"] == "on (front + back)"
    assert done["message"].endswith("· Pixel Match: on (front + back)")
    saved = json.loads((job.directory / finish_api.PIXEL_MATCH_NAME).read_text())
    assert saved == {"pixel_match": "on (front + back)"}


def test_a_post_launch_error_terminates_the_worker_before_releasing_the_slot(tmp_path, monkeypatch):
    """If the pid-record write fails after Popen, the child must be stopped and reaped
    before the active slot is freed; otherwise the worker keeps running and a second
    Finish can start."""
    import finish_api

    job = finish_api.FinishJob("2" * 32, tmp_path / "out" / "fox__finish__20261027-020202")
    manager = finish_api.FinishJobManager(tmp_path / "out", tmp_path / "out")
    manager.active = job.id

    calls = []

    class Worker:
        pid = 4242
        returncode = 0
        stdout = iter([])

        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout=None):
            calls.append(("wait", timeout, manager.active))
            return 0

        def poll(self):
            return None  # still alive until terminated

    def fake_terminate_group(pid):
        calls.append(("terminate_group", pid, manager.active))

    original_write_text = Path.write_text

    def fake_write_text(self, *args, **kwargs):
        if self.name == "pid":
            raise OSError("No space left on device")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(finish_api, "build_command",
                        lambda job, settings, resume=False, source=None, source_record=None: ["worker"])
    monkeypatch.setattr(finish_api.subprocess, "Popen", Worker)
    monkeypatch.setattr(finish_api.processes, "terminate_group", fake_terminate_group)
    monkeypatch.setattr(Path, "write_text", fake_write_text)

    finish_api.run_job(job, manager)

    assert job.status == "error"
    assert any(event.get("phase") == "error" for event in job.events)
    assert calls == [("terminate_group", 4242, job.id), ("wait", pytest.approx(30, rel=0.1), job.id)]
    assert manager.active is None


@pytest.mark.parametrize("backend", ["pixal3d", "sf3d"])
@pytest.mark.parametrize("record_name", ["source.provenance.json", "source.json"])
def test_finish_preserves_source_license_through_rig_lookup(tmp_path, monkeypatch, backend, record_name):
    import hashlib
    import sys

    import autorig_api

    asset = valid_glb(generator="the source backend")
    generated = tmp_path / "generated"
    generated.mkdir()
    source = {"backend": backend, "license": {"name": "Source terms", "conditions": ["credit"]},
              "model": {"identity": "source-model"}, "components": [{"component": "encoder"}],
              "output": {"sha256": hashlib.sha256(asset).hexdigest(), "classification": "commercial-conditional"},
              "source_record": {"padding": "x" * 21000}}
    (generated / record_name).write_text(json.dumps(source))
    manager = finish.FinishJobManager(tmp_path / "finish", generated)
    job = manager.create("source.glb", asset, None,
                         {"keep_mesh": True, "skip_paint": True, "skip_bake": True, "skip_compress": True})
    worker = finish.worker_module()
    monkeypatch.setattr(worker, "preflight", lambda *a: (None, None))
    command = finish.build_command(job, job.settings)
    monkeypatch.setattr(sys, "argv", command[2:])
    assert worker.main() == 0
    record = autorig_api.source_record_for(job.result_glb)
    assert record is not None
    assert record["source_record"] == source
    assert record["license"] == source["license"]
    assert record["output"]["sha256"] == hashlib.sha256(job.result_glb.read_bytes()).hexdigest()
    rig = autorig_api.rig_record("result.glb", job.result_glb.read_bytes(), record, "finish")
    assert rig["license"] == source["license"]
    assert rig["source_record"]["source_record"]["model"] == source["model"]
    assert finish.find_source_record(job.result_glb.read_bytes(), tmp_path / "finish") is not None
