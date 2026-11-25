"""The Animate tab's backend: what it offers, what it refuses, and the records it writes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from glb_fixtures import valid_glb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import autorig_api as aa


def _glb(path: Path, payload: bytes = valid_glb()) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


@pytest.fixture
def output(tmp_path):
    root = tmp_path / "output"
    _glb(root / "commercial" / "knight" / "knight.glb")
    _glb(root / "finish" / "knight__finish__20261003-135502" / "knight_40k.glb")
    _glb(root / "finish" / "knight__finish__20261003-135502" / "steps" / "1_retopo.glb")
    _glb(root / "animate" / "knight__rig__20261005-120000" / "knight_rigged.glb")
    _glb(root / "animate" / "knight__rig__20261005-120000" / "knight_walk.glb")
    _glb(root / "props" / "sheet" / "sheet.glb")
    return root


@pytest.fixture
def motions(tmp_path):
    root = tmp_path / "motions"
    root.mkdir()
    (root / "walk.npz").write_bytes(b"x")
    (root / "kick.npz").write_bytes(b"x")
    (root / "idle.npz").write_bytes(b"x")
    (root / "presets.json").write_text(json.dumps({"aliases": {"walk_old": "walk"}, "presets": [
        {"id": "walk", "label": "Walk", "seconds": 4},
        {"id": "kick", "label": "Kick", "seconds": 2, "speed": 1.25},
        {"id": "idle", "label": "Idle (restless)", "seconds": 7, "loop": False},
        {"id": "missing", "label": "No file"},
        {"id": "../evil", "label": "Bad id"},
    ]}))
    return root


def test_offers_generated_finished_and_rigged_models_only(output):
    models = {m["path"]: m["kind"] for m in aa.pickable_models(output)}
    assert models == {
        "commercial/knight/knight.glb": "generated",
        "finish/knight__finish__20261003-135502/knight_40k.glb": "finished",
        "animate/knight__rig__20261005-120000/knight_rigged.glb": "rigged",
    }


def test_resolve_refuses_anything_not_offered(output):
    assert aa.resolve_model("commercial/knight/knight.glb", output).is_file()
    for path in ("props/sheet/sheet.glb", "../secret.glb",
                 "animate/knight__rig__20261005-120000/knight_walk.glb"):
        with pytest.raises(ValueError):
            aa.resolve_model(path, output)


def test_presets_need_a_file_and_a_safe_id(motions):
    assert [p["id"] for p in aa.presets(motions)] == ["walk", "kick", "idle"]


def test_rig_job_copies_the_source_and_records_both_licences(tmp_path):
    manager = aa.AnimateJobManager(tmp_path / "animate", tmp_path)
    source = {"license": {"name": "MIT"}, "output": {"classification": "commercial"}}
    job = manager.create_rig("My Knight.glb", b"glTF-data", source, "generated")
    assert job.directory.name.startswith("My-Knight__rig__")
    assert (job.directory / "input" / "source.glb").read_bytes() == b"glTF-data"
    assert job.result.name == "My-Knight_rigged.glb"
    (rig, rig_cwd), (clean, _) = job.commands
    assert rig[1:3] == ["demo.py", "--input"] and "--use_transfer" in rig and rig_cwd == aa.SKINTOKENS
    # The clean-up rewrites SkinTokens' result in place, after it.
    assert clean[1].endswith("clean_skin_weights.py") and clean[-1] == str(job.result)
    assert job.record["license"] == {"name": "MIT"}
    assert job.record["classification"] == "commercial"
    assert job.record["rig"]["license"]["name"] == "MIT"
    with pytest.raises(RuntimeError, match="already running"):
        manager.create_rig("other.glb", b"x", None, "uploaded")


def test_bind_job_uses_rig_bind_and_notes_the_skeleton_is_kept(tmp_path):
    manager = aa.AnimateJobManager(tmp_path / "animate", tmp_path)
    job = manager.create_bind("My Rig.glb", b"glTF-with-skeleton", None, "uploaded")
    assert job.kind == "bind"
    assert job.directory.name.startswith("My-Rig__rig__")
    assert (job.directory / "input" / "source.glb").read_bytes() == b"glTF-with-skeleton"
    assert job.result.name == "My-Rig_rigged.glb"
    (command, cwd), = job.commands
    assert command[1].endswith("rig_bind.py") and "--input" in command and "--output" in command
    assert cwd == aa.REPO
    assert job.record["tool"] == "viewer Animate tab: SkinTokens bind to skeleton"
    assert job.record["rig"]["skeleton"] == "user's own, held fixed"
    with pytest.raises(RuntimeError, match="already running"):
        manager.create_bind("other.glb", b"x", None, "uploaded")


def test_kind_labels_cover_rig_bind_and_animate():
    assert aa._kind_label("rig")[2] == "Rigged"
    assert aa._kind_label("bind")[1] == "Binding weights to your skeleton"
    assert aa._kind_label("animate")[0] == "Starting"


def test_rig_human_job_writes_the_joints_and_runs_rig_human(tmp_path):
    manager = aa.AnimateJobManager(tmp_path / "animate", tmp_path)
    job = manager.create_rig_human("Hand Rig.glb", b"mesh", {"Hips": [0.5, 0.5, 0.52]}, None, "uploaded")
    assert job.kind == "rig_human"
    joints_file = job.directory / "input" / "joints.json"
    import json as _json
    assert _json.loads(joints_file.read_text(encoding="utf-8")) == {"Hips": [0.5, 0.5, 0.52]}
    (command, cwd), = job.commands
    assert command[1].endswith("rig_human.py") and "--joints" in command and str(joints_file) in command
    assert cwd == aa.REPO
    assert job.record["rig"]["skeleton"] == "placed by hand"


def test_fit_human_joints_reads_a_mesh_and_falls_back(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.icosphere(subdivisions=2)
    glb = tmp_path / "model.glb"
    mesh.export(str(glb))
    fitted = aa.fit_human_joints(glb)
    assert fitted is not None
    assert sorted(fitted) == sorted(aa.human_rig.joint_order())
    # A file that is not a GLB, and a missing one, both fall back to None.
    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"not a glb")
    assert aa.fit_human_joints(bad) is None
    assert aa.fit_human_joints(tmp_path / "missing.glb") is None


def _rigged_glb(path: Path, vertices, faces, bones) -> Path:
    """A GLB: one indexed triangle mesh plus a chain of bones (name, translation, parent)."""
    import struct

    import numpy as np
    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(faces, dtype=np.uint16).flatten()
    pos = verts.tobytes()
    idx = tris.tobytes()
    binary = pos + b"\x00" * (-len(pos) % 4) + idx + b"\x00" * (-len(idx) % 4)
    idx_off = len(pos) + (-len(pos) % 4)

    nodes = [{"name": "mesh", "mesh": 0}]
    for name, translation, parent in bones:
        index = len(nodes)
        nodes.append({"name": name, "translation": list(translation)})
        if parent is not None:
            nodes[parent + 1].setdefault("children", []).append(index)
    doc = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(vertices), "type": "VEC3",
             "min": verts.min(0).tolist(), "max": verts.max(0).tolist()},
            {"bufferView": 1, "componentType": 5123, "count": len(tris), "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos)},
            {"buffer": 0, "byteOffset": idx_off, "byteLength": len(idx)},
        ],
        "buffers": [{"byteLength": len(binary)}],
        "skins": [{"joints": list(range(1, len(nodes)))}],
    }
    blob = json.dumps(doc).encode()
    blob += b" " * (-len(blob) % 4)
    json_chunk = struct.pack("<II", len(blob), 0x4E4F534A) + blob
    bin_chunk = struct.pack("<II", len(binary), 0x004E4942) + binary
    body = json_chunk + bin_chunk
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)
    return path


def _box_faces():
    # The 12 triangles of a unit cube's surface, so trimesh sees a real volume.
    v = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    f = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
         [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]]
    return v, f


def _humanoid_bones():
    """A Y-up humanoid (hips, spine, chest, neck, head, two arms, two legs), `bone_N` names,
    as (name, translation, parent-index) rows in local space. The auto-rig names its bones
    this way, so the editor must map them by shape, not name."""
    bones: list[tuple[str, list[float], int | None]] = []

    def add(translation, parent):
        bones.append((f"bone_{len(bones)}", translation, parent))
        return len(bones) - 1

    hips = add([0.0, 1.0, 0.0], None)
    spine = add([0.0, 0.15, 0.0], hips)
    chest = add([0.0, 0.2, 0.0], spine)
    neck = add([0.0, 0.15, 0.0], chest)
    add([0.0, 0.1, 0.0], neck)  # head
    for side in (1, -1):
        b = add([0.08 * side, 0.1, 0.0], chest)
        for _ in range(3):
            b = add([0.2 * side, 0.0, 0.0], b)
    for side in (1, -1):
        b = add([0.1 * side, -0.05, 0.0], hips)
        b = add([0.0, -0.45, 0.0], b)
        b = add([0.0, -0.45, 0.0], b)
        add([0.0, -0.04, 0.12], b)
    return bones


def test_mixamo_soma_candidates_map_the_body():
    assert aa._mixamo_soma_candidates("Hips") == ["Hips"]
    assert aa._mixamo_soma_candidates("Spine") == ["Spine1"]
    assert aa._mixamo_soma_candidates("Spine2") == ["Chest"]
    assert aa._mixamo_soma_candidates("Neck") == ["Neck1", "Neck2"]
    assert aa._mixamo_soma_candidates("LeftUpLeg") == ["LeftLeg"]
    assert aa._mixamo_soma_candidates("LeftLeg") == ["LeftShin"]
    assert aa._mixamo_soma_candidates("LeftFoot") == ["LeftFoot"]
    assert aa._mixamo_soma_candidates("LeftHandThumb3") == ["LeftHandThumb3"]
    assert aa._mixamo_soma_candidates("RightHandIndex1") == ["RightHandIndex2"]
    assert aa._mixamo_soma_candidates("RightHandPinky3") == ["RightHandPinky4"]


def test_rig_joint_fractions_maps_the_auto_rig_by_shape(tmp_path):
    vertices, faces = _box_faces()
    # Stretch the unit cube up to a standing-humanoid box in world space; the bones sit in it.
    vertices = [[x * 1.6 - 0.8, y * 1.7, z * 0.5 - 0.25] for x, y, z in vertices]
    glb = _rigged_glb(tmp_path / "r.glb", vertices, faces, _humanoid_bones())

    joints = aa.rig_joint_fractions(glb)
    assert joints is not None
    assert sorted(joints) == sorted(aa.human_rig.joint_order())
    # Head near the top, hips in the middle, feet at the bottom, hands out to the sides.
    assert joints["Head"][2] > joints["Neck"][2] > joints["Hips"][2]
    assert joints["LeftFoot"][2] < 0.15 and joints["RightFoot"][2] < 0.15
    assert joints["LeftHand"][0] > 0.8 and joints["RightHand"][0] < 0.2


def test_rig_joint_fractions_falls_back_without_a_skeleton(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box()
    plain = tmp_path / "plain.glb"
    mesh.export(str(plain))
    assert aa.rig_joint_fractions(plain) is None
    assert aa.rig_joint_fractions(tmp_path / "missing.glb") is None


def test_animation_needs_a_rigged_model_a_preset_and_a_sane_spread(tmp_path, motions):
    manager = aa.AnimateJobManager(tmp_path / "animate", motions)
    run = tmp_path / "animate" / "knight__rig__20261005-120000"
    rigged = _glb(run / "knight_rigged.glb")
    with pytest.raises(ValueError, match="rigged"):
        manager.create_animation(_glb(tmp_path / "plain.glb"), "walk", 0)
    with pytest.raises(ValueError, match="preset"):
        manager.create_animation(rigged, "missing", 0)
    with pytest.raises(ValueError, match="between"):
        manager.create_animation(rigged, "walk", 90)
    with pytest.raises(ValueError, match="number"):
        manager.create_animation(rigged, "walk", "wide")
    job = manager.create_animation(rigged, "walk", 12)
    assert job.result == run / "knight_walk.glb"
    (command, _), = job.commands
    assert command[command.index("--arm-spread") + 1] == "12"
    assert "--loop" in command and "--in-place" in command
    assert job.record["motion"]["made_with"] == "Kimodo"
    assert job.record["motion"]["loop"] is True and job.record["motion"]["grounded"] is True
    assert job.record["motion"]["in_place"] is True
    assert "MIT" in job.record["motion"]["license"]["name"]


def test_animate_command_loops_repeating_clips_and_leaves_actions_alone(tmp_path):
    rigged = tmp_path / "knight_rigged.glb"
    motion = tmp_path / "walk.npz"
    walk = aa.animate_command(rigged, motion, tmp_path / "walk.glb", 0, 1.0,
                              aa.repeats("walk"), aa.repeats("walk"))
    kick = aa.animate_command(rigged, motion, tmp_path / "kick.glb", 0, 1.0,
                              aa.repeats("kick"), aa.repeats("kick"))
    assert "--loop" in walk and "--in-place" in walk
    assert "--loop" not in kick and "--in-place" not in kick
    # Grounding is on by default, so the studio passes no opt-out.
    assert "--no-ground" not in walk


def test_an_old_move_name_still_finds_the_move_that_replaced_it(tmp_path, motions):
    # Run, Wave, Sit down and Strafe left were renamed or folded into their second take; a saved
    # plan or an agent asking by the old name must still get the move, not "no such preset".
    assert aa.resolve_move("run") == "run_b" and aa.resolve_move("strafe_left") == "strafe_left_b"
    assert aa.resolve_move("walk") == "walk"           # a current name is left alone
    shipped = {p["id"] for p in aa.presets()}
    for old in ("idle_b", "fight_stance", "punch_combo", "run", "wave", "sit", "strafe_left"):
        assert aa.resolve_move(old) in shipped, old    # every alias lands on a move that exists
    # Every move id 0.5.8 shipped still works, except the moves dropped on purpose.
    dropped = {"kata", "kata_b", "sword_swing", "sword_swing_b", "roundhouse", "roundhouse_b",
               "flying_kick", "flying_kick_b"}
    v058 = {"cheer", "cheer_b", "dance", "dance_b", "death", "death_b", "fight_stance",
            "fight_stance_b", "hit_react", "hit_react_b", "idle", "idle_b", "jump", "jump_b",
            "kick", "kick_b", "punch", "punch_b", "punch_combo", "punch_combo_b", "roll", "roll_b",
            "run", "run_b", "sit", "sit_b", "sneak", "sneak_b", "strafe_left", "strafe_left_b",
            "walk", "walk_b", "walk_woman", "wave", "wave_b"} | dropped
    for old in sorted(v058 - dropped):
        assert aa.resolve_move(old) in shipped, old
    manager = aa.AnimateJobManager(tmp_path / "animate", motions)
    run = tmp_path / "animate" / "knight__rig__20261005-120000"
    rigged = _glb(run / "knight_rigged.glb")
    job = manager.create_animation(rigged, "walk_old", 0)      # the fixture's alias
    assert job.result == run / "knight_walk.glb"
    manager.finish(job)
    with pytest.raises(ValueError, match="The moves are"):
        manager.create_animation(rigged, "nonsense", 0)


def test_each_move_says_whether_it_stays_in_place(tmp_path, motions):
    # Loops travel nowhere in a game; one-offs keep their travel. presets.json says which, per move.
    shipped = {p["id"]: p for p in aa.presets()}
    for move in ("idle", "walk", "run_b", "sneak", "strafe_left_b", "strafe_right", "walk_woman"):
        assert aa.default_in_place(shipped[move]) is True, move
    for move in ("jump", "roll", "punch", "death", "cheer", "robot_dance", "sit_b"):
        assert aa.default_in_place(shipped[move]) is False, move


def test_the_in_place_switch_reaches_the_retarget_command(tmp_path, motions):
    manager = aa.AnimateJobManager(tmp_path / "animate", motions)
    run = tmp_path / "animate" / "knight__rig__20261005-120000"
    rigged = _glb(run / "knight_rigged.glb")

    walk = manager.create_animation(rigged, "walk", 0)              # the move's own default
    (command, _), = walk.commands
    assert "--in-place" in command and walk.record["motion"]["in_place"] is True
    manager.finish(walk)

    kick = manager.create_animation(rigged, "kick", 0)               # a one-off: travels by default
    (command, _), = kick.commands
    assert "--in-place" not in command and kick.record["motion"]["in_place"] is False
    manager.finish(kick)

    with pytest.raises(ValueError, match="true or false"):
        manager.create_animation(rigged, "walk", 0, in_place="yes")

    stuck = manager.create_animation(rigged, "kick", 0, in_place=True)  # flipped on
    (command, _), = stuck.commands
    assert "--in-place" in command and stuck.record["motion"]["in_place"] is True


def test_a_move_that_keeps_its_whole_clip_only_has_its_seam_blended(tmp_path, motions):
    """The restless idle is 7 s of distinct beats; trimming it cut 5 s out, so it keeps every
    frame and the retargeter only blends the seam. It still passes --loop for that blend."""
    manager = aa.AnimateJobManager(tmp_path / "animate", motions)
    run = tmp_path / "animate" / "knight__rig__20261005-120000"
    rigged = _glb(run / "knight_rigged.glb")
    job = manager.create_animation(rigged, "idle", 0)
    (command, _), = job.commands
    assert "--loop" in command and "--keep-whole" in command
    assert job.record["motion"]["loop"] is False and job.record["motion"]["keep_whole"] is True
    manager.finish(job)
    # A normal loop is blended but trimmed, and a one-off gets neither flag.
    walk = manager.create_animation(rigged, "walk", 0)
    (command, _), = walk.commands
    assert "--loop" in command and "--keep-whole" not in command
    manager.finish(walk)
    kick = manager.create_animation(rigged, "kick", 0)
    (command, _), = kick.commands
    assert "--loop" not in command and "--keep-whole" not in command


def test_repeating_presets_are_listed_in_one_place():
    for preset in ("idle", "walk", "run", "sneak", "strafe",
                   "walk_b", "run_b", "walk_woman", "strafe_left"):
        assert aa.repeats(preset), preset
    for preset in ("kick", "jump", "punch", "wave", "death", "roll"):
        assert not aa.repeats(preset), preset


def test_a_move_can_keep_its_whole_clip(tmp_path):
    # Trimming to the shortest repeat cut a 7 s look-left, look-right idle to 1.7 s and dropped the
    # looks; `"loop": false` keeps every frame. Without the field, the name rule still decides.
    assert aa.loops({"id": "idle", "loop": False}) is False
    assert aa.loops({"id": "crouch_sneak", "loop": True}) is True
    assert aa.loops({"id": "walk"}) is True and aa.loops({"id": "kick"}) is False

    motions = tmp_path / "motions"
    motions.mkdir()
    (motions / "idle.npz").write_bytes(b"x")
    (motions / "presets.json").write_text(json.dumps({"presets": [
        {"id": "idle", "label": "Idle", "seconds": 7, "in_place": True, "loop": False}]}))
    rigged = _glb(tmp_path / "animate" / "knight__rig__20261005-120000" / "knight_rigged.glb")
    job = aa.AnimateJobManager(tmp_path / "animate", motions).create_animation(rigged, "idle", 0)
    (command, _), = job.commands
    # It keeps every frame but still passes --loop, so the retargeter blends the seam.
    assert "--loop" in command and "--keep-whole" in command and "--in-place" in command
    assert job.record["motion"]["loop"] is False and job.record["motion"]["keep_whole"] is True


def test_the_shipped_moves_keep_whole_clips_and_strike_faster():
    shipped = {p["id"]: p for p in aa.presets()}
    for move in ("idle", "idle_restless", "sprint", "zombie_walk"):   # beats, or already a loop
        assert aa.loops(shipped[move]) is False, move
    assert aa.loops(shipped["crouch_sneak"]) is True
    assert shipped["punch"]["speed"] == shipped["punch_b"]["speed"] == 1.3
    assert shipped["one_two"]["speed"] == 1.5


def _write_glb_script(payload: bytes | None = None) -> str:
    """A worker stand-in that writes a real GLB, so the generator stamp can read it."""
    return f"open(r'{{out}}', 'wb').write(bytes.fromhex('{(payload or valid_glb()).hex()}'))"


def _fake_job(tmp_path, script: str) -> aa.AnimateJob:
    result = tmp_path / "out.glb"
    command = [sys.executable, "-c", script.format(out=result)]
    return aa.AnimateJob("0" * 32, "rig", tmp_path, result, [(command, tmp_path)], 1.0,
                         {"license": {"name": "MIT"}})


def test_a_finished_job_writes_its_provenance(tmp_path):
    job = _fake_job(tmp_path, _write_glb_script() + "; print('hello')")
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    assert job.status == "done"
    assert json.loads(job.result.with_suffix(".provenance.json").read_text())["license"]
    assert job.events[-1]["phase"] == "done"
    assert "hello" in job.log_path.read_text()


def test_done_is_said_only_after_the_done_event_names_the_file(tmp_path, monkeypatch):
    """A plan polls the status and reads the last event: if "done" came first (the fit check takes
    seconds), the plan saw no file and skipped the move. Lost a robot's rig on 2026-10-25."""
    job = _fake_job(tmp_path, _write_glb_script())
    seen = []

    def slow_fit(path):
        seen.append(job.status)          # what a poll would see while the fit check runs
        return {}

    monkeypatch.setattr(aa, "rig_fit", slow_fit)
    real_emit = job.emit

    def emit(event):
        if event.get("phase") == "done":
            seen.append(job.status)      # what a poll would see as the path arrives
        real_emit(event)

    job.emit = emit
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    assert seen and "done" not in seen
    assert job.status == "done" and job.events[-1]["phase"] == "done"


def test_a_job_that_writes_nothing_is_an_error_with_its_log(tmp_path):
    job = _fake_job(tmp_path, "import sys; print('boom'); sys.exit(3)")
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    assert job.status == "error"
    assert "code 3" in job.events[-1]["message"]
    assert "boom" in job.events[-1]["log_tail"]


def test_progress_estimate_never_claims_done():
    assert aa.estimated_pct(0, 70) == 0
    assert aa.estimated_pct(35, 70) == 50
    assert aa.estimated_pct(500, 70) == 95


def test_a_post_launch_error_terminates_the_worker_before_releasing_the_slot(tmp_path, monkeypatch):
    """If the worker process is alive when an exception fires, it must be stopped and reaped
    before the active slot is freed."""
    result = tmp_path / "out.glb"
    command = [sys.executable, "-c", "print('working')"]
    job = aa.AnimateJob("2" * 32, "rig", tmp_path, result, [(command, tmp_path)], 1.0,
                        {"license": {"name": "MIT"}})
    manager = aa.AnimateJobManager(tmp_path, tmp_path)
    manager.active = job.id

    calls = []

    class RaisingStdout:
        def __iter__(self):
            return self

        def __next__(self):
            raise OSError("broken pipe")

    class Worker:
        pid = 4242
        returncode = 0
        stdout = RaisingStdout()

        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout=None):
            calls.append(("wait", timeout, manager.active))
            return 0

        def poll(self):
            return None

    def fake_terminate_group(pid):
        calls.append(("terminate_group", pid, manager.active))

    original_write_text = Path.write_text

    def fake_write_text(self, *args, **kwargs):
        if self.name == "pid":
            raise OSError("No space left on device")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(aa.subprocess, "Popen", Worker)
    monkeypatch.setattr(aa.processes, "terminate_group", fake_terminate_group)
    monkeypatch.setattr(Path, "write_text", fake_write_text)

    aa.run_job(job, manager)

    assert job.status == "error"
    assert calls == [("terminate_group", 4242, job.id), ("wait", pytest.approx(30, rel=0.1), job.id)]
    assert manager.active is None


def test_an_animate_run_records_its_pid_then_clears_it(tmp_path):
    from Ar3DRAHub import processes

    directory = tmp_path / "clip"
    directory.mkdir()
    result = directory / "out.glb"
    result.write_bytes(valid_glb())
    script = (
        "import time\n"
        "from pathlib import Path\n"
        "time.sleep(0.2)\n"
        "print(Path('pid').read_text().strip())\n"
    )
    job = aa.AnimateJob(
        "ab" * 16, "animate", directory, result,
        [([sys.executable, "-c", script], directory)], 1.0, {"license": "test"},
    )
    manager = aa.AnimateJobManager(tmp_path, tmp_path / "motions")
    manager.jobs[job.id] = job
    manager.active = job.id

    aa.run_job(job, manager)

    assert job.status == "done", job.log_lines
    assert job.log_lines[0] == processes.pid_record(job.process.pid)
    assert not (directory / "pid").exists()


def test_source_record_prefers_a_sidecar_that_names_a_licence(tmp_path):
    glb = _glb(tmp_path / "m.glb")
    (tmp_path / "m.json").write_text(json.dumps({"seed": 1}))
    assert aa.source_record_for(glb) is None
    (tmp_path / "m.retopo-repaint.json").write_text(json.dumps({"license": {"name": "X"}}))
    assert aa.source_record_for(glb)["license"]["name"] == "X"


def test_friendly_names_drop_run_stamps_and_keep_the_face_count(output):
    names = {m["kind"]: m["name"] for m in aa.pickable_models(output)}
    assert names["finished"] == "knight · 40k"
    assert names["rigged"] == "knight"
    long = aa.friendly_name("a-cute-anime-girl-standing-in-a-perfect-t-pose__20261003__pixal3d")
    assert long.startswith("a cute anime girl") and long.endswith("…") and len(long) == 40


def test_a_failing_first_step_stops_the_chain(tmp_path):
    result = tmp_path / "out.glb"
    marker = tmp_path / "second-ran"
    commands = [([sys.executable, "-c", "import sys; sys.exit(2)"], tmp_path),
                ([sys.executable, "-c", f"open(r'{marker}', 'w')"], tmp_path)]
    job = aa.AnimateJob("1" * 32, "rig", tmp_path, result, commands, 1.0, {})
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    assert job.status == "error" and not marker.exists()


def test_speed_defaults_to_the_presets_own_and_can_be_overridden(tmp_path, motions):
    run = tmp_path / "animate" / "knight__rig__20261005-120000"
    rigged = _glb(run / "knight_rigged.glb")
    for preset, asked, expected in (("walk", None, "1"), ("kick", None, "1.25"), ("kick", 1.5, "1.5")):
        manager = aa.AnimateJobManager(tmp_path / "animate", motions)
        job = manager.create_animation(rigged, preset, 0, asked)
        (command, _), = job.commands
        assert command[command.index("--speed") + 1] == expected
        assert job.record["motion"]["speed"] == float(expected)
    with pytest.raises(ValueError, match="between"):
        aa.AnimateJobManager(tmp_path / "animate", motions).create_animation(rigged, "walk", 0, 5)


def test_a_hidden_second_take_always_has_a_visible_first_one():
    """A take kept alongside another stays hidden (`take` 1) and keeps its `_b` id. When a move was
    kept as its second take only (Jog is Run B), that take becomes the move: it shows, id unchanged."""
    path = Path(__file__).resolve().parents[1] / "Ar3DRAHub" / "motions" / "presets.json"
    shipped = json.loads(path.read_text())["presets"]
    primary = {p["id"] for p in shipped if not p.get("take")}
    assert "walk_woman" in primary and "run_b" in primary and "walk_b" not in primary
    for preset in shipped:
        if preset.get("take"):
            assert str(preset["id"]).endswith("_b"), preset["id"]
            assert str(preset["id"]).removesuffix("_b") in primary, preset["id"]


def test_the_shipped_preset_moves_are_all_in_the_repo():
    """The moves live in git: a fresh clone with an empty motions folder offers no moves."""
    shipped = aa.presets()
    assert len(shipped) >= 29, len(shipped)  # the reviewed move set, takes A and B where both are kept
    assert {"idle", "walk", "run_b", "dance", "strafe_right"} <= {p["id"] for p in shipped}
    assert not {"roundhouse", "flying_kick", "kata", "sword_swing", "kick_b"} & {p["id"] for p in shipped}
    assert {"sprint", "crouch_sneak", "zombie_walk", "robot_dance", "one_two", "death_2"} <= {
        p["id"] for p in shipped}


def test_free_ports_are_two_different_ports_that_bind():
    import socket
    ports = aa.free_ports(2)
    assert len(set(ports)) == 2
    for port in ports:
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", port))  # where SkinTokens' Blender server binds


def test_a_rig_job_hands_skintokens_free_ports_not_fixed_ones(tmp_path):
    # Windows can reserve SkinTokens' fixed ports for Hyper-V/WSL (#193): every run picks
    # its own, through the variables the patched spec.py reads.
    job = _fake_job(tmp_path, "import os; open(r'{out}', 'w').write("
                              "os.environ['SKINTOKENS_PORT'] + ' ' + os.environ['SKINTOKENS_BPY_PORT'])")
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    port, bpy_port = (int(p) for p in job.result.read_text().split())
    assert port != bpy_port


def test_an_older_install_gets_the_port_edit_before_it_rigs(tmp_path, monkeypatch):
    spec = tmp_path / "src" / "server" / "spec.py"
    spec.parent.mkdir(parents=True)
    spec.write_text("import os\nPORT = 59875\nBPY_PORT = 59876\n")
    monkeypatch.setattr(aa, "port_edits", lambda: [e for e in _portable().EDITS
                                                   if "SKINTOKENS_" in e.new])
    assert aa.ensure_port_patch(tmp_path) is None
    assert 'os.environ.get("SKINTOKENS_BPY_PORT"' in spec.read_text()
    assert aa.ensure_port_patch(tmp_path) is None  # already there: nothing to do


def test_a_tree_the_port_edit_no_longer_fits_still_rigs(tmp_path):
    # Upstream moved on: the rig keeps upstream's ports (today's behaviour) and says why.
    (tmp_path / "src" / "server").mkdir(parents=True)
    (tmp_path / "src" / "server" / "spec.py").write_text("PORT = 1234\n")
    assert "port" in aa.ensure_port_patch(tmp_path).lower()


def _portable():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import patch_skintokens_portable
    return patch_skintokens_portable
