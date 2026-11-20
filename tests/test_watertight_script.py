"""The `scripts/watertight.py` wrapper: its two decisions, not its arithmetic.

The rebuild itself is covered by `tests/test_watertight.py`. What can go wrong *here* and
nowhere else is finding the input's licence record, and the shape of the sidecar written
beside the result -- both cheap to get wrong in a way that only shows up weeks later,
when someone asks what made a file and the answer is missing.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import watertight as cli            # noqa: E402  (the wrapper, not Ar3DRAHub/watertight)

# `finish_api` imports its siblings from `viewer/`, so that folder has to be importable the
# way the studio runs it.
VIEWER = REPO / "viewer"
if str(VIEWER) not in sys.path:
    sys.path.insert(0, str(VIEWER))

from viewer import finish_api       # noqa: E402


def write_cube(path: Path, size: float = 1.0) -> Path:
    import trimesh

    trimesh.creation.box(extents=(size, size, size)).export(path)
    return path


@pytest.fixture(scope="module")
def cube(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cube")
    return write_cube(directory / "cube.glb")


# --- finding the input's licence record -------------------------------------------------

def test_find_source_record_prefers_an_explicit_path(tmp_path):
    mesh = tmp_path / "model.glb"
    explicit = tmp_path / "elsewhere.json"
    sidecar = tmp_path / "model.provenance.json"
    for path in (explicit, sidecar):
        path.write_text("{}", encoding="utf-8")
    assert cli.find_source_record(mesh, explicit) == explicit
    assert cli.find_source_record(mesh, sidecar) == sidecar


def test_find_source_record_falls_back_to_a_sibling_sidecar(tmp_path):
    mesh = tmp_path / "model.glb"
    mesh.write_bytes(b"")
    sidecar = tmp_path / "model.provenance.json"
    sidecar.write_text("{}", encoding="utf-8")
    assert cli.find_source_record(mesh, None) == sidecar


def test_find_source_record_falls_back_to_the_finish_run_s_record(tmp_path):
    """A finish run keeps the original upload's record in `input/source.json`, while the
    mesh handed to this tool sits one directory up under `finish/`."""
    steps = tmp_path / "run" / "finish"
    steps.mkdir(parents=True)
    mesh = steps / "refined.glb"
    mesh.write_bytes(b"")
    record = tmp_path / "run" / "input" / "source.json"
    record.parent.mkdir(parents=True)
    record.write_text("{}", encoding="utf-8")
    assert cli.find_source_record(mesh, None) == record


def test_find_source_record_says_none_rather_than_guessing(tmp_path):
    mesh = tmp_path / "model.glb"
    mesh.write_bytes(b"")
    assert cli.find_source_record(mesh, None) is None


# --- the sidecar -------------------------------------------------------------------------

class FakeResult:
    """Just the fields `run_summary` reads. Not a re-derivation of `WatertightResult`:
    the numbers are meaningless here, and what is under test is the record's shape."""

    def __init__(self):
        self.voxel = 0.00291065428169276
        self.shells_before = 97
        self.shells_after = 1
        self.boundary_edges = 0
        self.notes = []
        self.faces = np.zeros((1234, 3), dtype=np.int64)


def source_record() -> dict:
    return {
        "schema_version": 1,
        "license": {"name": "MIT (TRELLIS.2)", "conditions": []},
        "output": {"classification": "commercial-clear"},
        "model": {"backend": "trellis2"},
        "components": [],
    }


def test_run_summary_names_both_face_counts_and_the_voxel():
    summary = cli.run_summary(FakeResult(), 5000, 300)
    assert summary["faces_in"] == 5000
    assert summary["faces_out"] == 1234
    assert summary["shells_before"] == 97
    assert summary["shells_after"] == 1
    assert summary["across"] == 300
    assert summary["voxel"] == pytest.approx(0.00291065428169276)


def test_sidecar_keeps_the_input_licence_and_records_the_tool(tmp_path):
    source = write_cube(tmp_path / "source.glb")
    output = tmp_path / "result.glb"
    write_cube(output, size=1.1)
    written = cli.write_sidecar(source, output, source_record(),
                                cli.run_summary(FakeResult(), 12, 300))
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert written.name == "result.provenance.json"
    assert payload["operation"] == "watertight"
    # The licence is the input's, unchanged: an MIT instrument adds no condition.
    assert "MIT (TRELLIS.2)" in payload["license"]["name"]
    assert payload["output"]["classification"] == "commercial-clear"
    assert payload["watertight"]["faces_out"] == 1234
    assert payload["watertight"]["across"] == 300


def test_sidecar_does_not_turn_the_tool_into_a_licence_term(tmp_path):
    """The failure this guards: 'MIT' appearing in the licence *name*, which would read as
    a second licence on the output.

    The tool is recorded under `component_records`, not `components`: `components` is
    rebuilt from the source records' own lists, so a tool listed there would read as
    something the source used.
    """
    source = write_cube(tmp_path / "source.glb")
    output = tmp_path / "result.glb"
    write_cube(output, size=1.1)
    payload = json.loads(cli.write_sidecar(
        source, output, source_record(), cli.run_summary(FakeResult(), 12, 300)
    ).read_text(encoding="utf-8"))

    assert "watertight" not in payload["license"]["name"].lower()
    tools = [c["component"] for c in payload["component_records"]]
    assert "Ar3DRAHub/watertight.py" in tools
    # And nothing was smuggled into the input's own component list.
    assert payload["components"] == []


def test_the_tool_never_downgrades_the_classification(tmp_path):
    """An MIT tool adds no condition, so a commercial-clear input must stay that way.

    The bug this exists for: `write_derived_sidecar` takes the output's class from every
    record it is given, and reads a missing class as "unverified" -- so an unnamed tool
    quietly turns a clear input into an unverified output.
    """
    source = write_cube(tmp_path / "source.glb")
    output = tmp_path / "result.glb"
    write_cube(output, size=1.1)
    payload = json.loads(cli.write_sidecar(
        source, output, source_record(), cli.run_summary(FakeResult(), 12, 300)
    ).read_text(encoding="utf-8"))
    assert payload["output"]["classification"] == "commercial-clear"
    assert payload["license"].get("verified") is not False


# --- the command line ---------------------------------------------------------------------

def test_check_only_reports_and_writes_nothing(cube, tmp_path, capsys):
    output = tmp_path / "out.glb"
    code = run_cli([str(cube), str(output), "--check-only"])
    assert code == 0
    assert not output.exists()
    assert "watertight" in capsys.readouterr().out


def test_a_cube_comes_back_watertight(cube, tmp_path):
    import trimesh

    output = tmp_path / "out.glb"
    assert run_cli([str(cube), str(output), "--across", "24"]) == 0
    rebuilt = trimesh.load(output, force="mesh", process=False)
    assert rebuilt.is_watertight
    assert rebuilt.volume == pytest.approx(1.0, abs=0.22)


# --- the external tool ---------------------------------------------------------------------

def test_find_wativo_prefers_the_vendored_install(tmp_path, monkeypatch):
    """`bootstrap_wativo.py` puts it in `vendor/wativo`, and the studio looks there too --
    one install, so a command-line run and an interface run use the same one."""
    monkeypatch.setattr(cli, "WTIVO_VENDOR", tmp_path / "vendor-wativo")
    cli.WTIVO_VENDOR.mkdir()
    (cli.WTIVO_VENDOR / "wtivo.py").write_text("", encoding="utf-8")
    onpath = tmp_path / "onpath"
    onpath.write_bytes(b"")
    monkeypatch.delenv("WTIVO_HOME", raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: str(onpath))
    assert cli.find_wativo(None) == cli.WTIVO_VENDOR / "wtivo.py"


def test_find_wativo_prefers_the_argument(tmp_path, monkeypatch):
    binary = tmp_path / "wativo"
    binary.write_bytes(b"")
    monkeypatch.setenv("WTIVO_HOME", str(tmp_path / "from-env"))
    assert cli.find_wativo(binary) == binary


def test_find_wativo_falls_back_to_the_environment(tmp_path, monkeypatch):
    """`$WTIVO_HOME` is the same escape hatch the studio reads: one variable system, so a
    person following the studio's hint is found by the command line too."""
    monkeypatch.setattr(cli, "WTIVO_VENDOR", tmp_path / "absent")
    home = tmp_path / "wtivo"
    home.mkdir()
    (home / "wtivo.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("WTIVO_HOME", str(home))
    assert cli.find_wativo(None) == home / "wtivo.py"


def test_find_wativo_falls_back_to_path(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "WTIVO_VENDOR", tmp_path / "absent")
    binary = tmp_path / "onpath"
    binary.write_bytes(b"")
    monkeypatch.setenv("WTIVO_HOME", str(tmp_path / "missing"))
    monkeypatch.setattr(cli.shutil, "which", lambda name: str(binary))
    assert cli.find_wativo(None) == binary


def test_find_wativo_python_takes_the_vendored_venv(tmp_path, monkeypatch):
    """A `.py` found in the vendored folder is run by the venv beside it, never by the
    studio's 3.11: the extension is a 3.12 `.pyd` and 3.11 cannot load it."""
    monkeypatch.delenv("WTIVO_PYTHON", raising=False)
    venv = tmp_path / ".venv" / "Scripts"
    venv.mkdir(parents=True)
    (venv / "python.exe").write_bytes(b"")
    assert cli.find_wativo_python(tmp_path) == venv / "python.exe"


def test_find_wativo_python_prefers_an_explicit_override(tmp_path, monkeypatch):
    explicit = tmp_path / "a-python.exe"
    explicit.write_bytes(b"")
    monkeypatch.setenv("WTIVO_PYTHON", str(explicit))
    assert cli.find_wativo_python(tmp_path) == explicit


def test_find_wativo_python_says_none_rather_than_returning_ours(tmp_path, monkeypatch):
    monkeypatch.delenv("WTIVO_PYTHON", raising=False)
    assert cli.find_wativo_python(tmp_path) is None


def test_find_wativo_says_none_when_the_argument_points_nowhere(tmp_path, monkeypatch):
    """An explicit path that does not exist is a mistake worth reporting, not a fallback."""
    monkeypatch.setenv("WTIVO_HOME", str(tmp_path / "from-env"))
    assert cli.find_wativo(tmp_path / "nope") is None


def test_wativo_command_passes_the_chosen_quality(tmp_path):
    """WTiVo takes `--input`/`--output` (not positionals) and `--input-res`/`--final-res`
    for its resolution. The caller picks one of two measured points on WTiVo's own scale;
    the CPU engine's `--across` is never mapped onto it. Checked against `wtivo.py` in
    SimarglV-WatertightVoxel."""
    command = cli.wativo_command([str(tmp_path / "wativo")], Path("in.glb"), Path("out.glb"), 512, None)
    assert command == [str(tmp_path / "wativo"), "--input", "in.glb", "--output", "out.glb",
                       "--input-res", "512", "--final-res", "512"]
    assert "--resolution" not in command and "--across" not in command


def test_wativo_has_no_voxel_size_and_says_so(tmp_path, capsys):
    """WTiVo sizes its grid from the model's extent and the chosen resolution: there is no
    voxel flag to pass. Dropping it silently would look like it had been honoured."""
    command = cli.wativo_command([str(tmp_path / "wativo")], Path("in.glb"), Path("out.glb"), 512, 0.002)
    assert "--voxel-size" not in command and "--voxel" not in command
    assert "no --voxel-size" in capsys.readouterr().out


def test_the_external_run_keeps_wtivo_s_output_on_stdout(tmp_path, monkeypatch):
    """WTiVo redirects its stdout to a log file when it is not a tty; WTIVO_SUBPROCESS is
    its own escape hatch, and without it the run's log shows nothing of the tool."""
    (tmp_path / "out.glb").write_bytes(b"")
    (tmp_path / "in.glb").write_bytes(b"")
    seen = {}

    class Completed:
        returncode = 0
        stdout = ""

    def fake_run(command, **kwargs):
        seen["env"] = kwargs.get("env")
        return Completed()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    cli.run_external([str(tmp_path / "python.exe")], tmp_path / "in.glb", tmp_path / "out.glb", 512, None)
    assert seen["env"].get("WTIVO_SUBPROCESS") == "1"


def test_the_external_cleanup_drops_crumbs_the_tool_leaves(tmp_path):
    """WTiVo at 512 leaves the body plus specks; the promise is one closed body. A shell
    below KEEP_FRACTION of the biggest is a crumb, and it goes -- the body stays."""
    import trimesh

    body = trimesh.creation.icosphere(subdivisions=3)          # 1,280 faces
    crumb = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    crumb.apply_translation([2.0, 0.0, 0.0])
    messy = trimesh.util.concatenate([body, crumb])
    path = tmp_path / "messy.glb"
    messy.export(path)

    summary = cli._drop_crumbs(path)
    assert summary == {"dropped_shells": 1, "dropped_faces": 12, "shells_after": 1}
    reloaded = trimesh.load(path, force="mesh", process=False)
    assert len(reloaded.split(only_watertight=False)) == 1


def test_a_clean_external_output_is_left_byte_for_byte(tmp_path):
    """A result that is already one body must not be rewritten: the tool's file stays the
    tool's file, bit for bit, and only a messy one gets cleaned."""
    import trimesh

    path = tmp_path / "clean.glb"
    trimesh.creation.icosphere(subdivisions=2).export(path)
    before = path.read_bytes()
    assert cli._drop_crumbs(path) is None
    assert path.read_bytes() == before


def test_the_external_record_names_the_interpreter_and_script(tmp_path, monkeypatch):
    """`binary` lied: it held the python.exe path. The record should say what ran, at
    which resolution, and that our cleanup tidied the result."""
    from types import SimpleNamespace

    script = tmp_path / "wtivo.py"
    script.write_text("", encoding="utf-8")
    python = tmp_path / "python.exe"
    python.write_bytes(b"")
    source = write_cube(tmp_path / "source.glb")
    output = write_cube(tmp_path / "out.glb", size=1.1)
    args = SimpleNamespace(wativo_script=script, wativo_python=python,
                           wativo=None, across=300, wativo_res=1536, voxel=None)
    monkeypatch.setattr(cli, "run_external", lambda argv, src, out, res, voxel: None)
    monkeypatch.setattr(cli, "_drop_crumbs",
                        lambda path: {"dropped_shells": 3, "dropped_faces": 36, "shells_after": 1})
    assert cli._run_external(source, output, args, source_record()) == 0
    payload = json.loads(output.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    tool = payload["external_tools"][0]
    assert tool["interpreter"].endswith("python.exe")
    assert tool["script"].endswith("wtivo.py")
    assert "binary" not in tool
    assert payload["parameters"]["resolution"] == 1536
    assert "across" not in payload["parameters"]
    assert payload["postprocess"] == {
        "component": "Ar3DRAHub/watertight.py (keep_body)",
        "purpose": "dropped whole shells the external tool left detached",
        "license": "MIT",
        "dropped_shells": 3, "dropped_faces": 36, "shells_after": 1,
    }


def test_the_external_tool_keeps_the_input_licence_and_names_itself(tmp_path):
    """The whole point of `external_tools`: the output is not relicensed, and the record
    still says the file went through a GPL tool."""
    from Ar3DRAHub.provenance import WTIVO_COMPONENT, write_external_tool_sidecar

    source = write_cube(tmp_path / "source.glb")
    output = write_cube(tmp_path / "out.glb", size=1.1)
    written = write_external_tool_sidecar(source, output, operation="watertight",
                                          tool=WTIVO_COMPONENT,
                                          source_record=source_record(),
                                          parameters={"across": 300})
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert payload["license"]["name"] == "MIT (TRELLIS.2)"
    assert payload["output"]["classification"] == "commercial-clear"
    assert payload["external_tools"][0]["component"].startswith("WTiVo")
    assert "GPL" in payload["external_tools"][0]["license"]
    assert payload["components"] == []


def test_the_external_tool_is_not_merged_into_the_licence_name(tmp_path):
    """The failure this guards: collapsing GPL into the licence *name*, which reads as a
    licence on the output. WTiVo ran; it did not relicense anything."""
    from Ar3DRAHub.provenance import WTIVO_COMPONENT, write_external_tool_sidecar

    source = write_cube(tmp_path / "source.glb")
    output = write_cube(tmp_path / "out.glb", size=1.1)
    payload = json.loads(write_external_tool_sidecar(
        source, output, operation="watertight", tool=WTIVO_COMPONENT,
        source_record=source_record()).read_text(encoding="utf-8"))

    assert "GPL" not in payload["license"]["name"]
    assert payload["license"]["name"] == "MIT (TRELLIS.2)"


def test_a_missing_external_tool_says_what_to_do(cube, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WTIVO_HOME", str(tmp_path / "missing"))
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    output = tmp_path / "out.glb"
    with pytest.raises(SystemExit) as raised:
        run_cli([str(cube), str(output), "--tool", "wativo", "--wativo", str(tmp_path / "nope")])
    message = str(raised.value)
    assert "WTiVo not found" in message
    assert "--tool own" in message
    assert not output.exists()


def test_find_wativo_home_prefers_the_vendored_checkout(tmp_path, monkeypatch):
    """The bootstrap puts WTiVo in `vendor/wativo`; that is where the studio looks first."""
    monkeypatch.setattr(finish_api, "WTIVO_VENDOR", tmp_path / "vendor-wativo")
    vendored = finish_api.WTIVO_VENDOR
    vendored.mkdir()
    (vendored / "wtivo.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("WTIVO_HOME", str(tmp_path / "elsewhere"))
    assert finish_api.find_wativo_home() == vendored


def test_find_wativo_home_takes_a_checkout_or_the_script(tmp_path, monkeypatch):
    monkeypatch.setattr(finish_api, "WTIVO_VENDOR", tmp_path / "absent")
    for name in ("WTIVO_HOME", "WTIVO_DIR"):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "wtivo"
    home.mkdir()
    (home / "wtivo.py").write_text("", encoding="utf-8")
    assert finish_api.find_wativo_home(home) == home
    assert finish_api.find_wativo_home(home / "wtivo.py") == home


def test_find_wativo_home_reads_the_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(finish_api, "WTIVO_VENDOR", tmp_path / "absent")
    home = tmp_path / "wtivo"
    home.mkdir()
    (home / "wtivo.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("WTIVO_HOME", str(home))
    assert finish_api.find_wativo_home() == home


def test_find_wativo_home_says_none_rather_than_guessing(tmp_path, monkeypatch):
    monkeypatch.setattr(finish_api, "WTIVO_VENDOR", tmp_path / "absent")
    for name in ("WTIVO_HOME", "WTIVO_DIR"):
        monkeypatch.delenv(name, raising=False)
    assert finish_api.find_wativo_home() is None


def test_the_studio_prefers_an_explicit_wativo_python(tmp_path, monkeypatch):
    """`$WTIVO_PYTHON` wins over the vendored venv, here and in the command line: an
    explicit choice beats a guess, and the two must agree on which interpreter runs."""
    venv = tmp_path / ".venv" / "Scripts"
    venv.mkdir(parents=True)
    (venv / "python.exe").write_bytes(b"")
    explicit = tmp_path / "other.exe"
    explicit.write_bytes(b"")
    monkeypatch.setenv("WTIVO_PYTHON", str(explicit))
    assert finish_api.find_wativo_python(tmp_path) == explicit


def test_find_wativo_python_falls_back_to_the_vendored_venv(tmp_path, monkeypatch):
    """`vendor/wativo/.venv` is the install the bootstrap makes: its own 3.12, its own
    torch 2.8.0 pin. Nothing on the studio's side can stand in for it."""
    monkeypatch.delenv("WTIVO_PYTHON", raising=False)
    venv = tmp_path / ".venv" / "Scripts"
    venv.mkdir(parents=True)
    (venv / "python.exe").write_bytes(b"")
    assert finish_api.find_wativo_python(tmp_path) == venv / "python.exe"


def test_wativo_readiness_explains_a_missing_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(finish_api, "WTIVO_VENDOR", tmp_path / "absent")
    for name in ("WTIVO_HOME", "WTIVO_DIR"):
        monkeypatch.delenv(name, raising=False)
    ready, reason = finish_api.wativo_readiness()
    assert ready is False
    assert "WTIVO_HOME" in reason


def test_the_engine_choice_reaches_the_worker_command(monkeypatch, tmp_path):
    """`external` must pass the interpreter and the script, not a bare name: the studio
    runs 3.11 and WTiVo's extension is built for 3.12, so which Python runs it is the
    whole point."""
    home = tmp_path / "wtivo"
    home.mkdir()
    (home / "wtivo.py").write_text("", encoding="utf-8")
    python = home / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    monkeypatch.setattr(finish_api, "WTIVO_VENDOR", tmp_path / "absent")
    monkeypatch.setenv("WTIVO_HOME", str(home))

    command = finish_api.watertight_command(Path("in.glb"), Path("out.glb"), 300, "external")
    assert "--tool" in command and "wativo" in command
    assert "--wativo-python" in command and "--wativo-script" in command
    assert "--wativo-res" in command and "512" in command
    assert str(python) in command
    own = finish_api.watertight_command(Path("in.glb"), Path("out.glb"), 300, "own")
    assert "--tool" not in own and "--wativo-res" not in own


def test_choosing_the_external_engine_without_one_says_so(tmp_path, monkeypatch):
    """Better a clear refusal than a command line that fails on a missing file."""
    monkeypatch.setattr(finish_api, "WTIVO_VENDOR", tmp_path / "absent")
    for name in ("WTIVO_HOME", "WTIVO_DIR"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="WTIVO_HOME"):
        finish_api.watertight_command(Path("in.glb"), Path("out.glb"), 300, "external")


def run_cli(argv: list[str]) -> int:
    """`main()` without a subprocess: argparse reads `sys.argv`, and the module is imported."""
    import sys as _sys

    saved = _sys.argv
    _sys.argv = ["watertight.py", *argv]
    try:
        return cli.main()
    finally:
        _sys.argv = saved
