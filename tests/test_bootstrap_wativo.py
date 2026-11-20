"""Tests for scripts/bootstrap_wativo.py.

Nothing here downloads anything. The interesting parts are the two that decide whether an
install counts as present -- where the venv is, and whether the extension actually loads --
because getting either wrong costs a run rather than a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import bootstrap_wativo as boot  # noqa: E402


def test_the_venv_python_is_inside_the_vendored_checkout(tmp_path, monkeypatch):
    """The studio finds WTiVo by looking in vendor/wativo, so the venv has to be there
    too -- a venv somewhere else would be invisible to it."""
    monkeypatch.setattr(boot, "VENDOR", tmp_path / "vendor" / "wativo")
    assert boot.venv_python().parent.parent.parent == boot.VENDOR


def test_installed_needs_both_the_checkout_and_the_interpreter(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "VENDOR", tmp_path / "wativo")
    assert boot.installed() is False            # nothing yet
    (boot.VENDOR / "wtivo.py").parent.mkdir(parents=True, exist_ok=True)
    (boot.VENDOR / "wtivo.py").write_text("", encoding="utf-8")
    assert boot.installed() is False            # checkout but no interpreter
    boot.venv_python().parent.mkdir(parents=True, exist_ok=True)
    boot.venv_python().write_bytes(b"")
    assert boot.installed() is True


def test_an_unsupported_platform_is_named_not_guessed(monkeypatch):
    """WTiVo's extension is a Windows .pyd. On anything else, saying so beats a zip that
    unpacks into something that cannot run."""
    monkeypatch.setattr(boot.sys, "platform", "linux")
    reason = boot.supported()
    assert reason and "Windows" in reason
    monkeypatch.setattr(boot.sys, "platform", "win32")
    assert boot.supported() is None


def test_the_announcement_states_backend_route_size_and_licence():
    """`AGENTS.md`: a download path must name what it is about to fetch and how big it
    is before asking. A missing size is how somebody finds 7 GB on their disk."""
    text = boot.announcement()
    assert "WTiVo" in text
    assert "NVIDIA" in text
    assert "7.1 GB" in text and "21 MB" in text
    assert "MIT" in text and "CGAL" in text


def test_the_announcement_leaves_out_what_was_not_asked_for():
    text = boot.announcement(code=False)
    assert "github.com" not in text
    assert "7.1 GB" in text


def test_the_torch_pin_is_the_version_the_extension_was_built_against():
    """2.8.0 exactly: the .pyd is compiled against it, and a different minor version
    fails as 'DLL load failed', which reads like a broken install."""
    assert boot.TORCH == "2.8.0"
    assert boot.SOURCE_PYTHON == "3.12"
    assert "cu128" in boot.CUDA_INDEX


def test_verification_runs_the_extension_rather_than_trusting_the_files(tmp_path, monkeypatch):
    """A file being present is not the same as a DLL loading. The probe imports
    `cppmodules` through the venv's own interpreter, which is what a run does."""
    monkeypatch.setattr(boot, "VENDOR", tmp_path / "wativo")
    (boot.VENDOR / "build").mkdir(parents=True)
    seen: dict[str, list[str]] = {}

    class Completed:
        returncode = 0
        stdout = "2.8.0+cu128\n"
        stderr = ""

    def fake_run(command, **kwargs):
        seen["command"] = command
        return Completed()

    monkeypatch.setattr(boot.subprocess, "run", fake_run)
    assert boot.verify() is True
    assert str(boot.venv_python()) == seen["command"][0]
    joined = " ".join(seen["command"])
    assert "cppmodules" in joined and "torch" in joined


def test_verification_reports_a_failure_instead_of_claiming_success(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(boot, "VENDOR", tmp_path / "wativo")

    class Completed:
        returncode = 1
        stdout = ""
        stderr = "OSError: [WinError 126] caffe2_nvrtc.dll\n"

    monkeypatch.setattr(boot.subprocess, "run", lambda command, **kw: Completed())
    assert boot.verify() is False
    assert "will not load" in capsys.readouterr().out
