"""The Setup card for WTiVo: what it promises, and whether the button can keep it.

The Watertight step already had a working engine of its own. This card is the optional
faster one, so the things worth checking are the ones that would make it a trap: a size
that is not stated, a machine it claims to run on but cannot, or a Remove that deletes
the studio's own files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import backend_catalog as bc
import download_api as dl


@pytest.fixture()
def card():
    return bc.BY_ID["wativo"]


def test_the_card_states_what_it_downloads(card):
    """`AGENTS.md`: name the route, state the size, before anything is fetched."""
    assert card.bytes_expected > 7 * 1024 ** 3
    assert any("torch" in w.label for w in card.weights)


def test_the_size_is_the_measured_one_not_a_guess(card):
    """Setup shows this number over the button, and the card's own words must agree with
    it. It was 2.5 GB from the wheel list; a real install came to 7.1 GB, because torch's
    CUDA libraries are most of it."""
    venv = [w for w in card.weights if "environment" in w.label][0]
    assert venv.bytes_expected >= 7 * 1024 ** 3
    assert "7 GB" in card.tradeoff
    assert "2.5 GB" not in card.tradeoff


def test_it_runs_on_an_nvidia_card_and_not_elsewhere(card):
    assert card.runs_on == (bc.NVIDIA,)
    assert card.runs_here(bc.APPLE) is False
    assert card.runs_here(bc.AMD) is False
    # Windows only: the extension is a .pyd. A Linux NVIDIA box must be told so on the
    # card, not discover it when the installer refuses (os_family() says "linux").
    assert card.runs_here(bc.NVIDIA, family="linux") is False
    assert card.runs_here(bc.NVIDIA, family="windows") is True


def test_the_setup_button_has_a_command_behind_it(card):
    assert card.automated_setup is (card.id in dl.COMMANDS)
    command = dl.COMMANDS["wativo"]
    assert command[1].endswith("bootstrap_wativo.py")
    # The browser already asked; a second prompt on a stdin nobody is attached to hangs.
    assert command[-1] == "--yes"


def test_installed_means_the_checkout_and_an_interpreter(card):
    """The script alone cannot run: the extension needs the venv's own 3.12."""
    probes = [p.name for p in card.build_probes]
    assert "wtivo.py" in probes
    assert "python.exe" in probes or "python" in probes


def test_it_is_marked_installed_only_when_both_halves_exist(tmp_path, monkeypatch):
    import dataclasses

    card = bc.BY_ID["wativo"]
    home = tmp_path / "wativo"
    (home / ".venv" / "Scripts").mkdir(parents=True)
    python = home / ".venv" / "Scripts" / "python.exe"
    python.write_bytes(b"")
    script = home / "wtivo.py"
    cut = dataclasses.replace(card, build_probes=(python, script))
    assert cut.built_on(bc.NVIDIA) is False          # no script yet
    script.write_text("", encoding="utf-8")
    assert cut.built_on(bc.NVIDIA) is True


def test_removal_would_not_reach_the_studios_own_files(card):
    """Remove deletes what the card lists, so every path must stay inside its own folder."""
    home = Path(card.weights[0].path).parent
    assert home.name == "wativo"
    for weight in card.weights:
        assert str(weight.path).startswith(str(home))


def test_the_licence_line_names_both_halves(card):
    """MIT is the code; the native extension is CGAL's. A card that says only 'MIT' would
    hide the one fact someone checking would want."""
    assert "MIT" in card.license_name
    assert "CGAL" in card.license_name
