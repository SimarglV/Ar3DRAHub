"""Tests for scripts/bootstrap_marigold_v2.py: what it says before a byte moves, and consent.

The venv build and the weights download are the real, expensive steps; here both are
always replaced, so no test can reach the network or create a real vendor/marigold-v2.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "viewer")]

import backend_catalog  # noqa: E402
import bootstrap_marigold_v2 as boot  # noqa: E402


def test_announcement_names_backend_route_size_and_licence():
    text = boot.announcement()
    assert "Marigold V2" in text and "diffusers" in text and "NVIDIA" in text
    assert "22.6 GB" in text and "MIT" in text
    for repo, _, size in boot.WEIGHTS:
        assert repo in text and f"{size:.2f}" in text


def test_catalogue_card_quotes_the_same_sources():
    card = backend_catalog.BY_ID["marigold-v2"]
    assert [w.source for w in card.weights] == [repo for repo, _, _ in boot.WEIGHTS]


def test_nothing_downloads_without_a_yes_and_nobody_to_ask(monkeypatch, capsys):
    monkeypatch.setattr(boot.sys, "stdin", io.StringIO(""))
    assert boot.main([]) == 1
    assert "Refusing to download" in capsys.readouterr().out


def test_a_yes_builds_the_venv_then_the_weights(monkeypatch):
    done = []
    monkeypatch.setattr(boot.host, "find_uv", lambda: "uv")
    monkeypatch.setattr(boot, "build_env", lambda uv: done.append("env"))
    monkeypatch.setattr(boot, "install_weights", lambda: done.append("weights"))
    assert boot.main(["--yes"]) == 0 and done == ["env", "weights"]


def test_weights_only_skips_the_venv(monkeypatch):
    done = []
    monkeypatch.setattr(boot, "build_env", lambda uv: done.append("env"))
    monkeypatch.setattr(boot, "install_weights", lambda: done.append("weights"))
    assert boot.main(["--yes", "--weights-only"]) == 0 and done == ["weights"]
