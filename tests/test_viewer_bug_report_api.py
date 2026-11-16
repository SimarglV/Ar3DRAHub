"""Copy a failure's details: the home folder becomes ~, so nothing pasted carries the user's name."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import bug_report_api as br

HOME = "/Users/ada"


def test_the_home_folder_becomes_a_tilde_everywhere():
    assert br.scrub(f"{HOME}/Ar3DRAHub and {HOME}", HOME) == "~/Ar3DRAHub and ~"
    assert br.scrub(r"C:\Users\ada\Ar3DRAHub", r"C:\Users\ada") == r"~\Ar3DRAHub"


def test_copy_text_has_the_message_and_the_details_scrubbed():
    text = br.copy_text("Finish crashed", f"OSError: {HOME}/Ar3DRAHub/output/x.glb", home=HOME)
    assert text.startswith("Finish crashed") and "OSError: ~/Ar3DRAHub/output/x.glb" in text
    assert HOME not in text


def test_copy_text_defaults_to_the_real_home():
    # No home passed: the current user's home is used, whatever it is.
    text = br.copy_text("broke", "details here")
    assert text.startswith("broke") and "details here" in text


def test_every_spelling_of_a_windows_home_folder_is_hidden():
    # A failed command prints its paths doubled (C:\\Users\\ada); tools also write C:/Users/ada,
    # and Windows does not care about case.
    home = r"C:\Users\ada"
    text = (r"File C:\Users\ada\lab.py; ['C:\\Users\\ada\\.local\\bin\\uv.EXE']; "
            r"C:/Users/ada/x; c:\users\ADA\y")
    scrubbed = br.scrub(text, home)
    assert "ada" not in scrubbed.lower()
    assert scrubbed.startswith(r"File ~\lab.py; ['~\\.local")


def test_a_pasted_message_is_scrubbed_too():
    # A Setup failure copied into the message box carried the user's name in every path.
    home = r"C:\Users\ada"
    text = br.copy_text(r"Setup failed: C:\Users\ada\.local\bin\uv.EXE",
                        r"log at C:\\Users\\ada\\x", home=home)
    assert "ada" not in text
    assert text.startswith(r"Setup failed: ~\.local")


def test_scrub_hides_only_the_whole_home_folder():
    assert br.scrub(r"C:\Users\ada\x and C:\Users\adam\y", r"C:\Users\ada") == r"~\x and C:\Users\adam\y"
    assert br.scrub("/Users/ada/x /Users/adam/y /Users/ada.b in /Users/ada.", "/Users/ada") == \
        "~/x /Users/adam/y /Users/ada.b in ~."
    assert br.scrub("/Users/ada/x", "/Users/ada/") == "~/x"           # a trailing slash changes nothing


def test_scrub_leaves_text_alone_when_home_is_a_root():
    """Some services run with home at / or C:\\: replacing that would turn every path into ~."""
    for home in ("/", "C:\\", "", "C:"):
        assert br.scrub("/opt/lab/run.log C:\\lab\\x", home) == "/opt/lab/run.log C:\\lab\\x"
