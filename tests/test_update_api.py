"""The version is reported without an update check: no GitHub, no network."""

from __future__ import annotations

import update_api
from Ar3DRAHub import __version__


def test_reports_the_current_version_and_never_checks():
    result = update_api.check(current="0.5.9")
    assert result == {"schema_version": 1, "enabled": False, "current": "0.5.9",
                      "latest": None, "url": None, "newer": False, "command": ""}


def test_defaults_to_the_package_version():
    assert update_api.check()["current"] == __version__


def test_older_call_shapes_still_work_without_a_network():
    # The old check accepted these; the disabled check ignores them and never fetches.
    result = update_api.check(current="0.2.0", now=1_000_000.0, cache="ignored", fetch=None,
                              env={"I3D_NO_UPDATE_CHECK": "1"}, repo="ignored", family="windows")
    assert result["enabled"] is False and result["newer"] is False
