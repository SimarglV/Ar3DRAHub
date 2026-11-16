"""The current version, reported without an update check.

The daily GitHub releases check is gone: this repo no longer phones the upstream repo. `check`
keeps the shape it always had (so the About page's version line and the notices path still
work) but is always disabled and never touches the network.
"""

from __future__ import annotations

from typing import Any

import Ar3DRAHub


def check(current: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Always disabled: the current version, and no "newer" answer to chase."""
    return {"schema_version": 1, "enabled": False, "current": current or Ar3DRAHub.__version__,
            "latest": None, "url": None, "newer": False, "command": ""}
