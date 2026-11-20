"""Copy a failure's details with the home folder hidden.

The studio's "Copy details" button posts the failed step's log here and gets back the same
text with the home folder replaced by ~ everywhere, so nothing pasted anywhere carries the
user's name in a path.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def scrub(text: str, home: str) -> str:
    """`text` with the home folder shown as ~, so no path carries the user's name.

    Every spelling a Windows path takes: as typed, with / for \\, doubled \\\\ (how Python
    prints a path inside a list, as in a failed command), and any letter case."""
    home = home.rstrip("\\/")
    if len(home) < 3:   # no home, or a root like / or C: (some services): nothing personal to hide
        return text
    spellings = {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}
    pattern = "|".join(re.escape(s) for s in sorted(spellings, key=len, reverse=True))
    # Only the whole folder name: a home of /Users/ada leaves /Users/adam and /Users/ada.b alone.
    return re.sub(f"(?:{pattern})(?![\\w-]|\\.\\w)", "~", text, flags=re.IGNORECASE if "\\" in home else 0)


def copy_text(message: str, details: str, home: str | None = None) -> str:
    """The scrubbed message and details, ready to paste anywhere."""
    home = str(Path.home()) if home is None else home
    message, details = scrub(message, home), scrub(details, home)
    return f"{message.strip()}\n\n{details}".strip()
