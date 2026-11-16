"""The root-level Windows batch files: what breaks them without running the install.

cmd.exe reads a .bat line by line, and inside a parenthesised ``if (...)`` block it
counts ``(`` and ``)``. An ``echo`` that prints a parenthesis inside such a block
(e.g. ``echo ... exactly (the bpy wheels) ...``) makes the parser see a nested block
and die with ``. was unexpected at this time.`` — which, on a double-clicked installer,
closes the window before any message is read. This test is the cheap guard for that
class of bug: no ``echo`` line may print an unescaped, unquoted parenthesis while it is
inside a block.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BAT_FILES = sorted(REPO.glob("*.bat"))


def _bare_paren_lines(text: str) -> dict[int, int]:
    """Map line number -> block depth at line start, for echo lines with a bare paren.

    Tracks ``(``/``)`` depth across the file with cmd's two literal escapes: ``^``
    escapes the next character, and text inside ``"`` is literal. A line is an ``echo``
    only when it is ``echo`` followed by whitespace (so ``echo.``, ``echo(`` and the
    other blank-line idioms are ignored).
    """
    depth = 0
    hits: dict[int, int] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped[:4].lower() == "echo" and stripped[4:5].isspace():
            # Any bare ( or ) in the text to print?
            in_quote = False
            for i, ch in enumerate(stripped):
                if ch == '"' and (i == 0 or stripped[i - 1] != "^"):
                    in_quote = not in_quote
                elif not in_quote and ch in "()":
                    if stripped[i - 1] != "^":
                        hits[lineno] = depth
                        break

        # Advance the block depth for the whole file, not just echo lines.
        in_quote = False
        for i, ch in enumerate(line):
            if ch == "^":
                continue  # next char is literal; skip it too
            if ch == '"' and (i == 0 or line[i - 1] != "^"):
                in_quote = not in_quote
            elif not in_quote and ch == "(":
                depth += 1
            elif not in_quote and ch == ")":
                depth -= 1
    return hits


def test_no_bat_echo_prints_a_parenthesis_inside_a_block():
    for path in BAT_FILES:
        text = path.read_text(encoding="utf-8")
        hits = _bare_paren_lines(text)
        for lineno, depth in hits.items():
            assert depth == 0, (
                f"{path.name}:{lineno} echoes a parenthesis inside an if/else block "
                f"(depth {depth}); cmd will abort with '. was unexpected at this time.' "
                "Remove the parentheses or escape them with ^."
            )


def test_install_bat_fetches_python_310_via_uv_not_a_system_python():
    """uv downloads the right CPython for the venv, so the installer must not depend on
    a Python 3.10 already being on PATH (a stale `py -3.10` registry entry used to fail
    silently). The venv is seeded so the unchanged `python -m pip` steps still work."""
    text = (REPO / "install.bat").read_text(encoding="utf-8")
    assert "uv venv" in text
    assert "--seed" in text
    assert "--python 3.10" in text
    assert "--no-project" in text  # avoid the studio's requires-python (>=3.11) warning
    assert "py -3.10" not in text
    assert "-m venv" not in text


def test_install_bat_keeps_the_motion_build_constraint():
    """Motion has a legacy setup.py that needs setuptools<81, so the install must stay on
    --no-build-isolation with an env setuptools pinned below 81."""
    text = (REPO / "install.bat").read_text(encoding="utf-8")
    assert '--no-build-isolation' in text
    assert '"setuptools<81"' in text
