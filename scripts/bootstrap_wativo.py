#!/usr/bin/env python3
"""Install WTiVo, the fast watertight rebuild, into vendor/wativo with its own interpreter.

The studio's Watertight step has two engines. The default one is our own
(`Ar3DRAHub/watertight.py`, MIT, a few minutes on the CPU). This installs the optional
second one: WTiVo (SimarglV's WatertightVoxel), which rebuilds on the GPU in seconds.

It cannot share the studio's environment. WTiVo ships a native extension built for
**CPython 3.12** with **torch 2.8.0 + CUDA 12.8** pinned, and the studio runs 3.11 with
its own torch. So the checkout goes in vendor/wativo/ and gets a `.venv` of its own, the
way marigold-v2 does. The studio finds both by itself; nothing has to be exported.

Two things are downloaded here: the checkout (~21 MB, from GitHub) and the wheel set for
the venv (torch ~2.5 GB compressed; the finished venv is ~7.1 GB on disk). `AGENTS.md`
says a download path must name the backend, name the route, state the size, and require
an affirmative answer -- so it prints all of that and stops, unless `--yes` is given.

    python scripts/bootstrap_wativo.py            # says what it wants, then asks
    python scripts/bootstrap_wativo.py --yes      # for the viewer and for scripts
    python scripts/bootstrap_wativo.py --code-only

WTiVo is Windows x64 with an NVIDIA card only; the extension is a `.pyd`. On anything
else this says so and installs nothing.

Licence note, because the record in the sidecar says it too: WTiVo's own code is MIT, but
its native extension is built on CGAL's 3D Delaunay triangulation, which is
GPL-3.0-or-later. It runs as a separate process, so nothing links GPL into this repo or
into the models we ship -- but a file rebuilt by it was rebuilt by a GPL tool, and the
`.provenance.json` beside that file names it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "vendor" / "wativo"

# The upstream project. A zip, not a clone: there is no history to keep, only the files,
# and the `build/` folder with the prebuilt `.pyd` files comes with it.
ARCHIVE = "https://github.com/SimarglV/WatertightVoxel-ComfyuiNode/archive/refs/heads/main.zip"
APPROXIMATE_CODE_MB = 21

# The extension is compiled against these. torch must match exactly: a different minor
# version gives "DLL load failed while importing cppmodules", which reads like a broken
# install rather than a version mismatch, and that costs an hour.
SOURCE_PYTHON = "3.12"
TORCH = "2.8.0"
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
PACKAGES = ["numpy", "trimesh"]
# Measured after a real install, torch's CUDA libraries included. The wheel *download* is
# ~2.5 GB; the installed venv is what the Setup card reports, and this must agree with it.
APPROXIMATE_VENV_GB = 7.1

LICENCE = (
    "WTiVo code: MIT.\n"
    "  Its native extension is built on CGAL (GPL-3.0-or-later); it runs as a separate\n"
    "  process, so nothing here links it or ships it."
)


def venv_python() -> Path:
    return VENDOR / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def installed() -> bool:
    """A checkout and an interpreter, both where the studio looks for them."""
    return (VENDOR / "wtivo.py").is_file() and venv_python().is_file()


def supported() -> str | None:
    """Why this machine cannot run WTiVo, or None if it can."""
    if not sys.platform.startswith("win"):
        return ("WTiVo's native extension is a Windows x64 .pyd; there is no build for this "
                "platform. The Watertight step's own engine still works everywhere.")
    return None


def announcement(code: bool = True, venv: bool = True) -> str:
    """Exactly what is about to be fetched, before anything is."""
    lines = ["", "About to install:", "",
             "  backend: WTiVo (WatertightVoxel, SimarglV)",
             "  route:   prebuilt CUDA extension, on an NVIDIA card"]
    if code:
        lines += ["  code:    ~21 MB  github.com/SimarglV/WatertightVoxel-ComfyuiNode",
                  f"           -> {VENDOR}"]
    if venv:
        lines += [f"  venv:    {venv_python()}",
                  f"           CPython {SOURCE_PYTHON} + torch=={TORCH} (CUDA 12.8) "
                  f"+ {'/'.join(PACKAGES)}",
                  f"           ~{APPROXIMATE_VENV_GB:.1f} GB on disk once installed "
                  "(the torch wheel download is ~2.5 GB)"]
    lines += ["", "  licence: " + LICENCE, ""]
    return "\n".join(lines)


def fetch_checkout() -> None:
    """Download the project zip and unpack it into vendor/wativo."""
    if (VENDOR / "wtivo.py").is_file():
        print(f"already present: {VENDOR}", flush=True)
        return
    VENDOR.parent.mkdir(parents=True, exist_ok=True)
    archive = VENDOR.parent / "wtivo-main.zip"
    print(f"downloading {ARCHIVE}", flush=True)
    try:
        urllib.request.urlretrieve(ARCHIVE, archive)
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"could not download the WTiVo checkout: {exc}") from exc
    staging = VENDOR.parent / "_wtivo_unpack"
    shutil.rmtree(staging, ignore_errors=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(staging)
    # GitHub wraps everything in one folder named <repo>-<branch>.
    (root,) = [entry for entry in staging.iterdir() if entry.is_dir()]
    VENDOR.mkdir(parents=True, exist_ok=True)
    for entry in root.iterdir():
        target = VENDOR / entry.name
        if target.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target)
    shutil.rmtree(staging, ignore_errors=True)
    archive.unlink(missing_ok=True)
    print(f"unpacked into {VENDOR}", flush=True)


def build_env(uv: str) -> None:
    """Make the venv and install what the extension needs."""
    if not venv_python().is_file():
        subprocess.run([uv, "venv", str(VENDOR / ".venv"), "--python", SOURCE_PYTHON],
                       check=True)
    python = str(venv_python())
    subprocess.run([uv, "pip", "install", "--python", python, *PACKAGES], check=True)
    subprocess.run([uv, "pip", "install", "--python", python, f"torch=={TORCH}",
                    "--index-url", CUDA_INDEX], check=True)


def verify() -> bool:
    """Import the extension the way `wtivo.py` does, not the way a bare import does.

    Three things have to be true before `import cppmodules` works, and only the first is
    obvious: the venv must have torch 2.8.0, the `build/` folder must be on `sys.path`,
    and -- on Windows -- it must also be on the DLL search path, because the extension
    statically pulls in `libgmp-10.dll` and the TBB runtime that sit beside it. A plain
    `sys.path` insert gives "DLL load failed ... or one of its dependencies", which names
    the module rather than the DLL and sends you looking in the wrong place.

    Checked here rather than discovered during a run: this takes a second, and the
    alternative is an exit code fifteen minutes in.
    """
    probe = (
        "import os, sys, torch\n"
        "root = r'{root}'\n"
        "build = os.path.join(root, 'build')\n"
        "sys.path.insert(0, build)\n"
        "add = getattr(os, 'add_dll_directory', None)\n"
        "handle = add(build) if add else None\n"
        "import cppmodules\n"
        "print(torch.__version__)\n"
    ).format(root=VENDOR)
    completed = subprocess.run([str(venv_python()), "-c", probe],
                               capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        print(f"WTiVo's native extension will not load: {detail[-1] if detail else 'unknown'}")
        print("Usually torch differs from the pinned version, or the GPU needs the "
              "non-Blackwell build from build/ForNonBlackwellgpu*.rar.")
        return False
    print(f"WTiVo is ready (torch {completed.stdout.strip()}).")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the viewer and for scripts.")
    parser.add_argument("--code-only", action="store_true",
                        help="Fetch the checkout only, leaving the venv.")
    parser.add_argument("--venv-only", action="store_true",
                        help="Build the venv only, assuming the checkout is already there.")
    args = parser.parse_args(argv)

    reason = supported()
    if reason is not None:
        print(reason)
        return 1

    code = not args.venv_only
    venv = not args.code_only
    print(announcement(code=code, venv=venv))

    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing installed.")
            return 1

    if code:
        fetch_checkout()
    if venv:
        uv = shutil.which("uv")
        if uv is None:
            raise SystemExit("uv is required. Install it (install.ps1 / install.sh), then rerun.")
        build_env(uv)

    if installed() and not verify():
        return 1
    if installed():
        print("\nDone. The studio's Watertight step now offers the GPU engine.")
        print(f"    {venv_python()} {VENDOR / 'wtivo.py'} --input in.glb --output out.glb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
