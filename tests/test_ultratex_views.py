"""The UltraTex views preparer: silhouette tiles, reference framing and the eval
JSON entry. Pure parts only; no Blender, no network."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ultratex_views.py"


def _load():
    spec = importlib.util.spec_from_file_location("ultratex_views", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


views = _load()


def _tile(path: Path, alpha_box: int) -> None:
    alpha = Image.new("L", (16, 16), 0)
    alpha.paste(255, (alpha_box, alpha_box, 16 - alpha_box, 16 - alpha_box))
    rgba = Image.merge("RGBA", (alpha, alpha, alpha, alpha))
    path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(path, "WEBP")


def test_build_target_tiles_copies_the_silhouette_into_white_tiles(tmp_path):
    for index in range(6):
        _tile(tmp_path / "bump_normal_world" / f"{index:03d}.webp", alpha_box=index + 1)
    written = views.build_target_tiles(tmp_path)
    assert written == 12
    for subdir in ("albedo", "roughness_metallic"):
        for index in range(6):
            with Image.open(tmp_path / subdir / f"{index:03d}.webp").convert("RGBA") as tile:
                r, g, b, a = tile.split()
                assert r.getextrema() == (255, 255) and g.getextrema() == (255, 255)
                assert b.getextrema() == (255, 255)          # white RGB
                expected = Image.open(tmp_path / "bump_normal_world" / f"{index:03d}.webp")
                assert a.tobytes() == expected.convert("RGBA").getchannel("A").tobytes()


def test_frame_reference_crops_pads_and_squares():
    subject = Image.new("RGBA", (100, 100), (255, 255, 255, 0))
    alpha = Image.new("L", (100, 100), 0)
    alpha.paste(255, (25, 40, 75, 60))       # 50x20 subject, not square
    subject.putalpha(alpha)
    framed = views.frame_reference(subject, pad=1.2, size=128)
    assert framed.size == (128, 128)
    box = framed.getchannel("A").getbbox()
    assert box is not None
    # The subject stays centred after the crop+pad, with transparent corners.
    assert (box[0] + box[2]) / 2 == pytest.approx(64, abs=4)
    assert (box[1] + box[3]) / 2 == pytest.approx(64, abs=4)
    assert framed.getchannel("A").getpixel((0, 0)) == 0
    assert framed.getchannel("A").getpixel((64, 64)) == 255


def test_frame_reference_refuses_an_empty_subject():
    empty = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="no visible subject"):
        views.frame_reference(empty, pad=1.2, size=128)


def test_reference_tiles_fill_every_bucket(tmp_path):
    framed = Image.new("RGBA", (32, 32), (255, 255, 255, 255))
    written = views.reference_tiles(framed, tmp_path)
    assert written == len(views.REFERENCE_DIRS) * views.REFERENCE_FILES
    for folder in views.REFERENCE_DIRS:
        for index in range(views.REFERENCE_FILES):
            assert (tmp_path / folder / f"{index:03d}.webp").is_file()


def test_eval_entry_names_the_prompt_and_the_absolute_dir(tmp_path):
    entry = views.eval_entry(tmp_path / "spike")
    assert entry["prompt"] == ""                     # image-guided texturing
    assert Path(entry["image_dir"]).is_absolute()
    assert entry["id"] == "spike"


def test_render_command_runs_blender_headless():
    blender = Path("/bin/blender")
    command = views.render_command(Path("a.glb"), Path("out"), 512, blender)
    assert command[:7] == [str(blender), "--background", "--factory-startup",
                           "--python-exit-code", "1", "--python",
                           str(Path(views.REPO) / "scripts" / "blender_render_texverse_views.py")]
    assert command[-3:] == ["a.glb", "out", "512"]


def test_render_command_carries_the_guided_flag_when_asked():
    command = views.render_command(Path("a.glb"), Path("out"), 512, Path("blender.exe"), guided=True)
    assert command[-1] == "guided"
    plain = views.render_command(Path("a.glb"), Path("out"), 512, Path("blender.exe"))
    assert "guided" not in plain


def test_guided_dir_and_json_sit_beside_the_canonical_ones():
    output = Path("steps") / "ultratex-views"
    assert views.guided_dir(output) == Path("steps") / "ultratex-views-guided"
    canonical_json = Path("steps") / "ultratex-views.json"
    assert views.guided_json_path(canonical_json) == Path("steps") / "ultratex-views-guided.json"


def test_prepare_views_guided_writes_two_eval_entries(tmp_path, monkeypatch):
    mesh = tmp_path / "mesh.glb"
    image = tmp_path / "ref.png"
    output = tmp_path / "ultratex-views"
    mesh.write_bytes(b"glb")
    from PIL import Image as _Image
    _Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(image)

    calls = []
    monkeypatch.setattr(views.subprocess, "run", lambda command, check: calls.append(command))
    monkeypatch.setattr(views, "build_target_tiles", lambda renders: 12)
    monkeypatch.setattr(views, "reference_tiles", lambda framed, renders: 16)
    from Ar3DRAHub import matte
    monkeypatch.setattr(matte, "is_matted", lambda opened: True)
    monkeypatch.setattr(matte, "cut_out",
                        lambda rgb: _Image.new("RGBA", (4, 4), (255, 0, 0, 255)))
    monkeypatch.setattr(views, "frame_reference",
                        lambda cut, pad, size: _Image.new("RGBA", (16, 16), (0, 0, 0, 255)))

    written = views.prepare_views(mesh, image, output, 512, Path("blender.exe"), 1.2,
                                  tmp_path / "ultratex-views.json", guided=True)
    assert written == [tmp_path / "ultratex-views.json",
                       tmp_path / "ultratex-views-guided.json"]
    import json as _json
    entry = _json.loads(written[1].read_text(encoding="utf-8"))[0]
    assert entry["image_dir"].endswith("ultratex-views-guided")
    assert [call[-1] == "guided" for call in calls] == [False, True]
