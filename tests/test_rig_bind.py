"""The "keep my skeleton" path: coverage bind, then SkinTokens holds the skeleton fixed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rig_bind as rb


def test_demo_command_holds_the_skeleton_and_transfers():
    skintokens = Path("/skintokens")
    command = rb.demo_command(skintokens, Path("/in.glb"), Path("/out.glb"))
    assert command[0] == str(skintokens / "demo.py")
    assert "--use_skeleton" in command
    assert "--use_transfer" in command
    assert "--input" in command and "--output" in command


def test_clean_command_rewrites_in_place():
    clean, source, result = Path("/clean.py"), Path("/in.glb"), Path("/out.glb")
    assert rb.clean_command(clean, source, result) == \
        [str(clean), "--in", str(source), "--out", str(result)]


def test_free_ports_are_distinct():
    ports = rb.free_ports(2)
    assert len(ports) == 2
    assert ports[0] != ports[1]
    assert all(isinstance(p, int) and p > 0 for p in ports)


def test_port_env_names_the_two_skin_tokens_ports():
    env = rb.port_env()
    assert "SKINTOKENS_PORT" in env and "SKINTOKENS_BPY_PORT" in env
    assert env["SKINTOKENS_PORT"] != env["SKINTOKENS_BPY_PORT"]
