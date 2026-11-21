#!/usr/bin/env python3
"""Replace flash-attn with an SDPA fallback in the Hi3D v1.0 checkout, so refinement runs on Windows.

The official Hi3D stack pins flash-attn, which has no Windows build and no fallback
attention path. This patch makes the checkout prefer flash-attn when it is installed
(the measured Linux stack) and fall back to a torch SDPA implementation of the two
varlen entry points otherwise -- the Windows path. Both implement the same math
(softmax(QK^T/sqrt(d))V per packed span, non-causal), but the released weights were
measured on the flash-attn path, so the SDPA path is this repo's own port, not a
stack the authors validated.

The varlen helpers live in this module, importable and unit-tested; the shim file
written into the checkout is generated from their source, so the two cannot drift.

    python scripts/patch_hi3d_sdpa.py --root vendor/hi3d
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "hi3d"

#: The two files whose `_flash_attn` loader gets the SDPA fallback.
ATTENTION_FILES = (
    "sparc3d/nn/sparse/attention/full_attn.py",
    "sparc3d/models/autoencoders/blocks.py",
)

#: Written beside the package; imported only when flash-attn is absent.
SHIM_PATH = "sparc3d/_sdpa_varlen.py"

#: Once the loader says this, the patch is applied; re-running must be a no-op.
PATCH_MARKER = "from sparc3d import _sdpa_varlen"


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, *,
                                     dropout_p: float = 0.0,
                                     softmax_scale: float | None = None,
                                     causal: bool = False):
    """Self-attention over the packed spans of ``qkv`` [N, 3, H, D].

    Drop-in for flash-attn's varlen qkvpacked entry point, run as one fused SDPA
    call *per span* and concatenated. A span is a self-contained batch item: its
    queries attend its own keys and values only (non-causal, flash-attn's default).
    ``dropout_p`` must be 0 (inference).

    The previous port padded every span to one length and passed one mask the size
    of the whole packed sequence. For the transformer that sequence is every voxel
    (110,000 in the third pass), so the mask grew to tens of gigabytes and PyTorch
    could no longer use its flash or memory-efficient attention kernels — the ones
    that keep the softmax in float32 the way flash-attn does. It fell back to the
    math kernel, whose half-precision softmax was not what the weights were
    measured against, and the high-resolution passes lost whole slabs of geometry.
    Splitting by span needs no mask at all, so the fused kernels apply again.
    """
    if dropout_p:
        raise ValueError("the SDPA port supports dropout_p=0 only (inference)")
    if causal:
        raise ValueError("the SDPA port implements non-causal varlen attention")
    cu = cu_seqlens.to(dtype=torch.int64)
    batch = int(cu.numel()) - 1
    if batch < 1 or qkv.shape[0] != int(cu[-1]):
        raise ValueError("cu_seqlens does not describe qkv")
    if qkv.shape[1] != 3:
        raise ValueError("qkvpacked expects [N, 3, H, D]")
    if max_seqlen <= 0:
        return qkv.new_empty(0, qkv.shape[2], qkv.shape[3])
    scale = softmax_scale if softmax_scale is not None else qkv.shape[3] ** -0.5
    head_dim = qkv.shape[3]
    outputs = []
    for b in range(batch):
        start, end = int(cu[b]), int(cu[b + 1])
        if end <= start:
            continue
        span = qkv[start:end].permute(1, 2, 0, 3)               # [3, H, L, D]
        outputs.append(
            F.scaled_dot_product_attention(                      # [1, H, L, D]
                span[0].unsqueeze(0), span[1].unsqueeze(0), span[2].unsqueeze(0),
                scale=scale,
            )[0]
        )
    if not outputs:
        return qkv.new_empty(0, qkv.shape[2], head_dim)
    return torch.cat(outputs, dim=1).transpose(0, 1)             # [N, H, D]


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k,
                                    max_seqlen_q, max_seqlen_k, *,
                                    dropout_p: float = 0.0,
                                    softmax_scale: float | None = None,
                                    causal: bool = False):
    """Cross-attention: packed queries [Nq, H, D] over packed kv [Nk, 2, H, D].

    Drop-in for flash-attn's varlen kvpacked entry point, run as one fused SDPA
    call per query span. Query and key spans are described separately; within a
    batch item every query attends every key of the same item (non-causal).
    ``dropout_p`` must be 0 (inference). No mask: see the qkvpacked docstring for
    why a whole-sequence mask forced a half-precision fallback.
    """
    if dropout_p:
        raise ValueError("the SDPA port supports dropout_p=0 only (inference)")
    if causal:
        raise ValueError("the SDPA port implements non-causal varlen attention")
    cu_q = cu_seqlens_q.to(dtype=torch.int64)
    cu_k = cu_seqlens_k.to(dtype=torch.int64)
    batch = int(cu_q.numel()) - 1
    if batch < 1 or int(cu_k.numel()) - 1 != batch:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must share the batch size")
    if q.shape[0] != int(cu_q[-1]) or kv.shape[0] != int(cu_k[-1]):
        raise ValueError("cu_seqlens does not describe q or kv")
    if kv.shape[1] != 2:
        raise ValueError("kvpacked expects [N, 2, H, D]")
    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        return q.new_empty(0, q.shape[1], q.shape[2])
    scale = softmax_scale if softmax_scale is not None else q.shape[2] ** -0.5
    outputs = []
    for b in range(batch):
        q_start, q_end = int(cu_q[b]), int(cu_q[b + 1])
        k_start, k_end = int(cu_k[b]), int(cu_k[b + 1])
        if q_end <= q_start or k_end <= k_start:
            continue
        queries = q[q_start:q_end].transpose(0, 1)              # [H, Lq, D]
        keys_values = kv[k_start:k_end].permute(1, 2, 0, 3)     # [2, H, Lk, D]
        outputs.append(
            F.scaled_dot_product_attention(                      # [1, H, Lq, D]
                queries.unsqueeze(0),
                keys_values[0].unsqueeze(0), keys_values[1].unsqueeze(0),
                scale=scale,
            )[0]
        )
    if not outputs:
        return q.new_empty(0, q.shape[1], q.shape[2])
    return torch.cat(outputs, dim=1).transpose(0, 1)            # [Nq, H, D]


def _shim_source() -> str:
    """The self-contained shim module, generated from the two functions above."""
    functions = "\n\n".join(
        inspect.getsource(fn) for fn in (
            flash_attn_varlen_qkvpacked_func,
            flash_attn_varlen_kvpacked_func,
        )
    )
    return (
        '"""SDPA fallback for flash-attn\'s varlen functions (Ar3DRAHub Windows port).\n'
        "\n"
        "The checkout calls flash-attn's ``flash_attn_varlen_qkvpacked_func`` and\n"
        "``flash_attn_varlen_kvpacked_func``; this module provides the same entry\n"
        "points on torch's fused ``scaled_dot_product_attention`` over dense padded\n"
        "spans. Generated by ``scripts/patch_hi3d_sdpa.py`` -- do not edit by hand.\n"
        '"""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "import torch\n"
        "from torch.nn import functional as F\n"
        "\n\n"
        f"{functions}\n"
    )


def _patched_loader() -> str:
    """The ``_flash_attn`` body the patch installs in place of the original."""
    return (
        "def _flash_attn():\n"
        '    """Import flash-attn, or this repo\'s SDPA fallback where it has no build."""\n'
        "    try:\n"
        "        import flash_attn\n"
        "    except ImportError:\n"
        "        from sparc3d import _sdpa_varlen as flash_attn\n"
        "    return flash_attn\n"
    )


#: The exact original loader each patched file ships; the anchor the patch replaces.
ORIGINAL_LOADERS = (
    (
        "sparc3d/nn/sparse/attention/full_attn.py",
        'def _flash_attn():\n'
        '    """Import flash-attn when sparse attention is first executed."""\n'
        "    try:\n"
        "        import flash_attn\n"
        "    except ImportError as exc:\n"
        '        raise ImportError(\n'
        '            "Sparc3DTransformer attention requires flash-attn; "\n'
        '            "install sparc3d[inference]"\n'
        "        ) from exc\n"
        "    return flash_attn\n",
    ),
    (
        "sparc3d/models/autoencoders/blocks.py",
        'def _flash_attn():\n'
        '    """Import flash-attn when serialized attention is first constructed."""\n'
        "    try:\n"
        "        import flash_attn\n"
        "    except ImportError as exc:\n"
        '        raise ImportError(\n'
        '            "SparconvVAE serialized attention requires flash-attn; "\n'
        '            "install sparc3d[inference]"\n'
        "        ) from exc\n"
        "    return flash_attn\n",
    ),
)


def is_patched(root: Path) -> bool:
    """Whether every attention file already carries the SDPA fallback."""
    shim = root / SHIM_PATH
    if not shim.is_file():
        return False
    for relative, _anchor in ORIGINAL_LOADERS:
        if PATCH_MARKER not in (root / relative).read_text(encoding="utf-8"):
            return False
    return True


def apply(root: Path) -> bool:
    """Install the SDPA fallback into the checkout. Returns whether anything changed.

    Idempotent: a second run changes nothing. Raises RuntimeError when a file does
    not carry the expected original loader -- patching a changed host file would
    silently produce a checkout nobody can reason about.
    """
    root = root.expanduser().resolve()
    changed = False
    for relative, anchor in ORIGINAL_LOADERS:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(
                f"{path} is missing; refusing to patch an incomplete checkout"
            )
        text = path.read_text(encoding="utf-8")
        if PATCH_MARKER in text:
            continue
        if anchor not in text:
            raise RuntimeError(
                f"{path} does not contain the original _flash_attn loader; "
                "refusing to patch a file this repo does not know"
            )
        text = text.replace(anchor, _patched_loader(), 1)
        path.write_text(text, encoding="utf-8")
        changed = True
    shim = root / SHIM_PATH
    source = _shim_source()
    if not shim.is_file() or shim.read_text(encoding="utf-8") != source:
        shim.write_text(source, encoding="utf-8")
        changed = True
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="the Hi3D checkout to patch")
    args = parser.parse_args(argv)
    try:
        changed = apply(args.root)
    except (OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"SDPA fallback: {'applied' if changed else 'already present'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
