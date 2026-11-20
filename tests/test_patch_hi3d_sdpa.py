"""The SDPA attention port: exact attention math over packed spans, and an
idempotent, anchored patch. Nothing here touches the network or a GPU."""

from __future__ import annotations

import importlib.util
import sys

import patch_hi3d_sdpa as patch
import pytest
import torch


def _manual_qkvpacked(qkv, cu_seqlens):
    """The reference: softmax(QK^T/sqrt(d))V per span, non-causal, in float64."""
    batch = len(cu_seqlens) - 1
    outputs = []
    for b in range(batch):
        start, end = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
        span = qkv[start:end].double()          # [L, 3, H, D]
        q, k, v = span[:, 0], span[:, 1], span[:, 2]
        scores = torch.einsum("lhd,mhd->hlm", q, k) / (q.shape[-1] ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hlm,mhd->lhd", weights, v))
    return torch.cat(outputs, dim=0)


def test_qkvpacked_matches_exact_attention():
    torch.manual_seed(0)
    lengths = [2, 1, 3, 2]
    cu = torch.tensor([0, *torch.cumsum(torch.tensor(lengths), 0).tolist()])
    qkv = torch.randn(int(cu[-1]), 3, 2, 8)
    out = patch.flash_attn_varlen_qkvpacked_func(qkv, cu, max(lengths))
    expected = _manual_qkvpacked(qkv, cu)
    assert out.shape == expected.shape
    assert torch.allclose(out.double(), expected, atol=1e-6)


def test_qkvpacked_handles_unequal_spans_with_padding():
    torch.manual_seed(1)
    cu = torch.tensor([0, 4, 5])
    qkv = torch.randn(5, 3, 2, 4)
    out = patch.flash_attn_varlen_qkvpacked_func(qkv, cu, 4)
    expected = _manual_qkvpacked(qkv, cu)
    assert torch.allclose(out.double(), expected, atol=1e-6)


def _manual_kvpacked(q, kv, cu_q, cu_k):
    batch = len(cu_q) - 1
    outputs = []
    for b in range(batch):
        q_span = q[cu_q[b]:cu_q[b + 1]].double()      # [Lq, H, D]
        k = kv[cu_k[b]:cu_k[b + 1], 0].double()       # [Lk, H, D]
        v = kv[cu_k[b]:cu_k[b + 1], 1].double()
        scores = torch.einsum("lhd,mhd->hlm", q_span, k) / (q_span.shape[-1] ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hlm,mhd->lhd", weights, v))
    return torch.cat(outputs, dim=0)


def test_kvpacked_matches_exact_attention():
    torch.manual_seed(2)
    cu_q = torch.tensor([0, 2, 3])
    cu_k = torch.tensor([0, 3, 6])
    q = torch.randn(3, 2, 8)
    kv = torch.randn(6, 2, 2, 8)
    out = patch.flash_attn_varlen_kvpacked_func(q, kv, cu_q, cu_k, 2, 3)
    expected = _manual_kvpacked(q, kv, cu_q, cu_k)
    assert torch.allclose(out.double(), expected, atol=1e-6)


def test_qkvpacked_honours_the_caller_softmax_scale():
    torch.manual_seed(3)
    cu = torch.tensor([0, 2, 5])
    qkv = torch.randn(5, 3, 2, 4)
    out = patch.flash_attn_varlen_qkvpacked_func(qkv, cu, 3, softmax_scale=0.25)
    # The same inputs without the override, rescaled by the chosen factor:
    # softmax(x/s) is not a linear rescale, so compare against an independent
    # reference computed with that exact scale.
    expected = _manual_qkvpacked_scaled(qkv, cu, 0.25)
    assert torch.allclose(out.double(), expected, atol=1e-6)


def _manual_qkvpacked_scaled(qkv, cu_seqlens, scale):
    batch = len(cu_seqlens) - 1
    outputs = []
    for b in range(batch):
        span = qkv[cu_seqlens[b]:cu_seqlens[b + 1]].double()
        q, k, v = span[:, 0], span[:, 1], span[:, 2]
        scores = torch.einsum("lhd,mhd->hlm", q, k) * scale
        weights = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hlm,mhd->lhd", weights, v))
    return torch.cat(outputs, dim=0)


def test_qkvpacked_refuses_dropout_and_causal():
    cu = torch.tensor([0, 2])
    qkv = torch.randn(2, 3, 2, 4)
    with pytest.raises(ValueError, match="dropout_p=0"):
        patch.flash_attn_varlen_qkvpacked_func(qkv, cu, 2, dropout_p=0.1)
    with pytest.raises(ValueError, match="non-causal"):
        patch.flash_attn_varlen_qkvpacked_func(qkv, cu, 2, causal=True)


def test_qkvpacked_rejects_wrong_shapes():
    cu = torch.tensor([0, 2])
    with pytest.raises(ValueError, match="cu_seqlens"):
        patch.flash_attn_varlen_qkvpacked_func(torch.randn(3, 3, 2, 4), cu, 2)
    with pytest.raises(ValueError, match="qkvpacked"):
        patch.flash_attn_varlen_qkvpacked_func(torch.randn(2, 4, 2, 4), cu, 2)


def test_patch_is_idempotent_and_anchored(tmp_path):
    for relative, anchor in patch.ORIGINAL_LOADERS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"header\n{anchor}\nfooter\n", encoding="utf-8")
    assert patch.apply(tmp_path) is True
    assert patch.is_patched(tmp_path)
    assert patch.apply(tmp_path) is False          # second run changes nothing
    for relative, _ in patch.ORIGINAL_LOADERS:
        text = (tmp_path / relative).read_text(encoding="utf-8")
        assert patch.PATCH_MARKER in text
        assert "flash_attn_varlen_qkvpacked_func" not in text.split("_flash_attn")[1] \
            or patch.PATCH_MARKER in text
    shim = tmp_path / patch.SHIM_PATH
    assert shim.is_file()
    assert "flash_attn_varlen_kvpacked_func" in shim.read_text(encoding="utf-8")


def test_patch_refuses_a_host_file_it_does_not_know(tmp_path):
    path = tmp_path / "sparc3d" / "nn" / "sparse" / "attention" / "full_attn.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("totally different file\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not contain the original"):
        patch.apply(tmp_path)


def test_patch_refuses_a_checkout_missing_attention_files(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        patch.apply(tmp_path)


def test_generated_shim_imports_and_runs():
    """The shim the patch writes is the artifact that ships: import and run it."""
    source = patch._shim_source()
    spec = importlib.util.spec_from_loader("shim_under_test", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules["shim_under_test"] = module
    exec(compile(source, "sparc3d/_sdpa_varlen.py", "exec"), module.__dict__)
    try:
        cu = torch.tensor([0, 2])
        out = module.flash_attn_varlen_qkvpacked_func(torch.randn(2, 3, 2, 4), cu, 2)
        assert out.shape == (2, 2, 4)
    finally:
        sys.modules.pop("shim_under_test", None)
