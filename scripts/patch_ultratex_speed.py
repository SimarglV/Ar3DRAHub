#!/usr/bin/env python3
"""Empty-prompt fast path for the vendored UltraTex inference checkout.

Two independent speedups, both exact when every prompt is empty (the
Ar3DRAHub pipeline textures from reference images, so the prompt always is):

1. The empty-prompt embedding is computed ONCE, before the 9B denoiser
   loads, and the Qwen text encoder is then released.  Qwen never shares
   VRAM with FLUX.2-klein-9B; co-residency on a 32 GB card is what made
   every encoder call cost ~30 s instead of ~1 s (measured 2026-11-23,
   Windows, RTX 5090).
2. CFG doubling is skipped: with an empty prompt the unconditional and
   conditional branches are identical, so one forward per step is exact
   and half the cost.

`--fast_empty_prompt` defaults to False, so non-empty prompts keep the
untouched upstream path.  Every edit is narrow, idempotent, and fails
closed when upstream no longer matches the audited anchors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "ultratex" / "UltraTex"


def replace_exact(source: str, old: str, new: str, *, count: int, label: str) -> tuple[str, bool]:
    """Replace an audited anchor or fail instead of producing a partial port."""
    if new in source:
        return source, False
    found = source.count(old)
    if found != count:
        raise RuntimeError(f"{label}: expected {count} source anchors, found {found}")
    return source.replace(old, new, count), True


def patch_file(path: Path, transform) -> bool:
    """Patch one file. Vendored files are CRLF; match on LF and restore CRLF.

    Reads and writes with ``newline=""`` so Windows never mangles the bytes
    behind our back (write_text would translate LF to CRLF a second time).
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw = handle.read()
    had_crlf = "\r\n" in raw
    source = raw.replace("\r\n", "\n")
    patched, changed = transform(source)
    if changed:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(patched.replace("\n", "\r\n") if had_crlf else patched)
    return changed


GET_MODELS = """def get_models(model_name: str, device: torch.device):
    dit = load_flow_model(model_name, device="cpu")
    ae = load_ae(model_name, device=device)
    text_encoder = load_text_encoder(model_name, device=device)
    return dit, ae, text_encoder"""
GET_MODELS_PATCHED = """def get_models(model_name: str, device: torch.device, load_te: bool = True):
    dit = load_flow_model(model_name, device="cpu")
    ae = load_ae(model_name, device=device)
    text_encoder = load_text_encoder(model_name, device=device) if load_te else None
    return dit, ae, text_encoder"""

ENCODE_SIGNATURE = "def encode_conditions(batch: dict, ae, text_encoder, device: torch.device):"
ENCODE_SIGNATURE_PATCHED = (
    "def encode_conditions(\n"
    "    batch: dict, ae, text_encoder, device: torch.device,\n"
    "    ctx_override: torch.Tensor | None = None,\n"
    "):"
)

ENCODE_CTX = '    ctx = text_encoder(list(batch["txt"])).to(torch.bfloat16)'
ENCODE_CTX_PATCHED = """    if ctx_override is None:
        if text_encoder is None:
            raise ValueError("ctx_override is required when text_encoder is None")
        ctx = text_encoder(list(batch["txt"])).to(torch.bfloat16)
    else:
        ctx = ctx_override.to(torch.bfloat16)"""

ENCODE_CALL = """        latent_hw,
    ) = encode_conditions(batch, ae, text_encoder, accelerator.device)"""
ENCODE_CALL_PATCHED = """        latent_hw,
    ) = encode_conditions(
        batch, ae, text_encoder, accelerator.device, ctx_override=ctx_override
    )"""

INFERENCE_SIGNATURE = """    remove_background: bool = True,
    drop_background_tokens: bool = False,
    num_steps: int = 25,
) -> Image.Image:"""
INFERENCE_SIGNATURE_PATCHED = """    remove_background: bool = True,
    drop_background_tokens: bool = False,
    num_steps: int = 25,
    use_cfg: bool = True,
    ctx_override: torch.Tensor | None = None,
) -> Image.Image:"""

CFG_BLOCK = """    # FLUX.2 Klein Base is undistilled and uses text CFG. Keep both branches
    # conditioned on the same selected image tokens; only the text differs.
    empty_ctx = text_encoder([""] * x.shape[0]).to(torch.bfloat16)
    empty_ctx, empty_ctx_ids = batched_prc_txt(empty_ctx)
    cfg_ctx = torch.cat([empty_ctx, ctx], dim=0)
    cfg_ctx_ids = torch.cat([empty_ctx_ids, ctx_ids], dim=0)"""
CFG_BLOCK_PATCHED = """    # FLUX.2 Klein Base is undistilled and uses text CFG. Keep both branches
    # conditioned on the same selected image tokens; only the text differs.
    # Empty-prompt fast path: the two CFG branches are identical then, so the
    # unconditional embedding is never computed and one forward suffices.
    if use_cfg:
        empty_ctx = text_encoder([""] * x.shape[0]).to(torch.bfloat16)
        empty_ctx, empty_ctx_ids = batched_prc_txt(empty_ctx)
        cfg_ctx = torch.cat([empty_ctx, ctx], dim=0)
        cfg_ctx_ids = torch.cat([empty_ctx_ids, ctx_ids], dim=0)"""

CFG_LOOP = """        cfg_x_in = torch.cat([x_in, x_in], dim=0)
        cfg_x_in_ids = torch.cat([x_in_ids, x_in_ids], dim=0)
        cfg_t_vec = torch.cat([t_vec, t_vec], dim=0)
        pred_cfg = model(
            x=cfg_x_in,
            x_ids=cfg_x_in_ids,
            timesteps=cfg_t_vec,
            ctx=cfg_ctx,
            ctx_ids=cfg_ctx_ids,
            guidance=None,
        )[:, : x_foreground.shape[1]]
        pred_uncond, pred_cond = pred_cfg.chunk(2)
        pred = pred_uncond + guidance * (pred_cond - pred_uncond)"""
CFG_LOOP_PATCHED = """        if use_cfg:
            cfg_x_in = torch.cat([x_in, x_in], dim=0)
            cfg_x_in_ids = torch.cat([x_in_ids, x_in_ids], dim=0)
            cfg_t_vec = torch.cat([t_vec, t_vec], dim=0)
            pred_cfg = model(
                x=cfg_x_in,
                x_ids=cfg_x_in_ids,
                timesteps=cfg_t_vec,
                ctx=cfg_ctx,
                ctx_ids=cfg_ctx_ids,
                guidance=None,
            )[:, : x_foreground.shape[1]]
            pred_uncond, pred_cond = pred_cfg.chunk(2)
            pred = pred_uncond + guidance * (pred_cond - pred_uncond)
        else:
            # Single-stream path: with an empty prompt both CFG branches are
            # identical, so one forward per step is exact and half the cost.
            pred = model(
                x=x_in,
                x_ids=x_in_ids,
                timesteps=t_vec,
                ctx=ctx,
                ctx_ids=ctx_ids,
                guidance=None,
            )[:, : x_foreground.shape[1]]"""

TRAINER_EDITS = (
    ("get_models text-encoder switch", GET_MODELS, GET_MODELS_PATCHED, 1),
    ("encode_conditions signature", ENCODE_SIGNATURE, ENCODE_SIGNATURE_PATCHED, 1),
    ("encode_conditions ctx override", ENCODE_CTX, ENCODE_CTX_PATCHED, 1),
    ("inference signature", INFERENCE_SIGNATURE, INFERENCE_SIGNATURE_PATCHED, 1),
    ("inference encode_conditions call", ENCODE_CALL, ENCODE_CALL_PATCHED, 1),
    ("inference CFG block", CFG_BLOCK, CFG_BLOCK_PATCHED, 1),
    ("inference CFG loop", CFG_LOOP, CFG_LOOP_PATCHED, 1),
)


def patch_trainer(root: Path, name: str) -> bool:
    path = root / name

    def transform(source: str) -> tuple[str, bool]:
        changed = False
        for label, old, new, count in TRAINER_EDITS:
            source, did = replace_exact(source, old, new, count=count, label=f"{name}: {label}")
            changed |= did
        return source, changed

    return patch_file(path, transform)


ARGS_FLAG = """    inference_drop_background_tokens: bool = True
    inference_remove_background: bool = True"""
ARGS_FLAG_PATCHED = """    inference_drop_background_tokens: bool = True
    inference_remove_background: bool = True
    # Empty-prompt fast path: encode the (empty) prompt once before the DiT
    # loads, then release the text encoder so it never shares VRAM with the
    # 9B denoiser. One CFG-free forward per step. Refuses non-empty prompts.
    fast_empty_prompt: bool = False"""

MODEL_LOAD = "    model, ae, text_encoder = trainer.get_models(args.model_name, accelerator.device)"
MODEL_LOAD_PATCHED = """    model, ae, text_encoder = trainer.get_models(
        args.model_name, accelerator.device, load_te=not args.fast_empty_prompt
    )"""

ENCODER_EVAL = """    ae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()"""
ENCODER_EVAL_PATCHED = """    ae.requires_grad_(False).eval()
    if text_encoder is not None:
        text_encoder.requires_grad_(False).eval()"""

MAIN_ANCHOR = "def main(args: InferenceArgs) -> None:"
MAIN_HELPER = '''def encode_empty_prompt_once(trainer, base_dataset, model_name: str, device) -> torch.Tensor:
    """Embed the empty prompt, then release the text encoder before the DiT loads.

    Guards the fast path: every sample must carry an empty prompt, and the
    embedder is unloaded as soon as the embedding exists so Qwen never
    shares VRAM with the 9B denoiser.
    """
    import gc

    prompt = base_dataset[0]["prompt"]
    prompts = prompt if isinstance(prompt, (list, tuple)) else [prompt]
    if any(str(entry).strip() for entry in prompts):
        raise ValueError("--fast_empty_prompt requires every prompt to be empty")
    text_encoder = trainer.load_text_encoder(model_name, device=device)
    try:
        return text_encoder([""]).to(torch.bfloat16)
    finally:
        del text_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


'''

MAIN_LOAD = """    model, ae, text_encoder = load_inference_models(
        args, accelerator, checkpoint_file, trainer
    )"""
MAIN_LOAD_PATCHED = """    if args.fast_empty_prompt:
        fast_ctx_override = encode_empty_prompt_once(
            trainer, base_dataset, args.model_name, accelerator.device
        )
    else:
        fast_ctx_override = None

    model, ae, text_encoder = load_inference_models(
        args, accelerator, checkpoint_file, trainer
    )"""

INFERENCE_CALL = """            drop_background_tokens=args.inference_drop_background_tokens,
            num_steps=args.inference_num_steps,
        )"""
INFERENCE_CALL_PATCHED = """            drop_background_tokens=args.inference_drop_background_tokens,
            num_steps=args.inference_num_steps,
            use_cfg=not args.fast_empty_prompt,
            ctx_override=fast_ctx_override,
        )"""

MANIFEST_LINE = '''            "remove_background": args.inference_remove_background,'''
MANIFEST_LINE_PATCHED = '''            "remove_background": args.inference_remove_background,
            "empty_prompt_fast": args.fast_empty_prompt,'''

RUNNER_EDITS = (
    ("fast_empty_prompt argument", ARGS_FLAG, ARGS_FLAG_PATCHED, 1),
    ("text-encoder skip in loader", MODEL_LOAD, MODEL_LOAD_PATCHED, 1),
    ("text-encoder eval guard", ENCODER_EVAL, ENCODER_EVAL_PATCHED, 1),
    ("embed-once helper", MAIN_ANCHOR, MAIN_HELPER + MAIN_ANCHOR, 1),
    ("embed-once call before model load", MAIN_LOAD, MAIN_LOAD_PATCHED, 1),
    ("inference call flags", INFERENCE_CALL, INFERENCE_CALL_PATCHED, 1),
    ("manifest entry", MANIFEST_LINE, MANIFEST_LINE_PATCHED, 1),
)


def patch_runner(root: Path) -> bool:
    path = root / "inference_flux2.py"

    def transform(source: str) -> tuple[str, bool]:
        changed = False
        for label, old, new, count in RUNNER_EDITS:
            source, did = replace_exact(source, old, new, count=count, label=f"inference_flux2.py: {label}")
            changed |= did
        return source, changed

    return patch_file(path, transform)


def is_patched(root: Path) -> bool:
    """True once both trainers and the runner carry the fast path."""
    for name in ("train_flux2.py", "train_flux2_mr.py"):
        try:
            source = (root / name).read_text(encoding="utf-8")
        except OSError:
            return False
        if "use_cfg: bool = True" not in source:
            return False
    try:
        source = (root / "inference_flux2.py").read_text(encoding="utf-8")
    except OSError:
        return False
    return "fast_empty_prompt: bool = False" in source


def apply(root: Path) -> list[str]:
    if not (root / "inference_flux2.py").is_file():
        raise RuntimeError(f"not an UltraTex source tree: {root}")
    steps = (
        ("trainer albedo", lambda: patch_trainer(root, "train_flux2.py")),
        ("trainer mr", lambda: patch_trainer(root, "train_flux2_mr.py")),
        ("inference runner", lambda: patch_runner(root)),
    )
    changed = []
    for label, function in steps:
        if function():
            changed.append(label)
            print(f"applied : {label}")
        else:
            print(f"present : {label}")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    apply(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
