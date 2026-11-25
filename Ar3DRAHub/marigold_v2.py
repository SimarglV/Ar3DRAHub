"""Marigold V2 as a native predictor: image -> depth, normals, or albedo.

Runs the reference single-step dense predictors from
https://github.com/huawei-bayerlab/marigold-v2 (MIT) on a
local Qwen-Image-Edit-2509 backbone. The backbone is loaded from a GGUF build of the
same 2509 DiT (already quantized, so no bitsandbytes NF4 pass and no fp32/bf16 dtype
mismatch); each Marigold checkpoint is a rank-128 LoRA adapter plus, for normals and
albedo, a fine-tuned VAE decoder.

The heavy imports (torch, diffusers, peft) are deferred so the pure parts -- path
resolution, resolution rounding, saving -- import in the studio's own environment and
stay testable there. The predictor itself runs under its own venv
(scripts/bootstrap_marigold_v2.py), invoked as a subprocess like the other model
backends, because the studio environment does not carry these packages.

Weights layout (from scripts/bootstrap_marigold_v2.py, or copied in by hand):

    <models>/marigold-v2/
      Qwen-Image-Edit-2509/{model_index.json, vae/*, transformer/config.json}
      gguf/Qwen-Image-Edit-2509-Q6_K.gguf
      Marigold-V2/{depth/Log-stage2, normals, albedo}/trainables.safetensors
      Marigold-V2/qwen_text_embeddings/*.pt

Licensing: MIT. The Qwen-Image-Edit-2509
backbone keeps its own licence (Qwen), which the provenance record names separately.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

from Ar3DRAHub.models_home import vendor_models

# The models folder the predictor reads. Default: models/marigold-v2 in the repo's
# models home; MARIGOLD_V2_MODELS_DIR points it at a copy on another disk.
MODELS_DIR_ENV = "MARIGOLD_V2_MODELS_DIR"

QWEN_DIR_NAME = "Qwen-Image-Edit-2509"
GGUF_NAME = "Qwen-Image-Edit-2509-Q6_K.gguf"
CHECKPOINTS_DIR_NAME = "Marigold-V2"
EMBED_SUBDIR = "qwen_text_embeddings"

# Fixed rectified-flow timestep the models were fine-tuned at (single step).
TIMESTEP = 499.0
SIZE_MULTIPLE = 16

MODALITIES = ("depth", "normals", "albedo")
DEFAULT_DEPTH_VARIANT = "Log-stage2"

# Depth checkpoints differ in how depth is parameterized before the VAE; log and
# linear grow with distance, disparity shrinks.
DEPTH_VARIANTS = {
    "Log-stage2": True,
    "Log-stage1": True,
    "Log-layered": True,
    "Uniform-base": True,
    "Uniform-layered": True,
    "Disparity-base": False,
    "Disparity-layered": False,
}

EMBED_PREFIX = {
    "depth": "qwen_edit_2509_qwen_depth_realimg512",
    "normals": "qwen_edit_2509_qwen_normals_dummy512",
    "albedo": "qwen_edit_2509_qwen_albedo_rgb_dummy512",
}

# LoRA configuration shared by every released checkpoint.
LORA_RANK = 128
LORA_ALPHA = 128
LORA_DROPOUT = 0.0
LORA_INIT = "gaussian"
LORA_TARGET_MODULES = (
    "img_in", "txt_in", "to_q", "to_k", "to_v", "to_out.0",
    "attn.add_k_proj", "attn.add_v_proj", "attn.add_q_proj", "attn.to_add_out",
    "norm.linear", "a_to_out", "b_to_out", "img_mlp.net.0.proj", "img_mlp.net.2",
    "txt_mlp.net.0.proj", "txt_mlp.net.2", "ff_a.0", "ff_a.2", "ff_b.0", "ff_b.2",
    "norm_out.linear", "proj_out",
)


@dataclass(frozen=True)
class ModalitySpec:
    """One released checkpoint plus everything needed to run and read it."""

    modality: str
    variant: str
    subfolder: str
    embed_prefix: str
    far_is_high: bool = True

    @property
    def label(self) -> str:
        return f"{self.modality}/{self.variant}" if self.variant else self.modality


def get_spec(modality: str, depth_variant: str = DEFAULT_DEPTH_VARIANT) -> ModalitySpec:
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}; expected one of {MODALITIES}")
    if modality == "depth":
        if depth_variant not in DEPTH_VARIANTS:
            raise ValueError(
                f"unknown depth checkpoint {depth_variant!r}; expected one of "
                f"{sorted(DEPTH_VARIANTS)}"
            )
        return ModalitySpec("depth", depth_variant, f"depth/{depth_variant}",
                            EMBED_PREFIX["depth"], DEPTH_VARIANTS[depth_variant])
    return ModalitySpec(modality, "", modality, EMBED_PREFIX[modality])


def models_dir(environ: dict[str, str] | None = None) -> Path:
    """The Marigold V2 weights folder, defaulting to the repo's models home."""
    environ = os.environ if environ is None else environ
    value = environ.get(MODELS_DIR_ENV)
    if value:
        return Path(value).expanduser()
    return vendor_models("marigold-v2")


def resolve_paths(models: Path, modality: str,
                  depth_variant: str = DEFAULT_DEPTH_VARIANT) -> tuple[ModalitySpec, Path, Path, Path, Path, Path]:
    """The files one prediction needs, as (spec, qwen_dir, gguf, checkpoint, embeds, mask)."""
    spec = get_spec(modality, depth_variant)
    qwen_dir = models / QWEN_DIR_NAME
    gguf = models / "gguf" / GGUF_NAME
    checkpoint = models / CHECKPOINTS_DIR_NAME / spec.subfolder / "trainables.safetensors"
    embeds = (models / CHECKPOINTS_DIR_NAME / EMBED_SUBDIR
              / f"{spec.embed_prefix}_prompt_embeds.pt")
    mask = (models / CHECKPOINTS_DIR_NAME / EMBED_SUBDIR
            / f"{spec.embed_prefix}_prompt_mask.pt")
    return spec, qwen_dir, gguf, checkpoint, embeds, mask


def missing_files(models: Path, modality: str,
                  depth_variant: str = DEFAULT_DEPTH_VARIANT) -> list[str]:
    """Every file a prediction needs but does not have, as names under the models dir."""
    spec, qwen_dir, gguf, checkpoint, embeds, mask = resolve_paths(
        models, modality, depth_variant)
    required = (
        (qwen_dir / "vae" / "config.json"),
        (qwen_dir / "vae" / "diffusion_pytorch_model.safetensors"),
        (qwen_dir / "transformer" / "config.json"),
        gguf, checkpoint, embeds, mask,
    )
    # Posix spelling so the bootstrap's `need()` and the CLI's display agree with a test
    # on Windows, where `relative_to` would otherwise return backslashes.
    return [path.relative_to(models).as_posix() for path in required if not path.is_file()]


def _round_up(value: int, multiple: int = SIZE_MULTIPLE) -> int:
    return max(multiple, (int(value) + multiple - 1) // multiple * multiple)


def target_size(height: int, width: int, max_side: int = 1024) -> tuple[int, int]:
    """Inference resolution, always a multiple of 16 as the DiT requires."""
    if max(height, width) > max_side > 0:
        scale = max_side / float(max(height, width))
        return _round_up(round(height * scale)), _round_up(round(width * scale))
    return _round_up(height), _round_up(width)


# --- Resizing (upstream transforms: Lanczos for the input, bilinear for the output) ----

def lanczos_resize_chw(array, out_hw: tuple[int, int]):
    """Lanczos-resize a CHW array in [0, 1] or [-1, 1] to (out_h, out_w)."""
    import numpy as np
    from PIL import Image

    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if array.shape[1:] == (out_h, out_w):
        return array
    channels = [
        np.asarray(Image.fromarray(channel, mode="F").resize((out_w, out_h), Image.LANCZOS),
                   dtype=np.float32)
        for channel in array
    ]
    return np.stack(channels, axis=0)


def linear_resize_chw(array, out_hw: tuple[int, int]):
    """Bilinear-resize a 2D or CHW array back to (out_h, out_w)."""
    import numpy as np
    from PIL import Image

    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if array.shape[-2:] == (out_h, out_w):
        return array
    if array.ndim == 2:
        return np.asarray(Image.fromarray(array, mode="F").resize((out_w, out_h),
                                                                  Image.BILINEAR),
                          dtype=np.float32)
    return np.stack([linear_resize_chw(channel, out_hw) for channel in array], axis=0)


def normalize_depth(depth) -> tuple:
    """A (min, max) that maps depth to a displayable grayscale, or None for a flat map."""
    lo, hi = float(depth.min()), float(depth.max())
    return None if hi <= lo else (lo, hi)


def normals_to_image(normals):
    """Camera-space unit normals in [-1, 1] -> RGB in [0, 1]."""
    import numpy as np
    return (np.moveaxis(normals, 0, -1) + 1.0) * 0.5


def save_prediction(array, out_dir: Path, stem: str, modality: str) -> tuple[Path, Path]:
    """Write one prediction as .npy plus a PNG. Returns (npy, png) paths."""
    import numpy as np
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    npy = out_dir / f"{stem}_{modality}.npy"
    png = out_dir / f"{stem}_{modality}.png"
    np.save(npy, array)
    if modality == "depth":
        lo, hi = float(array.min()), float(array.max())
        norm = (array - lo) / (hi - lo) if hi > lo else np.zeros_like(array)
        Image.fromarray((norm * 255).astype(np.uint8), mode="L").save(png)
    else:
        hwc = np.moveaxis(array, 0, -1)
        if hwc.min() < -0.01:  # normals in [-1, 1]
            hwc = (hwc + 1.0) * 0.5
        Image.fromarray((np.clip(hwc, 0, 1) * 255).astype(np.uint8), mode="RGB").save(png)
    return npy, png


# --- The heavy, deferred path: everything that needs torch/diffusers/peft ------------

def _load_prompt(embeds_path: Path, mask_path: Path):
    import torch

    def load(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            return torch.load(path, map_location="cpu", weights_only=False)

    embeds = load(embeds_path).contiguous()
    mask = load(mask_path).contiguous()
    if mask.dtype != torch.bool:
        mask = mask > 0
    return embeds[:1], mask[:1]


def _vae_scale_factor(vae) -> int:
    temporal = vae.config.get("temperal_downsample", None)
    return 2 ** len(temporal) if temporal is not None else 8


def _latent_stats(vae, ref):
    import torch

    shape = (1, vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean, device=ref.device, dtype=ref.dtype)
    std = torch.tensor(vae.config.latents_std, device=ref.device, dtype=ref.dtype)
    return mean.view(shape), (1.0 / std).view(shape)


class Predictor:
    """A loaded Marigold V2 backbone with one checkpoint applied.

    One backbone serves all three modalities; `apply_checkpoint` swaps the LoRA adapter
    and the (possibly fine-tuned) VAE decoder in place.
    """

    def __init__(self, models: Path, modality: str, depth_variant: str = DEFAULT_DEPTH_VARIANT,
                 device=None):
        import torch

        self.models = models
        self.spec, qwen_dir, gguf, checkpoint, embeds, mask = resolve_paths(
            models, modality, depth_variant)
        self.prompt_paths = (embeds, mask)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load(qwen_dir, gguf)
        self.apply_checkpoint(checkpoint)

    def use(self, modality: str, depth_variant: str = DEFAULT_DEPTH_VARIANT) -> None:
        """Switch the shared backbone to another modality (swaps adapter + decoder)."""
        self.spec, _, _, checkpoint, embeds, mask = resolve_paths(
            self.models, modality, depth_variant)
        self.prompt_paths = (embeds, mask)
        self.apply_checkpoint(checkpoint)

    def _load(self, qwen_dir: Path, gguf: Path):
        import torch
        from diffusers import AutoencoderKLQwenImage, GGUFQuantizationConfig, \
            QwenImageTransformer2DModel
        from peft import LoraConfig

        self.dtype = torch.bfloat16
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            str(qwen_dir), subfolder="vae", local_files_only=True, torch_dtype=self.dtype,
            low_cpu_mem_usage=True, use_safetensors=True).to(self.device, dtype=self.dtype)
        self.vae.requires_grad_(False)
        self.vae.eval()
        # The fine-tuned decoders ship with the checkpoints; keep a pristine copy to
        # restore on each modality switch.
        self._pristine_vae = {k: v.detach().to("cpu", copy=True)
                              for k, v in self.vae.state_dict().items()
                              if k.startswith(("decoder.", "post_quant_conv."))}

        self.transformer = QwenImageTransformer2DModel.from_single_file(
            str(gguf),
            quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
            config=str(qwen_dir), subfolder="transformer",
            torch_dtype=self.dtype, local_files_only=True).to(self.device)
        # proj_out is not adapted (see the reference component_loader.py).
        targets = [t for t in LORA_TARGET_MODULES if "proj_out" not in t]
        self.transformer.add_adapter(LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            init_lora_weights=LORA_INIT, target_modules=list(targets)))
        for name, param in self.transformer.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(torch.bfloat16)
        self.transformer.requires_grad_(False)
        self.transformer.eval()

    def apply_checkpoint(self, checkpoint: Path):
        from safetensors.torch import load_file

        state = load_file(str(checkpoint), device="cpu")
        transformer_state = {k[len("Diffuser."):]: v
                             for k, v in state.items() if k.startswith("Diffuser.")}
        vae_state = {k[len("VAE."):]: v for k, v in state.items() if k.startswith("VAE.")}
        if not transformer_state:
            raise RuntimeError(f"{checkpoint.name} holds no 'Diffuser.*' weights; "
                               "it is not a Marigold V2 checkpoint.")
        _, unexpected = self.transformer.load_state_dict(transformer_state, strict=False)
        if unexpected:
            raise RuntimeError(
                f"{len(unexpected)} checkpoint tensors do not fit the LoRA adapter "
                f"(first: {unexpected[0]}). Check the diffusers and peft versions.")
        self.vae.load_state_dict(self._pristine_vae, strict=False)
        if vae_state:
            _, unexpected = self.vae.load_state_dict(vae_state, strict=False)
            if unexpected:
                raise RuntimeError(
                    f"{len(unexpected)} VAE tensors in the checkpoint are unexpected "
                    f"(first: {unexpected[0]}).")
        self.vae.to(self.device, dtype=self.dtype)

    def predict(self, image: Path, max_side: int = 1024, seed: int = 2025):
        """One image -> one float32 array ([H, W] depth, [3, H, W] normals/albedo)."""
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(image) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
        orig_h, orig_w = rgb.shape[:2]
        rgb_norm = (torch.from_numpy(rgb).permute(2, 0, 1) * 2.0 - 1.0)  # [-1, 1]

        out_h, out_w = target_size(orig_h, orig_w, max_side)
        if (out_h, out_w) != (orig_h, orig_w):
            rgb_norm = torch.from_numpy(lanczos_resize_chw(
                rgb_norm.numpy(), (out_h, out_w)))

        prompt = _load_prompt(*self.prompt_paths)
        generator = torch.Generator(device="cpu").manual_seed(seed % (2 ** 63))
        with torch.no_grad():
            latents = self._encode(rgb_norm.unsqueeze(0), generator)
            latents = self._step(latents, prompt)
            decoded = self._decode(latents)
        prediction = self._adapt(decoded)
        if self.spec.modality == "depth":
            prediction = prediction[0, 0]
        else:
            prediction = prediction[0]
        prediction = prediction.cpu().numpy().astype(np.float32)
        if prediction.shape[-2:] != (orig_h, orig_w):
            prediction = linear_resize_chw(prediction, (orig_h, orig_w))
        return np.ascontiguousarray(prediction)

    def _encode(self, pixels, generator):
        x = pixels.to(device=self.vae.device, dtype=self.vae.dtype).unsqueeze(2)
        latents = self.vae.encode(x).latent_dist.sample(generator=generator)
        mean, std_inv = _latent_stats(self.vae, latents)
        return (latents - mean) * std_inv

    def _decode(self, latents):
        mean, std_inv = _latent_stats(self.vae, latents)
        return self.vae.decode(latents / std_inv + mean).sample[:, :, 0].float()

    def _step(self, latents, prompt):
        import torch
        from diffusers import QwenImageEditPipeline

        device = next(self.transformer.parameters()).device
        run_dtype = torch.bfloat16
        prompt_embeds_cpu, prompt_mask_cpu = prompt
        prompt_embeds = prompt_embeds_cpu.to(device=device, dtype=run_dtype, non_blocking=True)
        prompt_mask = prompt_mask_cpu.to(device=device, dtype=torch.bool, non_blocking=True)

        latents_4d = latents[:, :, 0]
        batch, channels, height, width = latents_4d.shape
        packed = QwenImageEditPipeline._pack_latents(
            latents_4d, batch_size=batch, num_channels_latents=channels,
            height=height, width=width).to(run_dtype)
        timestep = torch.full((batch,), TIMESTEP, device=device, dtype=run_dtype) / 1000.0
        call_kwargs = dict(
            hidden_states=packed, timestep=timestep,
            encoder_hidden_states=prompt_embeds, encoder_hidden_states_mask=prompt_mask,
            img_shapes=[[(1, height // 2, width // 2)]] * batch,
            guidance=None, attention_kwargs={},
        )
        accepts_txt_seq_lens = "txt_seq_lens" in _forward_params(self.transformer)
        if accepts_txt_seq_lens:
            call_kwargs["txt_seq_lens"] = prompt_mask.sum(dim=1).tolist()
        model_pred = self.transformer(**call_kwargs, return_dict=False)[0]
        scale = _vae_scale_factor(self.vae)
        model_pred = QwenImageEditPipeline._unpack_latents(
            model_pred, height=height * scale, width=width * scale, vae_scale_factor=scale)
        return latents.to(self.vae.dtype) - model_pred.to(self.vae.dtype)

    def _adapt(self, decoded):
        import torch

        if self.spec.modality == "depth":
            return decoded.float().mean(dim=1, keepdim=True)
        if self.spec.modality == "normals":
            values = decoded.float()
            magnitude = torch.linalg.vector_norm(values, dim=1, keepdim=True)
            normals = values / magnitude.clamp_min(1.0e-6)
            return torch.where(magnitude > 1.0e-6, normals, torch.zeros_like(normals))
        albedo = (decoded.float() + 1.0) * 0.5
        return albedo.clamp_min(0.0).pow(1.0 / 2.2)


@functools.lru_cache(maxsize=1)
def _forward_params(transformer) -> frozenset:
    import inspect
    return frozenset(inspect.signature(type(transformer).forward).parameters)
