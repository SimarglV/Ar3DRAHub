# Dropping in your own models

The studio reads two kinds of model from the `models/` folder (the one on the Setup page —
`models/` by default, or wherever you moved it). You can add models by dropping files in, no
download step needed.

## Picture models (2D)

A picture model is a folder in `models/pictures/<name>/` with a small `model.json` that names
its files. Two runtimes are supported, and the studio picks which one from the manifest.

### Raw checkpoints (`.safetensors` / `.ckpt`), through diffusers

Drop one file (a full checkpoint: Stable Diffusion 1.x or SDXL, with the text encoder and VAE
bundled), or a diffusers-format folder (one with `model_index.json`), and point `diffusion` at it:

```
models/pictures/my-model/
    model.safetensors      <- the checkpoint (or model.ckpt)
    model.json
```

```json
{
  "label": "My Model",
  "diffusion": "model.safetensors",
  "license": { "name": "...", "url": "..." }
}
```

Only `diffusion` is required; it may be a single file or a subfolder. The studio runs it with
torch + diffusers (the "Diffusers" card on Setup & Status installs the software, no model
weights). This is the runtime ComfyUI / Stable Diffusion WebUI checkpoints run on, so most
civitai-style checkpoints load as-is.

You can also swap in your own VAE or text encoders — useful for a UNet-only checkpoint, or a
custom VAE. Point `vae`, `text_encoder` and (for SDXL / FLUX, which have two) `text_encoder_2`
at a diffusers-format folder (one with a `config.json`) or a single `.safetensors` file:

```json
{
  "label": "My Model",
  "diffusion": "unet.safetensors",
  "text_encoder": "text-encoder/",
  "text_encoder_2": "text-encoder-2/",
  "vae": "vae/"
}
```

Text-to-image and the back/side views work with any checkpoint. The **brush** (repainting one
painted area) is inpainting: it needs a checkpoint that understands a mask — an inpainting
checkpoint (`stable-diffusion-inpainting`, SDXL-inpainting) or a diffusers inpainting folder. A
plain text-to-image checkpoint is listed and draws, but its brush is refused.

### GGUF (FLUX.2 klein / Qwen-Image), through stable-diffusion.cpp

Three files — a GGUF diffusion model, a GGUF text encoder and a VAE:

```
models/pictures/my-flux-8b/
    model.gguf             <- the diffusion model (GGUF)
    clip.gguf              <- the text encoder (GGUF)
    vae.safetensors        <- the VAE (safetensors)
    model.json
```

```json
{
  "label": "My FLUX 8B",
  "diffusion": "model.gguf",
  "text_encoder": "clip.gguf",
  "vae": "vae.safetensors"
}
```

This runtime loads only the architectures `stable-diffusion.cpp` knows — today FLUX.2 klein and
Qwen-Image. A GGUF of another architecture is listed but will not run.

### Shared bits

`label` defaults to the folder name; `license` is optional — without it the model shows as
"verify its licence", and every picture it makes carries that note in its `.provenance.json`.
Set `"runtime"` to `"diffusers"` or `"sdcpp"` to force one, or leave it out and the extension
decides (`.safetensors`/`.ckpt`/a folder → diffusers, otherwise the three-GGUF form).

Once the folder is there, the model appears in the picture-engine picker everywhere: Create,
"Draw the back again", "Draw the left/right side", and the brush.

## 3D backends

3D models already live in `models/<route>/` — `models/pixal3d-cpp`, `models/hunyuan-cuda`,
`models/ultratex`, `models/hi3d`, and so on. Each backend looks for its own files there (or in
the Hugging Face cache). Dropping the files a backend expects into its folder makes it ready,
the same as running its bootstrap. There is no single "3D diffusion file" — each backend is its
own code port with its own files, so there is nothing generic to drop.
