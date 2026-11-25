# Credits

This lab builds on other people's models and ports. Thank you.

- [raven38/pixal3d.cpp](https://github.com/raven38/pixal3d.cpp): Pixal3D in C++/GGML
  (MIT).
- [Microsoft TRELLIS.2](https://huggingface.co/microsoft/TRELLIS.2-4B) (MIT), running on
  [pedronaugusto/trellis2-apple](https://github.com/pedronaugusto/trellis2-apple) and
  Pedro Naugusto's Metal kernels.
- [ZimengXiong/Hunyuan3D-MLX](https://github.com/ZimengXiong/Hunyuan3D-MLX) (MIT) and
  [dgrauet](https://github.com/dgrauet)'s Hunyuan3D shape port, for
  [Tencent Hunyuan3D](https://huggingface.co/tencent/Hunyuan3D-2.1) (MIT).
- [Stability AI's Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d) (MIT).
- [Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (MIT),
  run by [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
  (MIT). Built with Qwen.
- [Marigold V2](https://huggingface.co/huawei-bayerlab/marigold-v2-0) (MIT) over a
  [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) backbone, for
  the Finish de-light and (optional) surface-normals pass.
- [VAST-AI SkinTokens](https://github.com/VAST-AI-Research/SkinTokens) (MIT): the
  studio's auto-rig, which gives a humanoid its skeleton and skin weights.
- [NVIDIA Kimodo](https://github.com/nv-tlabs/kimodo): the studio's preset moves were
  made with it. All MIT.
- [UniMate](https://linzhanmou.com/unimate/) (SIGGRAPH Asia 2026): text-to-motion for any
  skeleton, the studio's Motion step. MIT.
- [Apple MLX](https://github.com/ml-explore/mlx), [rembg](https://github.com/danielgatis/rembg),
  [TinyCLIP](https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M) and
  [three.js](https://threejs.org), all MIT.

## Licences travel with the output

- Qwen-Image: the pictures you make are yours
  ([Qwen's statement](https://x.com/QwenDevs/status/2101917379785838660), 21 Sept 2026).
- Every model is recorded as MIT in the provenance record.
- Every model is recorded as MIT, motion included.
- Every run writes a `.provenance.json` saying which licences apply to that asset.

## Known issues

Only the ones everyone can hit:

- **Hunyuan's paint stage stalls above ~500,000 faces.** Keep `decimation_target` at or
  under 500,000.
- **TRELLIS.2 can get colours badly wrong on flat or vector-style art.** See
  [picking a picture](trellis2-flat-illustration-colour-drift.md).
- **NVIDIA: Pixal3D's ready-made build needs driver 575 or newer.** On an older driver the
  installer compiles it instead, if the CUDA toolkit is installed.
- **Animate: humanoids in an A-pose or T-pose only.** Four-legged creatures are coming.
- **Animate: some humanoids are wrongly called "not a humanoid".** Extra bones (hair, a
  cape, a sword) no longer cause it. If it happens, please attach the rigged GLB to an issue.
- **Pixel Match can look worse than the raw model on some assets.** Each step in the
  studio has its own Download GLB, so the raw model is always there. See
  [Pixel Match](pixel-match.md) for when that happens and what to check.
- **Animate: feet can slide a little, and elbows and knees can bend like rubber.**
  Generated meshes have no extra edges at the joints; a cleaner retopology helps.
- **Windows has had limited testing.** More testers wanted: tell us how it went.
