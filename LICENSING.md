# Third-Party Assets and License Registry

This file reproduces Table 11 from arXiv:2605.15178v1 (SANA-WM, NVIDIA 2026-05-14).
All entries describe the public terms of assets and tools used by our reproduction.

| Asset / Tool | Use in this work | Public license or terms |
|---|---|---|
| SpatialVID-HQ [87] | Real-video training source | CC-BY-NC-SA 4.0; gated Hugging Face access requires agreement to non-commercial terms. |
| DL3DV-10K [86] | Static 3D scenes, GT poses, and 3DGS augmentation | Custom DL3DV project terms; access requires accepting the dataset terms rather than a standard open-source license. |
| OmniWorld [89] | Synthetic/game data and held-out camera-control validation | CC-BY-NC-SA 4.0 on the public Hugging Face release. |
| Sekai [90] | Game and walking-video training sources | Public project/data release; no standard license clearly specified — follow project access terms. |
| MiraData [88] | Long real-video training source | Public project release; repository lists GPL-3.0, while source videos may remain subject to their original hosting terms. |
| ViPE [13] | Camera-pose annotation engine | Apache-2.0 code release, third-party components under their respective licenses. |
| Pi3X / Pi3 [14] | Pose/depth recovery and evaluation | BSD-3-Clause code; public model weights are released for non-commercial use (CC BY-NC 4.0). |
| MoGe-2 [15] | Metric-scale depth prior | Public Microsoft MoGe release; repository licensing includes permissive MIT/Apache-style terms — check model card for weight terms. |
| FCGS [94] | 3D Gaussian Splatting reconstruction for DL3DV augmentation | Public research code; no standard license clearly specified on the project page. |
| DiFix3D [95] | Refinement of 3DGS-rendered videos | NVIDIA research release; governed by NVIDIA terms for research/non-commercial use. |
| Qwen3.5 VLM [102] | Content filtering and scene-static captioning | Apache-2.0 public model/code release. |
| Nano Banana Pro [16] | First-frame evaluation images (benchmark only — not training) | Google/Gemini product terms apply; generated images marked with SynthID. |
| LTX-2 / LTX-2.3 [10] | LTX2 VAE and long-video refiner initialization | LTX-2 Community License Agreement for model weights, code, and related materials. |

## Reproduction notes (NOT from paper)

The following are choices our reproduction makes, with rationale:
- **VIPE repository**: `github.com/nv-tlabs/vipe` (verified existing on 2026-05-25; PyPI package `nvidia-vipe`)
- **Pi3 / Pi3X repository**: `github.com/yyfz/Pi3` (verified existing; ICLR 2026)
- **MoGe-2 repository**: `github.com/microsoft/MoGe` (verified existing; CVPR'25 Oral)
- **SANA upstream**: `github.com/NVlabs/Sana` (SANA-WM marked "coming soon" as of 2026-05-25)
- **Pi3X model weights**: CC-BY-NC-4.0 → **this reproduction is non-commercial only**.
