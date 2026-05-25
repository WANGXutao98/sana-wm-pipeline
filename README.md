# SANA-WM Data Annotation Pipeline (Reproduction)

End-to-end data annotation pipeline reproducing **arXiv:2605.15178v1** (SANA-WM, NVIDIA 2026-05-14): 6 stages that take 7 public video sources to ~213K WebDataset shards with metric-scale 6-DoF poses + scene-static captions, suitable for training the SANA-WM DiT (961-frame, 720p, 16fps).

## Pipeline overview

| Stage | Module | Paper passage |
|---|---|---|
| 01 Ingest | `stage01_ingest/` | §5.1, App. D.1 (720p / 16fps / 961 camera frames) |
| 02 Pose annotation | `stage02_pose/` | §4, App. B.1 (VIPE + Pi3X + MoGe-2, three modes) |
| 03 3DGS augmentation (DL3DV only) | `stage03_3dgs_aug/` | App. B.2 (FCGS + 40 trajs + DiFix3D) |
| 04 Filter | `stage04_filter/` | App. B.3 (Table 6 thresholds + VLM flagging) |
| 05 Caption | `stage05_caption/` | §4 (Qwen3.5-VL scene-static, no camera verbs) |
| 06 Pack | `stage06_pack/` | WebDataset (.tar shards) |

## Quick start

```bash
# 1. Create env (persistent path; never use /root/.local which is ephemeral on this host)
conda create -p /mnt/afs/davidwang/miniconda3/envs/sana_wm python=3.10 -c conda-forge -y
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh && conda activate sana_wm

# 2. Install package + deps
pip install -e ".[dev]"

# 3. Run unit tests (verifies all paper-fixed constants)
pytest tests/ -v

# 4. Verify a shard against paper hard constraints
python scripts/verify_consistency.py path/to/shards/
```

## Paper-fixed constants

All numerical constants in `configs/pipeline.yaml` trace to specific paper passages — see inline YAML comments. Tests in `tests/test_smoke.py::test_pipeline_config_loads` lock these against drift.

Key:
- 1280 × 720 / 16 fps / 961 camera frames (§5.1, App. D.1)
- LTX2-VAE C=128 latent channels (§5.1)
- 8 raw frames per VAE temporal stride (§3.3 fine branch)
- EMA momentum 0.99 for depth fusion (App. B.1)
- Umeyama 80th-percentile inlier filter (App. B.1)
- FOV ∈ [25°, 120°], focal divergence ≤ 0.20, scale CV ≤ 2.0 (App. B.3)
- DiFix3D: 1 step, prompt "remove degradation", timestep 199, guidance 0 (App. B.2)

## Third-party tools — see LICENSING.md

This reproduction is non-commercial only, because Pi3X model weights are CC-BY-NC-4.0 and several training corpora (SpatialVID-HQ, OmniWorld) are CC-BY-NC-SA 4.0.

Real repository URLs (verified 2026-05-25):
- VIPE: github.com/nv-tlabs/vipe (Apache-2.0)
- Pi3 / Pi3X: github.com/yyfz/Pi3
- MoGe-2: github.com/microsoft/MoGe
- SANA upstream: github.com/NVlabs/Sana (SANA-WM marked "coming soon")

## Persistence on this host (CMCC + AFS)

- **Persistent**: `/mnt/afs/davidwang/workspace/...` and `/mnt/afs/davidwang/miniconda3/envs/...`
- **Ephemeral (lost on reboot)**: `/root/.local/`, `/tmp/`, container hot disks outside `filestorage` mounts

Conda env MUST be created under `/mnt/afs/davidwang/miniconda3/envs/`; pip --user installs to `/root/.local/` will not survive reboots.

## Production scaling (CMCC 64×H100)

For the full 212,975-clip annotation, see `docs/TROUBLESHOOTING.md` and the deployment runbook at `/mnt/afs/davidwang/workspace/docker-images/cmcc/docs/`.

## License

Code: Apache-2.0 (this reproduction).
Data + model-weight licenses: see `LICENSING.md`.
