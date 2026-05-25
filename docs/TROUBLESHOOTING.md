# Troubleshooting

## env / package issues

### `conda activate sana_wm` fails with `EnvironmentNameNotFound`
The env was created at an ephemeral path (`/root/.local/conda/envs/...`) that disappears on reboot. Recreate at the persistent path:
```bash
conda create -p /mnt/afs/davidwang/miniconda3/envs/sana_wm python=3.10 -c conda-forge -y
```

### `git status` fails with `fatal: detected dubious ownership`
Add safe.directory:
```bash
git config --global --add safe.directory /mnt/afs/davidwang/workspace/sana_wm_pipeline
```

### `ffmpeg: command not found`
Install static binaries via pip and copy into the project `.bin/`:
```bash
pip install static-ffmpeg && static_ffmpeg -y
cp /root/.local/lib/python3.10/site-packages/static_ffmpeg/bin/linux/{ffmpeg,ffprobe} .bin/
chmod +x .bin/*
export PATH=$PWD/.bin:$PATH
```

(See user memory `env_ffmpeg_install.md` — conda ffmpeg lacks libx264.so.138 on this host.)

### `pip install` is slow
Use Tsinghua mirror globally:
```bash
mkdir -p ~/.pip && cat > ~/.pip/pip.conf <<EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
timeout = 120

[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```

## Stage-02 pose annotation issues

### VIPE OOM on 60s 720p clip
Stage-02 default-mode pose annotation needs chunk-based inference + halo exchange. The plain-VIPE wrapper in `stage02_pose/mode_default.py` does not yet chunk; for production wire to VIPE's `--chunk-size` argument.

### MoGe-2 metric scale drifts in over-exposed scenes
Raise EMA momentum from default 0.99 toward 0.995 to dampen frame-to-frame jumps. Be aware the paper used 0.99; deviating from it is a reproduction risk.

### Umeyama returns NaN
Source points are degenerate (zero variance). Verify the input correspondences have spatial spread — common cause is a static camera with all "track points" at the same image location.

## Stage-03 3DGS augmentation issues

### FCGS renders contain "floaters"
DiFix3D refinement is required and is sensitive: `timestep=199`, `guidance=0` are the exact paper values. Earlier timesteps lose detail; later timesteps fail to remove the artefacts.

### `coverage_gate` rejects too many trajectories
Lower thresholds rarely help — the paper's 70% / 65% bounds were tuned for DL3DV's scene density. Try alternative trajectory families (the paper samples 30 from 8 families; orbit / spiral usually pass first).

## Stage-05 captioning issues

### Qwen3.5-VL emits camera-motion verbs
The post-processor in `stage05_caption/postprocess.py` should detect via `has_camera_verb` and retry with a stronger prompt. The reproduction's fallback uses Qwen2.5-VL if 3.5 is not yet available.

## Production scaling (CMCC 64×H100)

See `/mnt/afs/davidwang/workspace/docker-images/cmcc/docs/CMCC_DEPLOY_RUNBOOK.md` for the full deployment runbook (14-step SOP). Key reminders:
- `--dereference` when tarring (training data has symlinks → AFS)
- Hot disk (e.g. `/root/work/<userspace>`) for env unpack; `filestorage` for persistent backups
- `source $ENV/bin/activate` (NOT `conda activate`) for conda-pack envs

### Expected per-clip annotation time on single H100
- Stage-01 normalize: ~8 s
- Stage-02 pose (default mode, VIPE+Pi3X+MoGe-2): ~3 min (PRIMARY BOTTLENECK)
- Stage-04 filter: ~12 s
- Stage-05 caption: ~4 s
- Stage-06 pack: ~1 s

Full 212K corpus on 1 H100 ≈ 460 days; on 64×H100 ≈ 7.2 days (roughly 1 GPU-week per source).
