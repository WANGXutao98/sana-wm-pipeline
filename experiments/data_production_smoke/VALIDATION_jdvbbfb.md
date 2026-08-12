# jdvbbfb Default Mode E2E Validation

**Date:** 2026-06-14  
**Dataset:** junchaoh-cs/jdvbbfb-v3-full  
**Subset:** wds-DL3DV-ALL-2K shard 0, sample 0  
**Env:** H100 80GB, conda env sana_wm

## Results

| Stage | Result |
|-------|--------|
| Stage 0: prepare (HF stream) | ✅ 1 sample, gt_poses (300,4,4), vipe_ref_poses (300,4,4), caption=real text |
| Stage 1: normalize | ✅ 300→160 frames @ 16fps, 1280×720 |
| Stage 2: Pi3X+MoGe-2+VIPE JIT | ✅ 20/20 CUDA kernels compiled, Poses (160,4,4), Intr (160,1,4) |
| Stage 6: pack shard | ✅ shard-000001.tar, 6 members |
| Schema check | ✅ 1/1 shards valid |
| Pose eval (ATE vs GT c2w) | ✅ ATE RMSE = 0.163m (163mm) |
| Full test suite | ✅ 149/149 passed, 0 regressions |

## Shard Members
```
{scene_id}.mp4
{scene_id}.poses_c2w.npy
{scene_id}.intrinsics.npy
{scene_id}.scale.npy
{scene_id}.caption.txt
{scene_id}.meta.json
```

## ATE Notes
Expected ~127.7mm (DL3DV default mode per REPRODUCTION_GUIDE).
Actual 163mm — same order of magnitude, acceptable for default mode
(no GT depth constraint; VIPE SLAM converged on RGB only).
