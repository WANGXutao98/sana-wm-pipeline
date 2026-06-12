# 操作记录：DL3DV 端到端管线实施

**日期**：2026-06-12  
**计划文件**：`docs/superpowers/plans/2026-06-11-dl3dv-e2e-pipeline.md`  
**最终状态**：✅ 全部完成，141 tests passed

---

## 修改的已有文件（6 个）

### Fix 3: `configs/filter_thresholds.yaml`
- 新增 `DL3DV` 数据源（第 68-75 行）
- `unimatch_flow: null`，`dover: null`（DL3DV smoke 环境未安装）
- `vmaf_motion: [0.5, 100]`，`color_saturation: [0, 180]`，`scene_cuts_max: 2`

### Fix 4: `src/sana_wm_pipeline/stage05_caption/qwen35_vl_runner.py`
- 新增模块常量 `CAPTION_FALLBACK`（第 21 行）
- `caption_clip()` 内部：
  - 捕获 `RuntimeError`（VLM 不可用）→ 返回 fallback
  - `strip_offending_sentences(last) or CAPTION_FALLBACK`

### Fix 5: `src/sana_wm_pipeline/stage06_pack/schema.py`
- `validate(strict_frames=True)` — False 时仅校验 ndim/shape[1:]，不要求精确 961 帧
- `intrinsics_NVD` 和 `scale_per_frame` 同样加了 strict 分支

### Fix 5: `src/sana_wm_pipeline/stage06_pack/webdataset_writer.py`
- `ShardWriter.__init__(strict_frames=True)` 参数
- `write()` 传 `strict_frames=self._strict_frames` 给 `validate()`

### Fix 1: `src/sana_wm_pipeline/stage02_pose/mode_default.py`
- `VIPE_PIPELINE = "vipe_cached_depth"`（原 `sana_wm_pose_only`）
- 新增 `_precompute_depth_cache()` 函数（Phase A：Pi3X+MoGe-2 EMA fusion）
- `run_default()` 改为两阶段：预计算→设 env var→VIPE SLAM→finally 清理

### 测试修复
- `tests/test_apply_table6.py`：`test_yaml_has_all_six_sources` → `test_yaml_has_all_sources`（expected 集合加入 `DL3DV`）
- `tests/test_pose_modes.py`：`test_default_mode_loads_artifact` 新增 monkeypatch `_precompute_depth_cache` + 设置 env vars

---

## 新建文件（7 个）

### `scripts/pi3x_infer_cli.py`（Fix 2）
- Pi3X CLI 脚本，供 `mode_gtpose.py` 调用
- 接口：`--video --emit-cams --emit-points [--chunk 16 --stride 8]`
- 核心：`recover_intrinsics_from_rays()` 从 rays 最小二乘恢复 K(3×3)
- 输出格式：`cams_pi3x.json = {"frames": [{"center": [x,y,z], "K": [[...]]}, ...]}`

### `experiments/data_production_smoke/download_dl3dv.sh`
- 从 HuggingFace DL3DV/DL3DV-ALL-2K 下载 5 个 smoke test 场景

### `experiments/data_production_smoke/prepare_dl3dv.py`
- 输入：`<scene>/images/*.png` + `transforms.json`
- 输出：`video.mp4`（ffmpeg）、`gt_poses.npy`（T,4,4）、`gt_intrinsics.npy`（4,）、`orig_fps.txt`
- DL3DV 用 OpenCV c2w 约定，与管线一致，无需坐标转换

### `experiments/data_production_smoke/run_e2e_default.sh`
- Default 模式端到端（Stage 01→02_default→05→06，Stage 04 在 smoke 中跳过）
- 使用 heredoc Python 调用各 Stage 模块

### `experiments/data_production_smoke/run_e2e_gtpose.sh`
- GT-pose 模式端到端
- Stage 02 使用 `mode_gtpose.run_gtpose(pi3x_cmd=scripts/pi3x_infer_cli.py)`

### `experiments/data_production_smoke/verify_and_eval.py`
- `--mode schema`：校验每个 tar 含 6 个必要文件
- `--mode pose-eval`：evo 计算 ATE RMSE/RTE，生成 3-view 轨迹对比图
- 降级处理：evo/matplotlib 不可用时优雅跳过

### `experiments/data_production_smoke/run_sana_wm_inference.py`
- 从 shard 提取 sample → 准备 first_frame.png + intrinsics_flat(4,) → 调用 inference_sana_wm.py
- 对比生成视频 vs DL3DV GT：逐帧 PSNR/SSIM + side-by-side MP4
- `--sample-limit N` 控制处理数量

---

## 下一步（待实际运行）

按计划执行顺序：
```bash
# Day 1 下午：下载数据（~2h）
bash experiments/data_production_smoke/download_dl3dv.sh

# 准备场景
python experiments/data_production_smoke/prepare_dl3dv.py \
  /mnt/afs/davidwang/workspace/data/dl3dv_smoke/1K/*/

# Day 1 晚：Default 模式端到端（5 场景，每场景 30-60 min）
bash experiments/data_production_smoke/run_e2e_default.sh

# Day 2 上午：GT-pose 模式
bash experiments/data_production_smoke/run_e2e_gtpose.sh

# Day 2 下午：验证
python experiments/data_production_smoke/verify_and_eval.py \
  --mode schema --shards-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default

python experiments/data_production_smoke/verify_and_eval.py \
  --mode pose-eval \
  --shards-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default \
  --scenes-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke

# Day 3 下午：SANA-WM 推理（需先确认权重）
python experiments/data_production_smoke/run_sana_wm_inference.py \
  --shards-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default \
  --sana-dir /mnt/afs/davidwang/workspace/Sana \
  --output-dir /mnt/afs/davidwang/workspace/data/sana_wm_results \
  --sample-limit 1
```
