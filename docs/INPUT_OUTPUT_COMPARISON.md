# 数据标注输入输出对比

> **创建日期**：2026-08-12  
> **目的**：对比当前流程与 sana-wm-data-clean 的输入输出

---

## 一、输入输出对比

### 1.1 当前流程 (sana_wm_pipeline)

#### 输入
```
单样本输入：
/path/to/video.mp4                # 原始视频（任意分辨率/帧率/编码）

可选GT输入（特定数据集）：
/path/to/gt_poses.npy             # GT轨迹 (N,4,4) 或 (N,3,4) 或 (N,3)
/path/to/gt_intrinsics.npy        # GT内参 (N,4) 或 (4,)
/path/to/gt_depth.npy             # GT深度 (N,H,W)（OmniWorld数据集）
```

#### 输出（完整6阶段）
```
output/
├── stage01_ingest/
│   ├── video_normalized.mp4      # 720p, 16fps, 961帧
│   └── ingest_meta.json
├── stage02_pose/
│   ├── poses_c2w.npy             # (961, 4, 4) camera-to-world
│   ├── intrinsics.npy            # (961, 4) [fx, fy, cx, cy]
│   ├── scale.npy                 # (961,) 逐帧尺度因子
│   └── pose_meta.json
├── stage04_filter/
│   └── filter_result.json        # UniMatch光流 + DOVER质量评分
├── stage05_caption/
│   └── caption.txt               # Scene-static描述
└── stage06_pack/
    └── 00000000.tar              # WebDataset shard
        ├── video.mp4
        ├── poses_c2w.npy
        ├── intrinsics.npy
        ├── scale.npy
        └── caption.txt
```

**关键特点**：
- ✅ 完整6阶段pipeline
- ✅ 支持3种模式（default/gt_pose/gt_depth）
- ✅ 集成QC系统
- ❌ **Stage 2深度融合算法有bug**（已定位）

---

### 1.2 sana-wm-data-clean（参考实现）

#### 输入
```
单样本输入：
/path/to/video.mp4                # 原始视频

命令行示例：
sana-wm-camera input.mp4 \
  --out /tmp/camera_out \
  --mode default \
  --max-frames 64
  
可选GT输入（gt_pose模式）：
--gt-poses /path/to/poses.npy
--gt-intrinsics /path/to/intrinsics.npy
```

#### 输出（仅位姿标注）
```
/tmp/camera_out/
├── <clip-id>.poses.npy           # (N, 4, 4) camera-to-world, float64
├── <clip-id>.intrinsics.npy      # (N, 4) [fx,fy,cx,cy], float64
├── result.json                   # 包含QC指标
└── <clip-id>/                    # VIPE中间结果（可选）
    ├── depth/
    ├── vipe_out/
    └── intr_pf.npy
```

**result.json 内容**：
```json
{
  "clip_id": "sample_001",
  "video": "/path/to/input.mp4",
  "mode": "default",
  "backend": "vipe_cli",
  "convention": "camera-to-world (c2w), OpenCV axes",
  "poses": {
    "path": "/tmp/camera_out/sample_001.poses.npy",
    "shape": [961, 4, 4],
    "dtype": "float64"
  },
  "intrinsics": {
    "path": "/tmp/camera_out/sample_001.intrinsics.npy",
    "shape": [961, 4],
    "dtype": "float64",
    "layout": ["fx", "fy", "cx", "cy"]
  },
  "scale": {
    "count": 961,
    "min": 0.98,
    "max": 1.05
  },
  "camera_qc": {
    "passed": true,
    "fov_x_deg": 67.3,
    "fov_y_deg": 42.1,
    "focal_divergence": 0.03,
    "scale_cov": 0.15,
    "thresholds": {
      "fov_deg": [25.0, 120.0],
      "focal_div_max": 0.20,
      "scale_cov_max": 2.0
    }
  }
}
```

**关键特点**：
- ✅ **核心算法100%对齐论文**
- ✅ 自带相机QC检查
- ❌ 只有位姿标注（无ingest/filter/caption/pack）
- ❌ 单样本CLI，无批量处理框架

---

## 二、VIPE环境分析

### 2.1 当前环境状态

**已安装的VIPE**：
```bash
# 位置
/mnt/afs/davidwang/miniconda3/envs/sana_wm/bin/vipe

# 版本
vipe, version 1.1.0

# 源码
/mnt/afs/davidwang/workspace/sana_wm_pipeline/third_party/vipe/
```

**补丁目录**：
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/third_party/vipe_patch/
├── depth_backend_pi3x_moge2.py        # Pi3X+MoGe-2深度后端
├── ba_per_frame_intrinsics.py         # 逐帧内参BA补丁
└── sana_wm_pose_only.yaml             # VIPE配置文件
```

### 2.2 VIPE环境对比

| 维度 | 当前sana_wm环境 | sana-wm-data-clean要求 |
|------|----------------|----------------------|
| **VIPE安装位置** | conda环境内 | 独立.venv-vipe/ |
| **VIPE版本** | 1.1.0 | 1.1.0 |
| **补丁文件** | third_party/vipe_patch/ | vipe_patches/ |
| **深度后端** | depth_backend_pi3x_moge2.py | pi3x_moge_depth.py |
| **内参BA** | ba_per_frame_intrinsics.py | apply_perframe_intrinsics_ba.py |
| **配置文件** | sana_wm_pose_only.yaml | sanawm_pipeline.yaml |

**关键发现**：
- ✅ **VIPE版本相同**（1.1.0）
- ✅ **补丁功能相同**（Pi3X+MoGe-2 + 逐帧内参BA）
- ⚠️ **文件名不同，但内容功能一致**

### 2.3 补丁是否已应用？

需要检查VIPE源码是否已经包含补丁修改：

```bash
# 检查1：深度后端是否注册
grep -r "pi3x_moge2" /mnt/afs/davidwang/workspace/sana_wm_pipeline/third_party/vipe/vipe/priors/depth/

# 检查2：逐帧内参BA是否集成
grep -r "per.*frame.*intrinsics" /mnt/afs/davidwang/workspace/sana_wm_pipeline/third_party/vipe/
```

---

## 三、结论与建议

### 3.1 输入输出兼容性

✅ **两套代码输出格式兼容**：
- 都输出 `poses.npy` (N,4,4) + `intrinsics.npy` (N,4)
- 坐标系统一致（camera-to-world, OpenCV轴）
- dtype一致（float64）

**差异点**：
- 当前代码额外输出 `scale.npy`（逐帧）
- 参考实现把scale放在 `result.json` 的 `scale_factors` 数组中

### 3.2 VIPE环境结论

**判断**：当前 sana_wm 环境中的VIPE **基本等同于** sana-wm-data-clean要求的环境

**理由**：
1. ✅ VIPE版本相同（1.1.0）
2. ✅ 补丁文件存在且功能一致
3. ⚠️ **但需要验证补丁是否已应用到VIPE源码**

**下一步行动**：
```bash
# 验证补丁是否生效
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/third_party/vipe_patch
grep -A 20 "class.*Pi3XMoGe2" depth_backend_pi3x_moge2.py

# 检查VIPE源码中是否有对应实现
find ../vipe -name "*.py" -exec grep -l "Pi3XMoGe2" {} \;
```

### 3.3 最终建议

**方案A：直接使用参考实现**（如果补丁未应用）
- 优点：算法100%正确
- 缺点：需要独立环境，配置复杂

**方案B：修复当前代码**（如果补丁已应用且VIPE可用）
- 优点：复用现有环境和pipeline
- 缺点：需要验证修复完整性

**建议执行顺序**：
1. 先验证当前VIPE补丁是否生效（5分钟）
2. 如果生效，只修复 `depth_fusion.py` 算法（1小时）
3. 如果未生效，考虑直接用参考实现（但需要重新配置VIPE）

---

## 四、快速验证脚本

```bash
# 1. 验证VIPE补丁状态
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python -c "
import sys
sys.path.insert(0, 'third_party/vipe')
try:
    from vipe.priors.depth import make_depth_model
    model = make_depth_model('pi3x_moge2')
    print('✅ Pi3X+MoGe-2后端已注册')
except:
    print('❌ 补丁未应用，需要重新配置')
"

# 2. 测试当前代码能否调用VIPE
conda activate sana_wm
vipe infer testdata/sekai-real-walking-hq__*.mp4 \
  -o /tmp/vipe_test \
  --pipeline sana_wm_pose_only
```

---

**等待验证结果后再决定最终方案。**
