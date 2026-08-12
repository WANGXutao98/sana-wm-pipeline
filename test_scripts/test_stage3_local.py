#!/usr/bin/env python3
"""本机测试 Stage 3：UniMatch + DOVER（不使用永久 CPU 方案）

测试目标：
1. 验证 FP16 GPU 模式 + 降采样优化
2. 测试 sekai-real-walking 样本
3. 验证处理速度和显存占用
4. 确认输出分数合理性
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "models" / "DOVER"))

print("="*80)
print("Stage 3 本机测试：UniMatch + DOVER")
print("="*80)

# ============================================================================
# 1. GPU 信息
# ============================================================================
print("\n[1/6] GPU 信息")
print("-"*80)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}")
    print(f"显存: {gpu_total:.1f} GB")
    print(f"初始占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
else:
    print("❌ 未检测到 GPU")
    sys.exit(1)

# ============================================================================
# 2. 加载测试数据
# ============================================================================
print("\n[2/6] 加载测试数据")
print("-"*80)

test_video = Path(__file__).parent.parent / "testdata" / "sekai-real-walking-hq__FP8j6WfkTY_0085528_0087328.mp4"

if not test_video.exists():
    print(f"❌ 测试视频不存在: {test_video}")
    sys.exit(1)

print(f"测试视频: {test_video.name}")
print(f"文件大小: {test_video.stat().st_size / 1024 / 1024:.2f} MB")

# 读取视频
mp4_bytes = test_video.read_bytes()
print(f"✅ 已加载视频（{len(mp4_bytes) / 1024 / 1024:.2f} MB）")

# ============================================================================
# 3. 解码视频（测试降采样）
# ============================================================================
print("\n[3/6] 解码视频")
print("-"*80)

from sana_wm_pipeline.qc.stage3_gpu import _decode_frames

t0 = time.time()

# 测试原始分辨率
print("测试 A: 原始分辨率（max_resolution=None）")
frames_orig = _decode_frames(mp4_bytes, max_resolution=None)
t1 = time.time()

if frames_orig is not None:
    T, H, W, C = frames_orig.shape
    print(f"  帧数: {T}")
    print(f"  分辨率: {H}×{W}")
    print(f"  解码耗时: {(t1-t0)*1000:.0f} ms")
    print(f"  数据大小: {frames_orig.nbytes / 1024**2:.1f} MB")
else:
    print("  ❌ 解码失败")
    sys.exit(1)

# 测试降采样
print("\n测试 B: 降采样到 480p（max_resolution=480）")
t0 = time.time()
frames_480p = _decode_frames(mp4_bytes, max_resolution=480)
t1 = time.time()

if frames_480p is not None:
    T, H, W, C = frames_480p.shape
    print(f"  帧数: {T}")
    print(f"  分辨率: {H}×{W}")
    print(f"  解码耗时: {(t1-t0)*1000:.0f} ms")
    print(f"  数据大小: {frames_480p.nbytes / 1024**2:.1f} MB")

    # 验证降采样（短边应该接近 480）
    if min(H, W) <= 480:
        print(f"  ✅ 降采样成功")
        orig_pixels = frames_orig.shape[1] * frames_orig.shape[2]
        down_pixels = H * W
        reduction = (1 - down_pixels / orig_pixels) * 100
        print(f"  像素数减少: {reduction:.1f}%")
    else:
        print(f"  ⚠️ 降采样未完全生效")
else:
    print("  ❌ 解码失败")
    sys.exit(1)

# 选择用于测试的视频（使用降采样版本，且截取前 240 帧避免 MIG 显存不足）
frames_rgb = frames_480p[:240]  # 只取前 240 帧（15 秒），MIG 分区显存有限
print(f"\n✅ 使用降采样版本进行测试（{frames_rgb.shape[1]}×{frames_rgb.shape[2]}，{frames_rgb.shape[0]} 帧）")

# ============================================================================
# 4. 加载 UniMatch
# ============================================================================
print("\n[4/6] 加载 UniMatch 模型")
print("-"*80)

try:
    from sana_wm_pipeline.qc.stage3_gpu import load_unimatch_fn

    unimatch_dir = Path(__file__).parent.parent / "models" / "unimatch"
    if not unimatch_dir.exists():
        print(f"❌ UniMatch 模型目录不存在: {unimatch_dir}")
        sys.exit(1)

    t0 = time.time()
    flow_fn = load_unimatch_fn(str(unimatch_dir), device="cuda")
    t1 = time.time()

    print(f"✅ UniMatch 加载完成（{t1-t0:.1f} 秒）")
    print(f"显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

except Exception as e:
    print(f"❌ UniMatch 加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 5. 加载 DOVER
# ============================================================================
print("\n[5/6] 加载 DOVER 模型")
print("-"*80)

try:
    from sana_wm_pipeline.qc.stage3_gpu import load_dover_fn

    t0 = time.time()
    # 本机 MIG 分区显存有限，使用 FP32
    dover_fn = load_dover_fn(device="cuda", use_fp16=False)
    t1 = time.time()

    print(f"✅ DOVER 加载完成（{t1-t0:.1f} 秒）")
    print(f"显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

except Exception as e:
    print(f"❌ DOVER 加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 6. 运行质检
# ============================================================================
print("\n[6/6] 运行质检")
print("-"*80)

# 清理显存
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print(f"视频规格: {frames_rgb.shape[0]} 帧, {frames_rgb.shape[1]}×{frames_rgb.shape[2]}")

# UniMatch 光流
print("\n计算 UniMatch 光流...")
try:
    from sana_wm_pipeline.stage04_filter.visual_metrics import unimatch_flow_magnitude

    t0 = time.time()
    flow_val = unimatch_flow_magnitude(frames_rgb, flow_fn, fps=16)
    t1 = time.time()

    mem_peak = torch.cuda.max_memory_allocated() / 1024**3

    print(f"✅ UniMatch 完成")
    print(f"   光流幅度: {flow_val:.2f} 像素/帧")
    print(f"   耗时: {(t1-t0)*1000:.0f} ms")
    print(f"   峰值显存: {mem_peak:.2f} GB")

    # 清理显存
    torch.cuda.empty_cache()

except Exception as e:
    print(f"❌ UniMatch 失败: {e}")
    import traceback
    traceback.print_exc()
    flow_val = None

# DOVER 质量
print("\n计算 DOVER 质量...")
try:
    from sana_wm_pipeline.stage04_filter.visual_metrics import dover_score

    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    dover_val = dover_score(frames_rgb, dover_fn, fps=16)
    t1 = time.time()

    mem_peak = torch.cuda.max_memory_allocated() / 1024**3

    print(f"✅ DOVER 完成")
    print(f"   质量分数: {dover_val:.4f}")
    print(f"   耗时: {(t1-t0)*1000:.0f} ms")
    print(f"   峰值显存: {mem_peak:.2f} GB")

except Exception as e:
    print(f"❌ DOVER 失败: {e}")
    import traceback
    traceback.print_exc()
    dover_val = None

# 色彩饱和度
print("\n计算色彩饱和度...")
try:
    from sana_wm_pipeline.stage04_filter.visual_metrics import mean_saturation

    t0 = time.time()
    sat_val = mean_saturation(frames_rgb)
    t1 = time.time()

    print(f"✅ 饱和度完成")
    print(f"   饱和度: {sat_val:.2f}")
    print(f"   耗时: {(t1-t0)*1000:.0f} ms")

except Exception as e:
    print(f"❌ 饱和度计算失败: {e}")
    sat_val = None

# ============================================================================
# 7. 结果汇总
# ============================================================================
print("\n" + "="*80)
print("测试结果汇总")
print("="*80)

print(f"\n样本: {test_video.name}")
print(f"视频规格: {frames_rgb.shape[0]} 帧, {frames_rgb.shape[1]}×{frames_rgb.shape[2]}")

print(f"\n质检分数:")
print(f"  UniMatch 光流: {flow_val:.2f} 像素/帧" if flow_val else "  UniMatch 光流: N/A")
print(f"  DOVER 质量:    {dover_val:.4f}" if dover_val else "  DOVER 质量:    N/A")
print(f"  色彩饱和度:    {sat_val:.2f}" if sat_val else "  色彩饱和度:    N/A")

# 与 Table 6 阈值对比（sekai-real-walking）
print(f"\nTable 6 阈值对比（Sekai_Real_Walking）:")
print(f"  光流范围:     [3, 80]")
print(f"  质量范围:     [0.40, 1.0]")
print(f"  饱和度范围:   [0, 180]")

if flow_val and dover_val and sat_val:
    passed = []
    failed = []

    if 3 <= flow_val <= 80:
        passed.append("光流")
    else:
        failed.append(f"光流({flow_val:.2f} NOT in [3,80])")

    if 0.40 <= dover_val <= 1.0:
        passed.append("DOVER")
    else:
        failed.append(f"DOVER({dover_val:.4f} NOT in [0.40,1.0])")

    if 0 <= sat_val <= 180:
        passed.append("饱和度")
    else:
        failed.append(f"饱和度({sat_val:.2f} NOT in [0,180])")

    print(f"\n判定结果:")
    if len(failed) == 0:
        print(f"  ✅ 通过所有检查")
    else:
        print(f"  ❌ 未通过: {', '.join(failed)}")
        print(f"  ✅ 通过: {', '.join(passed)}")

print("\n" + "="*80)
print("测试完成")
print("="*80)
