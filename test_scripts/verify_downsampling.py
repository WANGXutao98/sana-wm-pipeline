#!/usr/bin/env python3
"""验证降采样功能是否生效

直接调用 process_sample_stage3 测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 测试 1：验证 _decode_frames 函数签名
print("="*80)
print("测试 1：验证 _decode_frames 函数")
print("="*80)

try:
    from sana_wm_pipeline.qc import stage3_gpu
    import inspect

    sig = inspect.signature(stage3_gpu._decode_frames)
    print(f"✅ _decode_frames 函数签名: {sig}")

    params = list(sig.parameters.keys())
    if 'max_resolution' in params:
        print(f"✅ max_resolution 参数存在")
        default = sig.parameters['max_resolution'].default
        print(f"   默认值: {default}")
    else:
        print(f"❌ max_resolution 参数不存在")
        print(f"   当前参数: {params}")
except Exception as e:
    print(f"❌ 错误: {e}")

# 测试 2：验证 process_sample_stage3 中的调用
print(f"\n{'='*80}")
print("测试 2：验证 process_sample_stage3 代码")
print("="*80)

try:
    import inspect
    source = inspect.getsource(stage3_gpu.process_sample_stage3)

    # 查找 _decode_frames 调用
    if 'max_resolution' in source:
        print(f"✅ process_sample_stage3 中包含 max_resolution 参数")
        # 提取相关行
        for i, line in enumerate(source.split('\n')):
            if '_decode_frames' in line and 'max_resolution' in line:
                print(f"   行 {i}: {line.strip()}")
    else:
        print(f"❌ process_sample_stage3 中未使用 max_resolution")
        # 提取相关行
        for i, line in enumerate(source.split('\n')):
            if '_decode_frames' in line:
                print(f"   行 {i}: {line.strip()}")
except Exception as e:
    print(f"❌ 错误: {e}")

# 测试 3：实际测试降采样
print(f"\n{'='*80}")
print("测试 3：实际测试降采样功能")
print("="*80)

try:
    import numpy as np
    import io

    # 读取测试视频
    test_video = "/mnt/afs/davidwang/workspace/sana_wm_pipeline/testdata/sekai-real-walking-hq__FP8j6WfkTY_0085528_0087328.mp4"

    if Path(test_video).exists():
        with open(test_video, 'rb') as f:
            mp4_bytes = f.read()

        print(f"✅ 加载测试视频: {len(mp4_bytes)/1024/1024:.2f} MB")

        # 测试不降采样
        print(f"\n测试 A: 不降采样（原始分辨率）")
        frames_orig = stage3_gpu._decode_frames(mp4_bytes, max_resolution=None)
        if frames_orig is not None:
            print(f"   分辨率: {frames_orig.shape[1]}×{frames_orig.shape[2]}")
            print(f"   帧数: {frames_orig.shape[0]}")

        # 测试降采样到 480p
        print(f"\n测试 B: 降采样到 480p (max_resolution=640)")
        frames_480p = stage3_gpu._decode_frames(mp4_bytes, max_resolution=640)
        if frames_480p is not None:
            print(f"   分辨率: {frames_480p.shape[1]}×{frames_480p.shape[2]}")
            print(f"   帧数: {frames_480p.shape[0]}")

            # 验证降采样
            if max(frames_480p.shape[1], frames_480p.shape[2]) <= 640:
                print(f"   ✅ 降采样成功")
            else:
                print(f"   ❌ 降采样失败")

        # 对比
        if frames_orig is not None and frames_480p is not None:
            orig_pixels = frames_orig.shape[1] * frames_orig.shape[2]
            down_pixels = frames_480p.shape[1] * frames_480p.shape[2]
            reduction = (1 - down_pixels / orig_pixels) * 100
            print(f"\n   像素数减少: {reduction:.1f}%")
    else:
        print(f"❌ 测试视频不存在: {test_video}")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*80}")
print("测试完成")
print("="*80)
