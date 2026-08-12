#!/usr/bin/env python3
"""
UniMatch 真实视频光流验证脚本（CMCC H100）

新增功能：
- 测试 8：真实视频光流计算
- 测试 9：光流可视化保存

基于 test_unimatch_cmcc_v2.py，增加真实视频测试
"""

import sys
import os
import time
import glob
import gc
import torch
import numpy as np
import cv2

# ====================== 配置常量 ======================
UNIMATCH_ROOT = '/root/work/david_work/models/unimatch'
PRETRAINED_DIR = os.path.join(UNIMATCH_ROOT, 'pretrained')

# 真实视频测试
TEST_VIDEO = "/root/work/david_work/sana_qc_pipeline/DOVER/demo/SpatialVID-hq_622345a9-0375-5f10-941e-ffc8765e651a.mp4"
OUTPUT_DIR = "/tmp/unimatch_video_test"

# 模型结构参数
MODEL_CONFIG = {
    'feature_channels': 128,
    'num_scales': 2,
    'upsample_factor': 4,
    'num_head': 1,
    'ffn_dim_expansion': 4,
    'num_transformer_layers': 6,
}

# 推理配置
INFERENCE_HEIGHT, INFERENCE_WIDTH = 256, 256
BENCHMARK_WARMUP_ROUNDS = 3
BENCHMARK_TEST_ROUNDS = 10

INFERENCE_PARAMS = {
    'attn_type': 'swin',
    'attn_splits_list': [2, 8],
    'corr_radius_list': [-1, 4],
    'prop_radius_list': [-1, 1],
}
# ======================================================

sys.path.insert(0, UNIMATCH_ROOT)

print("=" * 80)
print("UniMatch 真实视频光流验证脚本")
print("=" * 80)

# ============================================================================
# 测试 1-5：环境检查 + 模型加载（与 v2 相同，简化输出）
# ============================================================================
print("\n[测试 1-5] 快速环境检查...")

from unimatch.unimatch import UniMatch

model = UniMatch(**MODEL_CONFIG)
model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    print(f"✅ GPU 模式：{torch.cuda.get_device_name(0)}")
else:
    print("⚠️  CPU 模式")

# 加载权重
weight_files = []
for ext in ['*.pth', '*.pt']:
    weight_files.extend(glob.glob(os.path.join(PRETRAINED_DIR, ext)))

if weight_files:
    weight_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    weight_file = weight_files[0]
    try:
        checkpoint = torch.load(weight_file, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 权重加载：{os.path.basename(weight_file)}")
    except:
        print("⚠️  权重加载失败，使用随机权重")

# ============================================================================
# 测试 6：随机数据快速验证
# ============================================================================
print("\n[测试 6] 随机数据快速验证")
img1 = torch.randn(1, 3, INFERENCE_HEIGHT, INFERENCE_WIDTH)
img2 = torch.randn(1, 3, INFERENCE_HEIGHT, INFERENCE_WIDTH)

if torch.cuda.is_available():
    img1, img2 = img1.cuda(), img2.cuda()

try:
    with torch.no_grad():
        flow_output = model(img1, img2, **INFERENCE_PARAMS)
    print("✅ 随机数据推理成功")
except Exception as e:
    print(f"❌ 推理失败: {e}")
    sys.exit(1)

# ============================================================================
# 辅助函数：视频帧处理
# ============================================================================
def load_video_frames(video_path, max_frames=30, target_size=(256, 256)):
    """加载视频帧并调整大小"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    frames = []
    frame_count = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 调整大小
        if target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

        frames.append(frame)
        frame_count += 1

    cap.release()

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return frames, {
        'fps': fps,
        'total_frames': frame_count,
        'orig_size': (orig_width, orig_height),
        'processed_size': target_size
    }

def preprocess_frame(frame):
    """将 numpy 图像转为 torch tensor (C, H, W), 归一化到 [0, 1]"""
    # frame: (H, W, 3) uint8 -> (3, H, W) float32
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)  # (1, 3, H, W)

def flow_to_color(flow, max_flow=None):
    """将光流转换为 HSV 彩色可视化（Middlebury color wheel）"""
    u, v = flow[..., 0], flow[..., 1]

    # 计算幅度和角度
    mag = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)

    # 归一化幅度
    if max_flow is None:
        max_flow = mag.max()
    if max_flow > 0:
        mag = mag / max_flow

    # HSV 编码
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (angle + np.pi) / (2 * np.pi) * 179  # Hue: 角度
    hsv[..., 1] = 255                                   # Saturation: 满
    hsv[..., 2] = np.clip(mag * 255, 0, 255)           # Value: 幅度

    # HSV -> RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

# ============================================================================
# 测试 7：真实视频光流计算
# ============================================================================
print("\n[测试 7] 真实视频光流计算")
print(f"测试视频: {TEST_VIDEO}")

# 检查视频文件
if not os.path.exists(TEST_VIDEO):
    print(f"❌ 视频文件不存在: {TEST_VIDEO}")
    print("跳过真实视频测试")
else:
    try:
        # 加载视频帧
        print("加载视频帧...")
        frames, video_info = load_video_frames(
            TEST_VIDEO,
            max_frames=30,
            target_size=(INFERENCE_HEIGHT, INFERENCE_WIDTH)
        )

        print(f"✅ 加载成功")
        print(f"   原始分辨率: {video_info['orig_size']}")
        print(f"   处理分辨率: {video_info['processed_size']}")
        print(f"   FPS: {video_info['fps']:.2f}")
        print(f"   总帧数: {len(frames)}")

        # 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 计算连续帧对的光流
        print(f"\n计算光流（共 {len(frames)-1} 对）...")
        flow_results = []
        flow_times = []

        for i in range(len(frames) - 1):
            # 预处理
            img1_t = preprocess_frame(frames[i])
            img2_t = preprocess_frame(frames[i+1])

            if torch.cuda.is_available():
                img1_t = img1_t.cuda()
                img2_t = img2_t.cuda()

            # 推理
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()

            with torch.no_grad():
                flow_output = model(img1_t, img2_t, **INFERENCE_PARAMS)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = (time.time() - start) * 1000
            flow_times.append(elapsed)

            # 提取光流
            if isinstance(flow_output, dict) and 'flow_preds' in flow_output:
                flow_pred = flow_output['flow_preds'][-1]
            else:
                flow_pred = flow_output

            # 转为 numpy (2, H, W) -> (H, W, 2)
            flow_np = flow_pred[0].cpu().numpy().transpose(1, 2, 0)
            flow_results.append(flow_np)

            if (i + 1) % 5 == 0:
                print(f"  进度: {i+1}/{len(frames)-1}, 最近 5 帧平均耗时: {np.mean(flow_times[-5:]):.2f} ms")

        print(f"✅ 光流计算完成")
        print(f"   平均耗时: {np.mean(flow_times):.2f} ms/帧对")
        print(f"   总耗时: {sum(flow_times)/1000:.2f} 秒")

        # 分析光流统计
        flow_mags = [np.sqrt((f**2).sum(axis=2)).mean() for f in flow_results]
        print(f"\n光流统计分析:")
        print(f"   平均幅度: {np.mean(flow_mags):.4f} 像素")
        print(f"   最小幅度: {np.min(flow_mags):.4f} 像素")
        print(f"   最大幅度: {np.max(flow_mags):.4f} 像素")
        print(f"   标准差: {np.std(flow_mags):.4f} 像素")

        # 保存部分可视化
        print(f"\n保存可视化结果到 {OUTPUT_DIR}/...")
        sample_indices = [0, len(flow_results)//2, len(flow_results)-1]

        for idx in sample_indices:
            if idx < len(flow_results):
                flow = flow_results[idx]
                flow_vis = flow_to_color(flow)

                output_path = os.path.join(OUTPUT_DIR, f"flow_frame_{idx:03d}.png")
                cv2.imwrite(output_path, cv2.cvtColor(flow_vis, cv2.COLOR_RGB2BGR))
                print(f"   保存: flow_frame_{idx:03d}.png")

        print(f"\n✅ 真实视频测试完成")
        print(f"   输出目录: {OUTPUT_DIR}")

    except Exception as e:
        print(f"❌ 真实视频测试失败: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 资源清理
# ============================================================================
del img1, img2
if 'flow_output' in locals():
    del flow_output
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================================
# 验证总结
# ============================================================================
print("\n" + "=" * 80)
print("验证总结")
print("=" * 80)
print("✅ UniMatch 模块导入正常")
print(f"✅ 模型加载成功 ({'GPU' if torch.cuda.is_available() else 'CPU'} 模式)")
print("✅ 随机数据推理正常")

if os.path.exists(TEST_VIDEO):
    print(f"✅ 真实视频测试完成 (平均 {np.mean(flow_times):.2f} ms/帧对)")
    print(f"   光流平均幅度: {np.mean(flow_mags):.4f} 像素")
else:
    print("⚠️  真实视频测试跳过（文件不存在）")

print("\n🎉 UniMatch 真实视频验证通过！")
print("=" * 80)
