#!/usr/bin/env python3
"""
Stage 3 单样本测试脚本（CMCC）

用途：验证端到端流程，确保所有模块正常工作
用法：python run_stage3_single_sample.py --sample-id <id> --video-path <path>
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/root/work/david_work/sana_qc_pipeline')

def main():
    parser = argparse.ArgumentParser(description='Stage 3 单样本测试')
    parser.add_argument('--sample-id', required=True, help='样本 ID')
    parser.add_argument('--video-path', required=True, help='视频文件路径')
    parser.add_argument('--caption-path', help='Caption 文件路径（可选）')
    parser.add_argument('--output', default='/tmp/stage3_single_test.jsonl', help='输出文件')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU 设备 ID')
    args = parser.parse_args()

    print("=" * 80)
    print("Stage 3 单样本测试")
    print("=" * 80)
    print(f"样本 ID: {args.sample_id}")
    print(f"视频路径: {args.video_path}")
    print(f"GPU: {args.gpu_id}")
    print()

    # 检查文件存在
    if not Path(args.video_path).exists():
        print(f"❌ 错误：视频文件不存在: {args.video_path}")
        return 1

    # 设置环境变量
    os.environ['TORCH_HOME'] = '/root/work/david_work/cache/torch'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # 加载模型（延迟导入，避免环境变量设置前加载）
    print("[1/5] 加载依赖...")
    import torch
    import numpy as np

    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 加载 Stage 3 模块
    print("[2/5] 初始化 Stage 3 模块...")

    # 方案 A：直接调用 stage3_gpu.py 的函数
    try:
        from sana_wm_pipeline.qc.stage3_gpu import load_unimatch_fn, load_dover_fn, load_qwen_fn

        print("  加载 UniMatch...")
        flow_fn = load_unimatch_fn(
            model_dir='/root/work/david_work/models/unimatch',
            device=f'cuda:{args.gpu_id}'
        )

        print("  加载 DOVER...")
        dover_fn = load_dover_fn(
            model_dir='/root/work/david_work/sana_qc_pipeline/DOVER',
            device=f'cuda:{args.gpu_id}'
        )

        print("  加载 Qwen3.5-9B...")
        vlm_call = load_qwen_fn(
            model_dir='/root/work/david_work/models/Qwen3.5-9B',
            device=f'cuda:{args.gpu_id}'
        )

        print("  ✅ 所有模型加载成功")
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()

    # 读取视频
    print("[3/5] 读取视频...")
    try:
        import av
        frames_rgb = []
        with av.open(args.video_path) as container:
            for frame in container.decode(video=0):
                frames_rgb.append(frame.to_ndarray(format='rgb24'))
        frames_rgb = np.array(frames_rgb)
        print(f"  ✅ 读取成功：{frames_rgb.shape}")
    except Exception as e:
        print(f"  ❌ 视频读取失败: {e}")
        return 1

    print()

    # 执行 Stage 3 检测
    print("[4/5] 执行质量检测...")
    result = {
        'sample_id': args.sample_id,
        'stage3': {
            'unimatch_flow': None,
            'dover': None,
            'vlm_entity_count': None,
            'vlm_quality': None,
            'caption_revised': None,
            'table6_accepted': None,
            'reasons': []
        }
    }

    # UniMatch 光流
    print("  [a] UniMatch 光流检测...")
    start = time.time()
    try:
        from sana_wm_pipeline.stage04_filter.visual_metrics import unimatch_flow_magnitude
        flow_val = unimatch_flow_magnitude(frames_rgb, flow_fn)
        result['stage3']['unimatch_flow'] = round(float(flow_val), 3)
        print(f"      光流幅度: {flow_val:.3f} 像素 ({time.time()-start:.2f}s)")
    except Exception as e:
        print(f"      ❌ 失败: {e}")
        result['stage3']['reasons'].append(f'unimatch_error: {e}')

    # DOVER 质量评分
    print("  [b] DOVER 质量评分...")
    start = time.time()
    try:
        from sana_wm_pipeline.stage04_filter.visual_metrics import dover_score
        dover_val = dover_score(frames_rgb, dover_fn)
        result['stage3']['dover'] = round(float(dover_val), 4)
        print(f"      质量分数: {dover_val:.4f} ({time.time()-start:.2f}s)")
    except Exception as e:
        print(f"      ❌ 失败: {e}")
        result['stage3']['reasons'].append(f'dover_error: {e}')

    # Qwen Caption 改写（如果提供了 caption）
    if args.caption_path and Path(args.caption_path).exists():
        print("  [c] Qwen Caption 改写...")
        start = time.time()
        try:
            with open(args.caption_path) as f:
                caption_text = f.read().strip()

            # 检查是否包含相机词汇
            camera_words = ['camera', 'panning', 'zooming', 'tilting', 'tracking', 'dolly']
            has_camera = any(word in caption_text.lower() for word in camera_words)

            if has_camera:
                from sana_wm_pipeline.stage04_filter.vlm_entity_quality import ENTITY_QUALITY_PROMPT
                prompt = ENTITY_QUALITY_PROMPT + f"\n\nCaption: {caption_text}"

                # 使用 8 个关键帧
                keyframes = [frames_rgb[i] for i in np.linspace(0, len(frames_rgb)-1, 8).astype(int)]
                raw = vlm_call(prompt, keyframes)

                # 解析 JSON
                parsed = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
                result['stage3']['caption_revised'] = parsed.get('caption_revised', caption_text)
                print(f"      原始: {caption_text[:50]}...")
                print(f"      改写: {result['stage3']['caption_revised'][:50]}...")
                print(f"      耗时: {time.time()-start:.2f}s")
            else:
                print(f"      无需改写（无相机词汇）")
                result['stage3']['caption_revised'] = caption_text
        except Exception as e:
            print(f"      ❌ 失败: {e}")
            result['stage3']['reasons'].append(f'caption_rewrite_error: {e}')
    else:
        print("  [c] 跳过 Caption 改写（未提供 caption 文件）")

    print()

    # 判定是否通过
    print("[5/5] 汇总结果...")
    stage3_pass = (
        result['stage3']['unimatch_flow'] is not None and
        result['stage3']['unimatch_flow'] < 50.0 and
        result['stage3']['dover'] is not None and
        result['stage3']['dover'] > 0.3 and
        len(result['stage3']['reasons']) == 0
    )
    result['stage3_pass'] = stage3_pass

    print(f"  UniMatch 光流: {result['stage3']['unimatch_flow']}")
    print(f"  DOVER 质量: {result['stage3']['dover']}")
    print(f"  Caption 改写: {'是' if result['stage3'].get('caption_revised') else '否'}")
    print(f"  Stage 3 通过: {'✅ 是' if stage3_pass else '❌ 否'}")

    if not stage3_pass:
        print(f"  拒绝原因: {result['stage3']['reasons']}")

    print()

    # 保存结果
    print(f"保存结果到: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 80)
    print("测试完成！")
    print("=" * 80)

    return 0

if __name__ == '__main__':
    sys.exit(main())
