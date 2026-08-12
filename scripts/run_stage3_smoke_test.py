#!/usr/bin/env python3
"""
Stage 3 单样本冒烟测试（CMCC 适配版）

改进点：
1. 支持从样本 ID 自动定位文件（通过 data_loader_cmcc.py）
2. 支持直接传入文件路径（兼容旧版）
3. 详细的性能分析和错误诊断

用法：
  # 方式 1：自动定位（推荐）
  python run_stage3_smoke_test.py \\
    --sample-id "SpatialVID-hq_05b84042-799c-55b1-8a0a-77a2911ecd18" \\
    --data-root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output

  # 方式 2：手动指定路径
  python run_stage3_smoke_test.py \\
    --video-path /path/to/sample.mp4 \\
    --caption-path /path/to/sample.caption.txt
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_environment(gpu_id: int = 0):
    """设置环境变量（必须在导入 PyTorch 前调用）"""
    os.environ['TORCH_HOME'] = '/root/work/david_work/cache/torch'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['PYTHONPATH'] = '/root/work/david_work/models/unimatch:' + os.environ.get('PYTHONPATH', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def load_models(gpu_id: int = 0):
    """加载三大模型"""
    print("[1/6] 加载依赖库...")
    import torch
    import numpy as np
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    print("[2/6] 初始化模型...")
    try:
        from sana_wm_pipeline.qc.stage3_gpu import load_unimatch_fn, load_dover_fn, load_qwen_fn

        print("  [a] 加载 UniMatch...")
        t0 = time.time()
        flow_fn = load_unimatch_fn(
            model_dir='/root/work/david_work/models/unimatch',
            device=f'cuda:{gpu_id}'
        )
        print(f"      ✅ 完成 ({time.time()-t0:.2f}s)")

        print("  [b] 加载 DOVER...")
        t0 = time.time()
        dover_fn = load_dover_fn(
            model_dir='/root/work/david_work/sana_qc_pipeline/DOVER',
            device=f'cuda:{gpu_id}'
        )
        print(f"      ✅ 完成 ({time.time()-t0:.2f}s)")

        print("  [c] 加载 Qwen3.5-9B...")
        t0 = time.time()
        vlm_call = load_qwen_fn(
            model_dir='/root/work/david_work/models/Qwen3.5-9B',
            device=f'cuda:{gpu_id}'
        )
        print(f"      ✅ 完成 ({time.time()-t0:.2f}s)")

        return flow_fn, dover_fn, vlm_call

    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def load_video(video_path: Path):
    """加载视频帧"""
    import numpy as np
    import av

    print(f"[3/6] 读取视频: {video_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    t0 = time.time()
    frames_rgb = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames_rgb.append(frame.to_ndarray(format='rgb24'))

    frames_rgb = np.array(frames_rgb)
    elapsed = time.time() - t0
    print(f"  ✅ 形状: {frames_rgb.shape}, 耗时: {elapsed:.2f}s")
    return frames_rgb


def run_stage3_checks(frames_rgb, caption_text, flow_fn, dover_fn, vlm_call, sample_id):
    """执行 Stage 3 的三项检测"""
    import numpy as np

    result = {
        'sample_id': sample_id,
        'stage3': {
            'unimatch_flow': None,
            'dover': None,
            'vlm_entity_count': None,
            'vlm_quality': None,
            'caption_revised': None,
            'table6_accepted': None,
            'reasons': []
        },
        'timings': {}
    }

    print("[4/6] 执行质量检测...")

    # 1. UniMatch 光流
    print("  [a] UniMatch 光流检测...")
    t0 = time.time()
    try:
        from sana_wm_pipeline.stage04_filter.visual_metrics import unimatch_flow_magnitude
        flow_val = unimatch_flow_magnitude(frames_rgb, flow_fn)
        result['stage3']['unimatch_flow'] = round(float(flow_val), 3)
        elapsed = time.time() - t0
        result['timings']['unimatch'] = round(elapsed, 3)
        print(f"      光流幅度: {flow_val:.3f} 像素 ({elapsed:.3f}s)")
    except Exception as e:
        print(f"      ❌ 失败: {e}")
        result['stage3']['reasons'].append(f'unimatch_error: {str(e)}')

    # 2. DOVER 质量评分
    print("  [b] DOVER 质量评分...")
    t0 = time.time()
    try:
        from sana_wm_pipeline.stage04_filter.visual_metrics import dover_score
        dover_val = dover_score(frames_rgb, dover_fn)
        result['stage3']['dover'] = round(float(dover_val), 4)
        elapsed = time.time() - t0
        result['timings']['dover'] = round(elapsed, 3)
        print(f"      质量分数: {dover_val:.4f} ({elapsed:.3f}s)")
    except Exception as e:
        print(f"      ❌ 失败: {e}")
        result['stage3']['reasons'].append(f'dover_error: {str(e)}')

    # 3. Qwen Caption 改写
    print("  [c] Qwen Caption 分析...")
    if caption_text:
        t0 = time.time()
        try:
            # 检查是否包含相机词汇
            camera_words = ['camera', 'panning', 'zooming', 'tilting', 'tracking', 'dolly',
                           'pan', 'zoom', 'tilt', 'rotate']
            has_camera = any(word in caption_text.lower() for word in camera_words)

            if has_camera:
                print(f"      检测到相机词汇，执行改写...")
                from sana_wm_pipeline.stage04_filter.vlm_entity_quality import ENTITY_QUALITY_PROMPT

                # 使用 8 个关键帧
                keyframe_indices = np.linspace(0, len(frames_rgb)-1, 8).astype(int)
                keyframes = [frames_rgb[i] for i in keyframe_indices]

                prompt = ENTITY_QUALITY_PROMPT + f"\n\nCaption: {caption_text}"
                raw = vlm_call(prompt, keyframes)

                # 解析 JSON
                try:
                    json_str = raw[raw.find("{"):raw.rfind("}")+1]
                    parsed = json.loads(json_str)
                    result['stage3']['caption_revised'] = parsed.get('caption_revised', caption_text)
                    result['stage3']['vlm_entity_count'] = parsed.get('entity_count', 0)
                    result['stage3']['vlm_quality'] = parsed.get('quality_score', 0.0)

                    elapsed = time.time() - t0
                    result['timings']['qwen'] = round(elapsed, 3)
                    print(f"      原始: {caption_text[:60]}...")
                    print(f"      改写: {result['stage3']['caption_revised'][:60]}...")
                    print(f"      耗时: {elapsed:.3f}s")
                except json.JSONDecodeError as e:
                    print(f"      ⚠️ JSON 解析失败，保留原始 caption")
                    result['stage3']['caption_revised'] = caption_text
            else:
                print(f"      无相机词汇，保留原始 caption")
                result['stage3']['caption_revised'] = caption_text
        except Exception as e:
            print(f"      ❌ 失败: {e}")
            result['stage3']['reasons'].append(f'caption_error: {str(e)}')
    else:
        print(f"      跳过（无 caption 文件）")

    return result


def main():
    parser = argparse.ArgumentParser(description='Stage 3 单样本冒烟测试（CMCC 适配版）')

    # 方式 1：自动定位
    parser.add_argument('--sample-id', help='样本 ID（需配合 --data-root）')
    parser.add_argument('--data-root', help='数据根目录')
    parser.add_argument('--completeness-csv', help='sample_completeness.csv 路径（可选）')

    # 方式 2：手动路径
    parser.add_argument('--video-path', help='视频文件路径（直接指定）')
    parser.add_argument('--caption-path', help='Caption 文件路径（可选）')

    # 通用参数
    parser.add_argument('--output', default='/tmp/stage3_smoke_test.json', help='输出文件')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU 设备 ID')

    args = parser.parse_args()

    print("=" * 80)
    print("Stage 3 单样本冒烟测试（CMCC 适配版）")
    print("=" * 80)
    print()

    # 设置环境
    setup_environment(args.gpu_id)

    # 确定文件路径
    video_path = None
    caption_path = None
    sample_id = None

    if args.sample_id and args.data_root:
        # 方式 1：自动定位
        print(f"[0/6] 定位样本文件...")
        print(f"  样本 ID: {args.sample_id}")
        print(f"  数据根目录: {args.data_root}")

        from scripts.data_loader_cmcc import Stage3DataLoaderCMCC

        loader = Stage3DataLoaderCMCC(
            data_root=args.data_root,
            completeness_csv=args.completeness_csv,
            build_index=True
        )

        files = loader.get_sample_files(args.sample_id)
        if not files:
            print(f"  ❌ 错误：样本 {args.sample_id} 未找到")
            return 1

        video_path = files['mp4']
        caption_path = files['caption']
        sample_id = args.sample_id

        print(f"  ✅ 视频: {video_path}")
        print(f"  ✅ Caption: {caption_path}")

        # 验证文件
        valid, missing = loader.verify_sample_files(sample_id)
        if not valid:
            print(f"  ❌ 错误：缺失文件: {missing}")
            return 1

        print()

    elif args.video_path:
        # 方式 2：手动路径
        video_path = Path(args.video_path)
        caption_path = Path(args.caption_path) if args.caption_path else None
        sample_id = video_path.stem

        if not video_path.exists():
            print(f"❌ 错误：视频文件不存在: {video_path}")
            return 1
    else:
        print("❌ 错误：必须指定以下之一：")
        print("  1. --sample-id + --data-root（自动定位）")
        print("  2. --video-path（手动指定）")
        return 1

    # 加载模型
    flow_fn, dover_fn, vlm_call = load_models(args.gpu_id)
    print()

    # 加载视频
    frames_rgb = load_video(video_path)
    print()

    # 加载 caption
    caption_text = None
    if caption_path and caption_path.exists():
        with open(caption_path, 'r') as f:
            caption_text = f.read().strip()

    # 执行检测
    result = run_stage3_checks(frames_rgb, caption_text, flow_fn, dover_fn, vlm_call, sample_id)
    print()

    # 判定是否通过
    print("[5/6] 评估结果...")
    stage3_pass = (
        result['stage3']['unimatch_flow'] is not None and
        result['stage3']['unimatch_flow'] < 50.0 and
        result['stage3']['dover'] is not None and
        result['stage3']['dover'] > 0.3 and
        len(result['stage3']['reasons']) == 0
    )
    result['stage3_pass'] = stage3_pass

    print(f"  UniMatch 光流: {result['stage3']['unimatch_flow']} (阈值: < 50.0)")
    print(f"  DOVER 质量: {result['stage3']['dover']} (阈值: > 0.3)")
    print(f"  Caption 改写: {'是' if result['stage3'].get('caption_revised') else '否'}")
    print(f"  Stage 3 通过: {'✅ 是' if stage3_pass else '❌ 否'}")

    if not stage3_pass:
        print(f"  拒绝原因: {result['stage3']['reasons']}")

    # 性能统计
    if result['timings']:
        print(f"\n  性能统计:")
        total_time = sum(result['timings'].values())
        for key, val in result['timings'].items():
            print(f"    {key}: {val:.3f}s")
        print(f"    总计: {total_time:.3f}s")

    print()

    # 保存结果
    print(f"[6/6] 保存结果到: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 80)
    print(f"冒烟测试完成！{'✅ 通过' if stage3_pass else '❌ 失败'}")
    print("=" * 80)

    return 0 if stage3_pass else 1


if __name__ == '__main__':
    sys.exit(main())
