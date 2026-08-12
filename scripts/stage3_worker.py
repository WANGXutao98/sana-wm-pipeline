#!/usr/bin/env python3
"""
Stage 3 Worker 脚本（单 GPU）

由 run_stage3_cmcc_full.py 调度启动，每个 worker 独立处理一部分样本

功能：
1. 加载模型（UniMatch、DOVER、Qwen）
2. 逐个处理分配的样本
3. 写入 JSONL 结果
4. 错误重试和日志记录
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
import logging

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def setup_environment(gpu_id: int):
    """设置环境变量"""
    os.environ['TORCH_HOME'] = '/root/work/david_work/cache/torch'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['PYTHONPATH'] = '/root/work/david_work/models/unimatch:' + os.environ.get('PYTHONPATH', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def load_models(gpu_id: int):
    """加载三大模型"""
    logger.info(f"GPU {gpu_id}: 开始加载模型...")

    try:
        from sana_wm_pipeline.qc.stage3_gpu import load_unimatch_fn, load_dover_fn, load_qwen_fn

        logger.info(f"GPU {gpu_id}: 加载 UniMatch...")
        flow_fn = load_unimatch_fn(
            model_dir='/root/work/david_work/models/unimatch',
            device='cuda:0'  # 因为 CUDA_VISIBLE_DEVICES 已设置，所以总是使用 cuda:0
        )

        logger.info(f"GPU {gpu_id}: 加载 DOVER...")
        dover_fn = load_dover_fn(
            model_dir='/root/work/david_work/sana_qc_pipeline/DOVER',
            device='cuda:0'
        )

        logger.info(f"GPU {gpu_id}: 加载 Qwen3.5-9B...")
        vlm_call = load_qwen_fn(
            model_dir='/root/work/david_work/models/Qwen3.5-9B',
            device='cuda:0'
        )

        logger.info(f"GPU {gpu_id}: ✅ 所有模型加载完成")
        return flow_fn, dover_fn, vlm_call

    except Exception as e:
        logger.error(f"GPU {gpu_id}: ❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def process_sample(
    sample_id: str,
    video_path: Path,
    caption_path: Path,
    flow_fn,
    dover_fn,
    vlm_call,
    config: dict
) -> dict:
    """处理单个样本"""
    import numpy as np
    import av

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
        'stage3_pass': False
    }

    try:
        # 1. 读取视频
        frames_rgb = []
        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                frames_rgb.append(frame.to_ndarray(format='rgb24'))
        frames_rgb = np.array(frames_rgb)

        if len(frames_rgb) == 0:
            result['stage3']['reasons'].append('video_empty')
            return result

        # 2. UniMatch 光流
        try:
            from sana_wm_pipeline.stage04_filter.visual_metrics import unimatch_flow_magnitude
            flow_val = unimatch_flow_magnitude(frames_rgb, flow_fn)
            result['stage3']['unimatch_flow'] = round(float(flow_val), 3)
        except Exception as e:
            result['stage3']['reasons'].append(f'unimatch_error: {str(e)[:100]}')

        # 3. DOVER 质量评分
        try:
            from sana_wm_pipeline.stage04_filter.visual_metrics import dover_score
            dover_val = dover_score(frames_rgb, dover_fn)
            result['stage3']['dover'] = round(float(dover_val), 4)
        except Exception as e:
            result['stage3']['reasons'].append(f'dover_error: {str(e)[:100]}')

        # 4. Qwen Caption 改写
        if caption_path.exists():
            try:
                with open(caption_path, 'r') as f:
                    caption_text = f.read().strip()

                # 检查是否包含相机词汇
                camera_words = ['camera', 'panning', 'zooming', 'tilting', 'tracking', 'dolly',
                               'pan', 'zoom', 'tilt', 'rotate']
                has_camera = any(word in caption_text.lower() for word in camera_words)

                if has_camera:
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
                    except json.JSONDecodeError:
                        result['stage3']['caption_revised'] = caption_text
                else:
                    result['stage3']['caption_revised'] = caption_text

            except Exception as e:
                result['stage3']['reasons'].append(f'caption_error: {str(e)[:100]}')

        # 5. 判定是否通过
        flow_ok = (
            result['stage3']['unimatch_flow'] is not None and
            result['stage3']['unimatch_flow'] < config['flow_threshold']
        )
        dover_ok = (
            result['stage3']['dover'] is not None and
            result['stage3']['dover'] > config['dover_threshold']
        )

        result['stage3_pass'] = flow_ok and dover_ok and len(result['stage3']['reasons']) == 0

    except Exception as e:
        result['stage3']['reasons'].append(f'processing_error: {str(e)[:100]}')
        logger.error(f"样本 {sample_id} 处理失败: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Stage 3 Worker（单 GPU）')
    parser.add_argument('--gpu-id', type=int, required=True, help='GPU ID')
    parser.add_argument('--data-root', required=True, help='数据根目录')
    parser.add_argument('--sample-list', required=True, help='样本 ID 列表（JSON 格式）')
    parser.add_argument('--output', required=True, help='输出 JSONL 文件')
    parser.add_argument('--flow-threshold', type=float, default=50.0, help='光流阈值')
    parser.add_argument('--dover-threshold', type=float, default=0.3, help='DOVER 阈值')
    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format=f'[GPU {args.gpu_id:02d}] %(asctime)s %(message)s',
        datefmt='%H:%M:%S'
    )

    logger.info(f"Worker 启动")
    logger.info(f"输出文件: {args.output}")

    # 设置环境
    setup_environment(args.gpu_id)

    # 解析样本列表
    sample_ids = json.loads(args.sample_list)
    logger.info(f"分配样本数: {len(sample_ids)}")

    # 加载数据索引
    from scripts.data_loader_cmcc import Stage3DataLoaderCMCC
    loader = Stage3DataLoaderCMCC(
        data_root=args.data_root,
        build_index=True
    )

    # 加载模型
    flow_fn, dover_fn, vlm_call = load_models(args.gpu_id)

    # 配置
    config = {
        'flow_threshold': args.flow_threshold,
        'dover_threshold': args.dover_threshold,
    }

    # 处理样本
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 如果输出文件已存在，加载已完成的样本
    completed = set()
    if output_path.exists():
        with open(output_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    completed.add(data['sample_id'])
        logger.info(f"已完成样本数: {len(completed)}")

    # 打开输出文件（追加模式）
    with open(output_path, 'a') as out_f:
        processed = 0
        skipped = 0
        failed = 0

        for i, sample_id in enumerate(sample_ids):
            # 跳过已完成的样本
            if sample_id in completed:
                skipped += 1
                continue

            # 获取文件路径
            files = loader.get_sample_files(sample_id)
            if not files:
                logger.warning(f"[{i+1}/{len(sample_ids)}] 样本 {sample_id} 未找到")
                failed += 1
                continue

            # 处理样本
            try:
                t0 = time.time()
                result = process_sample(
                    sample_id=sample_id,
                    video_path=files['mp4'],
                    caption_path=files['caption'],
                    flow_fn=flow_fn,
                    dover_fn=dover_fn,
                    vlm_call=vlm_call,
                    config=config
                )

                # 写入结果
                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                out_f.flush()  # 立即刷新，确保断点续传可用

                processed += 1
                elapsed = time.time() - t0

                # 每 10 个样本输出一次进度
                if processed % 10 == 0:
                    pass_status = '✅' if result['stage3_pass'] else '❌'
                    logger.info(f"[{i+1}/{len(sample_ids)}] {pass_status} {sample_id[:40]} ({elapsed:.2f}s)")

            except Exception as e:
                logger.error(f"[{i+1}/{len(sample_ids)}] 样本 {sample_id} 处理异常: {e}")
                failed += 1

    # 输出统计
    logger.info("=" * 60)
    logger.info(f"处理完成！")
    logger.info(f"  新处理: {processed}")
    logger.info(f"  已跳过: {skipped}")
    logger.info(f"  失败: {failed}")
    logger.info(f"  总计: {len(sample_ids)}")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
