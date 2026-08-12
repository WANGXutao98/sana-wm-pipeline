#!/usr/bin/env python3
"""
Stage 3 完整执行脚本（CMCC 48 GPU 并行）

功能：
1. 加载 sample_completeness.csv，获取完整样本列表
2. 构建样本文件索引（预扫描）
3. 将样本均分到 48 个 GPU
4. 启动 48 个独立 worker 进程
5. 支持断点续传（跳过已处理样本）
6. 实时进度监控

用法：
  python run_stage3_cmcc_full.py \\
    --data-root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \\
    --completeness-csv /path/to/sample_completeness.csv \\
    --output-dir /root/work/david_work/qc_output/stage3 \\
    --num-gpus 48

测试模式（100 样本，2 GPU）：
  python run_stage3_cmcc_full.py \\
    --data-root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \\
    --completeness-csv sample_completeness.csv \\
    --output-dir /tmp/stage3_test \\
    --num-gpus 2 \\
    --max-samples 100
"""

import sys
import os
import json
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict
import logging

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Stage 3 完整执行（48 GPU 并行）')

    # 数据参数
    parser.add_argument('--data-root', required=True, help='数据根目录')
    parser.add_argument('--completeness-csv', required=True, help='sample_completeness.csv 路径')

    # 输出参数
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--log-dir', help='日志目录（默认 <output-dir>/logs）')

    # 执行参数
    parser.add_argument('--num-gpus', type=int, default=48, help='GPU 数量（默认 48）')
    parser.add_argument('--max-samples', type=int, help='最大处理样本数（测试用，默认处理全部）')
    parser.add_argument('--resume', action='store_true', help='断点续传（跳过已完成样本）')

    # 质量阈值（可选）
    parser.add_argument('--flow-threshold', type=float, default=50.0, help='UniMatch 光流阈值')
    parser.add_argument('--dover-threshold', type=float, default=0.3, help='DOVER 质量阈值')

    return parser.parse_args()


def setup_output_dirs(output_dir: Path, log_dir: Path = None):
    """创建输出目录"""
    output_dir.mkdir(parents=True, exist_ok=True)

    if log_dir is None:
        log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, log_dir


def load_samples(data_root: Path, completeness_csv: Path, max_samples: int = None) -> List[str]:
    """加载可处理的样本列表"""
    from scripts.data_loader_cmcc import Stage3DataLoaderCMCC

    logger.info(f"加载数据索引: {data_root}")
    loader = Stage3DataLoaderCMCC(
        data_root=data_root,
        completeness_csv=completeness_csv,
        build_index=True
    )

    samples = loader.get_processable_samples()
    logger.info(f"可处理样本数: {len(samples)}")

    if max_samples:
        samples = samples[:max_samples]
        logger.info(f"限制样本数: {max_samples}")

    return samples


def load_completed_samples(output_dir: Path) -> set:
    """加载已完成的样本 ID（用于断点续传）"""
    completed = set()

    for jsonl_file in output_dir.glob('worker_*.jsonl'):
        try:
            with open(jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        completed.add(data['sample_id'])
        except Exception as e:
            logger.warning(f"读取 {jsonl_file} 失败: {e}")

    return completed


def distribute_workload(samples: List[str], num_gpus: int) -> List[Dict]:
    """将样本均匀分配到多个 GPU"""
    samples_per_gpu = len(samples) // num_gpus
    remainder = len(samples) % num_gpus

    workload = []
    start_idx = 0

    for gpu_id in range(num_gpus):
        # 前 remainder 个 GPU 多分配 1 个样本
        count = samples_per_gpu + (1 if gpu_id < remainder else 0)
        end_idx = start_idx + count

        workload.append({
            'gpu_id': gpu_id,
            'samples': samples[start_idx:end_idx],
            'count': count
        })

        start_idx = end_idx

    return workload


def launch_worker(
    gpu_id: int,
    sample_list: List[str],
    data_root: Path,
    output_file: Path,
    log_file: Path,
    config: Dict
) -> subprocess.Popen:
    """启动一个 worker 进程"""

    # 构建命令
    cmd = [
        'python', '-u',
        str(Path(__file__).parent / 'stage3_worker.py'),
        '--gpu-id', str(gpu_id),
        '--data-root', str(data_root),
        '--sample-list', json.dumps(sample_list),
        '--output', str(output_file),
        '--flow-threshold', str(config['flow_threshold']),
        '--dover-threshold', str(config['dover_threshold']),
    ]

    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 启动进程
    with open(log_file, 'w') as log_f:
        process = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env
        )

    logger.info(f"启动 worker {gpu_id:02d}: PID={process.pid}, 样本数={len(sample_list)}")
    return process


def monitor_progress(output_dir: Path, total_samples: int, check_interval: int = 30):
    """实时监控进度"""
    print("\n" + "=" * 80)
    print("实时进度监控（按 Ctrl+C 退出监控，worker 会继续运行）")
    print("=" * 80)

    try:
        while True:
            # 统计已完成样本数
            completed = len(load_completed_samples(output_dir))
            progress = completed / total_samples * 100 if total_samples > 0 else 0

            # 显示进度
            print(f"\r[{time.strftime('%H:%M:%S')}] 进度: {completed}/{total_samples} ({progress:.2f}%)", end='', flush=True)

            # 检查是否完成
            if completed >= total_samples:
                print("\n✅ 所有样本处理完成！")
                break

            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n\n⚠️ 监控已停止，但 worker 进程仍在后台运行")
        print(f"可以随时重新运行此脚本查看进度")


def main():
    args = parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print("=" * 80)
    print("Stage 3 完整执行脚本（CMCC 48 GPU 并行）")
    print("=" * 80)
    print(f"数据根目录: {args.data_root}")
    print(f"完整性 CSV: {args.completeness_csv}")
    print(f"输出目录: {args.output_dir}")
    print(f"GPU 数量: {args.num_gpus}")
    if args.max_samples:
        print(f"限制样本数: {args.max_samples}")
    print()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir) if args.log_dir else None
    output_dir, log_dir = setup_output_dirs(output_dir, log_dir)

    # 加载样本列表
    logger.info("步骤 1/5: 加载样本列表")
    samples = load_samples(
        Path(args.data_root),
        Path(args.completeness_csv),
        args.max_samples
    )

    if not samples:
        logger.error("没有可处理的样本，退出")
        return 1

    # 断点续传
    if args.resume:
        logger.info("步骤 2/5: 检查已完成样本（断点续传）")
        completed = load_completed_samples(output_dir)
        logger.info(f"已完成样本数: {len(completed)}")

        samples = [s for s in samples if s not in completed]
        logger.info(f"剩余待处理样本数: {len(samples)}")

        if not samples:
            logger.info("所有样本已处理完成，无需继续")
            return 0
    else:
        logger.info("步骤 2/5: 跳过断点续传检查")

    # 分配工作负载
    logger.info("步骤 3/5: 分配工作负载")
    workload = distribute_workload(samples, args.num_gpus)

    for job in workload:
        logger.info(f"GPU {job['gpu_id']:02d}: {job['count']} 个样本")

    # 检查 worker 脚本是否存在
    worker_script = Path(__file__).parent / 'stage3_worker.py'
    if not worker_script.exists():
        logger.error(f"Worker 脚本不存在: {worker_script}")
        logger.error("请先创建 stage3_worker.py 脚本")
        return 1

    # 启动 workers
    logger.info("步骤 4/5: 启动 workers")
    config = {
        'flow_threshold': args.flow_threshold,
        'dover_threshold': args.dover_threshold,
    }

    processes = []
    for job in workload:
        if job['count'] == 0:
            continue

        output_file = output_dir / f"worker_{job['gpu_id']:02d}.jsonl"
        log_file = log_dir / f"worker_{job['gpu_id']:02d}.log"

        process = launch_worker(
            gpu_id=job['gpu_id'],
            sample_list=job['samples'],
            data_root=Path(args.data_root),
            output_file=output_file,
            log_file=log_file,
            config=config
        )
        processes.append((job['gpu_id'], process))

    logger.info(f"已启动 {len(processes)} 个 worker 进程")
    print()

    # 等待一小段时间，确保所有 worker 都启动
    time.sleep(5)

    # 检查是否有 worker 立即失败
    failed = []
    for gpu_id, proc in processes:
        if proc.poll() is not None:
            failed.append(gpu_id)

    if failed:
        logger.error(f"以下 worker 启动失败: {failed}")
        logger.error("请检查日志文件排查问题")
        return 1

    # 监控进度
    logger.info("步骤 5/5: 监控执行进度")
    monitor_progress(output_dir, len(samples))

    # 等待所有 worker 完成
    logger.info("等待所有 worker 完成...")
    for gpu_id, proc in processes:
        proc.wait()
        if proc.returncode != 0:
            logger.warning(f"Worker {gpu_id:02d} 退出码非零: {proc.returncode}")

    print()
    logger.info("=" * 80)
    logger.info("执行完成！")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"日志目录: {log_dir}")
    logger.info("=" * 80)

    # 显示统计
    completed = len(load_completed_samples(output_dir))
    logger.info(f"已完成样本数: {completed}/{len(samples)}")

    if completed < len(samples):
        logger.warning(f"有 {len(samples) - completed} 个样本未完成，请检查日志")
        logger.warning("可以使用 --resume 参数重新运行以继续处理")

    return 0


if __name__ == '__main__':
    sys.exit(main())
