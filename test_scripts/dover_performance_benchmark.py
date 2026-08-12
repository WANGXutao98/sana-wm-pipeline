#!/usr/bin/env python3
"""DOVER 性能基准测试：对比不同策略的速度和显存占用

测试方案：
  A. 纯 GPU 模式（当前被放弃的方案）
  B. 纯 CPU 模式（当前的优化方案）
  C. 混合精度 GPU（float16）
  D. 智能分块（根据分辨率动态调整 chunk 大小）

目标：用数据说话，找到速度快且显存安全的最优方案
"""
import time
import numpy as np
import torch
import json
from pathlib import Path
from typing import Callable
import sys

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 添加 DOVER 到 path
DOVER_DIR = Path(__file__).parent.parent / "models" / "DOVER"
sys.path.insert(0, str(DOVER_DIR))

from sana_wm_pipeline.qc.stage3_gpu import load_dover_fn


def get_gpu_memory_mb() -> float:
    """获取当前 GPU 显存占用（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def generate_test_video(num_frames: int, height: int, width: int) -> np.ndarray:
    """生成测试视频数据（随机噪声）"""
    return np.random.randint(0, 256, (num_frames, height, width, 3), dtype=np.uint8)


def benchmark_dover_strategy(
    strategy_name: str,
    dover_fn: Callable,
    test_videos: list[tuple[str, np.ndarray]],
) -> dict:
    """对一个策略进行基准测试"""
    print(f"\n{'='*60}")
    print(f"测试策略: {strategy_name}")
    print(f"{'='*60}")

    results = []

    for video_name, frames in test_videos:
        T, H, W, _ = frames.shape
        print(f"\n测试视频: {video_name} ({T}帧, {H}x{W})")

        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        mem_before = get_gpu_memory_mb()

        # 执行推理
        try:
            start = time.time()
            score = dover_fn(frames)
            elapsed = time.time() - start

            mem_after = get_gpu_memory_mb()
            mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

            result = {
                "video": video_name,
                "frames": T,
                "resolution": f"{H}x{W}",
                "time_s": round(elapsed, 3),
                "score": round(score, 4),
                "mem_before_mb": round(mem_before, 1),
                "mem_after_mb": round(mem_after, 1),
                "mem_peak_mb": round(mem_peak, 1),
                "success": True,
                "error": None,
            }
            print(f"  ✅ 成功: {elapsed:.3f}s, score={score:.4f}, 峰值显存={mem_peak:.1f}MB")

        except Exception as e:
            result = {
                "video": video_name,
                "frames": T,
                "resolution": f"{H}x{W}",
                "time_s": None,
                "score": None,
                "mem_before_mb": round(mem_before, 1),
                "mem_after_mb": None,
                "mem_peak_mb": None,
                "success": False,
                "error": str(e),
            }
            print(f"  ❌ 失败: {e}")

        results.append(result)

    return {
        "strategy": strategy_name,
        "results": results,
        "summary": {
            "total_tests": len(results),
            "successes": sum(1 for r in results if r["success"]),
            "failures": sum(1 for r in results if not r["success"]),
            "avg_time_s": round(np.mean([r["time_s"] for r in results if r["time_s"]]), 3) if any(r["time_s"] for r in results) else None,
            "total_time_s": round(sum([r["time_s"] for r in results if r["time_s"]]), 3) if any(r["time_s"] for r in results) else None,
        }
    }


def load_dover_pure_gpu(model_dir: str) -> Callable:
    """策略 A：纯 GPU 模式（不做任何显存保护）"""
    from dover import DOVER
    import yaml

    config_path = Path(model_dir) / "dover.yml"
    weight_path = Path(model_dir) / "pretrained_weights" / "DOVER.pth"

    with open(config_path, "r") as f:
        dover_opt = yaml.safe_load(f)

    model = DOVER(**dover_opt["model"]["args"])
    model.load_state_dict(torch.load(weight_path, map_location="cuda", weights_only=False))
    model = model.to("cuda")
    model.eval()

    def dover_fn(frames_rgb: np.ndarray) -> float:
        t = torch.from_numpy(frames_rgb).float() / 255.0
        t = t.permute(3, 0, 1, 2).unsqueeze(0).to("cuda")  # 强制 GPU
        views = {"technical": t, "aesthetic": t}
        with torch.no_grad():
            results = model(views)
        return float(sum(r.mean().item() for r in results) / len(results))

    return dover_fn


def load_dover_pure_cpu(model_dir: str) -> Callable:
    """策略 B：纯 CPU 模式"""
    from dover import DOVER
    import yaml

    config_path = Path(model_dir) / "dover.yml"
    weight_path = Path(model_dir) / "pretrained_weights" / "DOVER.pth"

    with open(config_path, "r") as f:
        dover_opt = yaml.safe_load(f)

    model = DOVER(**dover_opt["model"]["args"])
    model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=False))
    model = model.to("cpu")
    model.eval()

    def dover_fn(frames_rgb: np.ndarray) -> float:
        t = torch.from_numpy(frames_rgb).float() / 255.0
        t = t.permute(3, 0, 1, 2).unsqueeze(0).to("cpu")  # 强制 CPU
        views = {"technical": t, "aesthetic": t}
        with torch.no_grad():
            results = model(views)
        return float(sum(r.mean().item() for r in results) / len(results))

    return dover_fn


def load_dover_fp16_gpu(model_dir: str) -> Callable:
    """策略 C：混合精度 GPU（float16）"""
    from dover import DOVER
    import yaml

    config_path = Path(model_dir) / "dover.yml"
    weight_path = Path(model_dir) / "pretrained_weights" / "DOVER.pth"

    with open(config_path, "r") as f:
        dover_opt = yaml.safe_load(f)

    model = DOVER(**dover_opt["model"]["args"])
    model.load_state_dict(torch.load(weight_path, map_location="cuda", weights_only=False))
    model = model.to("cuda").half()  # 转换为 float16
    model.eval()

    def dover_fn(frames_rgb: np.ndarray) -> float:
        t = torch.from_numpy(frames_rgb).float() / 255.0
        t = t.permute(3, 0, 1, 2).unsqueeze(0).to("cuda").half()  # float16
        views = {"technical": t, "aesthetic": t}
        with torch.no_grad():
            results = model(views)
        return float(sum(r.mean().item() for r in results) / len(results))

    return dover_fn


def load_dover_smart_chunk(model_dir: str) -> Callable:
    """策略 D：智能分块（根据分辨率动态选择设备）"""
    from dover import DOVER
    import yaml

    config_path = Path(model_dir) / "dover.yml"
    weight_path = Path(model_dir) / "pretrained_weights" / "DOVER.pth"

    with open(config_path, "r") as f:
        dover_opt = yaml.safe_load(f)

    model = DOVER(**dover_opt["model"]["args"])
    model.load_state_dict(torch.load(weight_path, map_location="cuda", weights_only=False))
    model = model.to("cuda")
    model.eval()

    def dover_fn(frames_rgb: np.ndarray) -> float:
        T, H, W, C = frames_rgb.shape

        # 动态判断：每个 chunk 独立决定使用 GPU 还是 CPU
        # 阈值：80 帧 × 720p = 80 × 1280 × 720 × 3 × 4 bytes ≈ 850MB
        estimated_vram_mb = (T * H * W * 3 * 4) / (1024 ** 2)

        # 如果估算显存 < 1GB，使用 GPU；否则使用 CPU
        if estimated_vram_mb < 1000:
            device = "cuda"
        else:
            device = "cpu"
            # 临时移动模型到 CPU
            model.cpu()

        t = torch.from_numpy(frames_rgb).float() / 255.0
        t = t.permute(3, 0, 1, 2).unsqueeze(0).to(device)
        views = {"technical": t, "aesthetic": t}
        with torch.no_grad():
            results = model(views)

        # 如果用了 CPU，移回 GPU
        if device == "cpu":
            model.cuda()

        return float(sum(r.mean().item() for r in results) / len(results))

    return dover_fn


def main():
    model_dir = "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER"

    # 定义测试视频（模拟真实场景）
    test_videos = [
        ("short_480p", generate_test_video(80, 480, 640)),      # 5秒, 480p
        ("short_720p", generate_test_video(80, 720, 1280)),     # 5秒, 720p
        ("short_1080p", generate_test_video(80, 1080, 1920)),   # 5秒, 1080p
        ("long_480p", generate_test_video(240, 480, 640)),      # 15秒, 480p
        ("long_720p", generate_test_video(240, 720, 1280)),     # 15秒, 720p
    ]

    print("\n" + "="*60)
    print("DOVER 性能基准测试")
    print("="*60)
    print(f"模型路径: {model_dir}")
    print(f"测试视频数量: {len(test_videos)}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    all_results = []

    # 策略 A: 纯 GPU
    try:
        print("\n加载策略 A: 纯 GPU 模式...")
        dover_fn_gpu = load_dover_pure_gpu(model_dir)
        result_gpu = benchmark_dover_strategy("A_Pure_GPU", dover_fn_gpu, test_videos)
        all_results.append(result_gpu)
    except Exception as e:
        print(f"❌ 策略 A 加载失败: {e}")

    # 策略 B: 纯 CPU
    try:
        print("\n加载策略 B: 纯 CPU 模式...")
        dover_fn_cpu = load_dover_pure_cpu(model_dir)
        result_cpu = benchmark_dover_strategy("B_Pure_CPU", dover_fn_cpu, test_videos)
        all_results.append(result_cpu)
    except Exception as e:
        print(f"❌ 策略 B 加载失败: {e}")

    # 策略 C: 混合精度 GPU
    try:
        print("\n加载策略 C: FP16 GPU 模式...")
        dover_fn_fp16 = load_dover_fp16_gpu(model_dir)
        result_fp16 = benchmark_dover_strategy("C_FP16_GPU", dover_fn_fp16, test_videos)
        all_results.append(result_fp16)
    except Exception as e:
        print(f"❌ 策略 C 加载失败: {e}")

    # 策略 D: 智能分块
    try:
        print("\n加载策略 D: 智能分块模式...")
        dover_fn_smart = load_dover_smart_chunk(model_dir)
        result_smart = benchmark_dover_strategy("D_Smart_Chunk", dover_fn_smart, test_videos)
        all_results.append(result_smart)
    except Exception as e:
        print(f"❌ 策略 D 加载失败: {e}")

    # 保存结果
    output_path = Path(__file__).parent / "dover_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"测试完成！结果已保存到: {output_path}")
    print(f"{'='*60}")

    # 打印汇总
    print("\n## 汇总对比")
    print(f"{'策略':<20} {'成功/总数':<12} {'平均耗时(s)':<15} {'总耗时(s)':<12}")
    print("-" * 60)
    for result in all_results:
        summary = result["summary"]
        avg_time = summary["avg_time_s"] if summary["avg_time_s"] else "N/A"
        total_time = summary["total_time_s"] if summary["total_time_s"] else "N/A"
        print(f"{result['strategy']:<20} {summary['successes']}/{summary['total_tests']:<10} {str(avg_time):<15} {str(total_time):<12}")


if __name__ == "__main__":
    main()
