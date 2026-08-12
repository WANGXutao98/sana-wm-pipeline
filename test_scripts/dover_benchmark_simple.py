#!/usr/bin/env python3
"""DOVER 性能基准测试（简化版）

直接使用 stage3_gpu.py 的加载函数，避免依赖问题
"""
import time
import numpy as np
import torch
import json
from pathlib import Path
import sys

# 添加 DOVER 和 src 到 path
DOVER_DIR = Path(__file__).parent.parent / "models" / "DOVER"
sys.path.insert(0, str(DOVER_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def get_gpu_memory_mb() -> float:
    """获取当前 GPU 显存占用（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def generate_test_video(num_frames: int, height: int, width: int) -> np.ndarray:
    """生成测试视频数据（随机噪声）"""
    print(f"  生成测试视频: {num_frames}帧 × {height}×{width}")
    return np.random.randint(0, 256, (num_frames, height, width, 3), dtype=np.uint8)


def test_dover_chunk(dover_fn, frames_rgb: np.ndarray, test_name: str) -> dict:
    """测试单个 chunk 的性能"""
    T, H, W, _ = frames_rgb.shape
    print(f"\n{'='*60}")
    print(f"测试: {test_name} ({T}帧, {H}×{W})")
    print(f"{'='*60}")

    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    mem_before = get_gpu_memory_mb()

    try:
        # 执行推理
        start = time.time()
        score = dover_fn(frames_rgb)
        elapsed = time.time() - start

        mem_after = get_gpu_memory_mb()
        mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

        print(f"✅ 成功: {elapsed:.3f}s, score={score:.4f}, 峰值显存={mem_peak:.1f}MB")

        return {
            "test_name": test_name,
            "frames": T,
            "resolution": f"{H}×{W}",
            "time_s": round(elapsed, 3),
            "score": round(score, 4),
            "mem_peak_mb": round(mem_peak, 1),
            "success": True,
            "error": None,
        }
    except Exception as e:
        print(f"❌ 失败: {e}")
        return {
            "test_name": test_name,
            "frames": T,
            "resolution": f"{H}×{W}",
            "time_s": None,
            "score": None,
            "mem_peak_mb": None,
            "success": False,
            "error": str(e),
        }


def main():
    print("\n" + "="*60)
    print("DOVER 性能基准测试（简化版）")
    print("="*60)

    model_dir = Path("/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER")

    # 测试用例：模拟 5 秒 chunk（80 帧）的不同分辨率
    test_cases = [
        ("480p_80frames", 80, 480, 640),
        ("720p_80frames", 80, 720, 1280),
        ("1080p_80frames", 80, 1080, 1920),
    ]

    results = []

    # ========== 策略 1：纯 GPU 模式 ==========
    print("\n" + "="*60)
    print("策略 1: 纯 GPU 模式")
    print("="*60)

    try:
        from sana_wm_pipeline.qc.stage3_gpu import load_dover_fn
        print("✅ 加载 DOVER（GPU 模式）...")
        dover_fn_gpu = load_dover_fn(device="cuda", dover_config_path=str(model_dir / "dover.yml"),
                                       dover_weight_path=str(model_dir / "pretrained_weights" / "DOVER.pth"))

        for test_name, T, H, W in test_cases:
            frames = generate_test_video(T, H, W)
            result = test_dover_chunk(dover_fn_gpu, frames, f"GPU_{test_name}")
            result["strategy"] = "GPU"
            results.append(result)

    except Exception as e:
        print(f"❌ 策略 1 失败: {e}")
        import traceback
        traceback.print_exc()

    # ========== 策略 2：纯 CPU 模式 ==========
    print("\n" + "="*60)
    print("策略 2: 纯 CPU 模式")
    print("="*60)

    try:
        from sana_wm_pipeline.qc.stage3_gpu import load_dover_fn
        print("✅ 加载 DOVER（CPU 模式）...")
        dover_fn_cpu = load_dover_fn(device="cpu", dover_config_path=str(model_dir / "dover.yml"),
                                       dover_weight_path=str(model_dir / "pretrained_weights" / "DOVER.pth"))

        for test_name, T, H, W in test_cases:
            frames = generate_test_video(T, H, W)
            result = test_dover_chunk(dover_fn_cpu, frames, f"CPU_{test_name}")
            result["strategy"] = "CPU"
            results.append(result)

    except Exception as e:
        print(f"❌ 策略 2 失败: {e}")
        import traceback
        traceback.print_exc()

    # 保存结果
    output_path = Path(__file__).parent / "dover_benchmark_simple_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"测试完成！结果已保存到: {output_path}")
    print(f"{'='*60}")

    # 打印汇总
    print("\n## 性能对比汇总")
    print(f"{'测试用例':<25} {'策略':<10} {'耗时(s)':<12} {'峰值显存(MB)':<15} {'状态':<10}")
    print("-" * 80)
    for r in results:
        status = "✅" if r["success"] else "❌"
        time_str = f"{r['time_s']:.3f}" if r['time_s'] else "N/A"
        mem_str = f"{r['mem_peak_mb']:.1f}" if r['mem_peak_mb'] else "N/A"
        print(f"{r['test_name']:<25} {r['strategy']:<10} {time_str:<12} {mem_str:<15} {status:<10}")

    # GPU vs CPU 速度对比
    gpu_results = [r for r in results if r["strategy"] == "GPU" and r["success"]]
    cpu_results = [r for r in results if r["strategy"] == "CPU" and r["success"]]

    if gpu_results and cpu_results:
        avg_gpu = sum(r["time_s"] for r in gpu_results) / len(gpu_results)
        avg_cpu = sum(r["time_s"] for r in cpu_results) / len(cpu_results)
        speedup = avg_cpu / avg_gpu if avg_gpu > 0 else 0

        print(f"\n## 加速比分析")
        print(f"GPU 平均耗时: {avg_gpu:.3f}s")
        print(f"CPU 平均耗时: {avg_cpu:.3f}s")
        print(f"GPU 加速比: {speedup:.1f}x")


if __name__ == "__main__":
    main()
