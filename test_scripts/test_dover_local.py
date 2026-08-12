#!/usr/bin/env python3
"""本机 DOVER 验证测试

目标：
1. 验证本机 DOVER 是否正常工作
2. 测试不同分辨率的显存占用
3. 验证 FP16 优化效果
4. 为 CMCC 问题提供对比数据
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER")


def get_gpu_memory_info():
    """获取 GPU 显存信息（MB）"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        free = total - allocated
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "total_mb": total,
            "free_mb": free,
        }
    return None


def print_gpu_memory(prefix=""):
    """打印 GPU 显存状态"""
    mem = get_gpu_memory_info()
    if mem:
        print(f"{prefix}GPU 显存: {mem['allocated_mb']:.0f} MB 已用 / {mem['total_mb']:.0f} MB 总计 "
              f"({mem['allocated_mb']/mem['total_mb']*100:.1f}%), 剩余: {mem['free_mb']:.0f} MB")


def generate_test_video(num_frames: int, height: int, width: int) -> np.ndarray:
    """生成测试视频（随机噪声）"""
    return np.random.randint(0, 256, (num_frames, height, width, 3), dtype=np.uint8)


def test_dover_config(config_name, frames, dover_fn):
    """测试特定配置"""
    T, H, W, _ = frames.shape
    print(f"\n{'='*80}")
    print(f"测试配置: {config_name}")
    print(f"视频规格: {T} 帧, {H}×{W}")
    print(f"{'='*80}")

    print_gpu_memory("开始前 - ")

    # 清理显存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        # 分块处理（模拟实际场景）
        chunk_size = 80  # 5 秒 @ 16fps
        num_chunks = (T + chunk_size - 1) // chunk_size

        print(f"分成 {num_chunks} 个 chunk（每个 {chunk_size} 帧）")

        scores = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, T)
            chunk = frames[start_idx:end_idx]

            chunk_start = time.time()
            score = dover_fn(chunk)
            chunk_time = time.time() - chunk_start

            mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"  Chunk {i+1}/{num_chunks} ({start_idx}-{end_idx}): "
                  f"score={score:.4f}, 耗时={chunk_time*1000:.0f}ms, 峰值显存={mem_peak:.0f}MB")

            scores.append(score)

            # 清理显存
            torch.cuda.empty_cache()

        avg_score = np.mean(scores)
        print(f"\n✅ 成功完成")
        print(f"   平均分数: {avg_score:.4f}")
        print_gpu_memory("   结束后 - ")

        return {
            "success": True,
            "avg_score": avg_score,
            "scores": scores,
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ OOM 错误")
            print(f"   错误: {e}")
            print_gpu_memory("   ")
            return {"success": False, "error": str(e)}
        else:
            raise


def main():
    print("="*80)
    print("本机 DOVER 验证测试")
    print("="*80)

    # GPU 信息
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print_gpu_memory("初始状态 - ")
    else:
        print("\n❌ 未检测到 GPU")
        return

    # 加载 DOVER
    print(f"\n{'='*80}")
    print("加载 DOVER 模型")
    print(f"{'='*80}")

    try:
        from sana_wm_pipeline.qc.stage3_gpu import load_dover_fn

        print("加载 DOVER（FP16 GPU 模式）...")
        t0 = time.time()
        dover_fn = load_dover_fn(device="cuda", use_fp16=True)
        t1 = time.time()
        print(f"✅ 加载完成（{t1-t0:.1f} 秒）")
        print_gpu_memory("")

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 测试用例
    test_cases = [
        ("480p_80帧", generate_test_video(80, 480, 640)),
        ("720p_80帧", generate_test_video(80, 720, 1280)),
        ("1080p_80帧", generate_test_video(80, 1080, 1920)),
        ("720p_160帧", generate_test_video(160, 720, 1280)),
    ]

    results = []
    for name, frames in test_cases:
        result = test_dover_config(name, frames, dover_fn)
        result["config"] = name
        results.append(result)

        # 测试间清理
        torch.cuda.empty_cache()
        time.sleep(1)

    # 汇总结果
    print(f"\n{'='*80}")
    print("测试汇总")
    print(f"{'='*80}")

    print(f"\n{'配置':<20} {'状态':<10} {'平均分数':<15}")
    print("-"*50)
    for r in results:
        status = "✅ 成功" if r["success"] else "❌ OOM"
        score = f"{r.get('avg_score', 0):.4f}" if r["success"] else "N/A"
        print(f"{r['config']:<20} {status:<10} {score:<15}")

    # 对比分析
    print(f"\n{'='*80}")
    print("对比分析")
    print(f"{'='*80}")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n成功: {len(successful)}/{len(results)}")
    print(f"失败: {len(failed)}/{len(results)}")

    if failed:
        print(f"\n❌ 失败的配置:")
        for r in failed:
            print(f"   - {r['config']}")
            print(f"     错误: {r.get('error', 'Unknown')}")

    # 与 CMCC 对比
    print(f"\n{'='*80}")
    print("与 CMCC 机器对比")
    print(f"{'='*80}")

    print(f"\nCMCC 问题：720p × 160 帧 OOM")
    print(f"本机测试：")

    for r in results:
        if "720p_160帧" in r["config"]:
            if r["success"]:
                print(f"   ✅ 本机成功处理 720p × 160 帧")
                print(f"   结论：本机配置正确，CMCC 机器可能有其他问题")
            else:
                print(f"   ❌ 本机也 OOM")
                print(f"   结论：720p × 160 帧在 H100 80GB 上确实会 OOM（FP16 模式）")
                print(f"   建议：必须降采样到 480p")


if __name__ == "__main__":
    main()
