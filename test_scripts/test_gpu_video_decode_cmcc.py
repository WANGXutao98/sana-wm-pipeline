#!/usr/bin/env python3
"""测试 GPU 视频解码性能 - CMCC 环境

对比三种解码方案：
1. PyAV (CPU) - 当前实现
2. TorchVision (GPU NVDEC) - 推荐方案
3. Decord (CPU/GPU) - 备选方案

测试内容：
- 解码速度对比
- 解码结果一致性
- 显存占用
- GPU 利用率
"""
import sys
import time
import io
from pathlib import Path
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def decode_with_pyav_cpu(mp4_bytes: bytes) -> tuple[np.ndarray | None, float]:
    """当前实现：PyAV CPU 解码"""
    import av

    start = time.perf_counter()
    try:
        frames = []
        with av.open(io.BytesIO(mp4_bytes)) as c:
            for pkt in c.demux(video=0):
                for f in pkt.decode():
                    frames.append(f.to_ndarray(format="rgb24"))
        result = np.array(frames, dtype=np.uint8) if frames else None
    except Exception as e:
        print(f"  ❌ PyAV 解码失败: {e}")
        result = None
    elapsed = time.perf_counter() - start
    return result, elapsed


def decode_with_torchvision_gpu(mp4_bytes: bytes, device: str = "cuda") -> tuple[np.ndarray | None, float]:
    """方案 A：TorchVision GPU 解码（推荐）"""
    try:
        import torch
        import torchvision
        from torchvision.io import read_video
    except ImportError as e:
        return None, -1.0

    start = time.perf_counter()
    try:
        # 方法 1：尝试从内存流解码（需要 torchvision >= 0.15）
        try:
            video_tensor, _, _ = torchvision.io.read_video(
                io.BytesIO(mp4_bytes),
                pts_unit='sec',
                output_format='TCHW'
            )
        except (TypeError, AttributeError):
            # 方法 2：Fallback - 写临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                f.write(mp4_bytes)
                temp_path = f.name
            try:
                video_tensor, _, _ = torchvision.io.read_video(
                    temp_path,
                    pts_unit='sec',
                    output_format='TCHW'
                )
            finally:
                import os
                os.unlink(temp_path)

        # 转换为 numpy (T, H, W, C) uint8
        frames_rgb = video_tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        result = frames_rgb
    except Exception as e:
        print(f"  ❌ TorchVision 解码失败: {e}")
        result = None
    elapsed = time.perf_counter() - start
    return result, elapsed


def decode_with_decord(mp4_bytes: bytes, device: str = "cuda") -> tuple[np.ndarray | None, float]:
    """方案备选：Decord 解码"""
    try:
        import decord
        from decord import VideoReader, cpu, gpu
    except ImportError:
        return None, -1.0

    start = time.perf_counter()
    try:
        # Decord 也需要文件路径
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(mp4_bytes)
            temp_path = f.name

        try:
            ctx = gpu(0) if device == "cuda" else cpu(0)
            vr = VideoReader(temp_path, ctx=ctx)
            frames = vr[:].asnumpy()  # (T, H, W, C) uint8
            result = frames
        finally:
            import os
            os.unlink(temp_path)
    except Exception as e:
        print(f"  ❌ Decord 解码失败: {e}")
        result = None
    elapsed = time.perf_counter() - start
    return result, elapsed


def compare_frames(frames1: np.ndarray, frames2: np.ndarray, name1: str, name2: str) -> bool:
    """对比两个解码结果是否一致"""
    if frames1 is None or frames2 is None:
        print(f"  ⚠️ {name1} vs {name2}: 其中一个为 None")
        return False

    if frames1.shape != frames2.shape:
        print(f"  ❌ {name1} vs {name2}: 形状不一致 {frames1.shape} vs {frames2.shape}")
        return False

    # 允许轻微的解码差异（不同解码器可能有亚像素差异）
    diff = np.abs(frames1.astype(np.float32) - frames2.astype(np.float32))
    max_diff = diff.max()
    mean_diff = diff.mean()

    if max_diff > 5:  # 容忍最大 5/255 的差异
        print(f"  ⚠️ {name1} vs {name2}: 差异较大 (max={max_diff:.2f}, mean={mean_diff:.4f})")
        return False

    print(f"  ✅ {name1} vs {name2}: 一致 (max_diff={max_diff:.2f}, mean_diff={mean_diff:.4f})")
    return True


def get_gpu_memory():
    """获取当前 GPU 显存占用"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3  # GB
    except:
        pass
    return 0.0


def main():
    print("=" * 80)
    print("GPU 视频解码性能测试 - CMCC 环境")
    print("=" * 80)

    # 检查环境
    print("\n[1/6] 环境检查")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  ❌ PyTorch 未安装")
        return

    try:
        import torchvision
        print(f"  TorchVision: {torchvision.__version__}")
    except ImportError:
        print("  ⚠️ TorchVision 未安装（方案 A 不可用）")

    try:
        import decord
        print(f"  Decord: {decord.__version__}")
    except ImportError:
        print("  ⚠️ Decord 未安装（备选方案不可用）")

    try:
        import av
        print(f"  PyAV: {av.__version__}")
    except ImportError:
        print("  ❌ PyAV 未安装")
        return

    # 准备测试视频
    print("\n[2/6] 加载测试视频")
    test_video_path = Path("/root/work/david_work/sana_qc_pipeline/DOVER/demo/SpatialVID-hq622345a9-0375-5f10-941e-ffc8765e651a.mp4")

    if not test_video_path.exists():
        # Fallback 到其他测试视频
        test_video_path = Path("/root/work/david_work/sana_qc_pipeline/DOVER/demo/17734.mp4")

    if not test_video_path.exists():
        print(f"  ❌ 测试视频不存在: {test_video_path}")
        return

    mp4_bytes = test_video_path.read_bytes()
    print(f"  视频路径: {test_video_path}")
    print(f"  视频大小: {len(mp4_bytes) / 1024 / 1024:.2f} MB")

    # 测试 PyAV (CPU) - 基线
    print("\n[3/6] 测试 PyAV (CPU) - 当前实现")
    mem_before = get_gpu_memory()
    frames_pyav, time_pyav = decode_with_pyav_cpu(mp4_bytes)
    mem_after = get_gpu_memory()

    if frames_pyav is not None:
        print(f"  ✅ 解码成功")
        print(f"     形状: {frames_pyav.shape}")
        print(f"     耗时: {time_pyav * 1000:.2f} ms")
        print(f"     显存变化: {(mem_after - mem_before) * 1024:.2f} MB")
    else:
        print(f"  ❌ 解码失败")
        return

    # 测试 TorchVision (GPU)
    print("\n[4/6] 测试 TorchVision (GPU) - 推荐方案")
    mem_before = get_gpu_memory()
    frames_torchvision, time_torchvision = decode_with_torchvision_gpu(mp4_bytes)
    mem_after = get_gpu_memory()

    if time_torchvision < 0:
        print(f"  ⚠️ TorchVision 不可用（未安装或版本不支持）")
    elif frames_torchvision is not None:
        print(f"  ✅ 解码成功")
        print(f"     形状: {frames_torchvision.shape}")
        print(f"     耗时: {time_torchvision * 1000:.2f} ms")
        print(f"     加速比: {time_pyav / time_torchvision:.2f}x")
        print(f"     显存变化: {(mem_after - mem_before) * 1024:.2f} MB")
    else:
        print(f"  ❌ 解码失败")

    # 测试 Decord
    print("\n[5/6] 测试 Decord - 备选方案")
    mem_before = get_gpu_memory()
    frames_decord, time_decord = decode_with_decord(mp4_bytes)
    mem_after = get_gpu_memory()

    if time_decord < 0:
        print(f"  ⚠️ Decord 不可用（未安装）")
    elif frames_decord is not None:
        print(f"  ✅ 解码成功")
        print(f"     形状: {frames_decord.shape}")
        print(f"     耗时: {time_decord * 1000:.2f} ms")
        print(f"     加速比: {time_pyav / time_decord:.2f}x")
        print(f"     显存变化: {(mem_after - mem_before) * 1024:.2f} MB")
    else:
        print(f"  ❌ 解码失败")

    # 对比解码结果
    print("\n[6/6] 验证解码结果一致性")
    if frames_torchvision is not None:
        compare_frames(frames_pyav, frames_torchvision, "PyAV", "TorchVision")
    if frames_decord is not None:
        compare_frames(frames_pyav, frames_decord, "PyAV", "Decord")

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    print(f"\n方案对比：")
    print(f"  PyAV (CPU):        {time_pyav * 1000:8.2f} ms  (基线)")
    if time_torchvision > 0:
        speedup = time_pyav / time_torchvision
        print(f"  TorchVision (GPU): {time_torchvision * 1000:8.2f} ms  ({speedup:.1f}x 加速)")
    else:
        print(f"  TorchVision (GPU):     不可用")

    if time_decord > 0:
        speedup = time_pyav / time_decord
        print(f"  Decord (GPU):      {time_decord * 1000:8.2f} ms  ({speedup:.1f}x 加速)")
    else:
        print(f"  Decord (GPU):          不可用")

    print(f"\n推荐方案：")
    if time_torchvision > 0 and frames_torchvision is not None:
        print(f"  ✅ TorchVision (GPU) - 速度快，与 PyTorch 集成好")
    elif time_decord > 0 and frames_decord is not None:
        print(f"  ✅ Decord (GPU) - 备选方案")
    else:
        print(f"  ⚠️ GPU 解码不可用，继续使用 PyAV (CPU)")
        print(f"     建议：安装 torchvision 或 decord 以启用 GPU 解码")

    print("\n预估 Stage 3 性能提升：")
    if time_torchvision > 0:
        speedup = time_pyav / time_torchvision
        current_time = 4.3 * 60  # 4.3 分钟/样本
        new_time = current_time / speedup
        print(f"  当前速度：~{current_time:.0f} 秒/样本")
        print(f"  预期速度：~{new_time:.0f} 秒/样本")
        print(f"  加速比：{speedup:.1f}x")
        print(f"  139 样本处理时间：{139 * new_time / 60:.1f} 分钟（当前 10 小时）")


if __name__ == "__main__":
    main()
