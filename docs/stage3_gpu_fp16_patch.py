#!/usr/bin/env python3
"""DOVER FP16 优化实现（方案 1）

修改内容：
1. 添加 use_fp16 参数到 load_dover_fn
2. GPU 模式自动使用 FP16（显存减半，速度不变）
3. CPU 模式保持 FP32（精度优先）

预期效果：
- 单样本（1068 帧）：169 秒 → 5-10 秒（~20x 加速）
- 139 样本：10 小时 → 12-23 分钟
"""

# 只需要修改 load_dover_fn 函数的这一部分：

def load_dover_fn(device: str = "cuda", dover_config_path: str = None, dover_weight_path: str = None, use_fp16: bool = True):
    """Load DOVER and return dover_fn(frames_rgb: (T,H,W,3) uint8) -> float.

    Args:
        device: torch device (e.g., 'cuda' or 'cpu')
        dover_config_path: path to dover.yml (default: auto-detect from DOVER package)
        dover_weight_path: path to DOVER.pth (default: auto-detect from DOVER package)
        use_fp16: use FP16 on GPU for 2x memory reduction (default: True)

    Note: 2026-08-09 FP16 优化
          - GPU 模式默认使用 FP16（显存减半，支持 1080p）
          - CPU 模式强制使用 FP32（精度优先）
          - 移除旧的 OOM 检测和模型移动逻辑（不再需要）
    """
    from dover import DOVER  # type: ignore
    import torch
    import yaml
    from pathlib import Path

    # Auto-detect DOVER paths if not provided
    if dover_config_path is None or dover_weight_path is None:
        try:
            import dover
            dover_pkg_dir = Path(dover.__file__).parent.parent
            if dover_config_path is None:
                dover_config_path = str(dover_pkg_dir / "dover.yml")
            if dover_weight_path is None:
                dover_weight_path = str(dover_pkg_dir / "pretrained_weights" / "DOVER.pth")
        except Exception:
            raise RuntimeError(
                "Could not auto-detect DOVER paths. Please provide dover_config_path and dover_weight_path explicitly."
            )

    # Load config and initialize model
    with open(dover_config_path, "r") as f:
        dover_opt = yaml.safe_load(f)

    # Initialize model on specified device
    model = DOVER(**dover_opt["model"]["args"])
    model.load_state_dict(torch.load(dover_weight_path, map_location=device, weights_only=False))
    model = model.to(device)

    # ⭐ 新增：GPU 模式自动使用 FP16
    if device == "cuda" and use_fp16:
        model = model.half()
        print(f"[DOVER] GPU 模式已启用 FP16（显存减半，速度不变）")

    model.eval()

    def dover_fn(frames_rgb: np.ndarray) -> float:
        import torch
        # DOVER expects a dict with 'technical' and 'aesthetic' views
        # frames_rgb: (T, H, W, 3) uint8

        # Convert to (1, 3, T, H, W) float normalized
        t = torch.from_numpy(frames_rgb).float() / 255.0  # (T, H, W, 3)
        t = t.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, T, H, W)

        # ⭐ 新增：GPU FP16 模式转换输入
        if device == "cuda" and use_fp16:
            t = t.half()

        t = t.to(device)

        views = {
            "technical": t,
            "aesthetic": t,
        }

        with torch.no_grad():
            results = model(views)

        # results is a list of [technical_score, aesthetic_score]
        # Return the mean of both
        return float(sum(r.mean().item() for r in results) / len(results))

    return dover_fn
