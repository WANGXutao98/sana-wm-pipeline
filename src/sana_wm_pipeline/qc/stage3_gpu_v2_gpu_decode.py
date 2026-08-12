# src/sana_wm_pipeline/qc/stage3_gpu.py
"""Stage 3: GPU-accelerated visual quality evaluation.

All heavy models (UniMatch, DOVER, Qwen) are injected as callables so the
module is importable without GPU. Use load_*_fn() helpers in the CMCC launcher.

2026-08-09 更新：添加 GPU 视频解码支持（TorchVision NVDEC）以解决性能瓶颈
"""
from __future__ import annotations
import io, json, tarfile, logging
from pathlib import Path
from typing import Any, Callable
import numpy as np

from sana_wm_pipeline.stage04_filter.visual_metrics import (
    unimatch_flow_magnitude, dover_score, mean_saturation,
)
from sana_wm_pipeline.stage04_filter.vlm_entity_quality import (
    ENTITY_QUALITY_PROMPT,
)
from sana_wm_pipeline.stage04_filter.apply_table6 import evaluate
from sana_wm_pipeline.qc.group_config import get_group_config

logger = logging.getLogger(__name__)

_CAPTION_REWRITE_SUFFIX = (
    "\n\nAdditionally, the caption below contains camera motion words "
    "(e.g., 'pans left', 'zooms in'). Rewrite it as a static scene description "
    "with no camera motion words. Output the rewritten caption in a JSON field "
    "\"caption_revised\" alongside the other fields."
)


def _decode_frames_gpu(mp4_bytes: bytes) -> np.ndarray | None:
    """使用 TorchVision + NVDEC 在 GPU 上解码视频（推荐）

    Args:
        mp4_bytes: MP4 视频字节流

    Returns:
        frames_rgb: (T, H, W, 3) uint8 numpy array，失败返回 None

    Note:
        - 需要 torchvision >= 0.15 支持从内存流解码
        - 自动 fallback 到临时文件方式（兼容旧版本）
        - 失败后自动 fallback 到 _decode_frames_cpu()
    """
    if not mp4_bytes:
        return None

    try:
        import torch
        import torchvision
    except ImportError:
        logger.debug("TorchVision 不可用，fallback 到 CPU 解码")
        return None

    try:
        # 方法 1：尝试从内存流解码（torchvision >= 0.15）
        try:
            video_tensor, _, _ = torchvision.io.read_video(
                io.BytesIO(mp4_bytes),
                pts_unit='sec',
                output_format='TCHW'
            )
        except (TypeError, AttributeError):
            # 方法 2：Fallback - 使用临时文件
            import tempfile
            import os
            temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            try:
                os.write(temp_fd, mp4_bytes)
                os.close(temp_fd)
                video_tensor, _, _ = torchvision.io.read_video(
                    temp_path,
                    pts_unit='sec',
                    output_format='TCHW'
                )
            finally:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

        # 转换为 numpy (T, H, W, C) uint8
        frames_rgb = video_tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return frames_rgb

    except Exception as e:
        logger.debug(f"GPU 解码失败: {e}，fallback 到 CPU")
        return None


def _decode_frames_cpu(mp4_bytes: bytes) -> np.ndarray | None:
    """使用 PyAV 在 CPU 上解码视频（原实现，作为 fallback）"""
    if not mp4_bytes:
        return None
    try:
        import av
        frames = []
        with av.open(io.BytesIO(mp4_bytes)) as c:
            for pkt in c.demux(video=0):
                for f in pkt.decode():
                    frames.append(f.to_ndarray(format="rgb24"))
        return np.array(frames, dtype=np.uint8) if frames else None
    except Exception:
        return None


def _decode_frames(mp4_bytes: bytes, prefer_gpu: bool = True) -> np.ndarray | None:
    """解码视频帧（自动选择 GPU 或 CPU）

    Args:
        mp4_bytes: MP4 视频字节流
        prefer_gpu: 优先使用 GPU 解码（默认 True）

    Returns:
        frames_rgb: (T, H, W, 3) uint8 numpy array，失败返回 None

    Note:
        - GPU 解码快 ~50 倍（NVDEC 硬件加速）
        - GPU 失败自动 fallback 到 CPU
        - 设置 prefer_gpu=False 强制使用 CPU（调试用）
    """
    if not mp4_bytes:
        return None

    # 尝试 GPU 解码
    if prefer_gpu:
        frames_gpu = _decode_frames_gpu(mp4_bytes)
        if frames_gpu is not None:
            return frames_gpu
        # GPU 失败，fallback 到 CPU
        logger.debug("GPU 解码失败，使用 CPU fallback")

    # CPU 解码
    return _decode_frames_cpu(mp4_bytes)


def process_sample_stage3(
    sample_id: str,
    tar_path: Path,
    group_name: str,
    flow_fn: Callable,
    dover_fn: Callable,
    vlm_call: Callable,
    table6_cfg: dict,
    has_camera_words: bool = False,
    skip_vlm: bool = False,
    prefer_gpu_decode: bool = True,  # 新增参数
) -> dict[str, Any]:
    """Run Stage 3 GPU checks on one sample. Returns merged result dict."""
    tar_path = Path(tar_path)
    cfg = get_group_config(group_name)
    stage3: dict[str, Any] = {
        "unimatch_flow": None, "dover": None,
        "vlm_entity_count": None, "vlm_quality": None,
        "table6_accepted": None, "caption_revised": None,
        "reasons": [],
    }

    try:
        with tarfile.open(tar_path, "r") as tf:
            mp4_bytes = tf.extractfile(tf.getmember(f"{sample_id}.mp4")).read()
            cap_bytes = tf.extractfile(tf.getmember(f"{sample_id}.caption.txt")).read()
    except Exception as e:
        stage3["reasons"].append(f"tar_read_error: {e}")
        return {"sample_id": sample_id, "stage3": stage3}

    caption_text = cap_bytes.decode("utf-8", errors="replace").strip()

    frames_rgb = _decode_frames(mp4_bytes, prefer_gpu=prefer_gpu_decode)
    if frames_rgb is None:
        stage3["reasons"].append("video_decode_failed")
        return {"sample_id": sample_id, "stage3": stage3}

    # UniMatch flow
    try:
        flow_val = unimatch_flow_magnitude(frames_rgb, flow_fn)
        stage3["unimatch_flow"] = round(float(flow_val), 3) if not np.isnan(flow_val) else None
    except Exception as e:
        stage3["reasons"].append(f"unimatch_error: {e}")

    # DOVER quality
    try:
        dover_val = dover_score(frames_rgb, dover_fn)
        stage3["dover"] = round(float(dover_val), 4) if not np.isnan(dover_val) else None
    except Exception as e:
        stage3["reasons"].append(f"dover_error: {e}")

    # Check if VLM is needed for this source
    need_vlm = False
    if cfg.table6_source is not None:
        source_cfg = table6_cfg.get("per_source", {}).get(cfg.table6_source, {})
        need_vlm = (source_cfg.get("vlm_entity") is not None or
                    source_cfg.get("vlm_quality") is not None or
                    has_camera_words)

    # Qwen VLM (entity + quality + optional caption rewrite)
    # Skip if not needed for this source to save time
    if need_vlm and not skip_vlm:
        try:
            prompt = ENTITY_QUALITY_PROMPT
            if has_camera_words:
                prompt = prompt + _CAPTION_REWRITE_SUFFIX + f"\n\nCaption: {caption_text}"
            keyframes = [frames_rgb[i] for i in np.linspace(0, len(frames_rgb) - 1, 8).astype(int)]
            raw = vlm_call(prompt, keyframes)
            parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            entity_count = (
                int(parsed.get("people", 0))
                + int(parsed.get("vehicles", 0))
                + int(parsed.get("animals", 0))
            )
            stage3["vlm_entity_count"] = entity_count
            stage3["vlm_quality"] = float(parsed.get("quality", -1.0))
            if has_camera_words and "caption_revised" in parsed:
                stage3["caption_revised"] = str(parsed["caption_revised"])
        except Exception as e:
            stage3["reasons"].append(f"vlm_error: {e}")
    else:
        stage3["reasons"].append("vlm_skipped: not needed for this source")

    # Table 6 evaluation
    if cfg.table6_source is not None:
        scores = {
            "unimatch_flow": stage3.get("unimatch_flow"),
            "dover": stage3.get("dover"),
            "vlm_entity_count": stage3.get("vlm_entity_count"),
            "vlm_quality": stage3.get("vlm_quality"),
            "color_saturation": round(mean_saturation(frames_rgb), 2),
        }
        try:
            t6_result = evaluate(cfg.table6_source, scores, table6_cfg)
            stage3["table6_accepted"] = t6_result["accepted"]
            if not t6_result["accepted"]:
                stage3["reasons"].extend(t6_result["reasons"])
        except KeyError:
            stage3["reasons"].append(f"table6_unknown_source: {cfg.table6_source}")

    return {"sample_id": sample_id, "caption_original": caption_text, "stage3": stage3}


def run_stage3(
    stage1_jsonl: Path,
    output_jsonl: Path,
    caption_overrides_jsonl: Path,
    flow_fn: Callable,
    dover_fn: Callable,
    vlm_call: Callable,
    table6_cfg: dict,
    prefer_gpu_decode: bool = True,  # 新增参数
) -> int:
    """Run Stage 3 on all non-failed samples from Stage 1. Single-process (GPU caller)."""
    stage1_jsonl = Path(stage1_jsonl)
    output_jsonl = Path(output_jsonl)
    caption_overrides_jsonl = Path(caption_overrides_jsonl)

    if not stage1_jsonl.exists():
        raise FileNotFoundError(f"stage1_jsonl not found: {stage1_jsonl}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(stage1_jsonl, encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout, \
         open(caption_overrides_jsonl, "w", encoding="utf-8") as cap_fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("verdict") == "fail":
                continue
            sid = rec["sample_id"]
            has_camera_words = bool(rec.get("metrics", {}).get("camera_words"))
            s3_rec = process_sample_stage3(
                sid, rec["tar_path"], rec.get("group", ""),
                flow_fn=flow_fn, dover_fn=dover_fn, vlm_call=vlm_call,
                table6_cfg=table6_cfg, has_camera_words=has_camera_words,
                prefer_gpu_decode=prefer_gpu_decode,
            )
            merged = dict(rec)
            merged["stage3"] = s3_rec["stage3"]
            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")

            # Write caption override sidecar if rewritten
            cap_revised = s3_rec["stage3"].get("caption_revised")
            if cap_revised:
                cap_orig = s3_rec.get("caption_original", "")
                cap_fout.write(json.dumps({
                    "sample_id": sid,
                    "caption_original": cap_orig,
                    "caption_revised": cap_revised,
                }, ensure_ascii=False) + "\n")

            total += 1
            if total % 1000 == 0:
                print(f"[stage3] {total} samples processed", flush=True)
    return total


# ── Model loader helpers (called by CMCC launcher, not imported in tests) ─────

def load_unimatch_fn(model_dir: str, device: str = "cuda"):
    """Load UniMatch and return flow_fn(img_a, img_b) -> (H,W,2) float32."""
    import sys
    import torch

    # Add parent directory to sys.path so unimatch package can be imported
    model_path = Path(model_dir)
    if str(model_path) not in sys.path:
        sys.path.insert(0, str(model_path))

    # Now import using standard import (unimatch package must be in sys.path)
    # This works even with empty __init__.py because we're importing the module directly
    from unimatch import unimatch as unimatch_module
    UniMatch = unimatch_module.UniMatch

    model = UniMatch(
        feature_channels=128, num_scales=2, upsample_factor=4,
        num_head=1, ffn_dim_expansion=4, num_transformer_layers=6,
        reg_refine=True, task="flow",
    ).to(device).eval()
    # Check for checkpoint in pretrained/ subdirectory first, then root
    ckpt_paths = [
        Path(model_dir) / "pretrained" / "gmflow-scale2-regrefine6-mixdata.pth",
        Path(model_dir) / "gmflow-scale2-regrefine6-mixdata.pth",
    ]
    ckpt = None
    for path in ckpt_paths:
        if path.exists():
            ckpt = path
            break
    if ckpt is None:
        raise FileNotFoundError(
            f"UniMatch checkpoint not found in: {[str(p) for p in ckpt_paths]}"
        )
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)

    def flow_fn(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        def prep(img):
            t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            # pad to 32-multiple
            _, _, H, W = t.shape
            pH = (32 - H % 32) % 32
            pW = (32 - W % 32) % 32
            return F.pad(t, (0, pW, 0, pH)), H, W

        ta, H, W = prep(img_a)
        tb, _, _ = prep(img_b)
        with torch.no_grad():
            result = model(ta, tb, attn_type="swin", attn_splits_list=[2, 8],
                           corr_radius_list=[-1, 4], prop_radius_list=[-1, 1],
                           num_reg_refine=6, task="flow")
        flow = result["flow_preds"][-1][0].permute(1, 2, 0).cpu().numpy()
        return flow[:H, :W]

    return flow_fn


def load_dover_fn(device: str = "cuda", dover_config_path: str = None, dover_weight_path: str = None):
    """Load DOVER and return dover_fn(frames_rgb: (T,H,W,3) uint8) -> float.

    Args:
        device: torch device (e.g., 'cuda' or 'cpu')
        dover_config_path: path to dover.yml (default: auto-detect from DOVER package)
        dover_weight_path: path to DOVER.pth (default: auto-detect from DOVER package)

    Note: H100 GPU is fully supported as of PyTorch 2.6.0+cu124.
          Previous CPU-only workaround has been removed (2026-08-07).
    """
    from dover import DOVER  # type: ignore
    import torch
    import yaml
    from pathlib import Path
    import warnings

    # Note: Previous H100 compatibility workaround removed (2026-08-07)
    # Testing confirmed DOVER works perfectly on H100 GPU with PyTorch 2.6.0+cu124
    # See: DOVER_H100_部署方案_CMCC实际执行记录.md

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

    # Load config and initialize model (correct way per DOVER repo)
    with open(dover_config_path, "r") as f:
        dover_opt = yaml.safe_load(f)

    # Initialize model on specified device
    model = DOVER(**dover_opt["model"]["args"])
    model.load_state_dict(torch.load(dover_weight_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()

    def dover_fn(frames_rgb: np.ndarray) -> float:
        import torch
        # DOVER expects a dict with 'technical' and 'aesthetic' views
        # frames_rgb: (T, H, W, 3) uint8
        # Convert to (1, 3, T, H, W) float32 normalized
        t = torch.from_numpy(frames_rgb).float() / 255.0  # (T, H, W, 3)
        t = t.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # (1, 3, T, H, W)
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


def load_qwen_fn(model_dir: str, device: str = "cuda"):
    """Load Qwen3.5-27B-VL and return vlm_call(prompt, keyframes) -> str."""
    from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
    import torch
    from PIL import Image

    # Use AutoModelForCausalLM with trust_remote_code for Qwen3.5
    # This works with older transformers versions
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    def vlm_call(prompt: str, keyframes: list) -> str:
        pil_imgs = [Image.fromarray(f) for f in keyframes]
        content = [{"type": "text", "text": prompt}]
        for img in pil_imgs:
            content.insert(-1, {"type": "image", "image": img})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=pil_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256)
        return processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    return vlm_call
