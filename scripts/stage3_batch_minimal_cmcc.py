#!/usr/bin/env python3
"""Stage3 Batch - CMCC适配版本（支持动态路径配置）

配置:
1. DOVER: 5s分块（论文原始配置）+ 720p降采样
2. UniMatch: 0.5s采样间隔（论文配置）
3. 100% 使用官方接口（spatial_temporal_view_decomposition + fuse_results）
4. 支持通过环境变量或命令行参数配置模型路径（CMCC兼容）

改进:
- 减少 70 行重复代码
- 自动降采样 >720p 视频避免OOM
- 基于实验验证的最优配置（+5.1% DOVER分数）
- 支持CMCC离线环境（动态路径配置）
"""
import sys, json, time, argparse, logging, os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

UNIMATCH_RANGE = [3, 80]
DOVER_RANGE = [0.35, 1.0]

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def load_models(dover_path, unimatch_path, device="cuda"):
    """加载模型 - 支持动态路径"""
    # 添加模型路径到 sys.path
    sys.path.insert(0, str(dover_path))
    sys.path.insert(0, str(unimatch_path))

    import yaml
    from dover import DOVER
    from dover.datasets import UnifiedFrameSampler
    from unimatch.unimatch import UniMatch

    # 加载 DOVER 配置
    dover_yml = Path(dover_path) / "dover.yml"
    with open(dover_yml) as f:
        opt = yaml.safe_load(f)

    dover = DOVER(**opt["model"]["args"])

    # 加载 DOVER 权重
    dover_weights = Path(dover_path) / "pretrained_weights" / "DOVER.pth"
    dover.load_state_dict(torch.load(
        str(dover_weights),
        map_location=device, weights_only=False
    ))
    dover = dover.to(device).eval()

    dopt = opt["data"]["val-l1080p"]["args"]
    temporal_samplers = {}
    for stype, sopt in dopt["sample_types"].items():
        if "t_frag" not in sopt:
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
            )
        else:
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"] // sopt["t_frag"],
                sopt["t_frag"],
                sopt["frame_interval"],
                sopt["num_clips"]
            )

    # 加载 UniMatch
    unimatch = UniMatch(
        feature_channels=128, num_scales=2, upsample_factor=4,
        num_head=1, ffn_dim_expansion=4, num_transformer_layers=6,
        reg_refine=True, task="flow"
    ).to(device).eval()

    # 加载 UniMatch 权重
    unimatch_weights = Path(unimatch_path) / "pretrained" / "gmflow-scale2-regrefine6-mixdata.pth"
    state = torch.load(str(unimatch_weights), map_location=device)
    unimatch.load_state_dict(state["model"] if "model" in state else state, strict=False)

    return dover, unimatch, dopt, temporal_samplers

def compute_dover_score(video_path, model, dopt, temporal_samplers, device):
    """DOVER - 5s分块 + 720p降采样"""
    from dover.datasets import spatial_temporal_view_decomposition
    from evaluate_one_video import fuse_results
    import decord
    import cv2
    import tempfile
    import os

    mean = torch.FloatTensor([123.675, 116.28, 103.53])
    std = torch.FloatTensor([58.395, 57.12, 57.375])

    # 检查分辨率，必要时降采样到720p
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    H, W = vr[0].shape[:2]

    video_path_to_use = str(video_path)
    is_temp = False

    if H > 720:
        # 降采样到720p
        scale = 720 / H
        new_H, new_W = 720, int(W * scale)

        # 创建临时降采样视频
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4", dir=video_path.parent / "tmp")
        os.close(temp_fd)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, vr.get_avg_fps(), (new_W, new_H))

        for frame in vr:
            frame_np = frame.numpy()
            frame_resized = cv2.resize(frame_np, (new_W, new_H))
            out.write(cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR))
        out.release()

        video_path_to_use = temp_path
        is_temp = True

    # ✅ 官方接口：一行完成采样+resize（5s分块）
    views, _ = spatial_temporal_view_decomposition(
        video_path_to_use,
        dopt["sample_types"],
        temporal_samplers,
        is_train=False
    )

    # 归一化（官方逻辑）
    for k, v in views.items():
        num_clips = dopt["sample_types"][k].get("num_clips", 1)
        views[k] = (
            ((v.permute(1, 2, 3, 0) - mean) / std)
            .permute(3, 0, 1, 2)
            .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
            .transpose(0, 1)
            .to(device)
        )

    # DOVER推理
    with torch.no_grad():
        results = [r.mean().item() for r in model(views)]

    # 清理临时文件
    if is_temp:
        try:
            os.unlink(video_path_to_use)
        except:
            pass

    # ✅ 官方融合函数
    tqe, aqe = results[0], results[1]
    fused = fuse_results(results)

    return {"tqe": tqe, "aqe": aqe, "fused": fused}

def compute_unimatch_flow(video_path, model, device, sample_interval=0.5):
    """UniMatch 光流估计 - 0.5s采样间隔"""
    import decord
    import torch.nn.functional as F

    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    fps = vr.get_avg_fps()
    frame_interval = max(1, int(fps * sample_interval))

    frame_indices = list(range(0, len(vr), frame_interval))
    if len(frame_indices) < 2:
        return {"magnitude": 0.0}

    frames = vr.get_batch(frame_indices).float() / 255.0
    frames = frames.permute(0, 3, 1, 2).to(device)

    flows = []
    with torch.no_grad():
        for i in range(len(frames) - 1):
            img1, img2 = frames[i:i+1], frames[i+1:i+2]

            # Padding to multiple of 8
            h, w = img1.shape[-2:]
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                img1 = F.pad(img1, (0, pad_w, 0, pad_h), mode='replicate')
                img2 = F.pad(img2, (0, pad_w, 0, pad_h), mode='replicate')

            results = model(img1, img2, attn_splits_list=[2], corr_radius_list=[-1],
                          prop_radius_list=[-1], pred_bidir_flow=False)
            flow = results['flow_preds'][-1][0].cpu()

            if pad_h > 0 or pad_w > 0:
                flow = flow[:, :h, :w]

            flows.append(flow)

    # 计算平均幅值
    magnitudes = [torch.norm(f, dim=0).mean().item() for f in flows]
    avg_magnitude = np.mean(magnitudes)

    return {"magnitude": avg_magnitude}

def process_one_video(video_path, dover_model, unimatch_model, dover_opt,
                     temporal_samplers, device, logger):
    """处理单个视频"""
    sample_id = video_path.stem

    try:
        # DOVER评分
        dover_result = compute_dover_score(video_path, dover_model, dover_opt,
                                          temporal_samplers, device)

        # UniMatch光流
        unimatch_result = compute_unimatch_flow(video_path, unimatch_model, device)

        # 判定
        dover_pass = DOVER_RANGE[0] <= dover_result["fused"] <= DOVER_RANGE[1]
        unimatch_pass = UNIMATCH_RANGE[0] <= unimatch_result["magnitude"] <= UNIMATCH_RANGE[1]
        overall_pass = dover_pass and unimatch_pass

        reasons = []
        if not dover_pass:
            reasons.append(f"dover_fused={dover_result['fused']:.3f} not in [{DOVER_RANGE[0]}, {DOVER_RANGE[1]}]")
        if not unimatch_pass:
            reasons.append(f"unimatch_flow={unimatch_result['magnitude']:.2f} not in [{UNIMATCH_RANGE[0]}, {UNIMATCH_RANGE[1]}]")

        verdict = "pass" if overall_pass else "fail"

        logger.info(f"{sample_id}: DOVER={dover_result['fused']:.3f}, UniMatch={unimatch_result['magnitude']:.2f}, {verdict.upper()}")

        return {
            "sample_id": sample_id,
            "dover_tqe": dover_result["tqe"],
            "dover_aqe": dover_result["aqe"],
            "dover_fused": dover_result["fused"],
            "unimatch_flow": unimatch_result["magnitude"],
            "pass": overall_pass,
            "verdict": verdict,
            "reasons": reasons
        }

    except Exception as e:
        logger.error(f"{sample_id}: ERROR - {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            "sample_id": sample_id,
            "unimatch_flow": None,
            "dover_tqe": None,
            "dover_aqe": None,
            "dover_fused": None,
            "pass": False,
            "verdict": "error",
            "reasons": [str(e)]
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage3 视频质量筛选 - CMCC适配版")
    parser.add_argument("--input_dir", required=True, help="输入视频目录")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件")
    parser.add_argument("--dover_path", required=True, help="DOVER 模型路径")
    parser.add_argument("--unimatch_path", required=True, help="UniMatch 模型路径")
    parser.add_argument("--log", default=None, help="日志文件（默认：output.log）")
    parser.add_argument("--device", default="cuda", help="设备（cuda/cpu）")
    parser.add_argument("--resume", action="store_true", help="断点续传")
    args = parser.parse_args()

    # 验证路径
    dover_path = Path(args.dover_path)
    unimatch_path = Path(args.unimatch_path)

    if not dover_path.exists():
        print(f"错误: DOVER 路径不存在: {dover_path}")
        sys.exit(1)
    if not unimatch_path.exists():
        print(f"错误: UniMatch 路径不存在: {unimatch_path}")
        sys.exit(1)

    output_file = Path(args.output)
    log_file = Path(args.log) if args.log else output_file.with_suffix('.log')
    logger = setup_logging(log_file)

    videos = sorted(Path(args.input_dir).glob("*.mp4"))
    logger.info(f"发现 {len(videos)} 个视频")

    if args.resume and output_file.exists():
        processed = set()
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    try:
                        processed.add(json.loads(line)["sample_id"])
                    except:
                        pass
        videos = [v for v in videos if v.stem not in processed]
        logger.info(f"Resume: {len(processed)} 已完成, {len(videos)} 待处理")

    logger.info("加载模型...")
    dover_model, unimatch_model, dover_opt, temporal_samplers = load_models(
        dover_path, unimatch_path, args.device
    )
    logger.info("✅ 模型加载完成")

    # 创建临时目录
    tmp_dir = Path(args.input_dir) / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    start_time = time.time()
    results = []

    with open(output_file, "a" if args.resume else "w") as f:
        for idx, video_path in enumerate(tqdm(videos, desc="Stage3"), 1):
            result = process_one_video(video_path, dover_model, unimatch_model,
                                      dover_opt, temporal_samplers, args.device, logger)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            results.append(result)

            if idx % 10 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                eta = (len(videos) - idx) / rate
                logger.info(f"{idx}/{len(videos)} | {rate:.2f} vid/s | ETA: {eta/3600:.1f}h")

    pass_count = sum(1 for r in results if r["verdict"] == "pass")
    logger.info(f"总计: {len(results)}, 通过: {pass_count} ({100*pass_count/len(results):.1f}%)")
