#!/usr/bin/env python3
"""Stage3 Batch - 修复版（正确的 DOVER 归一化）

修复点:
1. 添加 ImageNet 预处理归一化
2. 使用官方 fuse_results 公式
3. 分别输出 TQE/AQE 原始分数和融合分数
"""
import sys, json, time, argparse, logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

UNIMATCH_RANGE = [3, 80]
DOVER_RANGE = [0.35, 1.0]

# DOVER 官方归一化参数
IMAGENET_MEAN = torch.FloatTensor([123.675, 116.28, 103.53])
IMAGENET_STD = torch.FloatTensor([58.395, 57.12, 57.375])

def fuse_dover_results(tqe, aqe):
    """DOVER 官方融合公式"""
    x = (tqe - 0.1107) / 0.07355 * 0.6104 + \
        (aqe + 0.08285) / 0.03774 * 0.3896
    return 1 / (1 + np.exp(-x))

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def load_models(device="cuda"):
    sys.path.insert(0, "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER")
    sys.path.insert(0, "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/unimatch")

    import yaml
    from dover import DOVER
    from unimatch.unimatch import UniMatch

    with open("/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/dover.yml") as f:
        opt = yaml.safe_load(f)
    dover = DOVER(**opt["model"]["args"])
    dover.load_state_dict(torch.load(
        "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/pretrained_weights/DOVER.pth",
        map_location=device, weights_only=False
    ))
    dover = dover.to(device).eval()

    unimatch = UniMatch(
        feature_channels=128, num_scales=2, upsample_factor=4,
        num_head=1, ffn_dim_expansion=4, num_transformer_layers=6,
        reg_refine=True, task="flow"
    ).to(device).eval()
    state = torch.load(
        "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata.pth",
        map_location=device
    )
    unimatch.load_state_dict(state["model"] if "model" in state else state, strict=False)

    return dover, unimatch

def decode_video(video_path):
    import decord
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    return vr[:].numpy()

def compute_unimatch_flow(frames, model, device):
    import torch.nn.functional as F
    def prep(img):
        t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        _, _, H, W = t.shape
        pH = (32 - H % 32) % 32
        pW = (32 - W % 32) % 32
        return F.pad(t, (0, pW, 0, pH)), H, W

    fps = 16
    step = max(1, int(fps * 0.5))
    pairs = [(i, min(i + step, len(frames) - 1)) for i in range(0, len(frames) - step, step)]

    magnitudes = []
    for i, j in pairs:
        ta, H, W = prep(frames[i])
        tb, _, _ = prep(frames[j])
        with torch.no_grad():
            result = model(ta, tb, attn_type="swin", attn_splits_list=[2, 8],
                          corr_radius_list=[-1, 4], prop_radius_list=[-1, 1],
                          num_reg_refine=6, task="flow")
        flow = result["flow_preds"][-1][0].permute(1, 2, 0).cpu().numpy()[:H, :W]
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
        magnitudes.append(mag)

    return float(np.mean(magnitudes))

def compute_dover_score(frames, model, device):
    """正确的 DOVER 评分 - 使用 ImageNet 归一化 + fuse_results"""
    chunk_size = 32
    tqe_scores = []
    aqe_scores = []

    mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(device)
    std = IMAGENET_STD.view(1, 3, 1, 1).to(device)

    for i in range(0, len(frames), chunk_size):
        chunk = frames[i:i+chunk_size]
        if len(chunk) < 10:
            continue

        # 正确的预处理: ImageNet 归一化
        t = torch.from_numpy(chunk).float().to(device)
        t = t.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)
        t = t.permute(0, 2, 3, 4, 1)  # (1, T, H, W, C)
        t = (t - mean) / std
        t = t.permute(0, 4, 1, 2, 3)  # (1, C, T, H, W)

        views = {"technical": t, "aesthetic": t}
        with torch.no_grad():
            results = model(views)

        tqe = results[0].mean().item()
        aqe = results[1].mean().item()
        tqe_scores.append(tqe)
        aqe_scores.append(aqe)

        del t, views, results
        torch.cuda.empty_cache()

    # 平均各分块的 TQE/AQE
    avg_tqe = float(np.mean(tqe_scores))
    avg_aqe = float(np.mean(aqe_scores))

    # 使用官方融合公式
    fused_score = fuse_dover_results(avg_tqe, avg_aqe)

    return {
        "tqe": round(avg_tqe, 4),
        "aqe": round(avg_aqe, 4),
        "fused": round(fused_score, 4)
    }

def process_one_video(video_path, dover_model, unimatch_model, device, logger):
    try:
        frames = decode_video(video_path)
        flow_mag = compute_unimatch_flow(frames, unimatch_model, device)
        dover_result = compute_dover_score(frames, dover_model, device)

        flow_pass = UNIMATCH_RANGE[0] <= flow_mag <= UNIMATCH_RANGE[1]
        dover_pass = DOVER_RANGE[0] <= dover_result["fused"] <= DOVER_RANGE[1]
        final_pass = flow_pass and dover_pass

        return {
            "sample_id": video_path.stem,
            "unimatch_flow": round(flow_mag, 3),
            "dover_tqe": dover_result["tqe"],
            "dover_aqe": dover_result["aqe"],
            "dover_fused": dover_result["fused"],
            "verdict": "pass" if final_pass else "fail",
            "reasons": [] if final_pass else [
                f"unimatch={flow_mag:.3f} not in {UNIMATCH_RANGE}" if not flow_pass else None,
                f"dover={dover_result['fused']:.4f} not in {DOVER_RANGE}" if not dover_pass else None,
            ]
        }
    except Exception as e:
        logger.error(f"Error: {video_path.name}: {e}")
        return {
            "sample_id": video_path.stem,
            "unimatch_flow": None,
            "dover_tqe": None,
            "dover_aqe": None,
            "dover_fused": None,
            "verdict": "error",
            "reasons": [str(e)]
        }

def load_processed_ids(output_file):
    if not output_file.exists():
        return set()
    processed = set()
    with open(output_file) as f:
        for line in f:
            if line.strip():
                try:
                    processed.add(json.loads(line)["sample_id"])
                except:
                    pass
    return processed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output)
    log_file = Path(args.log) if args.log else output_file.with_suffix('.log')

    logger = setup_logging(log_file)

    videos = sorted(input_dir.glob("*.mp4"))
    logger.info(f"Found {len(videos)} videos")

    if args.resume:
        processed_ids = load_processed_ids(output_file)
        logger.info(f"Resume: {len(processed_ids)} processed")
        videos = [v for v in videos if v.stem not in processed_ids]
        logger.info(f"Remaining: {len(videos)}")

    logger.info("Loading models...")
    dover_model, unimatch_model = load_models(args.device)
    logger.info("✅ Models loaded")

    start_time = time.time()
    results = []

    with open(output_file, "a" if args.resume else "w") as f:
        for idx, video_path in enumerate(tqdm(videos, desc="Stage3-Fixed"), 1):
            result = process_one_video(video_path, dover_model, unimatch_model, args.device, logger)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            results.append(result)

            if idx % 10 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                eta = (len(videos) - idx) / rate
                logger.info(f"{idx}/{len(videos)} | {rate:.2f} vid/s | ETA: {eta/3600:.1f}h")

    pass_count = sum(1 for r in results if r["verdict"] == "pass")
    fail_count = sum(1 for r in results if r["verdict"] == "fail")
    error_count = sum(1 for r in results if r["verdict"] == "error")

    logger.info(f"\n=== Summary ===")
    logger.info(f"Total: {len(results)}")
    logger.info(f"Pass:  {pass_count} ({100*pass_count/len(results):.1f}%)")
    logger.info(f"Fail:  {fail_count} ({100*fail_count/len(results):.1f}%)")
    logger.info(f"Error: {error_count} ({100*error_count/len(results):.1f}%)")
    logger.info(f"Time:  {(time.time()-start_time)/3600:.2f}h")
