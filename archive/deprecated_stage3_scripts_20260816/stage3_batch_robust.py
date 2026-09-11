#!/usr/bin/env python3
"""Stage3 Batch - 批量处理 SpatialVID group_0001 (改进版)

改进点:
1. 添加断点续传 (检测已处理样本，跳过)
2. 每 10 个样本保存一次 checkpoint
3. 捕获所有异常，确保不会中断
4. 独立日志文件，不依赖 tee
"""
import sys, json, time, argparse, logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

UNIMATCH_RANGE = [3, 80]
DOVER_RANGE = [0.35, 1.0]

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_models(device="cuda"):
    sys.path.insert(0, "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER")
    sys.path.insert(0, "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/unimatch")

    import yaml
    from dover import DOVER
    from unimatch.unimatch import UniMatch

    # DOVER
    with open("/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/dover.yml") as f:
        opt = yaml.safe_load(f)
    dover = DOVER(**opt["model"]["args"])
    dover.load_state_dict(torch.load(
        "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/pretrained_weights/DOVER.pth",
        map_location=device, weights_only=False
    ))
    dover = dover.to(device).eval()

    # UniMatch
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
    chunk_size = 32
    scores = []
    for i in range(0, len(frames), chunk_size):
        chunk = frames[i:i+chunk_size]
        if len(chunk) < 10:
            continue
        t = torch.from_numpy(chunk).float() / 255.0
        t = t.permute(3, 0, 1, 2).unsqueeze(0).to(device)
        views = {"technical": t, "aesthetic": t}
        with torch.no_grad():
            results = model(views)
        chunk_score = sum(r.mean().item() for r in results) / len(results)
        scores.append(chunk_score)
        del t, views, results
        torch.cuda.empty_cache()
    return float(np.mean(scores))

def process_one_video(video_path, dover_model, unimatch_model, device, logger):
    try:
        frames = decode_video(video_path)
        flow_mag = compute_unimatch_flow(frames, unimatch_model, device)
        dover_score = compute_dover_score(frames, dover_model, device)

        flow_pass = UNIMATCH_RANGE[0] <= flow_mag <= UNIMATCH_RANGE[1]
        dover_pass = DOVER_RANGE[0] <= dover_score <= DOVER_RANGE[1]
        final_pass = flow_pass and dover_pass

        return {
            "sample_id": video_path.stem,
            "unimatch_flow": round(flow_mag, 3),
            "dover_score": round(dover_score, 4),
            "verdict": "pass" if final_pass else "fail",
            "reasons": [] if final_pass else [
                f"unimatch={flow_mag:.3f} not in {UNIMATCH_RANGE}" if not flow_pass else None,
                f"dover={dover_score:.4f} not in {DOVER_RANGE}" if not dover_pass else None,
            ]
        }
    except Exception as e:
        logger.error(f"Error processing {video_path.name}: {e}")
        return {
            "sample_id": video_path.stem,
            "unimatch_flow": None,
            "dover_score": None,
            "verdict": "error",
            "reasons": [str(e)]
        }

def load_processed_ids(output_file):
    """加载已处理的样本 ID"""
    if not output_file.exists():
        return set()
    processed = set()
    with open(output_file) as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    processed.add(data["sample_id"])
                except:
                    pass
    return processed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", default=None, help="Log file (default: <output>.log)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output)
    log_file = Path(args.log) if args.log else output_file.with_suffix('.log')

    logger = setup_logging(log_file)

    videos = sorted(input_dir.glob("*.mp4"))
    logger.info(f"Found {len(videos)} videos in {input_dir}")

    # 断点续传
    if args.resume:
        processed_ids = load_processed_ids(output_file)
        logger.info(f"Resume mode: {len(processed_ids)} already processed")
        videos = [v for v in videos if v.stem not in processed_ids]
        logger.info(f"Remaining: {len(videos)} videos")

    logger.info(f"Loading models on {args.device}...")
    dover_model, unimatch_model = load_models(args.device)
    logger.info("✅ Models loaded")

    logger.info(f"Processing {len(videos)} videos...")

    start_time = time.time()
    results = []

    with open(output_file, "a" if args.resume else "w") as f:
        for idx, video_path in enumerate(tqdm(videos, desc="Stage3"), 1):
            result = process_one_video(video_path, dover_model, unimatch_model, args.device, logger)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            results.append(result)

            # 每 10 个样本输出一次统计
            if idx % 10 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                eta = (len(videos) - idx) / rate
                logger.info(f"Progress: {idx}/{len(videos)} | Rate: {rate:.2f} vid/s | ETA: {eta/3600:.1f}h")

    # 最终统计
    pass_count = sum(1 for r in results if r["verdict"] == "pass")
    fail_count = sum(1 for r in results if r["verdict"] == "fail")
    error_count = sum(1 for r in results if r["verdict"] == "error")

    logger.info(f"\n=== Summary ===")
    logger.info(f"Total:  {len(results)}")
    logger.info(f"Pass:   {pass_count} ({100*pass_count/len(results):.1f}%)")
    logger.info(f"Fail:   {fail_count} ({100*fail_count/len(results):.1f}%)")
    logger.info(f"Error:  {error_count} ({100*error_count/len(results):.1f}%)")
    logger.info(f"Time:   {(time.time()-start_time)/3600:.2f}h")
    logger.info(f"Output: {output_file}")
