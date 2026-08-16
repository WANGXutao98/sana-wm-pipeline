#!/usr/bin/env python3
"""Stage3 Batch - 批量处理 SpatialVID group_0001

Usage:
    conda activate sana_qc
    python stage3_batch.py --input_dir <dir> --output <results.jsonl>
"""
import sys, json, time, argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

# Table 6 thresholds
UNIMATCH_RANGE = [3, 80]
DOVER_RANGE = [0.35, 1.0]

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

def process_one_video(video_path, dover_model, unimatch_model, device):
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
        return {
            "sample_id": video_path.stem,
            "unimatch_flow": None,
            "dover_score": None,
            "verdict": "error",
            "reasons": [str(e)]
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with .mp4 files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    videos = sorted(input_dir.glob("*.mp4"))

    print(f"[1/3] Loading models on {args.device}...")
    dover_model, unimatch_model = load_models(args.device)
    print(f"✅ Models loaded")

    print(f"\n[2/3] Processing {len(videos)} videos...")
    results = []
    with open(args.output, "w") as f:
        for video_path in tqdm(videos, desc="Stage3"):
            result = process_one_video(video_path, dover_model, unimatch_model, args.device)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            results.append(result)

    # Summary
    pass_count = sum(1 for r in results if r["verdict"] == "pass")
    fail_count = sum(1 for r in results if r["verdict"] == "fail")
    error_count = sum(1 for r in results if r["verdict"] == "error")

    print(f"\n[3/3] Summary:")
    print(f"  Total:  {len(results)}")
    print(f"  Pass:   {pass_count} ({100*pass_count/len(results):.1f}%)")
    print(f"  Fail:   {fail_count} ({100*fail_count/len(results):.1f}%)")
    print(f"  Error:  {error_count} ({100*error_count/len(results):.1f}%)")
    print(f"\n✅ Output written to: {args.output}")
