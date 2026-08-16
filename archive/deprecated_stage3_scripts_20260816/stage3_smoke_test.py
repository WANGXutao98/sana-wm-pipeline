#!/usr/bin/env python3
"""Stage3 Smoke Test - 单视频验证 Dover + UniMatch

Usage:
    conda activate sana_qc
    python stage3_smoke_test.py <video.mp4>
"""
import sys, time
import numpy as np
import torch
from pathlib import Path

# SpatialVID Table 6 thresholds
UNIMATCH_RANGE = [3, 80]
DOVER_RANGE = [0.35, 1.0]

def load_dover(weights_path, config_path, device="cuda"):
    import yaml
    dover_dir = Path(weights_path).parent.parent
    sys.path.insert(0, str(dover_dir))
    from dover import DOVER
    with open(config_path) as f:
        opt = yaml.safe_load(f)
    model = DOVER(**opt["model"]["args"])
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    return model.to(device).eval()

def load_unimatch(weights_path, device="cuda"):
    sys.path.insert(0, str(Path(weights_path).parent.parent))
    from unimatch.unimatch import UniMatch
    model = UniMatch(
        feature_channels=128, num_scales=2, upsample_factor=4,
        num_head=1, ffn_dim_expansion=4, num_transformer_layers=6,
        reg_refine=True, task="flow"
    ).to(device).eval()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state, strict=False)
    return model

def decode_video(video_path):
    import decord
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    frames = vr[:].numpy()  # (T, H, W, 3) uint8
    return frames

def compute_unimatch_flow(frames, model, device):
    import torch.nn.functional as F

    def prep(img):
        t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        _, _, H, W = t.shape
        pH = (32 - H % 32) % 32
        pW = (32 - W % 32) % 32
        return F.pad(t, (0, pW, 0, pH)), H, W

    # Sample every 0.5s (论文规定)
    fps = 16  # 假设 16fps
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
    # Use 32-frame chunks (2s @ 16fps) to avoid OOM
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
        # Clear GPU cache after each chunk
        del t, views, results
        torch.cuda.empty_cache()
    return float(np.mean(scores))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stage3_smoke_test.py <video.mp4>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    device = "cuda"

    print(f"[1/5] Loading models...")
    dover_model = load_dover(
        "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/pretrained_weights/DOVER.pth",
        "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/dover.yml",
        device
    )
    unimatch_model = load_unimatch(
        "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata.pth",
        device
    )
    print("✅ Models loaded")

    print(f"\n[2/5] Decoding video: {video_path.name}")
    t0 = time.time()
    frames = decode_video(video_path)
    print(f"✅ Decoded {len(frames)} frames in {time.time()-t0:.2f}s")

    print(f"\n[3/5] Computing UniMatch flow...")
    t0 = time.time()
    flow_mag = compute_unimatch_flow(frames, unimatch_model, device)
    print(f"✅ Flow magnitude: {flow_mag:.3f} (took {time.time()-t0:.2f}s)")

    print(f"\n[4/5] Computing DOVER score...")
    t0 = time.time()
    dover_score = compute_dover_score(frames, dover_model, device)
    print(f"✅ DOVER score: {dover_score:.4f} (took {time.time()-t0:.2f}s)")

    print(f"\n[5/5] Table 6 judgment (SpatialVID)")
    flow_pass = UNIMATCH_RANGE[0] <= flow_mag <= UNIMATCH_RANGE[1]
    dover_pass = DOVER_RANGE[0] <= dover_score <= DOVER_RANGE[1]
    final_pass = flow_pass and dover_pass

    print(f"  UniMatch: {'PASS' if flow_pass else 'FAIL'} ({flow_mag:.3f} ∈ {UNIMATCH_RANGE})")
    print(f"  DOVER:    {'PASS' if dover_pass else 'FAIL'} ({dover_score:.4f} ∈ {DOVER_RANGE})")
    print(f"  Final:    {'✅ PASS' if final_pass else '❌ FAIL'}")
