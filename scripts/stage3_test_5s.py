#!/usr/bin/env python3
"""Stage3 - 5s分块版本（论文原始配置 + 降采样防OOM）

改动:
1. DOVER 采样: 使用论文原始配置 (5s分块)
2. 降采样: 输入resize到480p防止OOM
3. 其他: 保持官方接口100%对齐
"""
import sys, json, argparse
from pathlib import Path
import torch

sys.path.insert(0, "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER")
sys.path.insert(0, "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/unimatch")

UNIMATCH_RANGE = [3, 80]
DOVER_RANGE = [0.35, 1.0]

def load_models(device="cuda"):
    import yaml
    from dover import DOVER
    from dover.datasets import UnifiedFrameSampler
    from unimatch.unimatch import UniMatch

    with open("/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/dover.yml") as f:
        opt = yaml.safe_load(f)

    dover = DOVER(**opt["model"]["args"])
    dover.load_state_dict(torch.load(
        "/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/pretrained_weights/DOVER.pth",
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

    return dover, unimatch, dopt, temporal_samplers

def compute_dover_score_5s_downsampled(video_path, model, dopt, temporal_samplers, device):
    """DOVER 5s分块 + 480p降采样"""
    from dover.datasets import spatial_temporal_view_decomposition
    from evaluate_one_video import fuse_results
    import decord
    import cv2
    import tempfile
    import os

    mean = torch.FloatTensor([123.675, 116.28, 103.53])
    std = torch.FloatTensor([58.395, 57.12, 57.375])

    # 降采样到480p
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    frames = vr[:].numpy()

    H, W = frames.shape[1:3]
    if H > 480:
        # 降采样到480p
        scale = 480 / H
        new_H, new_W = 480, int(W * scale)

        # 创建临时降采样视频
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.mp4')
        os.close(tmp_fd)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_path, fourcc, 16, (new_W, new_H))
        for frame in frames:
            resized = cv2.resize(frame, (new_W, new_H))
            out.write(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        out.release()

        video_path_to_use = tmp_path
        is_temp = True
    else:
        video_path_to_use = str(video_path)
        is_temp = False

    # 官方接口处理
    views, _ = spatial_temporal_view_decomposition(
        video_path_to_use,
        dopt["sample_types"],
        temporal_samplers,
        is_train=False
    )

    # 清理临时文件
    if is_temp:
        os.unlink(tmp_path)

    # 归一化
    for k, v in views.items():
        num_clips = dopt["sample_types"][k].get("num_clips", 1)
        views[k] = (
            ((v.permute(1, 2, 3, 0) - mean) / std)
            .permute(3, 0, 1, 2)
            .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
            .transpose(0, 1)
            .to(device)
        )

    with torch.no_grad():
        results = [r.mean().item() for r in model(views)]

    tqe, aqe = results[0], results[1]
    fused = fuse_results(results)

    return {"tqe": round(tqe, 4), "aqe": round(aqe, 4), "fused": round(fused, 4)}

def compute_unimatch_flow(video_path, model, device):
    """UniMatch - 保持不变"""
    import decord
    import torch.nn.functional as F
    import numpy as np

    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    frames = vr[:].numpy()

    def prep(img):
        t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
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
        magnitudes.append(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean())

    return float(np.mean(magnitudes))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    video_path = Path(args.video)

    print("Loading models...")
    dover_model, unimatch_model, dover_opt, temporal_samplers = load_models(args.device)
    print("✅ Models loaded")

    print(f"\nProcessing: {video_path.name}")

    flow_mag = compute_unimatch_flow(video_path, unimatch_model, args.device)
    print(f"UniMatch flow: {flow_mag:.3f}")

    dover_result = compute_dover_score_5s_downsampled(
        video_path, dover_model, dover_opt, temporal_samplers, args.device
    )
    print(f"DOVER TQE:   {dover_result['tqe']:.4f}")
    print(f"DOVER AQE:   {dover_result['aqe']:.4f}")
    print(f"DOVER fused: {dover_result['fused']:.4f}")

    flow_pass = UNIMATCH_RANGE[0] <= flow_mag <= UNIMATCH_RANGE[1]
    dover_pass = DOVER_RANGE[0] <= dover_result["fused"] <= DOVER_RANGE[1]

    print(f"\nVerdict: {'PASS' if (flow_pass and dover_pass) else 'FAIL'}")
    if not flow_pass:
        print(f"  - UniMatch: {flow_mag:.3f} not in {UNIMATCH_RANGE}")
    if not dover_pass:
        print(f"  - DOVER: {dover_result['fused']:.4f} not in {DOVER_RANGE}")

    # 输出JSON
    result = {
        "sample_id": video_path.stem,
        "config": "5s_downsampled_480p",
        "unimatch_flow": round(flow_mag, 3),
        "dover_tqe": dover_result["tqe"],
        "dover_aqe": dover_result["aqe"],
        "dover_fused": dover_result["fused"],
        "verdict": "pass" if (flow_pass and dover_pass) else "fail"
    }

    output_file = Path("/mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_5s.jsonl")
    with open(output_file, "w") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n✅ Result saved to: {output_file}")
