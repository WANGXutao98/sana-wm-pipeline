"""GT-pose pose-annotation mode (与官方代码 100% 对齐).

Paper App. B.1: "GT trajectory kept at FULL length (N frames). Pi3X runs on an
n_scale subsample ONLY to recover the single metric-scale scalar via Umeyama
on matched frames."

官方参考: sana-wm-data-clean/sana_wm_data/pose/stage.py:59-77

Targets: Sekai-Game, DL3DV.
Pipeline: trust the GT camera trajectory; run Pi3X for structure;
Umeyama-Sim(3) with 80%-inlier filter aligns Pi3X scene scale to GT.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ._common import PoseArtifact
from .umeyama import DEFAULT_INLIER_PERCENTILE, recover_metric_scale


def run_gtpose(
    clip_path: Path,
    gt_poses_path: Path,
    work_dir: Path,
    inlier_percentile: float = DEFAULT_INLIER_PERCENTILE,
    gt_intrinsics_path: Path | None = None,
) -> PoseArtifact:
    """GT-pose 模式（与官方代码对齐）

    Pipeline (官方 stage.py:59-77):
      1. Pi3X 在 n_scale 帧子集上运行（节省 GPU 内存）
      2. GT poses 加载完整 N 帧
      3. GT 子采样匹配 Pi3X 帧索引
      4. Umeyama 恢复 metric scale
      5. 返回完整 GT poses（N 帧）+ scale

    与官方的关键对齐点:
      - 使用 _real.pi3_infer() 直接调用（利用 @lru_cache）
      - 支持帧子采样（n_scale < N）
      - 优先使用 GT 内参
      - 输出完整 N 帧 poses

    Args:
        clip_path: 视频文件路径
        gt_poses_path: GT poses 文件路径（.npy，(N, 4, 4) 或 (N, 3, 4) 或 (N, 3)）
        work_dir: 工作目录（保留接口兼容性，实际不再需要）
        inlier_percentile: Umeyama inlier 百分位阈值（default: 80.0）
        gt_intrinsics_path: GT 内参文件路径（可选，.npz 包含 K_px 或直接 .npy）
                          如果提供，优先使用；否则从 gt_poses_path 推断

    Returns:
        PoseArtifact with:
          - poses_c2w: (N, 4, 4) GT poses（完整长度）
          - intrinsics: (N, 1, 4) [fx, fy, cx, cy]（优先 GT，fallback seed）
          - scale_per_frame: (N,) metric scale（单值广播）
          - depth_downsampled: None（GT-pose 不产生深度）
    """
    # 导入官方模块（与 mode_default.py 保持一致）
    from ..sana_wm_data_clean.pose import _real, adapters

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. 决定 Pi3X 运行帧数
    # 官方: N = rec.num_frames or 24
    #       max_frames = int(os.environ.get("SANA_WM_MAX_FRAMES", "64"))
    #       n_scale = min(N, max_frames)
    #
    # 注意：官方的 N 来自视频元数据（rec.num_frames），不是从 GT 文件读取
    # 但本地实现没有 ClipRecord，需要从其他地方获取视频帧数

    # 导入官方模块（与 mode_default.py 保持一致）
    from ..sana_wm_data_clean.pose import _real, adapters

    # 获取视频帧数（与官方 rec.num_frames 等价）
    import cv2
    cap = cv2.VideoCapture(str(clip_path))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 24
    cap.release()

    max_frames = int(os.environ.get("SANA_WM_MAX_FRAMES", "64"))
    n_scale = min(N, max_frames)
    print(f"[mode_gtpose] Video has {N} frames, Pi3X will run on {n_scale} frames (SANA_WM_MAX_FRAMES={max_frames})")

    # 2. Pi3X 推理（与官方 run_pi3x_trajectory 对齐）
    # 官方: pred_pos = adapters.run_pi3x_trajectory(rec.video_path, rec.clip_id, n_scale, models_cfg, dry)
    print(f"[mode_gtpose] Running Pi3X inference...")
    frames = adapters.read_frames(str(clip_path), n_scale)
    poses_pi3x, _depth = _real.pi3_infer(frames)  # 利用 @lru_cache
    pred_positions = poses_pi3x[:, :3, 3]  # (n_scale, 3) camera centers
    n_scale = pred_positions.shape[0]  # Pi3 可能返回更少（官方逻辑）
    print(f"[mode_gtpose] Pi3X returned {n_scale} camera positions")

    # 3. 加载 GT poses（完整 N 帧）
    # 官方: gt_poses = _load_gt_poses(rec, N)
    #       if gt_poses is not None: ...
    #       else: # dry-run
    gt_poses_full = _load_gt_poses(gt_poses_path) if gt_poses_path.exists() else None

    if gt_poses_full is not None:
        # === 正常分支：有 GT poses ===
        # 官方: gt_sub = gt_poses[adapters.even_indices(gt_poses.shape[0], n_scale)]
        #       s = recover_metric_scale(pred_pos, gt_sub[:, :3, 3], inlier_percentile=80.0)
        #       poses = gt_poses  # FULL length
        N_gt = len(gt_poses_full)
        print(f"[mode_gtpose] Loaded {N_gt} GT poses from {gt_poses_path.name}")

        # 4. GT 子采样匹配 Pi3X 帧索引（与官方对齐）
        gt_indices = adapters.even_indices(N_gt, n_scale)
        poses_gt_sub = gt_poses_full[gt_indices]
        gt_positions_sub = poses_gt_sub[:, :3, 3]
        print(f"[mode_gtpose] GT subsampled to {len(gt_positions_sub)} frames for alignment")
        print(f"[mode_gtpose]   Subsample indices: {gt_indices[:5].tolist()} ... {gt_indices[-5:].tolist()}")

        # 5. Umeyama Sim(3) 恢复 metric scale（80% inlier filter）
        s = recover_metric_scale(
            pred_positions, gt_positions_sub, inlier_percentile=inlier_percentile
        )
        print(f"[mode_gtpose] Umeyama scale: {s:.6f} (inlier_percentile={inlier_percentile}%)")

        # 6. 使用完整 GT poses
        poses_c2w = gt_poses_full

    else:
        # === dry-run 分支：无 GT poses（与官方对齐）===
        # 官方: gt_centers = 1.7 * np.asarray(pred_pos)
        #       s = recover_metric_scale(pred_pos, gt_centers, inlier_percentile=80.0)
        #       poses = _positions_to_poses(gt_centers)
        print(f"[mode_gtpose] ⚠️  No GT poses found at {gt_poses_path}, using dry-run mode")
        print(f"[mode_gtpose] Generating synthetic GT (1.7x scaled Pi3X trajectory)")

        gt_centers = 1.7 * np.asarray(pred_positions)
        s = recover_metric_scale(
            pred_positions, gt_centers, inlier_percentile=inlier_percentile
        )
        print(f"[mode_gtpose] Umeyama scale (dry-run): {s:.6f} (inlier_percentile={inlier_percentile}%)")

        # 从位置构造 poses
        poses_c2w = _positions_to_poses(gt_centers)
        print(f"[mode_gtpose] Generated {len(poses_c2w)} synthetic poses (dry-run)")

    # 7. 加载内参（优先 GT → Fallback seed）
    # 官方: m = poses.shape[0]
    #       intr = _load_real_intrinsics(rec, m)
    #       if intr is None: intr = _seed_intrinsics(rec, m)
    m = poses_c2w.shape[0]

    # 优先使用显式提供的 gt_intrinsics_path
    if gt_intrinsics_path is not None and gt_intrinsics_path.exists():
        intr = _load_gt_intrinsics(gt_intrinsics_path, m)
    else:
        # Fallback: 从 gt_poses_path 推断（原有逻辑）
        intr = _load_gt_intrinsics(gt_poses_path, m) if gt_poses_path.exists() else None

    if intr is None:
        print("[mode_gtpose] ⚠️  No GT intrinsics found, using seed intrinsics")
        intr = _seed_intrinsics(clip_path, m)
    else:
        print(f"[mode_gtpose] ✅ Using GT intrinsics (shape: {intr.shape})")

    # 8. Scale 广播到所有帧（与官方对齐）
    # 官方: scales = [s] * m
    # 注意：官方使用 Python list，后续保存到 rec.scale_factors = [float(x) for x in scales]
    # 本地返回 numpy array，调用方可以根据需要转换
    scale_per_frame = np.full(m, float(s), dtype=np.float32)

    print(f"[mode_gtpose] ✅ GT-pose mode completed:")
    print(f"[mode_gtpose]    Output poses: {poses_c2w.shape}")
    print(f"[mode_gtpose]    Intrinsics: {intr.shape}")
    print(f"[mode_gtpose]    Scale: {s:.6f} (broadcasted to {m} frames)")

    return PoseArtifact(
        poses_c2w=poses_c2w,             # FULL length (m 帧)
        intrinsics=intr,                 # (m, 1, 4)
        scale_per_frame=scale_per_frame, # (m,) numpy array，等价于官方的 [s] * m
        depth_downsampled=None,          # GT-pose 不产生深度
    )


def _load_gt_poses(gt_poses_path: Path) -> np.ndarray:
    """加载 GT poses 为 (N, 4, 4)，与官方 _load_gt_poses 对齐

    官方参考: sana-wm-data-clean/sana_wm_data/pose/stage.py:140-163

    Accepts:
        - (N, 4, 4): 完整的 cam2world 矩阵
        - (N, 3, 4): 旋转+平移（补齐为 4x4）
        - (N, 3): 只有位置（构造单位旋转）

    Args:
        gt_poses_path: GT poses 文件路径

    Returns:
        (N, 4, 4) cam2world poses
    """
    arr = np.load(gt_poses_path).astype(np.float64)
    m = arr.shape[0]
    poses = np.tile(np.eye(4), (m, 1, 1))

    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        poses = arr
    elif arr.ndim == 3 and arr.shape[1:] == (3, 4):
        poses[:, :3, :4] = arr
    elif arr.ndim == 2 and arr.shape[1] == 3:
        poses[:, :3, 3] = arr
    else:
        raise ValueError(f"Unrecognized GT pose array shape {arr.shape}")

    return poses.astype(np.float32)


def _positions_to_poses(positions: np.ndarray) -> np.ndarray:
    """从相机位置构造 poses（单位旋转），与官方 _positions_to_poses 对齐

    官方参考: sana-wm-data-clean/sana_wm_data/pose/stage.py:118-122

    Args:
        positions: (N, 3) camera positions

    Returns:
        (N, 4, 4) cam2world poses with identity rotation
    """
    n = positions.shape[0]
    poses = np.tile(np.eye(4), (n, 1, 1)).astype(np.float64)
    poses[:, :3, 3] = positions
    return poses.astype(np.float32)


def _load_gt_intrinsics(gt_poses_path: Path, n_frames: int) -> np.ndarray | None:
    """加载 GT 内参（DL3DV: K_px），与官方 _load_real_intrinsics 对齐

    官方参考: sana-wm-data-clean/sana_wm_data/pose/stage.py:125-137

    对于 DL3DV 数据集，尝试从 camera.npz 加载 K_px (N, 4) [fx, fy, cx, cy]。

    目录结构假设:
        .../scene_name/gt_poses.npy          ← gt_poses_path
        .../scene_name.camera.npz            ← 包含 K_px

    或者:
        .../poses/gt_poses.npy               ← gt_poses_path
        .../scene_name.camera.npz            ← 上级目录

    或者（直接传入 camera.npz）:
        .../scene_name.camera.npz            ← gt_poses_path 直接是 .npz 文件

    Args:
        gt_poses_path: GT poses 文件路径或直接 camera.npz 路径
        n_frames: 目标帧数（用于子采样）

    Returns:
        (n_frames, 1, 4) intrinsics 或 None（如果未找到）
    """
    # 如果直接传入 .npz 文件，直接加载
    if gt_poses_path.suffix == '.npz' and gt_poses_path.exists():
        try:
            data = np.load(gt_poses_path)
            if 'K_px' in data:
                K_gt = data['K_px']  # (M, 4) [fx, fy, cx, cy]

                # 子采样到 n_frames
                if K_gt.ndim == 1:
                    K_gt = K_gt[None, :]
                if K_gt.shape[-1] != 4:
                    return None

                idx = np.linspace(0, K_gt.shape[0] - 1, n_frames).round().astype(int)
                return K_gt[idx][:, None, :].astype(np.float32)  # (n_frames, 1, 4)
        except Exception as e:
            print(f"[mode_gtpose] Failed to load {gt_poses_path}: {e}")
            return None

    # 原有逻辑：从 gt_poses_path 推断 camera.npz 位置
    gt_dir = gt_poses_path.parent

    # 尝试多种可能的 camera.npz 位置
    candidates = [
        gt_dir / f"{gt_dir.name}.camera.npz",      # .../scene_name/scene_name.camera.npz
        gt_dir.parent / f"{gt_dir.name}.camera.npz", # .../scene_name.camera.npz (poses 在子目录)
    ]

    # 特殊处理：如果路径中包含 DL3DV 的长命名格式
    # 例如: DL3DV-ALL-2K_10K__<hash>__images_2
    if gt_dir.name.startswith("DL3DV"):
        candidates.append(gt_dir / f"{gt_dir.name}.camera.npz")

    for camera_npz in candidates:
        if camera_npz.exists():
            try:
                data = np.load(camera_npz)
                if 'K_px' in data:
                    K_gt = data['K_px']  # (M, 4) [fx, fy, cx, cy]

                    # 子采样到 n_frames（与官方 _load_real_intrinsics 对齐）
                    # 官方: idx = np.linspace(0, K.shape[0] - 1, n).round().astype(int)
                    if K_gt.ndim == 1:
                        K_gt = K_gt[None, :]
                    if K_gt.shape[-1] != 4:
                        continue

                    idx = np.linspace(0, K_gt.shape[0] - 1, n_frames).round().astype(int)
                    return K_gt[idx][:, None, :].astype(np.float32)  # (n_frames, 1, 4)
            except Exception as e:
                print(f"[mode_gtpose] Failed to load {camera_npz}: {e}")
                continue

    return None


def _seed_intrinsics(video_path: Path, n_frames: int) -> np.ndarray:
    """种子内参（moderate FoV 假设），与官方 _seed_intrinsics 对齐

    官方参考: sana-wm-data-clean/sana_wm_data/pose/stage.py:33-37
    官方逻辑: fx = fy = 0.9 * w  (moderate FoV starting guess)

    Args:
        video_path: 视频文件路径（用于读取分辨率）
        n_frames: 目标帧数

    Returns:
        (n_frames, 1, 4) intrinsics
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()

    # 官方逻辑
    fx = fy = 0.9 * w  # ~ moderate FoV starting guess
    cx, cy = w / 2.0, h / 2.0

    K = np.array([[fx, fy, cx, cy]], dtype=np.float32)
    return np.tile(K, (n_frames, 1, 1))  # (n_frames, 1, 4)
