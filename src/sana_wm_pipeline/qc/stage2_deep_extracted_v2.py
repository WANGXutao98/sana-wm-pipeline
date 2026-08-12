# src/sana_wm_pipeline/qc/stage2_deep_extracted_v2.py
"""Stage 2: deep targeted checks - 解压数据版本 v2（修复索引性能问题）"""
from __future__ import annotations
import io, json, random
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional
import numpy as np

try:
    import av
    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False

from sana_wm_pipeline.stage04_filter.scene_cut import count_scene_cuts

_FROZEN_THRESHOLD = 1e-4
_BLACK_BRIGHTNESS = 10


def count_video_frames_av(video_bytes: bytes) -> int:
    if not _AV_AVAILABLE:
        raise RuntimeError("pip install av")
    if not video_bytes:
        return 0
    try:
        with av.open(io.BytesIO(video_bytes)) as container:
            if not container.streams.video:
                return 0
            stream = container.streams.video[0]
            if stream.frames > 0:
                return int(stream.frames)
            return sum(1 for _ in container.decode(video=0))
    except Exception:
        return -1


def check_black_frame_ratio(video_bytes: bytes, brightness_threshold: int = _BLACK_BRIGHTNESS) -> float:
    if not _AV_AVAILABLE or not video_bytes:
        return 0.0
    total, black = 0, 0
    try:
        with av.open(io.BytesIO(video_bytes)) as container:
            for frame in container.decode(video=0):
                arr = frame.to_ndarray(format="gray")
                if arr.mean() < brightness_threshold:
                    black += 1
                total += 1
    except Exception:
        return 0.0
    return black / total if total else 0.0


def check_trajectory_frozen(poses_c2w: np.ndarray, frozen_threshold: float = _FROZEN_THRESHOLD) -> tuple[bool, float]:
    t = poses_c2w[:, :3, 3].astype(np.float64)
    if len(t) < 2:
        return False, 0.0
    steps = np.linalg.norm(np.diff(t, axis=0), axis=1)
    ratio = float((steps < frozen_threshold).mean())
    return ratio > 0.5, ratio


def count_scene_cuts_from_file(video_path: Path, threshold: float = 27.0) -> int:
    if not video_path.exists():
        return -1
    try:
        return count_scene_cuts(str(video_path), threshold=threshold)
    except Exception:
        return -1


def build_sample_index(data_root: Path, sample_ids: set[str]) -> dict[str, dict[str, Path]]:
    """
    一次性构建所有样本的索引（避免重复 glob）

    Args:
        data_root: 数据根目录
        sample_ids: 需要处理的样本 ID 集合

    Returns:
        {sample_id: {'mp4': Path, 'poses': Path, 'shard_dir': Path}}
    """
    print(f"[index] 开始构建样本索引（{len(sample_ids)} 个样本）...")
    index = {}

    # 扫描所有解压后的 shard 目录
    pattern = "final_wds-*/wds-*/w*/shard-*"
    shard_count = 0

    for shard_dir in data_root.glob(pattern):
        if not shard_dir.is_dir():
            continue
        if shard_dir.suffix in ['.tar', '.SUCCESS']:
            continue

        shard_count += 1

        # 扫描该 shard 下的所有 .mp4 文件
        for mp4_file in shard_dir.glob("*.mp4"):
            sample_id = mp4_file.stem

            # 只索引需要处理的样本
            if sample_id in sample_ids:
                index[sample_id] = {
                    'mp4': mp4_file,
                    'poses': mp4_file.with_suffix('.poses_c2w.npy'),
                    'shard_dir': shard_dir,
                }

        # 每 100 个 shard 输出一次进度
        if shard_count % 100 == 0:
            print(f"[index] 已扫描 {shard_count} 个 shard，已索引 {len(index)} 个样本")

    print(f"[index] 索引完成！扫描了 {shard_count} 个 shard，找到 {len(index)} / {len(sample_ids)} 个样本")
    return index


def deep_check_sample_with_index(
    sample_id: str,
    file_paths: dict[str, Path],
    group_name: str
) -> Optional[dict[str, Any]]:
    """
    使用预构建的索引进行深度检查

    Args:
        sample_id: 样本 ID
        file_paths: 预构建的文件路径 {'mp4': Path, 'poses': Path}
        group_name: 数据集分组名

    Returns:
        检查结果字典
    """
    from sana_wm_pipeline.qc.group_config import get_group_config

    cfg = get_group_config(group_name)
    stage2: dict[str, Any] = {
        "video_T": -1, "video_T_matches_npy": None,
        "black_frame_ratio": None, "scene_cuts": None,
        "traj_frozen": None, "frozen_ratio": None, "reasons": [],
    }

    mp4_path = file_paths['mp4']
    poses_path = file_paths['poses']

    try:
        # 检查视频文件
        if not mp4_path.exists():
            stage2["reasons"].append("mp4_not_found")
            return {"sample_id": sample_id, "stage2": stage2}

        # 读取视频字节
        try:
            video_bytes = mp4_path.read_bytes()
        except Exception as e:
            stage2["reasons"].append(f"mp4_read_error: {e}")
            return {"sample_id": sample_id, "stage2": stage2}

        # Frame count
        try:
            video_T = count_video_frames_av(video_bytes)
            stage2["video_T"] = video_T
        except Exception as e:
            stage2["reasons"].append(f"av_error: {e}")
            video_T = -1

        # Black frame ratio
        try:
            stage2["black_frame_ratio"] = round(check_black_frame_ratio(video_bytes), 4)
            if stage2["black_frame_ratio"] > 0.30:
                stage2["reasons"].append(f"black_frame_ratio={stage2['black_frame_ratio']:.2f} > 0.30")
        except Exception:
            pass

        # Scene cuts
        if cfg.max_scene_cuts is not None:
            n_cuts = count_scene_cuts_from_file(mp4_path)
            if n_cuts < 0:
                stage2["scene_cuts"] = None
                stage2["reasons"].append("scene_cuts_error: detection failed")
            else:
                stage2["scene_cuts"] = n_cuts
                if n_cuts > cfg.max_scene_cuts:
                    stage2["reasons"].append(f"scene_cuts={n_cuts} > {cfg.max_scene_cuts}")

        # Trajectory frozen
        try:
            if not poses_path.exists():
                stage2["reasons"].append("poses_not_found")
            else:
                poses = np.load(poses_path)
                npy_T = int(poses.shape[0])
                stage2["video_T_matches_npy"] = (video_T == npy_T)
                if video_T > 0 and video_T != npy_T:
                    stage2["reasons"].append(f"video_npy_T_mismatch: video={video_T} npy={npy_T}")
                frozen, ratio = check_trajectory_frozen(poses)
                stage2["traj_frozen"] = frozen
                stage2["frozen_ratio"] = round(ratio, 4)
                if frozen:
                    stage2["reasons"].append(f"traj_frozen: {ratio:.1%} frames stationary")
        except Exception as e:
            stage2["reasons"].append(f"poses_error: {e}")

    except Exception as e:
        stage2["reasons"].append(f"processing_error: {e}")

    return {"sample_id": sample_id, "stage2": stage2}


def _worker_fn_with_index(args: tuple) -> Optional[dict]:
    """Worker 函数：使用预构建索引"""
    sample_id, file_paths, group_name = args
    return deep_check_sample_with_index(sample_id, file_paths, group_name)


def run_stage2_extracted_v2(
    stage1_jsonl: Path,
    output_jsonl: Path,
    data_root: Path,
    sample_frac: float = 1.0,
    n_workers: int = 16,
) -> int:
    """
    Stage 2 深度检查 - 解压数据版本 v2（修复索引性能）

    主要改进：
    1. 一次性构建所有样本索引（避免每个样本都 glob）
    2. Worker 直接使用索引，不需要文件查找
    """
    stage1_jsonl = Path(stage1_jsonl)
    output_jsonl = Path(output_jsonl)
    data_root = Path(data_root)

    if not stage1_jsonl.exists():
        raise FileNotFoundError(f"stage1_jsonl not found: {stage1_jsonl}")
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # ===== 步骤 1：读取 Stage 1 结果并选择样本 =====
    selected_ids: set[str] = set()
    all_records: dict[str, dict] = {}
    sample_groups: dict[str, str] = {}  # sample_id -> group_name
    rng = random.Random(42)

    print(f"[stage2] 读取 Stage 1 结果: {stage1_jsonl}")
    with open(stage1_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec["sample_id"]
            all_records[sid] = rec
            sample_groups[sid] = rec.get("group", "")
            verdict = rec.get("verdict", "pass")

            if verdict == "fail":
                continue

            if verdict == "flag" or rng.random() < sample_frac:
                selected_ids.add(sid)

    if not selected_ids:
        output_jsonl.write_text("", encoding="utf-8")
        return 0

    print(f"[stage2] 选择了 {len(selected_ids)} 个样本进行检查")

    # ===== 步骤 2：构建样本索引（一次性 glob）=====
    sample_index = build_sample_index(data_root, selected_ids)

    # 找出缺失的样本
    missing = selected_ids - set(sample_index.keys())
    if missing:
        print(f"[stage2] ⚠️  {len(missing)} 个样本在数据目录中未找到（将跳过）")

    # ===== 步骤 3：准备任务列表 =====
    tasks = []
    for sid in selected_ids:
        if sid in sample_index:
            tasks.append((sid, sample_index[sid], sample_groups[sid]))

    print(f"[stage2] 开始处理 {len(tasks)} 个样本（并发: {n_workers}）")

    # ===== 步骤 4：多进程处理 =====
    processed = 0
    skipped = len(missing)

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        n_proc = min(max(1, n_workers), len(tasks))
        with Pool(processes=n_proc) as pool:
            for s2_rec in pool.imap_unordered(_worker_fn_with_index, tasks):
                if s2_rec is None:
                    skipped += 1
                    continue

                # 合并结果
                merged = dict(all_records[s2_rec["sample_id"]])
                merged["stage2"] = s2_rec["stage2"]
                fout.write(json.dumps(merged, ensure_ascii=False) + "\n")
                fout.flush()
                processed += 1

                if processed % 1000 == 0:
                    print(f"[stage2] 已处理: {processed}, 已跳过: {skipped}")

    print(f"[stage2] 完成！处理: {processed}, 跳过: {skipped}, 总计: {len(selected_ids)}")
    return processed
