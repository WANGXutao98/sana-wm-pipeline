# src/sana_wm_pipeline/qc/stage2_deep_extracted.py
"""Stage 2: deep targeted checks - 解压数据版本（直接读取文件，不依赖 tar）"""
from __future__ import annotations
import io, json, random, tempfile
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
_BLACK_BRIGHTNESS = 10  # pixel mean < this → black frame


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
        return -1  # -1 indicates error


def check_black_frame_ratio(video_bytes: bytes, brightness_threshold: int = _BLACK_BRIGHTNESS) -> float:
    """Fraction of frames with mean brightness < threshold."""
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
    """直接从文件运行 PySceneDetect，返回切换次数"""
    if not video_path.exists():
        return -1
    try:
        return count_scene_cuts(str(video_path), threshold=threshold)
    except Exception:
        return -1


def find_sample_files(sample_id: str, data_root: Path) -> Optional[dict[str, Path]]:
    """
    在解压目录中查找样本文件

    扫描模式: data_root/final_wds-*/wds-*/w*/shard-*/{sample_id}.*

    Returns:
        包含文件路径的字典，如果找不到则返回 None
        {
            'mp4': Path,
            'poses': Path,
            'shard_dir': Path,
        }
    """
    # 从 sample_id 提取数据集前缀（如 "SpatialVID-hq" 从 "SpatialVID-hq_UUID"）
    # 但有些数据集名称包含下划线，需要更智能的匹配

    # 策略：直接搜索包含该 sample_id 的 .mp4 文件
    pattern = f"final_wds-*/wds-*/w*/shard-*/{sample_id}.mp4"

    matches = list(data_root.glob(pattern))
    if matches:
        mp4_path = matches[0]
        shard_dir = mp4_path.parent

        return {
            'mp4': mp4_path,
            'poses': mp4_path.with_suffix('.poses_c2w.npy'),
            'shard_dir': shard_dir,
        }

    return None


def deep_check_sample_extracted(
    sample_id: str,
    data_root: Path,
    group_name: str
) -> Optional[dict[str, Any]]:
    """
    从解压目录读取文件进行深度检查

    Args:
        sample_id: 样本 ID
        data_root: 数据根目录（包含 final_wds-* 目录）
        group_name: 数据集分组名

    Returns:
        检查结果字典，如果样本不存在则返回 None
    """
    from sana_wm_pipeline.qc.group_config import get_group_config

    cfg = get_group_config(group_name)
    stage2: dict[str, Any] = {
        "video_T": -1, "video_T_matches_npy": None,
        "black_frame_ratio": None, "scene_cuts": None,
        "traj_frozen": None, "frozen_ratio": None, "reasons": [],
    }

    # 查找样本文件
    files = find_sample_files(sample_id, data_root)
    if files is None:
        # 文件不存在，返回 None（调用方会跳过）
        return None

    mp4_path = files['mp4']
    poses_path = files['poses']

    try:
        # 检查视频文件
        if not mp4_path.exists():
            stage2["reasons"].append("mp4_not_found")
            return {"sample_id": sample_id, "stage2": stage2}

        # 读取视频字节（用于帧数统计和黑帧检测）
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

        # Scene cuts (only for groups with max_scene_cuts limit)
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


def _worker_fn_extracted(args: tuple) -> Optional[dict]:
    """Worker 函数：从解压目录处理样本"""
    sample_id, data_root, group_name = args
    return deep_check_sample_extracted(sample_id, Path(data_root), group_name)


def run_stage2_extracted(
    stage1_jsonl: Path,
    output_jsonl: Path,
    data_root: Path,
    sample_frac: float = 1.0,  # 默认全量处理
    n_workers: int = 16,
) -> int:
    """
    Stage 2 深度检查 - 解压数据版本

    Args:
        stage1_jsonl: Stage 1 结果文件
        output_jsonl: Stage 2 输出文件
        data_root: 数据根目录（包含 final_wds-* 解压目录）
        sample_frac: Pass 样本采样比例（Flag 样本始终 100% 检查）
        n_workers: 并发进程数

    Returns:
        处理的样本数
    """
    stage1_jsonl = Path(stage1_jsonl)
    output_jsonl = Path(output_jsonl)
    data_root = Path(data_root)

    if not stage1_jsonl.exists():
        raise FileNotFoundError(f"stage1_jsonl not found: {stage1_jsonl}")
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    selected: list[tuple[str, str, str]] = []  # (sample_id, data_root, group_name)
    all_records: dict[str, dict] = {}
    rng = random.Random(42)

    # 读取 Stage 1 结果并选择样本
    with open(stage1_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec["sample_id"]
            all_records[sid] = rec
            verdict = rec.get("verdict", "pass")

            # 跳过 Stage 1 失败的样本
            if verdict == "fail":
                continue

            # Flag 样本 100% 检查，Pass 样本按采样率
            if verdict == "flag" or rng.random() < sample_frac:
                selected.append((sid, str(data_root), rec.get("group", "")))

    if not selected:
        output_jsonl.write_text("", encoding="utf-8")
        return 0

    print(f"[stage2] 开始处理 {len(selected)} 个样本（并发: {n_workers}）")

    # 多进程处理
    processed = 0
    skipped = 0

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        n_proc = min(max(1, n_workers), len(selected))
        with Pool(processes=n_proc) as pool:
            for s2_rec in pool.imap_unordered(_worker_fn_extracted, selected):
                # 如果返回 None，说明文件不存在，跳过
                if s2_rec is None:
                    skipped += 1
                    continue

                # 合并 Stage 1 和 Stage 2 结果
                merged = dict(all_records[s2_rec["sample_id"]])
                merged["stage2"] = s2_rec["stage2"]
                fout.write(json.dumps(merged, ensure_ascii=False) + "\n")
                processed += 1

                # 每 1000 个样本输出进度
                if processed % 1000 == 0:
                    print(f"[stage2] 已处理: {processed}, 已跳过: {skipped}")

    print(f"[stage2] 完成！处理: {processed}, 跳过: {skipped}, 总计: {len(selected)}")
    return processed
