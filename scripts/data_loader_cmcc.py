#!/usr/bin/env python3
"""
CMCC 数据加载器 - Stage 3 专用

功能：
1. 扫描解压后的 shard 目录，建立样本索引
2. 加载 sample_completeness.csv，过滤完整样本
3. 提供快速的样本 ID -> 文件路径映射
"""

import csv
from pathlib import Path
from typing import Dict, Set, Optional
import logging

logger = logging.getLogger(__name__)


class Stage3DataLoaderCMCC:
    """CMCC 数据加载器 - 为 Stage 3 提供样本路径映射"""

    def __init__(
        self,
        data_root: str | Path,
        completeness_csv: Optional[str | Path] = None,
        build_index: bool = True
    ):
        """
        初始化数据加载器

        Args:
            data_root: 数据根目录（包含 final_wds-* 目录）
            completeness_csv: sample_completeness.csv 路径（可选）
            build_index: 是否立即构建索引（默认 True）
        """
        self.data_root = Path(data_root)
        self.sample_index: Dict[str, Dict[str, Path]] = {}
        self.complete_samples: Set[str] = set()

        if not self.data_root.exists():
            raise FileNotFoundError(f"数据根目录不存在: {self.data_root}")

        # 构建样本索引
        if build_index:
            logger.info(f"开始扫描数据目录: {self.data_root}")
            self._build_index()
            logger.info(f"索引构建完成，共 {len(self.sample_index)} 个样本")

        # 加载完整性标记
        if completeness_csv:
            self._load_completeness(completeness_csv)
            logger.info(f"完整样本数: {len(self.complete_samples)}")

    def _build_index(self):
        """扫描所有 shard 目录，建立 sample_id -> 文件路径 映射"""
        # 查找所有解压后的 shard 目录
        # 模式: final_wds-*/wds-*/w*/shard-*/
        shard_pattern = "final_wds-*/wds-*/w*/shard-*"

        for shard_dir in self.data_root.glob(shard_pattern):
            if not shard_dir.is_dir():
                continue

            # 排除 .tar 文件和 .SUCCESS 标记
            if shard_dir.suffix in ['.tar', '.SUCCESS']:
                continue

            # 扫描该 shard 下的所有 .mp4 文件
            for mp4_file in shard_dir.glob("*.mp4"):
                sample_id = mp4_file.stem  # 去掉 .mp4 扩展名

                # 构建文件路径映射
                self.sample_index[sample_id] = {
                    'mp4': mp4_file,
                    'caption': mp4_file.with_suffix('.caption.txt'),
                    'intrinsics': mp4_file.with_suffix('.intrinsics.npy'),
                    'poses': mp4_file.with_suffix('.poses_c2w.npy'),
                    'meta': mp4_file.with_suffix('.meta.json'),
                    'scale': mp4_file.with_suffix('.scale.npy'),
                    'shard_dir': shard_dir,
                }

    def _load_completeness(self, csv_path: str | Path):
        """从 CSV 加载 complete=TRUE 的样本 ID"""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if row['complete'].strip().upper() == 'TRUE':
                    self.complete_samples.add(row['sample_id'].strip())

    def get_sample_files(self, sample_id: str) -> Optional[Dict[str, Path]]:
        """
        根据样本 ID 返回文件路径

        Returns:
            包含文件路径的字典，如果样本不存在则返回 None
            {
                'mp4': Path(...),
                'caption': Path(...),
                'shard_dir': Path(...),
                ...
            }
        """
        return self.sample_index.get(sample_id)

    def is_complete(self, sample_id: str) -> bool:
        """检查样本是否完整（基于 CSV）"""
        if not self.complete_samples:
            # 如果没有加载 CSV，默认认为所有样本完整
            return True
        return sample_id in self.complete_samples

    def get_processable_samples(self) -> list[str]:
        """
        返回可处理的样本 ID 列表（存在于索引中 且 标记为完整）

        Returns:
            样本 ID 列表
        """
        if self.complete_samples:
            # 如果有完整性标记，返回交集
            return [
                sample_id for sample_id in self.sample_index.keys()
                if sample_id in self.complete_samples
            ]
        else:
            # 否则返回所有索引中的样本
            return list(self.sample_index.keys())

    def verify_sample_files(self, sample_id: str) -> tuple[bool, list[str]]:
        """
        验证样本的必需文件是否存在

        Returns:
            (是否通过, 缺失文件列表)
        """
        files = self.get_sample_files(sample_id)
        if not files:
            return False, ["sample_not_in_index"]

        missing = []
        # 检查 Stage 3 必需的文件
        required = ['mp4', 'caption']
        for key in required:
            if not files[key].exists():
                missing.append(str(files[key]))

        return len(missing) == 0, missing

    def get_stats(self) -> dict:
        """返回统计信息"""
        return {
            'total_samples_indexed': len(self.sample_index),
            'complete_samples': len(self.complete_samples),
            'processable_samples': len(self.get_processable_samples()),
        }


def demo():
    """演示用法"""
    import sys

    if len(sys.argv) < 2:
        print("用法: python data_loader_cmcc.py <data_root> [completeness_csv]")
        print("示例: python data_loader_cmcc.py /root/work/.../jdvbbfb_output sample_completeness.csv")
        return

    data_root = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else None

    # 初始化加载器
    print(f"初始化数据加载器...")
    loader = Stage3DataLoaderCMCC(data_root, csv_path)

    # 显示统计
    stats = loader.get_stats()
    print(f"\n统计信息:")
    print(f"  索引样本数: {stats['total_samples_indexed']}")
    print(f"  完整样本数: {stats['complete_samples']}")
    print(f"  可处理样本数: {stats['processable_samples']}")

    # 显示前 5 个样本
    processable = loader.get_processable_samples()
    print(f"\n前 5 个可处理样本:")
    for sample_id in processable[:5]:
        files = loader.get_sample_files(sample_id)
        print(f"  {sample_id}")
        print(f"    视频: {files['mp4']}")
        print(f"    描述: {files['caption']}")

        # 验证文件存在
        valid, missing = loader.verify_sample_files(sample_id)
        if valid:
            print(f"    状态: ✅ 完整")
        else:
            print(f"    状态: ❌ 缺失文件: {missing}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    demo()
