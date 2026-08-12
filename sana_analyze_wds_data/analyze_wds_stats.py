#!/usr/bin/env python3
"""
WebDataset Statistics Analyzer

Scans WebDataset tar files and generates statistics reports.
Designed for analyzing sana-wm pipeline output.

Usage:
    python analyze_wds_stats.py --input-dir /path/to/jdvbbfb_output
"""

import argparse
import csv
import json
import os
import tarfile
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional
import sys


@dataclass
class ShardStats:
    """Statistics for a single shard (tar file)"""
    dataset: str
    worker: str
    shard: str
    samples: int
    size_bytes: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 * 1024 * 1024)


@dataclass
class DatasetStats:
    """Aggregated statistics for a dataset"""
    name: str
    total_samples: int
    total_bytes: int
    total_shards: int
    workers: int
    shards: List[ShardStats]

    @property
    def size_gb(self) -> float:
        return self.total_bytes / (1024 * 1024 * 1024)

    @property
    def avg_samples_per_shard(self) -> float:
        return self.total_samples / self.total_shards if self.total_shards > 0 else 0


class TarScanner:
    """Scans tar files and extracts metadata without decompression"""

    @staticmethod
    def scan_tar(tar_path: str) -> tuple[Set[str], int]:
        """
        Scan a tar file and extract sample keys.

        Args:
            tar_path: Path to the tar file

        Returns:
            Tuple of (set of sample keys, file size in bytes)
        """
        keys = set()
        try:
            with tarfile.open(tar_path, 'r') as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        # Extract key from filename (before first dot)
                        filename = os.path.basename(member.name)
                        if '.' in filename:
                            key = filename.split('.')[0]
                            keys.add(key)

            # Get file size
            size_bytes = os.path.getsize(tar_path)
            return keys, size_bytes

        except Exception as e:
            print(f"  Warning: Failed to read {tar_path}: {e}", file=sys.stderr)
            return set(), 0


class SampleIdentifier:
    """Identifies unique samples from file listings"""

    @staticmethod
    def extract_keys(members: List[str]) -> Set[str]:
        """
        Extract unique sample keys from file list.

        WebDataset convention: files like "0001234.jpg", "0001234.json"
        share the same key "0001234".

        Args:
            members: List of filenames

        Returns:
            Set of unique sample keys
        """
        keys = set()
        for member in members:
            filename = os.path.basename(member)
            if '.' in filename:
                key = filename.split('.')[0]
                keys.add(key)
        return keys


class StatisticsAggregator:
    """Aggregates statistics from multiple shards"""

    def __init__(self):
        self.shards: List[ShardStats] = []
        self.datasets: Dict[str, List[ShardStats]] = defaultdict(list)

    def add_shard(self, shard_stats: ShardStats):
        """Add statistics for a single shard"""
        self.shards.append(shard_stats)
        self.datasets[shard_stats.dataset].append(shard_stats)

    def get_dataset_stats(self, dataset_name: str) -> Optional[DatasetStats]:
        """Get aggregated statistics for a dataset"""
        if dataset_name not in self.datasets:
            return None

        shards = self.datasets[dataset_name]
        total_samples = sum(s.samples for s in shards)
        total_bytes = sum(s.size_bytes for s in shards)
        workers = len(set(s.worker for s in shards))

        return DatasetStats(
            name=dataset_name,
            total_samples=total_samples,
            total_bytes=total_bytes,
            total_shards=len(shards),
            workers=workers,
            shards=shards
        )

    def get_all_dataset_stats(self) -> List[DatasetStats]:
        """Get statistics for all datasets"""
        return [self.get_dataset_stats(name) for name in sorted(self.datasets.keys())]

    def get_global_stats(self) -> dict:
        """Get global statistics across all datasets"""
        return {
            'total_samples': sum(s.samples for s in self.shards),
            'total_bytes': sum(s.size_bytes for s in self.shards),
            'total_shards': len(self.shards),
            'num_datasets': len(self.datasets)
        }


class ReportGenerator:
    """Generates various report formats"""

    def __init__(self, aggregator: StatisticsAggregator, output_dir: str):
        self.aggregator = aggregator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_csv(self):
        """Generate CSV report with shard-level details"""
        csv_path = self.output_dir / 'shard_statistics.csv'

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'worker', 'shard', 'samples', 'size_bytes', 'size_mb', 'size_gb'])

            for shard in self.aggregator.shards:
                writer.writerow([
                    shard.dataset,
                    shard.worker,
                    shard.shard,
                    shard.samples,
                    shard.size_bytes,
                    f"{shard.size_mb:.2f}",
                    f"{shard.size_gb:.4f}"
                ])

        print(f"✓ CSV report saved to: {csv_path}")

    def generate_json(self, input_dir: str):
        """Generate JSON report with complete statistics"""
        json_path = self.output_dir / 'dataset_statistics.json'

        datasets_dict = {}
        for ds_stats in self.aggregator.get_all_dataset_stats():
            datasets_dict[ds_stats.name] = {
                'total_samples': ds_stats.total_samples,
                'total_bytes': ds_stats.total_bytes,
                'total_shards': ds_stats.total_shards,
                'workers': ds_stats.workers,
                'avg_samples_per_shard': round(ds_stats.avg_samples_per_shard, 1),
                'size_gb': round(ds_stats.size_gb, 2),
                'shards': [
                    {
                        'worker': s.worker,
                        'shard': s.shard,
                        'samples': s.samples,
                        'size_bytes': s.size_bytes
                    }
                    for s in ds_stats.shards
                ]
            }

        global_stats = self.aggregator.get_global_stats()

        report = {
            'scan_time': datetime.now().isoformat(),
            'input_dir': input_dir,
            'total_samples': global_stats['total_samples'],
            'total_bytes': global_stats['total_bytes'],
            'total_shards': global_stats['total_shards'],
            'datasets': datasets_dict
        }

        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✓ JSON report saved to: {json_path}")

    def generate_markdown(self, original_stats: Optional[dict] = None):
        """Generate Markdown comparison report"""
        md_path = self.output_dir / 'comparison_report.md'

        lines = [
            "# 数据处理统计报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 总体统计",
            ""
        ]

        global_stats = self.aggregator.get_global_stats()
        total_gb = global_stats['total_bytes'] / (1024**3)
        total_tb = global_stats['total_bytes'] / (1024**4)

        lines.extend([
            "| 指标 | 数值 |",
            "|------|------|",
            f"| 总样本数 | {global_stats['total_samples']:,} |",
            f"| 总数据量 | {total_gb:.2f} GB ({total_tb:.3f} TB) |",
            f"| 总 Shard 数 | {global_stats['total_shards']:,} |",
            f"| 数据集数量 | {global_stats['num_datasets']} |",
            ""
        ])

        lines.extend([
            "## 各数据集统计",
            "",
            "| 数据集 | 样本数 | 数据量(GB) | Shard数 | Worker数 |",
            "|--------|--------|-----------|---------|----------|"
        ])

        for ds_stats in self.aggregator.get_all_dataset_stats():
            lines.append(
                f"| {ds_stats.name} | {ds_stats.total_samples:,} | "
                f"{ds_stats.size_gb:.2f} | {ds_stats.total_shards} | {ds_stats.workers} |"
            )

        # Add comparison if original stats provided
        if original_stats:
            lines.extend([
                "",
                "## 与原始数据对比",
                "",
                "| 数据集 | 已处理 | 原始 | 完成率 |",
                "|--------|--------|------|--------|"
            ])

            for ds_stats in self.aggregator.get_all_dataset_stats():
                ds_name_clean = ds_stats.name.replace('final_wds-', '')
                original_count = original_stats.get(ds_name_clean, 0)
                if original_count > 0:
                    completion = (ds_stats.total_samples / original_count) * 100
                    status = "✓" if completion >= 99.0 else "⚠"
                    lines.append(
                        f"| {ds_name_clean} | {ds_stats.total_samples:,} | "
                        f"{original_count:,} | {completion:.1f}% {status} |"
                    )

        with open(md_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"✓ Markdown report saved to: {md_path}")

    def print_summary(self, original_stats: Optional[dict] = None):
        """Print summary to terminal"""
        print("\n" + "=" * 80)
        print("数据集处理统计报告")
        print("=" * 80)

        global_stats = self.aggregator.get_global_stats()
        total_gb = global_stats['total_bytes'] / (1024**3)
        total_tb = global_stats['total_bytes'] / (1024**4)

        print(f"\n总样本数: {global_stats['total_samples']:,}")
        print(f"总数据量: {total_gb:.2f} GB ({total_tb:.3f} TB)")
        print(f"总 Shard 数: {global_stats['total_shards']:,}")
        print(f"数据集数量: {global_stats['num_datasets']}")

        print("\n" + "=" * 80)
        print("各数据集统计")
        print("=" * 80)

        for ds_stats in self.aggregator.get_all_dataset_stats():
            print(f"\n【{ds_stats.name}】")
            print(f"  样本数: {ds_stats.total_samples:,}")
            print(f"  数据量: {ds_stats.size_gb:.2f} GB")
            print(f"  Shard 数: {ds_stats.total_shards}")
            print(f"  Worker 数: {ds_stats.workers}")
            print(f"  平均每 shard: {ds_stats.avg_samples_per_shard:.1f} samples")

        if original_stats:
            print("\n" + "=" * 80)
            print("与原始数据对比")
            print("=" * 80)
            print(f"\n{'数据集':<30} {'已处理':>12} {'原始':>12} {'完成率':>10}")
            print("-" * 80)

            for ds_stats in self.aggregator.get_all_dataset_stats():
                ds_name_clean = ds_stats.name.replace('final_wds-', '')
                original_count = original_stats.get(ds_name_clean, 0)
                if original_count > 0:
                    completion = (ds_stats.total_samples / original_count) * 100
                    status = "✓" if completion >= 99.0 else "⚠"
                    print(f"{ds_name_clean:<30} {ds_stats.total_samples:>12,} "
                          f"{original_count:>12,} {completion:>9.1f}% {status}")


def scan_dataset_directory(dataset_path: Path, dataset_name: str,
                          aggregator: StatisticsAggregator,
                          verbose: bool = False):
    """
    Scan a single dataset directory and collect statistics.

    Args:
        dataset_path: Path to the dataset directory (e.g., final_wds-SpatialVID-hq)
        dataset_name: Name of the dataset
        aggregator: Statistics aggregator to collect results
        verbose: Print detailed progress
    """
    # Find the wds-* subdirectory
    wds_dirs = list(dataset_path.glob('wds-*'))
    if not wds_dirs:
        print(f"  Warning: No wds-* directory found in {dataset_path}")
        return

    wds_dir = wds_dirs[0]

    # Find all worker directories (w000, w001, ...)
    worker_dirs = sorted([d for d in wds_dir.iterdir() if d.is_dir() and d.name.startswith('w')])

    if not worker_dirs:
        print(f"  Warning: No worker directories found in {wds_dir}")
        return

    total_shards = 0
    for worker_dir in worker_dirs:
        worker_name = worker_dir.name

        # Find all tar files
        tar_files = sorted(worker_dir.glob('*.tar'))

        for tar_file in tar_files:
            if verbose:
                print(f"    Scanning {worker_name}/{tar_file.name}...", end='\r')

            keys, size_bytes = TarScanner.scan_tar(str(tar_file))

            shard_stats = ShardStats(
                dataset=dataset_name,
                worker=worker_name,
                shard=tar_file.name,
                samples=len(keys),
                size_bytes=size_bytes
            )

            aggregator.add_shard(shard_stats)
            total_shards += 1

    if verbose:
        print(f"  ✓ Scanned {total_shards} shards")


def load_original_stats(stats_path: str) -> Optional[dict]:
    """Load original dataset statistics from JSON file"""
    if not stats_path or not os.path.exists(stats_path):
        return None

    try:
        with open(stats_path, 'r') as f:
            data = json.load(f)

        # Extract sample counts by dataset name
        stats = {}
        if 'datasets' in data:
            for ds_name, ds_data in data['datasets'].items():
                stats[ds_name] = ds_data.get('total_samples', 0)

        return stats
    except Exception as e:
        print(f"Warning: Failed to load original stats from {stats_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Analyze WebDataset tar files and generate statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python analyze_wds_stats.py --input-dir /path/to/jdvbbfb_output

  # With original stats comparison
  python analyze_wds_stats.py --input-dir /path/to/jdvbbfb_output \\
    --original-stats original_stats.json

  # Verbose mode
  python analyze_wds_stats.py --input-dir /path/to/jdvbbfb_output --verbose
        """
    )

    parser.add_argument('--input-dir', required=True,
                       help='Input directory containing final_wds-* subdirectories')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for reports (default: current directory)')
    parser.add_argument('--original-stats', default=None,
                       help='Path to original statistics JSON file for comparison')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress information')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Load original stats if provided
    original_stats = load_original_stats(args.original_stats)

    print("=" * 80)
    print("WebDataset Statistics Analyzer")
    print("=" * 80)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Scan started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Find all dataset directories
    dataset_dirs = sorted([d for d in input_dir.iterdir()
                          if d.is_dir() and d.name.startswith('final_wds-')])

    if not dataset_dirs:
        print(f"Error: No final_wds-* directories found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(dataset_dirs)} dataset(s):")
    for d in dataset_dirs:
        print(f"  - {d.name}")
    print()

    # Scan all datasets
    aggregator = StatisticsAggregator()
    start_time = datetime.now()

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        print(f"Processing {dataset_name}...")
        scan_dataset_directory(dataset_dir, dataset_name, aggregator, args.verbose)

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print(f"\n✓ Scan completed in {elapsed:.1f} seconds")

    # Generate reports
    print("\nGenerating reports...")
    reporter = ReportGenerator(aggregator, args.output_dir)
    reporter.generate_csv()
    reporter.generate_json(str(input_dir))
    reporter.generate_markdown(original_stats)

    # Print summary
    reporter.print_summary(original_stats)

    print("\n" + "=" * 80)
    print("All reports generated successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
