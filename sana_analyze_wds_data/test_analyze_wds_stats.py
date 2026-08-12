#!/usr/bin/env python3
"""
Quick test script for WebDataset Statistics Analyzer
Creates a minimal test dataset and runs the analyzer.
"""

import os
import sys
import tarfile
import tempfile
import shutil
from pathlib import Path


def create_test_tar(tar_path, num_samples=5):
    """Create a test tar file with dummy samples"""
    import io
    with tarfile.open(tar_path, 'w') as tar:
        for i in range(num_samples):
            # Create dummy files for each sample
            sample_key = f"{i:06d}"

            # Add .jpg file
            jpg_data = b"fake image data"
            info = tarfile.TarInfo(name=f"{sample_key}.jpg")
            info.size = len(jpg_data)
            tar.addfile(info, fileobj=io.BytesIO(jpg_data))

            # Add .json file
            json_data = b'{"key": "value"}'
            info = tarfile.TarInfo(name=f"{sample_key}.json")
            info.size = len(json_data)
            tar.addfile(info, fileobj=io.BytesIO(json_data))


def create_test_dataset(base_dir):
    """Create a minimal test dataset structure"""
    dataset_dir = base_dir / "final_wds-test-dataset"
    wds_dir = dataset_dir / "wds-test-dataset"

    # Create worker directories
    for worker_idx in range(2):  # Only 2 workers for testing
        worker_dir = wds_dir / f"w{worker_idx:03d}"
        worker_dir.mkdir(parents=True, exist_ok=True)

        # Create a few shards in each worker
        for shard_idx in range(2):  # Only 2 shards per worker
            tar_path = worker_dir / f"shard-{worker_idx:06d}-{shard_idx:06d}.tar"
            create_test_tar(tar_path, num_samples=10)

    return dataset_dir


def main():
    print("=" * 80)
    print("WebDataset Statistics Analyzer - Quick Test")
    print("=" * 80)
    print()

    # Create temporary test directory
    temp_dir = Path(tempfile.mkdtemp(prefix="wds_test_"))
    print(f"Creating test dataset in: {temp_dir}")

    try:
        # Create test dataset
        dataset_dir = create_test_dataset(temp_dir)
        print(f"✓ Test dataset created: {dataset_dir}")
        print()

        # Run the analyzer
        print("Running analyzer...")
        print()

        import subprocess
        result = subprocess.run([
            sys.executable,
            "analyze_wds_stats.py",
            "--input-dir", str(temp_dir),
            "--output-dir", str(temp_dir / "output"),
            "--verbose"
        ], capture_output=False, text=True)

        if result.returncode == 0:
            print()
            print("=" * 80)
            print("✓ Test passed! The analyzer works correctly.")
            print("=" * 80)
            print()
            print("Generated reports:")
            output_dir = temp_dir / "output"
            for file in output_dir.glob("*"):
                print(f"  - {file.name}")
        else:
            print()
            print("=" * 80)
            print("✗ Test failed! Check the error messages above.")
            print("=" * 80)
            sys.exit(1)

    finally:
        # Cleanup
        print()
        print(f"Cleaning up test directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("✓ Cleanup complete")


if __name__ == '__main__':
    main()
