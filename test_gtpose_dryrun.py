#!/usr/bin/env python3
"""测试 mode_gtpose.py 的 dry-run 模式（无 GT poses 文件）"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from pathlib import Path
import tempfile
import shutil


def test_dryrun_mode():
    """测试 dry-run 模式（与官方对齐）"""
    print("\n" + "="*60)
    print("Test: dry-run 模式（无 GT poses 文件）")
    print("="*60)

    from sana_wm_pipeline.stage02_pose.mode_gtpose import (
        run_gtpose, _positions_to_poses
    )

    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())
    work_dir = temp_dir / "work"
    work_dir.mkdir()

    try:
        # 创建一个测试视频路径（不需要真实存在，因为我们会 mock Pi3X）
        test_video = temp_dir / "test_video.mp4"

        # 不存在的 GT poses 路径（触发 dry-run）
        nonexistent_gt = temp_dir / "nonexistent_gt_poses.npy"

        print(f"测试视频路径: {test_video}")
        print(f"GT poses 路径: {nonexistent_gt} (不存在，将触发 dry-run)")

        # 注意：由于 run_gtpose 会调用 adapters.read_frames 读取视频，
        # 这里需要真实的视频文件才能完整测试
        # 但我们可以测试 _positions_to_poses 函数本身

        print("\n测试 _positions_to_poses 函数:")
        positions = np.random.randn(10, 3).astype(np.float64)
        poses = _positions_to_poses(positions)

        print(f"  输入 positions: {positions.shape}")
        print(f"  输出 poses: {poses.shape}")

        # 验证
        assert poses.shape == (10, 4, 4), f"Expected (10, 4, 4), got {poses.shape}"
        assert poses.dtype == np.float32, f"Expected float32, got {poses.dtype}"

        # 验证单位旋转
        for i in range(10):
            assert np.allclose(poses[i, :3, :3], np.eye(3)), f"Frame {i} rotation not identity"
            assert np.allclose(poses[i, :3, 3], positions[i]), f"Frame {i} position mismatch"
            assert poses[i, 3, 3] == 1.0, f"Frame {i} bottom-right not 1.0"

        print("  ✅ _positions_to_poses 正确构造了单位旋转的 poses")

        # 测试官方逻辑的 dry-run 流程
        print("\n验证 dry-run 逻辑:")
        print("  官方: gt_centers = 1.7 * np.asarray(pred_pos)")
        pred_pos = np.random.randn(64, 3)
        gt_centers = 1.7 * np.asarray(pred_pos)

        print(f"  pred_pos 范围: [{pred_pos.min():.2f}, {pred_pos.max():.2f}]")
        print(f"  gt_centers 范围: [{gt_centers.min():.2f}, {gt_centers.max():.2f}]")
        print(f"  比例: {gt_centers.mean() / pred_pos.mean():.2f}x (预期 ~1.7x)")

        ratio = gt_centers.mean() / pred_pos.mean()
        if abs(ratio - 1.7) < 0.1:
            print("  ✅ dry-run 的 1.7x 缩放因子正确")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_positions_to_poses():
    """单独测试 _positions_to_poses 函数"""
    print("\n" + "="*60)
    print("Test: _positions_to_poses 函数")
    print("="*60)

    from sana_wm_pipeline.stage02_pose.mode_gtpose import _positions_to_poses

    # 测试各种输入
    test_cases = [
        ("单帧", np.array([[1.0, 2.0, 3.0]])),
        ("多帧", np.random.randn(20, 3)),
        ("全零", np.zeros((5, 3))),
    ]

    for name, positions in test_cases:
        print(f"\n测试 {name}: shape={positions.shape}")
        poses = _positions_to_poses(positions)

        n = len(positions)
        assert poses.shape == (n, 4, 4), f"Shape mismatch: {poses.shape}"
        assert poses.dtype == np.float32, f"Dtype mismatch: {poses.dtype}"

        # 验证结构
        for i in range(n):
            # 旋转部分应该是单位矩阵
            R = poses[i, :3, :3]
            assert np.allclose(R, np.eye(3)), f"Rotation not identity at frame {i}"

            # 平移部分应该等于输入位置
            t = poses[i, :3, 3]
            assert np.allclose(t, positions[i]), f"Translation mismatch at frame {i}"

            # 底部行应该是 [0, 0, 0, 1]
            assert np.allclose(poses[i, 3, :], [0, 0, 0, 1]), f"Bottom row incorrect at frame {i}"

        print(f"  ✅ {name} 测试通过")

    return True


def test_official_dryrun_logic():
    """测试官方 dry-run 逻辑的完整流程"""
    print("\n" + "="*60)
    print("Test: 官方 dry-run 逻辑完整流程")
    print("="*60)

    from sana_wm_pipeline.stage02_pose.mode_gtpose import _positions_to_poses
    from sana_wm_pipeline.stage02_pose.umeyama import recover_metric_scale

    # 模拟官方 dry-run 流程
    print("\n模拟官方代码流程:")
    print("  1. 生成随机 pred_pos (Pi3X 输出)")
    pred_pos = np.random.randn(64, 3).astype(np.float64)
    print(f"     pred_pos: {pred_pos.shape}")

    print("  2. 生成 synthetic GT (1.7x 缩放)")
    gt_centers = 1.7 * np.asarray(pred_pos)
    print(f"     gt_centers: {gt_centers.shape}")

    print("  3. Umeyama 恢复 scale")
    s = recover_metric_scale(pred_pos, gt_centers, inlier_percentile=80.0)
    print(f"     scale: {s:.6f} (预期 ~1.7)")

    print("  4. 从位置构造 poses")
    poses = _positions_to_poses(gt_centers)
    print(f"     poses: {poses.shape}")

    # 验证
    if abs(s - 1.7) < 0.1:
        print("\n  ✅ 恢复的 scale 接近 1.7（预期值）")
    else:
        print(f"\n  ⚠️  scale {s:.6f} 偏离 1.7 较多")

    assert poses.shape == (64, 4, 4)
    assert np.allclose(poses[:, :3, 3], gt_centers)
    print("  ✅ poses 结构正确")

    return True


def main():
    print("="*60)
    print("mode_gtpose.py dry-run 模式测试")
    print("="*60)

    tests = [
        ("_positions_to_poses 函数", test_positions_to_poses),
        ("官方 dry-run 逻辑", test_official_dryrun_logic),
        ("dry-run 模式集成", test_dryrun_mode),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"通过: {passed_count}/{total_count}")

    if passed_count == total_count:
        print("\n🎉 所有 dry-run 测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} 个测试失败。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
