"""Unit tests for lite_geometric_check module."""
from __future__ import annotations
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '/mnt/afs/davidwang/workspace/sana_wm_pipeline/src')
from sana_wm_pipeline.qc.lite_geometric_check import (
    check_pose_geometry,
    check_pose_json,
)


def create_valid_data(n_frames: int = 10):
    """Create valid test data."""
    poses = np.eye(4)[None, :, :].repeat(n_frames, axis=0)
    intrinsics = np.array([[[800, 800, 640, 360]]] * n_frames)
    scale = np.ones(n_frames) * 0.5
    image_wh = (1280, 720)
    return poses, intrinsics, scale, image_wh


def test_valid_data():
    """Test 1: Valid data should pass all checks."""
    print("\n测试 1: 正常数据应该通过")
    poses, intrinsics, scale, image_wh = create_valid_data()
    result = check_pose_geometry(poses, intrinsics, scale, image_wh)

    assert result.passed, "Valid data should pass"
    assert result.level1_passed, "Level 1 should pass"
    assert result.level2_passed, "Level 2 should pass"
    assert result.so3_valid, "SO(3) should be valid"
    assert result.first_frame_aligned, "First frame should be aligned"
    assert result.no_nan_inf, "Should have no NaN/Inf"
    assert result.fov_valid, "FOV should be valid"
    assert result.focal_div_valid, "Focal divergence should be valid"
    assert result.scale_cv_valid, "Scale CV should be valid"

    print(f"  ✓ PASS: 所有检查通过")
    print(f"    SO(3): det={result.so3_det_mean:.8f}, orth_err={result.so3_orth_err:.3e}")
    print(f"    First frame: dev={result.first_frame_dev:.6f}")
    print(f"    FOV: [{result.fov_x_min:.1f}°, {result.fov_x_max:.1f}°] x [{result.fov_y_min:.1f}°, {result.fov_y_max:.1f}°]")
    print(f"    Focal div: {result.focal_div_max:.4f}")
    print(f"    Scale CV: {result.scale_cv:.4f}")


def test_so3_invalid_scale():
    """Test 2: SO(3) invalid - scaled rotation matrix (det ≠ 1)."""
    print("\n测试 2: SO(3) 失败 - 行列式 ≠ 1")
    poses, intrinsics, scale, image_wh = create_valid_data()
    poses[:, :3, :3] *= 2.0  # Scale → det = 8

    result = check_pose_geometry(poses, intrinsics, scale, image_wh)

    assert not result.passed, "Should fail"
    assert not result.level1_passed, "Level 1 should fail"
    assert not result.so3_valid, "SO(3) should be invalid"
    assert abs(result.so3_det_mean - 8.0) < 0.1, f"det should be ~8.0, got {result.so3_det_mean}"
    assert "SO(3) invalid" in result.failure_reasons[0]

    print(f"  ✓ FAIL (符合预期): SO(3) 检查失败")
    print(f"    det_mean={result.so3_det_mean:.6f}")
    print(f"    原因: {result.failure_reasons[0]}")


def test_first_frame_not_aligned():
    """Test 3: First frame not aligned to origin."""
    print("\n测试 3: 第一帧未对齐")
    poses, intrinsics, scale, image_wh = create_valid_data()
    poses[0, :3, 3] = [5.0, 0.0, 0.0]  # Translate 5m

    result = check_pose_geometry(poses, intrinsics, scale, image_wh)

    assert not result.passed, "Should fail"
    assert not result.level1_passed, "Level 1 should fail"
    assert not result.first_frame_aligned, "First frame should not be aligned"
    assert result.first_frame_dev > 4.9, f"dev should be ~5.0, got {result.first_frame_dev}"
    assert "First frame not aligned" in result.failure_reasons[0]

    print(f"  ✓ FAIL (符合预期): 第一帧对齐检查失败")
    print(f"    dev={result.first_frame_dev:.6f}")
    print(f"    原因: {result.failure_reasons[0]}")


def test_nan_values():
    """Test 4: Data contains NaN values."""
    print("\n测试 4: 包含 NaN 值")
    poses, intrinsics, scale, image_wh = create_valid_data()
    poses[5, 2, 3] = np.nan

    result = check_pose_geometry(poses, intrinsics, scale, image_wh)

    assert not result.passed, "Should fail"
    assert not result.level1_passed, "Level 1 should fail"
    assert not result.no_nan_inf, "Should detect NaN"
    assert len(result.nan_inf_fields) > 0
    assert "poses_c2w" in result.nan_inf_fields[0]

    print(f"  ✓ FAIL (符合预期): NaN 检查失败")
    print(f"    原因: {result.failure_reasons[0]}")


def test_fov_out_of_range():
    """Test 5: FOV out of valid range."""
    print("\n测试 5: FOV 超出范围")
    poses, intrinsics, scale, image_wh = create_valid_data()
    intrinsics = np.array([[[100, 100, 640, 360]]] * 10)  # Small focal length → large FOV

    result = check_pose_geometry(poses, intrinsics, scale, image_wh)

    assert not result.passed, "Should fail"
    assert not result.level2_passed, "Level 2 should fail"
    assert not result.fov_valid, "FOV should be invalid"
    assert result.fov_x_max > 120.0, f"FOV should exceed 120°, got {result.fov_x_max}"
    assert "FOV horizontal out of range" in result.failure_reasons[0]

    print(f"  ✓ FAIL (符合预期): FOV 检查失败")
    print(f"    水平 FOV: [{result.fov_x_min:.1f}°, {result.fov_x_max:.1f}°]")
    print(f"    原因: {result.failure_reasons[0]}")


def test_focal_divergence_high():
    """Test 6: Focal divergence too high (fx ≠ fy)."""
    print("\n测试 6: 焦距差异过大")
    poses, intrinsics, scale, image_wh = create_valid_data()
    intrinsics = np.array([[[800, 1000, 640, 360]]] * 10)  # |fx-fy| / avg = 0.222

    result = check_pose_geometry(poses, intrinsics, scale, image_wh)

    assert not result.passed, "Should fail"
    assert not result.level2_passed, "Level 2 should fail"
    assert not result.focal_div_valid, "Focal divergence should be invalid"
    assert result.focal_div_max > 0.20, f"Div should exceed 0.20, got {result.focal_div_max}"
    assert "Focal divergence too high" in result.failure_reasons[0]

    print(f"  ✓ FAIL (符合预期): 焦距差异检查失败")
    print(f"    最大差异: {result.focal_div_max:.4f}")
    print(f"    原因: {result.failure_reasons[0]}")


def test_scale_cv_high():
    """Test 7: Scale CV too high (unstable scale)."""
    print("\n测试 7: Scale CV 过大")
    poses, intrinsics, scale, image_wh = create_valid_data()
    # Create scale with high variance: most values small, one very large
    scale = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 50.0])
    # CV = std / mean ≈ 15.1 / 5.1 ≈ 2.96 > 2.0

    result = check_pose_geometry(poses, intrinsics, scale, image_wh)

    assert not result.passed, "Should fail"
    assert not result.level2_passed, "Level 2 should fail"
    assert not result.scale_cv_valid, "Scale CV should be invalid"
    assert result.scale_cv > 2.0, f"CV should exceed 2.0, got {result.scale_cv}"
    assert "Scale CV too high" in result.failure_reasons[0]

    print(f"  ✓ FAIL (符合预期): Scale CV 检查失败")
    print(f"    变异系数: {result.scale_cv:.4f}")
    print(f"    Scale: min={scale.min():.1f}, max={scale.max():.1f}, mean={scale.mean():.1f}")
    print(f"    原因: {result.failure_reasons[0]}")


def test_multiple_failures():
    """Test 8: Multiple checks fail simultaneously."""
    print("\n测试 8: 多个检查同时失败")
    poses, intrinsics, scale, image_wh = create_valid_data()

    # Create multiple failures
    poses[0, :3, 3] = [2.0, 0.0, 0.0]  # First frame not aligned
    intrinsics = np.array([[[100, 100, 640, 360]]] * 10)  # FOV out of range (both h and v)
    scale = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 50.0])  # High CV

    result = check_pose_geometry(poses, intrinsics, scale, image_wh)

    assert not result.passed, "Should fail"
    assert not result.level1_passed, "Level 1 should fail"
    assert not result.level2_passed, "Level 2 should fail"
    assert len(result.failure_reasons) >= 3, f"Should have at least 3 failures, got {len(result.failure_reasons)}"

    print(f"  ✓ FAIL (符合预期): 多个检查失败")
    print(f"    Level 1: {'PASS' if result.level1_passed else 'FAIL'}")
    print(f"    Level 2: {'PASS' if result.level2_passed else 'FAIL'}")
    print(f"    失败原因数量: {len(result.failure_reasons)}")
    for i, reason in enumerate(result.failure_reasons, 1):
        print(f"      {i}. {reason}")


def test_real_sample():
    """Test 9: Real sample from test data."""
    print("\n测试 9: 真实样例数据")
    json_path = Path("/mnt/afs/davidwang/workspace/sana_test_data/cmcc/run_20260827_125654/0a00f99d-9d9a-5265-9548-e97a34c1302c/vipe_work_default/pose_artifact_default.json")

    if not json_path.exists():
        print(f"  ⊘ SKIP: 文件不存在 {json_path}")
        return

    result = check_pose_json(json_path)

    assert result.passed, "Real sample should pass"
    assert result.level1_passed, "Level 1 should pass"
    assert result.level2_passed, "Level 2 should pass"
    assert result.num_frames == 35, f"Should have 35 frames, got {result.num_frames}"
    assert result.image_wh == (1280, 720), f"Should be 1280x720, got {result.image_wh}"

    print(f"  ✓ PASS: 真实样例通过所有检查")
    print(f"    帧数: {result.num_frames}")
    print(f"    图像尺寸: {result.image_wh[0]} x {result.image_wh[1]}")
    print(f"    SO(3): det={result.so3_det_mean:.8f}")
    print(f"    FOV: {result.fov_x_min:.1f}° x {result.fov_y_min:.1f}°")
    print(f"    Scale CV: {result.scale_cv:.4f}")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Lite Geometric Check 单元测试")
    print("=" * 70)

    tests = [
        test_valid_data,
        test_so3_invalid_scale,
        test_first_frame_not_aligned,
        test_nan_values,
        test_fov_out_of_range,
        test_focal_divergence_high,
        test_scale_cv_high,
        test_multiple_failures,
        test_real_sample,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 70)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
