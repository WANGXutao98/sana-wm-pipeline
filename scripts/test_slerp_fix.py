#!/usr/bin/env python3
"""快速验证Slerp插值范围修复"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def test_fix():
    """模拟修复前后的场景"""

    # 模拟32帧视频
    pose_inds = np.arange(32)
    print(f"✅ 视频帧数: {len(pose_inds)} (indices: 0-{pose_inds[-1]})")

    # 修复前：只按间隔选择
    KEYFRAME_INTERVAL = 4
    sparse_mask_old = (pose_inds % KEYFRAME_INTERVAL == 0)
    inds_sparse_old = pose_inds[sparse_mask_old]
    print(f"\n❌ 修复前:")
    print(f"   sparse_mask: {sparse_mask_old}")
    print(f"   inds_sparse: {inds_sparse_old}")
    print(f"   Slerp范围: [{inds_sparse_old.min()}, {inds_sparse_old.max()}]")
    print(f"   插值请求: [0, {len(pose_inds)-1}]")
    print(f"   问题: {len(pose_inds)-1} > {inds_sparse_old.max()} ❌")

    # 修复后：强制包含最后一帧
    sparse_mask_new = (pose_inds % KEYFRAME_INTERVAL == 0)
    if not sparse_mask_new[-1]:
        sparse_mask_new[-1] = True
    inds_sparse_new = pose_inds[sparse_mask_new]
    print(f"\n✅ 修复后:")
    print(f"   sparse_mask: {sparse_mask_new}")
    print(f"   inds_sparse: {inds_sparse_new}")
    print(f"   Slerp范围: [{inds_sparse_new.min()}, {inds_sparse_new.max()}]")
    print(f"   插值请求: [0, {len(pose_inds)-1}]")
    print(f"   验证: {len(pose_inds)-1} == {inds_sparse_new.max()} ✅")

    # 模拟Slerp插值
    print(f"\n🧪 Slerp插值测试:")
    # 创建随机旋转矩阵
    R_sparse = Rotation.random(len(inds_sparse_new), random_state=42)

    try:
        slerp = Slerp(inds_sparse_new, R_sparse)
        R_interp = slerp(np.arange(len(pose_inds)))
        print(f"   ✅ Slerp插值成功！")
        print(f"   输出shape: {R_interp.as_matrix().shape}")
    except ValueError as e:
        print(f"   ❌ Slerp插值失败: {e}")

    # 对比keyframe数量
    print(f"\n📊 稀疏化效果:")
    print(f"   原始帧数: {len(pose_inds)}")
    print(f"   修复前keyframes: {len(inds_sparse_old)} (减少 {(1-len(inds_sparse_old)/len(pose_inds))*100:.1f}%)")
    print(f"   修复后keyframes: {len(inds_sparse_new)} (减少 {(1-len(inds_sparse_new)/len(pose_inds))*100:.1f}%)")
    print(f"   增加的keyframes: {len(inds_sparse_new) - len(inds_sparse_old)} 个")

if __name__ == "__main__":
    test_fix()
