#!/usr/bin/env python3
"""
快速测试脚本：验证 Stage 2 解压版本是否正常工作

用法：
  python scripts/test_stage2_extracted.py
"""
from pathlib import Path
from sana_wm_pipeline.qc.stage2_deep_extracted import (
    find_sample_files,
    deep_check_sample_extracted,
    run_stage2_extracted
)

def test_find_sample_files():
    """测试文件查找功能"""
    print("=" * 60)
    print("测试 1: 文件查找功能")
    print("=" * 60)

    # 模拟数据根目录（需要根据实际情况调整）
    data_root = Path("/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output")

    if not data_root.exists():
        print(f"❌ 数据根目录不存在: {data_root}")
        print("请在 CMCC 机器上运行此测试")
        return False

    # 测试样本 ID（从 Stage 1 结果中随机选一个）
    test_sample_id = "SpatialVID-hq_05b84042-799c-55b1-8a0a-77a2911ecd18"

    print(f"查找样本: {test_sample_id}")
    files = find_sample_files(test_sample_id, data_root)

    if files:
        print("✅ 找到样本文件:")
        for key, path in files.items():
            exists = "✓" if path.exists() else "✗"
            print(f"   [{exists}] {key}: {path}")
        return True
    else:
        print(f"❌ 未找到样本: {test_sample_id}")
        print("提示: 请使用实际存在的 sample_id 进行测试")
        return False


def test_deep_check_sample():
    """测试单样本深度检查"""
    print("\n" + "=" * 60)
    print("测试 2: 单样本深度检查")
    print("=" * 60)

    data_root = Path("/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output")
    test_sample_id = "SpatialVID-hq_05b84042-799c-55b1-8a0a-77a2911ecd18"
    group_name = "wds-SpatialVID-hq"

    print(f"处理样本: {test_sample_id}")
    print(f"数据集: {group_name}")

    result = deep_check_sample_extracted(test_sample_id, data_root, group_name)

    if result:
        print("✅ 检查完成:")
        print(f"   视频帧数: {result['stage2']['video_T']}")
        print(f"   黑帧比例: {result['stage2']['black_frame_ratio']}")
        print(f"   轨迹冻结: {result['stage2']['traj_frozen']}")
        print(f"   问题数量: {len(result['stage2']['reasons'])}")
        if result['stage2']['reasons']:
            print(f"   问题列表: {result['stage2']['reasons']}")
        return True
    else:
        print("❌ 样本不存在或处理失败")
        return False


def test_run_stage2_small():
    """测试小批量处理（前 10 个样本）"""
    print("\n" + "=" * 60)
    print("测试 3: 小批量处理（前 10 个样本）")
    print("=" * 60)

    # 使用最小的数据集测试
    s1_path = Path("/root/work/david_work/qc_output_new/wds-sekai-game-drone/stage1_results.jsonl")
    s2_path = Path("/tmp/stage2_test_output.jsonl")
    data_root = Path("/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output")

    if not s1_path.exists():
        print(f"❌ Stage 1 结果不存在: {s1_path}")
        print("请在 CMCC 机器上运行此测试")
        return False

    print(f"Stage 1 输入: {s1_path}")
    print(f"Stage 2 输出: {s2_path}")
    print(f"数据根目录: {data_root}")

    # 创建临时 Stage 1 文件（只包含前 10 个样本）
    temp_s1 = Path("/tmp/stage1_test_10samples.jsonl")
    with open(s1_path, 'r') as fin, open(temp_s1, 'w') as fout:
        for i, line in enumerate(fin):
            if i >= 10:
                break
            fout.write(line)

    print(f"创建临时输入文件（10 个样本）: {temp_s1}")

    try:
        n = run_stage2_extracted(
            temp_s1,
            s2_path,
            data_root,
            sample_frac=1.0,
            n_workers=2  # 测试用 2 进程
        )

        if s2_path.exists():
            actual = sum(1 for _ in open(s2_path))
            print(f"✅ 处理完成:")
            print(f"   预期处理: 10 个样本")
            print(f"   实际处理: {actual} 个样本")
            print(f"   输出文件: {s2_path}")

            # 显示前 2 个结果
            print("\n前 2 个结果:")
            with open(s2_path) as f:
                for i, line in enumerate(f):
                    if i >= 2:
                        break
                    import json
                    rec = json.loads(line)
                    print(f"   [{i+1}] {rec['sample_id']}: {rec['stage2']['reasons'] or 'OK'}")

            return True
        else:
            print("❌ 输出文件未生成")
            return False

    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Stage 2 解压版本 - 功能测试")
    print("=" * 60)
    print()

    tests = [
        ("文件查找", test_find_sample_files),
        ("单样本检查", test_deep_check_sample),
        ("小批量处理", test_run_stage2_small),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")

    all_pass = all(r for _, r in results)
    print()
    if all_pass:
        print("🎉 所有测试通过！可以开始全量执行")
    else:
        print("⚠️  部分测试失败，请检查配置")

    return all_pass


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
