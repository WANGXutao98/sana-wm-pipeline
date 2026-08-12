def main():
    print("=" * 80)
    print("Stage 3 性能分析 - 详细计时 + 显存管理")
    print("=" * 80)

    # 显示 GPU 信息
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print_gpu_memory("初始状态 - ")
    else:
        print("\n⚠️ 未检测到 GPU")

    # 加载模型
    print("\n[模型加载]")
    t0 = time.perf_counter()

    print("  加载 UniMatch...")
    flow_fn = load_unimatch_fn(UNIMATCH_DIR, DEVICE)
    t1 = time.perf_counter()
    print(f"  ✅ UniMatch 加载完成 ({(t1-t0):.1f} 秒)")
    print_gpu_memory("  ")

    print("  加载 DOVER...")
    dover_fn = load_dover_fn(DEVICE)
    t2 = time.perf_counter()
    print(f"  ✅ DOVER 加载完成 ({(t2-t1):.1f} 秒)")
    print_gpu_memory("  ")

    print("  加载 Table 6 配置...")
    table6_cfg = load_thresholds(TABLE6_CFG)
    print(f"  ✅ 配置加载完成")

    print(f"\n  总加载时间: {(t2-t0):.1f} 秒")
    print_gpu_memory("  模型加载后 - ")

    # 加载测试样本
    print(f"\n[测试样本]")
    rec = load_sample_data(Path(STAGE12_JSONL), sample_idx=0)
    if not rec:
        print("  ❌ 无法加载样本")
        return
    print(f"  ✅ 加载样本: {rec['sample_id']}")

    # 详细计时
    try:
        timings = detailed_timing_single_sample(rec, flow_fn, dover_fn, table6_cfg)

        # 与已知问题对比
        print(f"\n{'='*80}")
        print(f"与问题现象对比")
        print(f"{'='*80}")
        print(f"\n已知问题：10 小时处理 139 样本 = 258 秒/样本")
        print(f"本次测试：{timings['total']/1000:.2f} 秒/样本")
        print(f"差异：    {258 - timings['total']/1000:.2f} 秒 未解释")

        if timings['total']/1000 < 10:
            print(f"\n✅ 单样本处理时间正常（<10 秒）")
            print(f"\n可能原因：")
            print(f"  1. 实际运行时有其他开销（多进程竞争、日志、监控）")
            print(f"  2. 某些样本特别慢（长视频、高分辨率）")
            print(f"  3. 网络文件系统延迟（在批量处理时更明显）")
            print(f"  4. 测试样本不代表平均情况")
            print(f"\n建议：")
            print(f"  1. 测试更多样本（特别是慢的那些）")
            print(f"  2. 在实际运行时添加详细日志")
            print(f"  3. 使用 py-spy profiling 找出瓶颈")
        else:
            print(f"\n⚠️ 单样本处理时间异常（>10 秒）")
            print(f"\n瓶颈已在上面的时间分布中显示")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n{'='*80}")
            print(f"❌ OOM 错误诊断")
            print(f"{'='*80}")
            print(f"\n错误信息: {e}")
            print_gpu_memory("\n当前显存状态 - ")

            print(f"\n诊断结果：")
            print(f"  1. 即使使用 FP16，当前配置仍然显存不足")
            print(f"  2. 可能原因：")
            print(f"     - UniMatch 和 DOVER 同时在 GPU 上占用过多显存")
            print(f"     - 视频分辨率过高（720p × 160 帧）")
            print(f"     - PyTorch 显存碎片化")

            print(f"\n解决方案：")
            print(f"  方案 A：降低分辨率（推荐）")
            print(f"    - 将视频降采样到 480p 或更低")
            print(f"    - 修改 _decode_frames 函数添加降采样")

            print(f"\n  方案 B：减小 chunk 大小")
            print(f"    - 将 5 秒 chunk 改为 2-3 秒")
            print(f"    - 修改 visual_metrics.py 的 DOVER_CHUNK_S")

            print(f"\n  方案 C：卸载 UniMatch")
            print(f"    - 在 DOVER 处理前卸载 UniMatch")
            print(f"    - 释放更多显存给 DOVER")

            print(f"\n  方案 D：使用 CPU 模式（最后手段）")
            print(f"    - dover_fn = load_dover_fn(device='cpu')")
            print(f"    - 速度会慢 10-40x")
        else:
            raise