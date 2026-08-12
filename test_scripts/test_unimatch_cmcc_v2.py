#!/usr/bin/env python3
"""
UniMatch 光流检测验证脚本（CMCC H100）- 最终修复版

修复记录：
- v1: 修复 num_scales 长度不匹配（attn_splits_list 等参数长度必须为 2）
- v2: 修复 attn_type 缺失（transformer 需要 attn_type 参数）

测试内容：
1. 运行环境与依赖检查
2. UniMatch 模型导入与参数验证
3. CPU/GPU 模型加载与显存占用测试
4. 预训练权重加载与兼容性校验
5. 随机张量推理功能验证
6. 推理性能基准测试
"""

import sys
import os
import time
import glob
import gc
import torch
import numpy as np

# ====================== 配置常量（可根据实际环境修改） ======================
UNIMATCH_ROOT = '/root/work/david_work/models/unimatch'
PRETRAINED_DIR = os.path.join(UNIMATCH_ROOT, 'pretrained')

# 模型结构参数
MODEL_CONFIG = {
    'feature_channels': 128,
    'num_scales': 2,
    'upsample_factor': 4,
    'num_head': 1,
    'ffn_dim_expansion': 4,
    'num_transformer_layers': 6,
}

# 推理与基准配置
INFERENCE_HEIGHT, INFERENCE_WIDTH = 256, 256
BENCHMARK_WARMUP_ROUNDS = 3   # GPU 预热轮次
BENCHMARK_TEST_ROUNDS = 10    # 正式计时轮次

# ⚠️ 修复 v2：添加 attn_type 参数
INFERENCE_PARAMS = {
    'attn_type': 'swin',                 # 注意力类型（swin 或 self_swin2d_cross_1d）
    'attn_splits_list': [2, 8],          # 长度必须等于 num_scales (2)
    'corr_radius_list': [-1, 4],         # 长度必须等于 num_scales (2)
    'prop_radius_list': [-1, 1],         # 长度必须等于 num_scales (2)
}
# ============================================================================

# 添加 UniMatch 到 Python 路径
sys.path.insert(0, UNIMATCH_ROOT)

print("=" * 80)
print("UniMatch 光流检测验证脚本（最终修复版 v2）")
print("=" * 80)

# ============================================================================
# 测试 1：环境检查
# ============================================================================
print("\n[测试 1] 环境检查")
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU 计算能力: {torch.cuda.get_device_capability(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"显存总量: {total_mem:.2f} GB")

# ============================================================================
# 测试 2：导入 UniMatch
# ============================================================================
print("\n[测试 2] 导入 UniMatch 模块")
try:
    from unimatch.unimatch import UniMatch
    print("✅ UniMatch 导入成功")
except ImportError as e:
    print(f"❌ UniMatch 导入失败: {e}")
    print("\n可能的原因：")
    print("1. unimatch 目录不在 PYTHONPATH 中")
    print("2. unimatch/__init__.py 缺失")
    print("3. 依赖包缺失（opencv-python, imageio 等）")
    sys.exit(1)

# ============================================================================
# 测试 3：模型创建与 CPU 验证
# ============================================================================
print("\n[测试 3] 模型创建与 CPU 验证")
try:
    model = UniMatch(**MODEL_CONFIG)
    model.eval()
    print("✅ CPU 模式模型创建成功")

    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params / 1e6:.2f} M")
    print(f"模型配置: num_scales={MODEL_CONFIG['num_scales']}")

except Exception as e:
    print(f"❌ CPU 模式模型创建失败: {e}")
    sys.exit(1)

# ============================================================================
# 测试 4：GPU 迁移与显存检查
# ============================================================================
print("\n[测试 4] GPU 模式加载与显存检查")
if not torch.cuda.is_available():
    print("⚠️  GPU 不可用，跳过 GPU 测试")
else:
    try:
        model = model.cuda()
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        print("✅ GPU 模式加载成功")
        print(f"模型显存占用: {memory_allocated:.2f} GB")

    except Exception as e:
        print(f"❌ GPU 模式加载失败: {e}")
        print("\n可能的原因：")
        print("1. CUDA 版本不兼容")
        print("2. 显存不足")
        print("3. PyTorch 未正确安装 CUDA 支持")
        sys.exit(1)

# ============================================================================
# 测试 5：加载预训练权重
# ============================================================================
print("\n[测试 5] 加载预训练权重")
weight_loaded = False
weight_file_name = "无"

# 查找权重文件并按修改时间排序（取最新）
weight_files = []
for ext in ['*.pth', '*.pt']:
    weight_files.extend(glob.glob(os.path.join(PRETRAINED_DIR, ext)))

if not weight_files:
    print(f"⚠️  未在 {PRETRAINED_DIR} 找到权重文件")
    print("将使用随机初始化权重进行测试")
else:
    # 按修改时间降序排列，取最新的权重文件
    weight_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    weight_file = weight_files[0]
    weight_file_name = os.path.basename(weight_file)
    print(f"找到权重文件: {weight_file}")
    print(f"文件大小: {os.path.getsize(weight_file) / 1024**2:.2f} MB")

    try:
        # 优先使用安全加载模式，失败则降级兼容
        try:
            checkpoint = torch.load(weight_file, map_location='cpu', weights_only=True)
        except TypeError:
            # 旧版本 PyTorch 不支持 weights_only 参数
            checkpoint = torch.load(weight_file, map_location='cpu')

        # 兼容不同的权重存储格式
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 加载权重并打印不匹配信息
        load_result = model.load_state_dict(state_dict, strict=False)
        weight_loaded = True
        print("✅ 权重加载成功")

        # 打印权重兼容性诊断
        if load_result.missing_keys:
            print(f"⚠️  缺失参数键 ({len(load_result.missing_keys)} 个):")
            print(f"   前 5 个: {load_result.missing_keys[:5]}")
        if load_result.unexpected_keys:
            print(f"⚠️  多余参数键 ({len(load_result.unexpected_keys)} 个):")
            print(f"   前 5 个: {load_result.unexpected_keys[:5]}")

    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        print("将使用随机初始化权重继续测试")

# ============================================================================
# 测试 6：随机数据推理验证
# ============================================================================
print("\n[测试 6] 随机数据推理测试")
inference_time = 0.0
try:
    # 生成随机输入（模拟两帧图像）
    img1 = torch.randn(1, 3, INFERENCE_HEIGHT, INFERENCE_WIDTH)
    img2 = torch.randn(1, 3, INFERENCE_HEIGHT, INFERENCE_WIDTH)

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

    print(f"推理参数配置:")
    print(f"  attn_type: {INFERENCE_PARAMS['attn_type']}")
    print(f"  attn_splits_list: {INFERENCE_PARAMS['attn_splits_list']}")
    print(f"  corr_radius_list: {INFERENCE_PARAMS['corr_radius_list']}")
    print(f"  prop_radius_list: {INFERENCE_PARAMS['prop_radius_list']}")

    # 同步后计时
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        flow_output = model(img1, img2, **INFERENCE_PARAMS)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time = (time.time() - start_time) * 1000

    # 解析输出
    if isinstance(flow_output, dict) and 'flow_preds' in flow_output:
        flow_pred = flow_output['flow_preds'][-1]  # 取最终上采样光流
    else:
        flow_pred = flow_output

    print("✅ 推理成功")
    print(f"输入形状: {tuple(img1.shape)}")
    print(f"输出光流形状: {tuple(flow_pred.shape)}")
    print(f"单次推理耗时: {inference_time:.2f} ms")
    print(f"光流数值范围: [{flow_pred.min():.2f}, {flow_pred.max():.2f}]")
    print(f"光流平均幅度: {torch.abs(flow_pred).mean():.4f}")

except Exception as e:
    print(f"❌ 推理失败: {e}")
    print("\n尝试其他 attn_type 参数...")

    # 尝试其他可能的 attn_type
    for alt_attn_type in ['self_swin2d_cross_1d', None]:
        print(f"\n尝试 attn_type = {alt_attn_type}")
        try:
            alt_params = INFERENCE_PARAMS.copy()
            if alt_attn_type is None:
                alt_params.pop('attn_type', None)
            else:
                alt_params['attn_type'] = alt_attn_type

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                flow_output = model(img1, img2, **alt_params)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time = (time.time() - start_time) * 1000

            if isinstance(flow_output, dict) and 'flow_preds' in flow_output:
                flow_pred = flow_output['flow_preds'][-1]
            else:
                flow_pred = flow_output

            print(f"✅ 成功！使用 attn_type={alt_attn_type}")
            print(f"单次推理耗时: {inference_time:.2f} ms")
            INFERENCE_PARAMS['attn_type'] = alt_attn_type
            break

        except Exception as e2:
            print(f"❌ 失败: {e2}")
            continue
    else:
        print("\n所有 attn_type 尝试均失败")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# 测试 7：性能基准测试
# ============================================================================
print("\n[测试 7] 性能基准测试")
benchmark_done = False
times = []

if torch.cuda.is_available():
    try:
        print(f"预热 {BENCHMARK_WARMUP_ROUNDS} 轮...")
        # GPU 预热（排除内核编译、上下文初始化开销）
        for _ in range(BENCHMARK_WARMUP_ROUNDS):
            with torch.no_grad():
                _ = model(img1, img2, **INFERENCE_PARAMS)
            torch.cuda.synchronize()

        print(f"正式计时 {BENCHMARK_TEST_ROUNDS} 轮...")
        for i in range(BENCHMARK_TEST_ROUNDS):
            torch.cuda.synchronize()
            start = time.time()

            with torch.no_grad():
                _ = model(img1, img2, **INFERENCE_PARAMS)

            torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)

        benchmark_done = True
        print(f"平均推理时间: {np.mean(times):.2f} ms")
        print(f"最小推理时间: {np.min(times):.2f} ms")
        print(f"最大推理时间: {np.max(times):.2f} ms")
        print(f"标准差: {np.std(times):.2f} ms")

    except Exception as e:
        print(f"⚠️  性能测试失败: {e}")
else:
    print("⚠️  GPU 不可用，跳过性能基准测试")

# ============================================================================
# 资源清理
# ============================================================================
del img1, img2
if 'flow_output' in locals():
    del flow_output
if 'flow_pred' in locals():
    del flow_pred
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================================
# 验证总结
# ============================================================================
print("\n" + "=" * 80)
print("验证总结")
print("=" * 80)
print("✅ UniMatch 模块导入正常")
print(f"✅ 模型创建成功 ({'GPU' if torch.cuda.is_available() else 'CPU'} 模式)")

if weight_loaded:
    print(f"✅ 预训练权重加载成功 ({weight_file_name})")
else:
    print("⚠️  未加载预训练权重（使用随机初始化）")

print(f"✅ 随机数据推理正常 ({inference_time:.2f} ms @ {INFERENCE_HEIGHT}x{INFERENCE_WIDTH})")
print(f"   使用参数: attn_type={INFERENCE_PARAMS.get('attn_type', 'None')}")

if benchmark_done:
    print(f"✅ 性能基准测试完成 (平均 {np.mean(times):.2f} ms)")
else:
    print("ℹ️  性能基准测试未执行")

print("\n🎉 UniMatch 环境验证通过！")
print("=" * 80)
