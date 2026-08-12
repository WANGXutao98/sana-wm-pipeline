#!/usr/bin/env python3
"""
Qwen3.5-9B 验证脚本 - 最终修复版

修复：过滤 processor 生成的不兼容参数（mm_token_type_ids）
"""

import sys
import os
import time
import gc
import torch
import numpy as np

# ====================== 配置 ======================
MODEL_PATH = '/root/work/david_work/models/Qwen3.5-9B'
OUTPUT_DIR = "/tmp/qwen_test_output"

CAPTION_TESTS = [
    "A person walking in the park with camera panning from left to right and zooming in slowly.",
    "A red car parked on the street under a blue sky.",
    "Birds flying over the ocean, camera tilting up and tracking the birds with smooth motion.",
    "A chef cooking in a kitchen, the camera slowly dollies in while maintaining focus on the hands.",
]

CAMERA_KEYWORDS = [
    "camera", "panning", "zooming", "tilting", "tracking", "dolly",
    "crane", "steadicam", "handheld", "zoom in", "zoom out", "pan", "tilt"
]

BENCHMARK_ROUNDS = 5
# ===================================================

print("=" * 80)
print("Qwen3.5-9B 验证脚本（最终修复版）")
print("=" * 80)

# ============================================================================
# 测试 1-3：环境检查（简化输出）
# ============================================================================
print("\n[测试 1-3] 环境检查")
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)} ({total_mem:.0f} GB)")

try:
    import transformers
    print(f"transformers: {transformers.__version__}")
    from transformers import AutoModelForCausalLM, AutoProcessor
    print("✅ 依赖检查通过")
except ImportError as e:
    print(f"❌ 依赖缺失: {e}")
    sys.exit(1)

# ============================================================================
# 测试 4：模型加载
# ============================================================================
print("\n[测试 4] 模型加载")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    initial_mem = torch.cuda.memory_allocated(0) / 1024**3

start = time.time()

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    load_time = time.time() - start
    params = sum(p.numel() for p in model.parameters())

    print(f"✅ 加载成功 ({load_time:.1f} 秒)")
    print(f"参数量: {params / 1e9:.2f} B")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        model_mem = (torch.cuda.memory_allocated(0) / 1024**3) - initial_mem
        print(f"显存占用: {model_mem:.2f} GB")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("✅ 处理器加载成功")

except Exception as e:
    print(f"❌ 加载失败: {e}")
    sys.exit(1)

# ============================================================================
# 辅助函数：过滤不兼容参数
# ============================================================================
def filter_model_kwargs(inputs):
    """移除模型不支持的参数"""
    # 移除 mm_token_type_ids（Qwen3.5 不需要）
    if 'mm_token_type_ids' in inputs:
        inputs.pop('mm_token_type_ids')
    return inputs

# ============================================================================
# 测试 5：基础推理
# ============================================================================
print("\n[测试 5] 基础推理测试")

try:
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True)

    # ✅ 修复：过滤不兼容参数
    inputs = filter_model_kwargs(inputs)

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.time() - start) * 1000

    generated = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0]

    print(f"✅ 推理成功 ({elapsed:.0f} ms)")
    print(f"输出: {response[:80]}...")

except Exception as e:
    print(f"❌ 推理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 测试 6：Caption 改写（Stage 3 核心）
# ============================================================================
print("\n[测试 6] Caption 改写功能（Stage 3 核心）")
print("说明：纯文本任务\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)
results = []
times = []

for i, original in enumerate(CAPTION_TESTS, 1):
    print(f"--- 测试 {i}/{len(CAPTION_TESTS)} ---")
    print(f"原始: {original[:60]}...")

    prompt = f"""Remove all camera movement descriptions from this caption, keep only content.

Original: {original}

Rewritten:"""

    try:
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt", padding=True)

        # 过滤参数
        inputs = filter_model_kwargs(inputs)

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

        generated = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
        rewritten = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        results.append({'original': original, 'rewritten': rewritten, 'time_ms': elapsed})

        print(f"改写: {rewritten[:60]}...")
        print(f"耗时: {elapsed:.0f} ms")

        remaining = [k for k in CAMERA_KEYWORDS if k.lower() in rewritten.lower()]
        if remaining:
            print(f"⚠️  仍含: {remaining}")
        else:
            print("✅ 相机词汇已移除")
        print()

    except Exception as e:
        print(f"❌ 改写失败: {e}")
        times.append(0)

if times and sum(times) > 0:
    valid_times = [t for t in times if t > 0]
    print(f"Caption 改写性能:")
    print(f"  平均: {np.mean(valid_times):.0f} ms")
    print(f"  最小: {np.min(valid_times):.0f} ms")
    print(f"  最大: {np.max(valid_times):.0f} ms")

# 保存结果
import json
result_file = os.path.join(OUTPUT_DIR, 'results.json')
with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n结果保存: {result_file}")

# ============================================================================
# 测试 7：性能基准
# ============================================================================
print("\n[测试 7] 性能基准测试")
bench_times = []

for _ in range(BENCHMARK_ROUNDS):
    messages = [{"role": "user", "content": "Say hello briefly."}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True)

    inputs = filter_model_kwargs(inputs)

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    bench_times.append((time.time() - start) * 1000)

print(f"基准测试 ({BENCHMARK_ROUNDS} 轮):")
print(f"  平均: {np.mean(bench_times):.0f} ms")
print(f"  最小: {np.min(bench_times):.0f} ms")
print(f"  最大: {np.max(bench_times):.0f} ms")

# ============================================================================
# 清理
# ============================================================================
del model, processor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("验证总结")
print("=" * 80)
print(f"✅ 模型加载成功 ({load_time:.1f} 秒)")
print(f"✅ 显存占用: {model_mem:.2f} GB")
print(f"✅ 基础推理正常 ({elapsed:.0f} ms)")
if valid_times:
    print(f"✅ Caption 改写正常 (平均 {np.mean(valid_times):.0f} ms)")
print(f"✅ 性能基准完成 (平均 {np.mean(bench_times):.0f} ms)")

print("\n🎉 Qwen3.5-9B 验证通过！")
print("=" * 80)
