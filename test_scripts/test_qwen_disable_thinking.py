#!/usr/bin/env python3
"""
Qwen3.5-9B 禁用思维链测试
使用 enable_thinking=False 参数直接输出结果
"""

import sys
import os
import time
import torch
import json

MODEL_PATH = '/root/work/david_work/models/Qwen3.5-9B'
OUTPUT_DIR = "/tmp/qwen_disable_thinking_test"

# 测试用例
TEST_CASES = [
    "A person walking in the park with camera panning from left to right and zooming in slowly.",
    "A red car parked on the street under a blue sky.",
    "Birds flying over the ocean, camera tilting up and tracking their movement.",
    "A chef cooking in a kitchen, the camera slowly dollies in while zooming out slightly.",
]

print("=" * 80)
print("Qwen3.5-9B 思维链禁用测试")
print("=" * 80)

# ============================================================================
# 加载模型
# ============================================================================
print("\n[1] 加载模型...")
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("✅ 模型加载完成\n")

def filter_model_kwargs(inputs):
    if 'mm_token_type_ids' in inputs:
        inputs.pop('mm_token_type_ids')
    return inputs

# ============================================================================
# 测试：对比启用/禁用思维链
# ============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
results = []

# 简化提示词
PROMPT_TEMPLATE = """Remove all camera movement descriptions from this caption.

Original: {caption}

Output only the cleaned caption:"""

for test_idx, caption in enumerate(TEST_CASES, 1):
    print("=" * 80)
    print(f"测试 {test_idx}/{len(TEST_CASES)}")
    print("=" * 80)
    print(f"原始: {caption}\n")

    for enable_thinking_flag in [True, False]:
        mode_name = "启用思维链" if enable_thinking_flag else "禁用思维链"
        print(f"--- {mode_name} ---")

        try:
            # 准备输入
            prompt = PROMPT_TEMPLATE.format(caption=caption)
            messages = [{"role": "user", "content": prompt}]

            # 关键：传入 enable_thinking 参数
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking_flag  # 🔑 关键参数
            )

            inputs = processor(text=[text], return_tensors="pt", padding=True)
            inputs = filter_model_kwargs(inputs)

            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            # 推理计时
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = (time.time() - start) * 1000

            # 解码输出
            generated = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
            output_text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

            # 统计
            output_tokens = len(generated[0])

            # 检查思维链标记
            thinking_markers = ["Thinking", "Process:", "Analyze", "Step", "<think>"]
            has_thinking = any(marker in output_text for marker in thinking_markers)

            # 检查相机词汇
            camera_words = ["camera", "panning", "zooming", "tilting", "tracking", "dolly", "pan", "tilt"]
            remaining_camera = [w for w in camera_words if w.lower() in output_text.lower()]

            result = {
                "test_case": caption,
                "enable_thinking": enable_thinking_flag,
                "output": output_text,
                "time_ms": elapsed,
                "output_tokens": output_tokens,
                "has_thinking_chain": has_thinking,
                "remaining_camera_words": remaining_camera,
                "success": not has_thinking and len(remaining_camera) == 0
            }
            results.append(result)

            # 打印结果
            print(f"输出: {output_text[:150]}{'...' if len(output_text) > 150 else ''}")
            print(f"耗时: {elapsed:.0f} ms")
            print(f"Tokens: {output_tokens}")
            print(f"思维链: {'❌ 有' if has_thinking else '✅ 无'}")
            print(f"相机词: {remaining_camera if remaining_camera else '✅ 已移除'}")
            print(f"状态: {'✅ 成功' if result['success'] else '⚠️ 需改进'}")
            print()

        except Exception as e:
            print(f"❌ 错误: {e}\n")
            results.append({
                "test_case": caption,
                "enable_thinking": enable_thinking_flag,
                "error": str(e),
                "success": False
            })

# ============================================================================
# 总结
# ============================================================================
print("=" * 80)
print("测试总结")
print("=" * 80)

enabled_results = [r for r in results if r.get("enable_thinking") == True and "error" not in r]
disabled_results = [r for r in results if r.get("enable_thinking") == False and "error" not in r]

if enabled_results:
    enabled_avg_time = sum(r["time_ms"] for r in enabled_results) / len(enabled_results)
    enabled_thinking = sum(1 for r in enabled_results if r["has_thinking_chain"])
    print(f"\n启用思维链 (enable_thinking=True):")
    print(f"  平均耗时: {enabled_avg_time:.0f} ms")
    print(f"  包含思维链: {enabled_thinking}/{len(enabled_results)} 个")

if disabled_results:
    disabled_avg_time = sum(r["time_ms"] for r in disabled_results) / len(disabled_results)
    disabled_thinking = sum(1 for r in disabled_results if r["has_thinking_chain"])
    disabled_success = sum(1 for r in disabled_results if r["success"])
    print(f"\n禁用思维链 (enable_thinking=False):")
    print(f"  平均耗时: {disabled_avg_time:.0f} ms")
    print(f"  包含思维链: {disabled_thinking}/{len(disabled_results)} 个")
    print(f"  完全成功: {disabled_success}/{len(disabled_results)} 个")

    if enabled_results:
        speedup = enabled_avg_time / disabled_avg_time
        print(f"\n⚡ 加速比: {speedup:.2f}x")

# 保存结果
result_file = os.path.join(OUTPUT_DIR, 'disable_thinking_results.json')
with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n详细结果已保存: {result_file}")

print("\n" + "=" * 80)
