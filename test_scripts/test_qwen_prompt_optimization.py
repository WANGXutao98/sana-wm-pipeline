#!/usr/bin/env python3
"""
Qwen3.5-9B Caption 改写优化测试

目标：通过优化提示词抑制"思维链"，直接输出结果
测试多种提示词策略，找到最优方案
"""

import sys
import os
import time
import torch
import json

MODEL_PATH = '/root/work/david_work/models/Qwen3.5-9B'
OUTPUT_DIR = "/tmp/qwen_prompt_test"

# 测试用例
TEST_CAPTION = "A person walking in the park with camera panning from left to right and zooming in slowly."

# ============================================================================
# 提示词策略（从简单到复杂）
# ============================================================================
PROMPT_STRATEGIES = {
    "strategy_1_direct": {
        "name": "直接指令（无解释）",
        "template": """Remove camera movements. Output ONLY the result.

Original: {caption}
Result:"""
    },

    "strategy_2_xml": {
        "name": "XML 格式约束",
        "template": """Remove all camera movement descriptions from this caption.

Input: {caption}

Output the rewritten caption in this format:
<caption>your rewritten caption here</caption>"""
    },

    "strategy_3_system": {
        "name": "系统角色约束",
        "template": """You are a caption rewriter. Your task is to remove camera movements and output ONLY the cleaned caption. No explanation, no reasoning, just the result.

Original caption: {caption}

Cleaned caption:"""
    },

    "strategy_4_fewshot": {
        "name": "Few-shot 示例",
        "template": """Remove camera movements from captions. Examples:

Input: "A cat playing with a ball, camera zooming in"
Output: "A cat playing with a ball"

Input: "Sunset over the ocean, camera panning left"
Output: "Sunset over the ocean"

Input: {caption}
Output:"""
    },

    "strategy_5_json": {
        "name": "JSON 格式约束",
        "template": """Remove camera movements and return JSON.

Input: {caption}

Output format: {{"result": "cleaned caption"}}

Output:"""
    },
}

# ============================================================================
# 加载模型
# ============================================================================
print("=" * 80)
print("Qwen3.5-9B 提示词优化测试")
print("=" * 80)

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
print("✅ 模型加载完成")

def filter_model_kwargs(inputs):
    if 'mm_token_type_ids' in inputs:
        inputs.pop('mm_token_type_ids')
    return inputs

# ============================================================================
# 测试各种提示词策略
# ============================================================================
print(f"\n[2] 测试用例: {TEST_CAPTION[:60]}...")
print(f"\n开始测试 {len(PROMPT_STRATEGIES)} 种提示词策略...\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)
results = []

for strategy_id, strategy in PROMPT_STRATEGIES.items():
    print("=" * 80)
    print(f"策略: {strategy['name']}")
    print("=" * 80)

    prompt = strategy['template'].format(caption=TEST_CAPTION)
    print(f"提示词预览:\n{prompt[:150]}...\n")

    try:
        # 准备输入
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
                max_new_tokens=128,  # 减少生成长度
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

        # 检查是否包含"思维链"关键词
        thinking_markers = ["Thinking", "Process:", "Analyze", "Step", "1.", "2.", "**"]
        has_thinking = any(marker in output_text for marker in thinking_markers)

        # 检查是否移除相机词汇
        camera_words = ["camera", "panning", "zooming", "pan", "zoom"]
        remaining_camera = [w for w in camera_words if w.lower() in output_text.lower()]

        result = {
            "strategy_id": strategy_id,
            "strategy_name": strategy["name"],
            "output": output_text,
            "time_ms": elapsed,
            "output_tokens": output_tokens,
            "has_thinking_chain": has_thinking,
            "remaining_camera_words": remaining_camera,
            "success": not has_thinking and len(remaining_camera) == 0
        }
        results.append(result)

        # 打印结果
        print(f"输出: {output_text[:200]}...")
        print(f"耗时: {elapsed:.0f} ms")
        print(f"生成 tokens: {output_tokens}")
        print(f"包含思维链: {'❌ 是' if has_thinking else '✅ 否'}")
        print(f"剩余相机词: {remaining_camera if remaining_camera else '✅ 无'}")
        print(f"评估: {'✅ 成功' if result['success'] else '⚠️  需改进'}")
        print()

    except Exception as e:
        print(f"❌ 测试失败: {e}\n")
        results.append({
            "strategy_id": strategy_id,
            "strategy_name": strategy["name"],
            "error": str(e),
            "success": False
        })

# ============================================================================
# 总结与推荐
# ============================================================================
print("=" * 80)
print("测试总结")
print("=" * 80)

successful = [r for r in results if r.get("success", False)]
no_thinking = [r for r in results if not r.get("has_thinking_chain", True)]

print(f"\n总测试数: {len(results)}")
print(f"完全成功: {len(successful)} 个")
print(f"无思维链: {len(no_thinking)} 个")

if successful:
    print("\n✅ 成功的策略:")
    for r in successful:
        print(f"  - {r['strategy_name']}: {r['time_ms']:.0f} ms, {r['output_tokens']} tokens")

    # 推荐最快的
    fastest = min(successful, key=lambda x: x['time_ms'])
    print(f"\n🏆 推荐策略: {fastest['strategy_name']}")
    print(f"   耗时: {fastest['time_ms']:.0f} ms")
    print(f"   输出: {fastest['output'][:100]}...")

elif no_thinking:
    print("\n⚠️  部分成功（无思维链但可能有相机词）:")
    for r in no_thinking:
        print(f"  - {r['strategy_name']}: {r['time_ms']:.0f} ms")
        print(f"    剩余相机词: {r.get('remaining_camera_words', [])}")
else:
    print("\n❌ 所有策略均未成功抑制思维链")
    print("建议: 考虑使用 Qwen2.5-VL 替代")

# 保存详细结果
result_file = os.path.join(OUTPUT_DIR, 'prompt_test_results.json')
with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n详细结果已保存: {result_file}")

# ============================================================================
# 性能对比
# ============================================================================
if results:
    valid_results = [r for r in results if 'time_ms' in r]
    if valid_results:
        times = [r['time_ms'] for r in valid_results]
        tokens = [r['output_tokens'] for r in valid_results]

        print(f"\n性能统计:")
        print(f"  平均耗时: {sum(times)/len(times):.0f} ms")
        print(f"  最快: {min(times):.0f} ms")
        print(f"  最慢: {max(times):.0f} ms")
        print(f"  平均 tokens: {sum(tokens)/len(tokens):.0f}")

print("\n" + "=" * 80)
