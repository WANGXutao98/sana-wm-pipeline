#!/usr/bin/env python3
"""
Qwen3.5-9B 后处理方案
如果无法禁用思维链，通过正则提取最终结果
"""

import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_PATH = '/root/work/david_work/models/Qwen3.5-9B'

def extract_final_caption(output_text):
    """
    从包含思维链的输出中提取最终结果

    尝试多种提取策略：
    1. 提取 "Output:" / "Result:" / "Final:" 之后的内容
    2. 提取最后一个句子（如果是完整句子）
    3. 移除明显的思维过程标记
    """
    # 策略 1：查找明确的输出标记
    patterns = [
        r'(?:Output|Result|Final|Cleaned caption|Rewritten):\s*["\']?([^"\n]+)["\']?',
        r'</think>\s*(.+?)(?:<|$)',  # </think> 标签之后
        r'Thinking Process:.*?(?:\n\n|\n-\s*)(.*?)(?:\n|$)',  # 思维过程后第一行
    ]

    for pattern in patterns:
        match = re.search(pattern, output_text, re.IGNORECASE | re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            # 验证：不包含思维过程关键词
            if not any(kw in candidate for kw in ["Analyze", "Step", "Process", "Thinking"]):
                return candidate

    # 策略 2：提取最后一个完整句子
    sentences = [s.strip() for s in output_text.split('\n') if s.strip() and len(s.strip()) > 10]
    if sentences:
        last_sentence = sentences[-1]
        if not any(kw in last_sentence for kw in ["Analyze", "Step", "Process", "Thinking", "1.", "2."]):
            return last_sentence

    # 策略 3：返回原文（失败）
    return output_text


# ============================================================================
# 测试
# ============================================================================
print("=" * 80)
print("Qwen3.5-9B 后处理提取测试")
print("=" * 80)

print("\n加载模型...")
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

# 测试用例
test_caption = "A person walking in the park with camera panning from left to right."

# 使用结构化提示词
prompt = f"""Remove camera movements from this caption. After your analysis, output the final result after "Final:".

Original: {test_caption}

Final:"""

messages = [{"role": "user", "content": prompt}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt", padding=True)
inputs = filter_model_kwargs(inputs)

if torch.cuda.is_available():
    inputs = inputs.to("cuda")

print("推理中...")
start = time.time()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

elapsed = (time.time() - start) * 1000

generated = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
raw_output = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

print(f"\n原始输出 ({elapsed:.0f} ms):")
print("-" * 80)
print(raw_output)
print("-" * 80)

# 后处理提取
extracted = extract_final_caption(raw_output)
print(f"\n提取结果:")
print(f"✅ {extracted}")

# 验证
camera_words = ["camera", "panning", "zooming"]
remaining = [w for w in camera_words if w.lower() in extracted.lower()]
print(f"\n验证: {'✅ 成功' if not remaining else f'⚠️ 仍含 {remaining}'}")
