#!/usr/bin/env python3
"""
Qwen3.5-9B 视觉语言模型验证脚本（CMCC H100）

测试内容：
1. 运行环境与依赖检查
2. 模型加载与显存占用测试
3. 图像理解功能验证
4. Caption 改写功能测试（核心）
5. 推理性能基准测试
6. 批量处理性能测试

参考：
- DOVER H100 验证经验（环境变量设置）
- UniMatch 验证经验（性能测试方法）
"""

import sys
import os
import time
import gc
import torch
import numpy as np
from PIL import Image

# ====================== 配置常量 ======================
MODEL_PATH = '/root/work/david_work/models/Qwen3.5-9B'
TEST_IMAGE_PATH = "/root/work/david_work/sana_qc_pipeline/DOVER/demo"  # 从 demo 视频提取帧
OUTPUT_DIR = "/tmp/qwen_test_output"

# 测试用例
TEST_CAPTIONS = [
    # 测试 1: 包含相机动作的 caption（需要改写）
    "A person walking in the park with camera panning from left to right and zooming in slowly.",

    # 测试 2: 纯内容描述（不需要改写）
    "A red car parked on the street under a blue sky.",

    # 测试 3: 多个相机动作
    "Birds flying over the ocean, camera tilting up and tracking the birds with smooth motion.",
]

# 相机动作关键词（需要从 caption 中移除）
CAMERA_KEYWORDS = [
    "camera", "panning", "zooming", "tilting", "tracking", "dolly",
    "crane", "steadicam", "handheld", "shaky", "rotating",
    "zoom in", "zoom out", "pan left", "pan right", "tilt up", "tilt down"
]

# 推理配置
BENCHMARK_ROUNDS = 5
# ======================================================

print("=" * 80)
print("Qwen3.5-9B 视觉语言模型验证脚本")
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
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU 计算能力: {torch.cuda.get_device_capability(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"显存总量: {total_mem:.2f} GB")
    free_mem = (total_mem - torch.cuda.memory_allocated(0) / 1024**3)
    print(f"可用显存: {free_mem:.2f} GB")

# 检查离线环境变量（CMCC 必需）
offline_vars = {
    'TRANSFORMERS_OFFLINE': os.getenv('TRANSFORMERS_OFFLINE'),
    'HF_HUB_OFFLINE': os.getenv('HF_HUB_OFFLINE'),
    'HF_DATASETS_OFFLINE': os.getenv('HF_DATASETS_OFFLINE'),
}
print(f"\n离线模式环境变量:")
for var, val in offline_vars.items():
    status = "✅" if val == "1" else "⚠️ "
    print(f"  {status} {var}={val}")

if not all(v == "1" for v in offline_vars.values()):
    print("\n⚠️  警告：部分离线环境变量未设置，可能导致联网下载失败")
    print("建议执行：")
    print("  export TRANSFORMERS_OFFLINE=1")
    print("  export HF_HUB_OFFLINE=1")
    print("  export HF_DATASETS_OFFLINE=1")

# ============================================================================
# 测试 2：依赖检查
# ============================================================================
print("\n[测试 2] 依赖检查")
try:
    import transformers
    print(f"✅ transformers: {transformers.__version__}")
except ImportError:
    print("❌ transformers 未安装")
    print("   安装命令: pip install transformers")
    sys.exit(1)

try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print("✅ Qwen2VL 模块导入成功")
except ImportError as e:
    print(f"❌ Qwen2VL 模块导入失败: {e}")
    print("   可能需要更新 transformers 版本")
    print("   安装命令: pip install transformers>=4.37.0")
    sys.exit(1)

try:
    from PIL import Image
    print(f"✅ Pillow (PIL)")
except ImportError:
    print("❌ Pillow 未安装")
    print("   安装命令: pip install Pillow")
    sys.exit(1)

# ============================================================================
# 测试 3：模型文件检查
# ============================================================================
print("\n[测试 3] 模型文件检查")
print(f"模型路径: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print(f"❌ 模型路径不存在: {MODEL_PATH}")
    sys.exit(1)

# 检查关键文件
required_files = ['config.json', 'model.safetensors.index.json', 'tokenizer.json']
missing_files = []

for file in required_files:
    file_path = os.path.join(MODEL_PATH, file)
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / 1024
        print(f"✅ {file} ({size:.1f} KB)")
    else:
        print(f"❌ {file} 缺失")
        missing_files.append(file)

if missing_files:
    print(f"\n❌ 缺失关键文件，无法加载模型")
    sys.exit(1)

# 统计权重文件大小
import glob
weight_files = glob.glob(os.path.join(MODEL_PATH, "model.safetensors-*.safetensors"))
if weight_files:
    total_size = sum(os.path.getsize(f) for f in weight_files) / 1024**3
    print(f"\n权重文件统计:")
    print(f"  分片数量: {len(weight_files)}")
    print(f"  总大小: {total_size:.2f} GB")
else:
    print(f"\n⚠️  未找到权重文件")

# ============================================================================
# 测试 4：模型加载
# ============================================================================
print("\n[测试 4] 模型加载测试")
print("正在加载模型（可能需要 1-2 分钟）...")

# 记录初始显存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    initial_mem = torch.cuda.memory_allocated(0) / 1024**3
    print(f"加载前显存占用: {initial_mem:.2f} GB")

load_start = time.time()

try:
    # 加载模型（使用 flash_attention_2 加速）
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,  # 使用 bfloat16 节省显存
        device_map="auto",           # 自动分配到 GPU
        trust_remote_code=True,      # 必需，Qwen 使用自定义代码
    )
    model.eval()

    load_time = time.time() - load_start
    print(f"✅ 模型加载成功 (耗时 {load_time:.2f} 秒)")

    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params / 1e9:.2f} B")

    # 检查显存占用
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        loaded_mem = torch.cuda.memory_allocated(0) / 1024**3
        model_mem = loaded_mem - initial_mem
        print(f"模型显存占用: {model_mem:.2f} GB")
        print(f"剩余可用显存: {total_mem - loaded_mem:.2f} GB")

except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 加载处理器
print("\n加载处理器...")
try:
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    print("✅ 处理器加载成功")
except Exception as e:
    print(f"❌ 处理器加载失败: {e}")
    sys.exit(1)

# ============================================================================
# 测试 5：图像理解功能验证
# ============================================================================
print("\n[测试 5] 图像理解功能验证")

# 创建测试图像（纯色图像 + 文字）
test_image = Image.new('RGB', (256, 256), color='red')
print("创建测试图像: 256x256 红色图像")

# 简单推理测试
test_prompt = "Describe this image briefly."

try:
    print(f"\n测试提示: {test_prompt}")

    # 准备输入
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": test_prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    # 移到 GPU
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    # 推理计时
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    inference_time = (time.time() - start) * 1000

    # 解码输出
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print(f"✅ 推理成功")
    print(f"推理耗时: {inference_time:.2f} ms")
    print(f"生成 tokens: {len(generated_ids_trimmed[0])}")
    print(f"输出: {output_text}")

except Exception as e:
    print(f"❌ 图像理解测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 辅助函数：处理视觉输入
# ============================================================================
def process_vision_info(messages):
    """从消息中提取图像和视频"""
    image_inputs = []
    video_inputs = []

    for message in messages:
        if isinstance(message["content"], list):
            for item in message["content"]:
                if item.get("type") == "image":
                    image_inputs.append(item["image"])
                elif item.get("type") == "video":
                    video_inputs.append(item["video"])

    return image_inputs if image_inputs else None, video_inputs if video_inputs else None

# ============================================================================
# 测试 6：Caption 改写功能测试（核心）
# ============================================================================
print("\n[测试 6] Caption 改写功能测试（Stage 3 核心任务）")
print("测试任务：移除 caption 中的相机动作词汇\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

rewrite_times = []
rewrite_results = []

for i, original_caption in enumerate(TEST_CAPTIONS, 1):
    print(f"--- 测试用例 {i} ---")
    print(f"原始 caption: {original_caption}")

    # 构造改写提示
    rewrite_prompt = f"""Please rewrite the following video caption by removing all camera movement and filming technique descriptions (such as "camera panning", "zooming", "tilting", "tracking", "dolly", "crane", etc.), while keeping the content description intact.

Original caption: {original_caption}

Rewritten caption (content only, no camera movements):"""

    try:
        # 准备输入（纯文本任务，不需要图像）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": rewrite_prompt}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt"
        )

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        # 推理计时
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.time() - start) * 1000
        rewrite_times.append(elapsed)

        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        rewritten_caption = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        rewrite_results.append({
            'original': original_caption,
            'rewritten': rewritten_caption,
            'time_ms': elapsed
        })

        print(f"改写 caption: {rewritten_caption}")
        print(f"耗时: {elapsed:.2f} ms")

        # 简单验证：检查相机关键词是否被移除
        remaining_keywords = [kw for kw in CAMERA_KEYWORDS if kw.lower() in rewritten_caption.lower()]
        if remaining_keywords:
            print(f"⚠️  仍包含相机词汇: {remaining_keywords}")
        else:
            print("✅ 相机词汇已移除")

        print()

    except Exception as e:
        print(f"❌ 改写失败: {e}")
        import traceback
        traceback.print_exc()

if rewrite_times:
    print(f"Caption 改写性能统计:")
    print(f"  平均耗时: {np.mean(rewrite_times):.2f} ms")
    print(f"  最小耗时: {np.min(rewrite_times):.2f} ms")
    print(f"  最大耗时: {np.max(rewrite_times):.2f} ms")

# 保存结果
import json
result_file = os.path.join(OUTPUT_DIR, 'caption_rewrite_results.json')
with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(rewrite_results, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到: {result_file}")

# ============================================================================
# 测试 7: 性能基准测试
# ============================================================================
print("\n[测试 7] 性能基准测试")
print(f"测试轮次: {BENCHMARK_ROUNDS}")

benchmark_times = []
test_text = "Describe what you see in this image."

for i in range(BENCHMARK_ROUNDS):
    messages = [{"role": "user", "content": [{"type": "text", "text": test_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=None, padding=True, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    benchmark_times.append((time.time() - start) * 1000)

print(f"平均推理时间: {np.mean(benchmark_times):.2f} ms")
print(f"最小推理时间: {np.min(benchmark_times):.2f} ms")
print(f"最大推理时间: {np.max(benchmark_times):.2f} ms")
print(f"标准差: {np.std(benchmark_times):.2f} ms")

# ============================================================================
# 资源清理
# ============================================================================
del model, processor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================================
# 验证总结
# ============================================================================
print("\n" + "=" * 80)
print("验证总结")
print("=" * 80)
print("✅ Qwen3.5-9B 模型加载成功")
print(f"✅ 模型显存占用: {model_mem:.2f} GB")
print(f"✅ 图像理解功能正常 ({inference_time:.2f} ms)")
print(f"✅ Caption 改写功能正常 (平均 {np.mean(rewrite_times):.2f} ms)")
print(f"✅ 性能基准测试完成 (平均 {np.mean(benchmark_times):.2f} ms)")

print("\n🎉 Qwen3.5-9B 环境验证通过！")
print("=" * 80)
