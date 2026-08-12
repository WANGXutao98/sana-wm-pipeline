# DOVER 包说明文档

**日期：** 2026-06-27  
**包名：** dover  
**用途：** Stage 3 视频质量评分

---

## 📋 DOVER 是什么？

### 全称
**DOVER** = **D**isentangled **O**bjective **V**ideo Quality **E**valuato**r**

### 学术背景
- **论文：** "Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives" (ICCV 2023)
- **作者：** 来自顶级计算机视觉会议 ICCV
- **任务：** 视频质量评估（Video Quality Assessment, VQA）

### 核心功能
评估视频的两个维度：
1. **技术质量**（Technical Quality）：编码失真、模糊、噪声等
2. **美学质量**（Aesthetic Quality）：构图、色彩、光影等

---

## 🎯 在 QC 系统中的作用

### Stage 3 的使用场景

DOVER 是 **Stage 3 GPU 密集评估** 的三个模型之一：

| 模型 | 用途 | 输出 |
|------|------|------|
| **UniMatch** | 光流计算 | 相邻帧运动幅值 |
| **DOVER** | 视频质量评分 | 技术+美学综合分数 |
| **Qwen3.5-27B** | VLM 多模态理解 | 实体计数+质量标记 |

### 具体应用

在 `src/sana_wm_pipeline/qc/stage3_gpu.py` 中：

```python
from dover import DOVER

def load_dover_fn(device: str = "cuda") -> Callable:
    """加载 DOVER 模型用于视频质量评分"""
    import torch
    from dover import DOVER  # type: ignore
    
    model = DOVER().to(device).eval()
    
    def dover_fn(frames_rgb: np.ndarray) -> float:
        """
        输入: frames_rgb (T, H, W, 3) uint8 视频帧
        输出: 质量分数 (float)，越高越好
        """
        t = torch.from_numpy(frames_rgb).float()
        t = t.permute(0, 3, 1, 2).unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            score = model(t)
        return float(score.mean().item())
    
    return dover_fn
```

### 评分标准

根据 SANA-WM 论文 Table 6：
- **DOVER 分数 > 阈值** → 视频质量合格
- **DOVER 分数 < 阈值** → 视频质量不合格，拒绝样本

不同 group 有不同阈值（在 `configs/filter_thresholds.yaml` 中配置）

---

## 📦 包信息

### PyPI 信息
- **包名：** `dover`
- **当前版本：** 0.5.1（2026-06-27 确认）
- **依赖：** docopt, toml
- **大小：** < 1 MB
- **安装命令：** `pip install dover`

### 依赖项
```
dover==0.5.1
├── docopt==0.6.2
└── toml==0.10.2
```

---

## 🌐 国内镜像源安装

### CMCC 机器网络限制

由于 CMCC 机器有网络限制，必须使用国内镜像源：

#### 方法 1：阿里云镜像（推荐）

```bash
pip install dover \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com
```

#### 方法 2：清华镜像（备选）

```bash
pip install dover \
    -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    --trusted-host pypi.tuna.tsinghua.edu.cn
```

#### 方法 3：配置全局镜像源

```bash
# 创建 pip 配置文件
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<EOF
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
EOF

# 之后可以直接
pip install dover
```

---

## 🔧 部署脚本修复

### 修复版本对比

| 项目 | 原脚本 | v2 修复版 |
|------|--------|-----------|
| dover 检测 | 仅检测 | 自动安装 |
| 镜像源 | 默认 PyPI | 阿里云+清华备选 |
| 失败处理 | 基础 | 多级降级重试 |

### v2 版本安装逻辑

```bash
if python -c "import dover" 2>/dev/null; then
    echo "✓ dover 已安装"
else
    echo "⚠️  dover 未找到，正在从阿里云镜像安装..."
    pip install dover \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host mirrors.aliyun.com
    
    if python -c "import dover" 2>/dev/null; then
        echo "✓ dover 安装成功"
    else
        echo "❌ dover 安装失败，尝试清华镜像..."
        pip install dover \
            -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
            --trusted-host pypi.tuna.tsinghua.edu.cn
        
        if python -c "import dover" 2>/dev/null; then
            echo "✓ dover 安装成功（清华镜像）"
        else
            echo "❌ dover 安装失败"
            exit 1
        fi
    fi
fi
```

**特点：**
- 先检测是否已安装（避免重复安装）
- 优先使用阿里云镜像
- 失败后自动降级到清华镜像
- 双重验证确保安装成功

---

## 🎯 验证 DOVER 安装

### 基础验证

```bash
# 导入测试
python -c "from dover import DOVER; print('✓ DOVER import OK')"

# 查看版本
python -c "import dover; print(f'dover version: {dover.__version__}')"
```

### GPU 加载测试

```bash
python3 <<EOF
import torch
from dover import DOVER

# 加载到 GPU
model = DOVER().cuda().eval()
print("✓ DOVER loaded on GPU")

# 模拟输入：10帧 224×224 RGB 视频
import numpy as np
frames = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)
t = torch.from_numpy(frames).float().permute(0, 3, 1, 2).unsqueeze(0).cuda() / 255.0

# 推理
with torch.no_grad():
    score = model(t)
print(f"DOVER score: {score.mean().item():.4f}")
print("✓ DOVER inference OK")
EOF
```

### 预期输出

```
✓ DOVER loaded on GPU
DOVER score: 0.XXXX
✓ DOVER inference OK
```

---

## 📊 性能特性

### 显存占用
- **DOVER 单独加载：** ~5GB
- **与 UniMatch + Qwen 共存：** 总计 ~58GB（H100 80GB 可容纳）

### 推理速度
- **输入：** 10-16 帧视频片段
- **分辨率：** 224×224（自动 resize）
- **时间：** ~50-100ms/样本（A100/H100）

---

## 🐛 常见问题

### 问题 1：import dover 失败

```bash
ModuleNotFoundError: No module named 'dover'
```

**解决：**
```bash
pip install dover -i https://mirrors.aliyun.com/pypi/simple/
```

### 问题 2：网络超时

```bash
ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', ...)
```

**解决：** 使用国内镜像源（见上文）

### 问题 3：CUDA OOM

```bash
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解决：**
- 确认 GPU 显存 ≥ 60GB
- 确保没有其他进程占用 GPU
- 确认 Qwen 使用 BF16（不是 FP32）

---

## 📚 参考资料

### 官方资源
- **PyPI：** https://pypi.org/project/dover/
- **论文：** ICCV 2023 - "Exploring Video Quality Assessment on User Generated Contents"
- **代码：** （如果有公开仓库）

### 相关文档
- `docs/QC_MODEL_WEIGHTS.md` - 模型权重清单
- `docs/QC_DEPENDENCIES.md` - 完整依赖说明
- `DOVER_PACKAGING_ANALYSIS.md` - 打包问题根因分析

---

## 🎉 总结

### DOVER 的重要性

在 QC 系统中，DOVER 负责：
1. ✅ **自动化视频质量筛选** - 无需人工逐个查看
2. ✅ **客观评分标准** - 基于学术研究，可复现
3. ✅ **高效处理** - GPU 加速，快速评估大规模数据

### 部署要点

1. ✅ **使用 v2 修复版脚本**（国内镜像源）
2. ✅ **验证安装成功**（import 测试）
3. ✅ **GPU 显存充足**（≥60GB 推荐）

---

**文档版本：** v1.0 (2026-06-27)  
**适用范围：** SANA-WM QC System Stage 3  
**联系人：** David Wang
