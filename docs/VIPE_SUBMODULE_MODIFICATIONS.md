# VIPE Submodule 修改记录（阶段2）

> **重要**: VIPE是第三方submodule，这些修改仅存在于本地工作树，未提交到submodule仓库。
> 部署时需要手动应用这些修改。

## 修改文件

### 1. `third_party/vipe/vipe/priors/depth/pi3xmoge.py` (新增)

**来源**: `sana-wm-data-clean/vipe_patches/pi3x_moge_depth.py`

**功能**: Pi3xMogeModel深度后端
- 加载预计算的融合深度 (fused.npy)
- 使用RGB 16x16签名匹配帧 (sig.npy)
- 通过 `SANA_WM_FUSED_DEPTH_DIR` 环境变量指定数据目录

**关键特性**:
- RGB 16x16 (768维) 签名，比灰度8x8 (64维) 更robust
- 近似重复帧可分离，避免BA获得错误帧的深度

### 2. `third_party/vipe/vipe/priors/depth/__init__.py` (修改)

**变更**: 在 `make_depth_model()` 中注册 `pi3xmoge` 模型

```python
elif model_name == "pi3xmoge":
    from .pi3xmoge import Pi3xMogeModel
    return Pi3xMogeModel()
```

### 3. `third_party/vipe/configs/pipeline/vipe_sanawm.yaml` (新增)

**来源**: `sana-wm-data-clean/vipe_patches/sanawm_pipeline.yaml`

**配置**:
```yaml
slam:
  keyframe_depth: pi3xmoge          # 使用Pi3xMogeModel
  optimize_intrinsics: true         # 启用内参优化
  ba:
    fused: false                    # 禁用fused CUDA kernel（支持逐帧内参）
```

## 部署说明

### 方式1: 直接复制（推荐用于CMCC）

```bash
# 在目标环境中
cd /path/to/sana_wm_pipeline

# 复制文件
cp sana-wm-data-clean/vipe_patches/pi3x_moge_depth.py \
   third_party/vipe/vipe/priors/depth/pi3xmoge.py

# 注册模型（手动编辑或使用patch）
# 编辑 third_party/vipe/vipe/priors/depth/__init__.py
# 在 make_depth_model() 最后一个 elif 后添加:
#   elif model_name == "pi3xmoge":
#       from .pi3xmoge import Pi3xMogeModel
#       return Pi3xMogeModel()

# 复制配置
cp sana-wm-data-clean/vipe_patches/sanawm_pipeline.yaml \
   third_party/vipe/configs/pipeline/vipe_sanawm.yaml
```

### 方式2: Git Patch（用于版本控制）

```bash
# 在本地生成patch
cd third_party/vipe
git diff > ../../vipe_modifications.patch

# 在目标环境应用
cd third_party/vipe
git apply ../../vipe_modifications.patch
```

## 验证

```bash
# 测试模型注册
python -c "
from vipe.priors.depth import make_depth_model
model = make_depth_model('pi3xmoge')
print(f'✅ pi3xmoge模型加载成功: {type(model)}')
"

# 测试配置文件
vipe infer --help | grep vipe_sanawm
# 应该在可用pipeline列表中看到 vipe_sanawm
```

## 相关环境变量

- `SANA_WM_FUSED_DEPTH_DIR`: Pi3xMogeModel加载预计算深度的目录
- `SANA_WM_PI3X_WEIGHTS`: Pi3X模型权重路径
- `SANA_WM_MOGE2_WEIGHTS`: MoGe-2模型权重路径

## 回退

如果需要回退到原始CachedDepthModel:

```bash
# mode_default.py 中修改:
pipeline: str = "vipe_cached_depth"  # 改回原配置

# 并使用旧的inline预计算（需要恢复 _precompute_depth_cache 函数）
```
