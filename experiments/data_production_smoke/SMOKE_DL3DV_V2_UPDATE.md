# DL3DV 冒烟测试脚本更新说明

**日期**: 2026-09-11  
**版本**: v2.0 - 自动扫描数据目录  
**状态**: ✅ 已更新

---

## 🔄 主要改动

### 改动前 (v1.0)

```bash
# 硬编码样本列表
cat > "$SAMPLES_FILE" << 'EOF'
DL3DV-...__images_2	303
DL3DV-...__images_2	301
...
EOF
```

**问题**:
- ❌ 需要手动维护样本列表
- ❌ 添加/删除样本需要修改脚本
- ❌ 帧数需要手动指定
- ❌ 容易出错

### 改动后 (v2.0)

```bash
# 自动扫描数据目录
for mp4_file in "$DL3DV_DIR"/*.mp4; do
    sample_id=$(basename "$mp4_file" .mp4)
    camera_file="$DL3DV_DIR/${sample_id}.camera.npz"
    
    # 自动读取帧数
    n_frames=$(python3 -c "
        import numpy as np
        data = np.load('$camera_file')
        print(data['c2w'].shape[0])
    ")
    
    echo "$sample_id	$n_frames" >> samples.tsv
done
```

**优势**:
- ✅ 自动发现所有样本
- ✅ 自动读取帧数
- ✅ 鲁棒的错误处理
- ✅ 支持动态样本集

---

## 📂 新的数据目录

### 路径

```bash
export DL3DV_DIR="/mnt/afs/davidwang/workspace/sana_test_data/smoke_dl3dv_test_sample"
```

**之前**: `/mnt/afs/.../Dl3dv/` (117 个样本的完整目录)  
**现在**: `/mnt/afs/.../smoke_dl3dv_test_sample/` (5 个测试样本)

### 目录结构

```
smoke_dl3dv_test_sample/
├── DL3DV-ALL-2K_10K__a1cc9c41...__images_2.mp4          (5.6M)
├── DL3DV-ALL-2K_10K__a1cc9c41...__images_2.camera.npz   (66K)
├── DL3DV-ALL-2K_10K__b137b3eb...__images_2.mp4          (6.5M)
├── DL3DV-ALL-2K_10K__b137b3eb...__images_2.camera.npz   (66K)
├── DL3DV-ALL-2K_10K__9eb0f6c5...__images_2.mp4          (19M)
├── DL3DV-ALL-2K_10K__9eb0f6c5...__images_2.camera.npz   (66K)
├── DL3DV-ALL-2K_10K__9d44d654...__images_2.mp4          (28M)
├── DL3DV-ALL-2K_10K__9d44d654...__images_2.camera.npz   (66K)
├── DL3DV-ALL-2K_10K__a42b0f54...__images_2.mp4          (48M)
└── DL3DV-ALL-2K_10K__a42b0f54...__images_2.camera.npz   (66K)
```

---

## 🔍 扫描逻辑详解

### 1. 验证数据目录

```bash
if [[ ! -d "$DL3DV_DIR" ]]; then
    echo "❌ 错误：数据目录不存在: $DL3DV_DIR"
    exit 1
fi
```

### 2. 扫描 .mp4 文件

```bash
for mp4_file in "$DL3DV_DIR"/*.mp4; do
    # 处理通配符未匹配的情况
    [[ -f "$mp4_file" ]] || continue
    
    sample_id=$(basename "$mp4_file" .mp4)
```

### 3. 检查配对的 camera.npz

```bash
camera_file="$DL3DV_DIR/${sample_id}.camera.npz"
if [[ ! -f "$camera_file" ]]; then
    echo "  ⚠️  跳过 $sample_id (缺少 camera.npz)"
    continue
fi
```

### 4. 自动读取帧数

```bash
n_frames=$(python3 -c "
import numpy as np
import sys
try:
    data = np.load('$camera_file')
    print(data['c2w'].shape[0])
except Exception as e:
    print('0', file=sys.stderr)
    sys.exit(1)
")
```

### 5. 验证有效性

```bash
if [[ "$n_frames" =~ ^[0-9]+$ ]] && [[ "$n_frames" -gt 0 ]]; then
    echo "$sample_id	$n_frames" >> samples.tsv
    echo "  ✅ $sample_id ($n_frames 帧)"
else
    echo "  ⚠️  跳过 $sample_id (无法读取帧数)"
fi
```

---

## 🛡️ 鲁棒性增强

### 错误处理

| 场景 | 处理方式 |
|------|----------|
| **数据目录不存在** | 提前退出，显示错误 |
| **无 .mp4 文件** | 通配符保护，不会报错 |
| **缺少 camera.npz** | 跳过该样本，继续处理 |
| **camera.npz 损坏** | Python 异常捕获，跳过 |
| **帧数无效** | 正则验证，跳过无效值 |
| **未找到任何样本** | 退出前显示帮助信息 |

### 输出示例

#### 成功扫描

```
=== 扫描数据目录中的样本 ===
  ✅ DL3DV-ALL-2K_10K__a1cc9c41...__images_2 (303 帧)
  ✅ DL3DV-ALL-2K_10K__b137b3eb...__images_2 (303 帧)
  ✅ DL3DV-ALL-2K_10K__9eb0f6c5...__images_2 (301 帧)
  ✅ DL3DV-ALL-2K_10K__9d44d654...__images_2 (303 帧)
  ✅ DL3DV-ALL-2K_10K__a42b0f54...__images_2 (301 帧)

共发现 5 个有效样本
样本列表已保存到: /mnt/afs/.../samples.tsv
```

#### 部分样本无效

```
=== 扫描数据目录中的样本 ===
  ✅ DL3DV-...__images_2 (303 帧)
  ⚠️  跳过 DL3DV-...__images_2 (缺少 camera.npz)
  ⚠️  跳过 DL3DV-...__images_2 (无法读取帧数)
  ✅ DL3DV-...__images_2 (301 帧)

共发现 2 个有效样本
```

#### 无有效样本

```
=== 扫描数据目录中的样本 ===
  ⚠️  跳过 sample1 (缺少 camera.npz)
  ⚠️  跳过 sample2 (无法读取帧数)

❌ 错误：未在 /mnt/afs/.../test_sample 中找到有效样本
   每个样本需要：
     - {sample_id}.mp4
     - {sample_id}.camera.npz (包含 'c2w' 数据)
```

---

## 🚀 使用方法

### 基本用法

```bash
# 直接运行，自动扫描 smoke_dl3dv_test_sample/
bash experiments/data_production_smoke/smoke_dl3dv.sh
```

### 使用自定义目录

```bash
# 临时修改数据目录
export DL3DV_DIR="/path/to/your/samples"
bash experiments/data_production_smoke/smoke_dl3dv.sh
```

### 添加新样本

```bash
# 1. 复制样本到测试目录
cp sample.mp4 sample.camera.npz smoke_dl3dv_test_sample/

# 2. 直接运行测试（自动发现）
bash experiments/data_production_smoke/smoke_dl3dv.sh
```

### 删除样本

```bash
# 1. 删除文件
rm smoke_dl3dv_test_sample/sample.*

# 2. 运行测试（自动跳过）
bash experiments/data_production_smoke/smoke_dl3dv.sh
```

---

## 📊 兼容性

### 向后兼容

- ✅ 输出格式不变（samples.tsv 仍然是 TSV 格式）
- ✅ Python 脚本无需修改
- ✅ 环境变量保持一致

### 数据要求

**每个样本必须包含**:
1. `{sample_id}.mp4` - 视频文件
2. `{sample_id}.camera.npz` - 包含 `c2w` 字段的 numpy 压缩文件

**可选**:
- `{sample_id}.camera.npz` 中的 `K_px` 字段（GT intrinsics）

---

## 🎯 优势总结

| 维度 | v1.0 (手动) | v2.0 (自动扫描) |
|------|-------------|-----------------|
| **样本管理** | 手动编辑脚本 | 自动发现 ✅ |
| **帧数获取** | 手动指定 | 自动读取 ✅ |
| **添加样本** | 修改脚本 | 复制文件即可 ✅ |
| **删除样本** | 修改脚本 | 删除文件即可 ✅ |
| **错误处理** | 基础 | 完善 ✅ |
| **可维护性** | 中等 | 高 ✅ |
| **易用性** | 中等 | 高 ✅ |

---

## ✅ 测试验证

```bash
# 运行测试
bash experiments/data_production_smoke/smoke_dl3dv.sh

# 预期输出
=== 扫描数据目录中的样本 ===
  ✅ ... (303 帧)
  ✅ ... (303 帧)
  ✅ ... (301 帧)
  ✅ ... (303 帧)
  ✅ ... (301 帧)

共发现 5 个有效样本

=== 验证样本文件完整性 ===
  ✅ ... (303 帧)
     video: 5.6M, camera: 66K
  ...

✅ 所有样本文件验证通过

=== 启动批量处理 ===
...
```

---

**更新完成！脚本现在更加鲁棒和易用！** 🐴
