# CMCC Conda 环境部署文件清单

**生成日期**: 2026-08-17  
**目标**: CMCC 无外网集群离线部署 sana_qc_clean & sana_wm_clean 环境

---

## 一、已生成文件

### 1. 环境压缩包（本地）

| 文件 | 大小 | MD5 | 用途 |
|------|------|-----|------|
| `sana_qc_clean.tar.gz` | 3.7GB | ✓ | QC 环境离线包 |
| `sana_wm_clean.tar.gz` | 3.7GB | ✓ | WM 环境离线包 |
| `sana_qc_clean.tar.gz.md5` | 67B | - | 校验文件 |
| `sana_wm_clean.tar.gz.md5` | 67B | - | 校验文件 |

**位置**: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/cmcc_deploy/conda_envs/`

---

### 2. 部署文档

| 文件 | 行数 | 用途 |
|------|------|------|
| `PACKAGING_PLAN.md` | 161 | 打包方案记录（已完成） |
| `CMCC_DEPLOYMENT_GUIDE.sh` | 273 | 一键自动部署脚本 |
| `CMCC_MANUAL_DEPLOYMENT.md` | 468 | 手动逐步部署指南 |
| `CMCC_USAGE_GUIDE.md` | 197 | CMCC 端使用指南（旧版，已被 MANUAL 替代） |
| `upload_to_modelscope.sh` | 75 | ModelScope 上传脚本 |
| `DEPLOYMENT_SUMMARY.md` | 本文件 | 文件清单与使用说明 |

**位置**: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/cmcc_deploy/conda_envs/`

---

## 二、部署流程

### 本地端（已完成）

1. ✅ 打包 sana_qc → sana_qc_clean.tar.gz (3.7GB, 57990 文件)
2. ✅ 打包 sana_wm → sana_wm_clean.tar.gz (3.7GB, 57990 文件)
3. ✅ 生成 MD5 校验文件
4. ✅ 生成部署脚本与文档
5. ⏳ **待执行**: 上传到 ModelScope

### CMCC 端（待执行）

选择以下任一方案：

#### 方案 A: 自动部署（推荐）

```bash
# 1. 下载部署脚本
cd /root/work/david_work
wget https://modelscope.cn/datasets/davidxwang/sana_spatialvid_smoke_data/resolve/master/conda_envs/CMCC_DEPLOYMENT_GUIDE.sh

# 2. 执行一键部署
bash CMCC_DEPLOYMENT_GUIDE.sh
```

**耗时**: 15-20 分钟  
**功能**: 自动完成下载、校验、解压、路径修复、可编辑包安装、验证

#### 方案 B: 手动部署

参考 `CMCC_MANUAL_DEPLOYMENT.md` 逐步执行 7 个步骤。

---

## 三、上传到 ModelScope

### 执行命令

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/cmcc_deploy/conda_envs

# 1. 登录 ModelScope（如未登录）
modelscope login --token YOUR_TOKEN

# 2. 执行上传
bash upload_to_modelscope.sh
```

### 上传内容

上传到数据集: `davidxwang/sana_spatialvid_smoke_data`

```
conda_envs/
├── sana_qc_clean.tar.gz         (3.7GB)
├── sana_wm_clean.tar.gz         (3.7GB)
├── sana_qc_clean.tar.gz.md5     (67B)
├── sana_wm_clean.tar.gz.md5     (67B)
└── CMCC_DEPLOYMENT_GUIDE.sh     (8KB) ← 同时上传部署脚本
```

---

## 四、CMCC 端部署后目录结构

```
/root/work/david_work/
├── conda_envs_download/              # 下载缓存（可删除）
│   ├── sana_qc_clean.tar.gz          (3.7GB)
│   ├── sana_wm_clean.tar.gz          (3.7GB)
│   └── *.md5
│
├── conda_envs/                        # 环境安装目录
│   ├── sana_qc_clean/                # QC 环境（8GB）
│   │   ├── bin/python                # Python 解释器
│   │   ├── lib/python3.10/           # 包目录
│   │   └── conda-meta/               # 元数据
│   │
│   ├── sana_wm_clean/                # WM 环境（7.5GB）
│   │   ├── bin/python
│   │   ├── lib/python3.10/
│   │   └── conda-meta/
│   │
│   ├── activate_qc.sh                # QC 快捷激活
│   └── activate_wm.sh                # WM 快捷激活
│
└── sana_wm_optimized/                # 项目代码
    └── sana_wm_pipeline/
        ├── third_party/vipe/         # vipe 源码（可编辑安装）
        ├── src/sana_wm_pipeline/     # 管线代码（可编辑安装）
        └── experiments/data_production_smoke/
            └── smoke_cmcc_pass.sh    # 冒烟测试脚本
```

**总磁盘占用**: 
- 下载目录: 7.4GB（可删除）
- 安装目录: 15.5GB（永久保留）

---

## 五、验证清单

CMCC 端部署完成后，依次验证：

### 基础环境验证

```bash
# 激活 sana_qc_clean
source /root/work/david_work/conda_envs/sana_qc_clean/bin/activate
python --version                      # 应显示 Python 3.10.x
python -c "import torch; print(torch.__version__)"  # 应显示 2.12.0+cu130
python -c "import torch; print(torch.cuda.is_available())"  # 应显示 True
conda deactivate

# 激活 sana_wm_clean
source /root/work/david_work/conda_envs/sana_wm_clean/bin/activate
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import vipe; print('vipe OK')"
python -c "import sana_wm_pipeline; print('sana_wm_pipeline OK')"
conda deactivate
```

### 冒烟测试验证

```bash
source /root/work/david_work/conda_envs/activate_wm.sh
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline
bash experiments/data_production_smoke/smoke_cmcc_pass.sh
```

**预期**: 所有测试样本成功生成 normalized.mp4 + pose_artifact_default.json + shard.tar

---

## 六、关键差异说明

### 环境重命名原因

| 本地环境 | CMCC 环境 | 原因 |
|----------|-----------|------|
| `sana_qc` | `sana_qc_clean` | 避免与 CMCC 集群已有环境冲突 |
| `sana_wm` | `sana_wm_clean` | 避免与 CMCC 集群已有环境冲突 |

### 路径修复必要性

**本地硬编码路径**: `/mnt/afs/davidwang/miniconda3/envs/sana_qc`  
**CMCC 目标路径**: `/root/work/david_work/conda_envs/sana_qc_clean`

**需要修复的位置**:
1. `conda-meta/*.json` - conda 包元数据
2. `bin/*` - 所有可执行文件的 shebang
3. `lib/**/*.py` 和 `*.pth` - Python 路径配置

**自动修复**: `CMCC_DEPLOYMENT_GUIDE.sh` 的步骤 4 自动完成  
**手动修复**: 参考 `CMCC_MANUAL_DEPLOYMENT.md` 的步骤 4

### 可编辑包处理

**包含的可编辑包**:
- `nvidia-vipe`: 位于 `third_party/vipe/`
- `sana-wm-pipeline`: 项目根目录

**处理方式**:
1. 压缩包包含原始安装（指向本地路径）
2. CMCC 端解压后重新执行 `pip install -e <path> --no-deps`
3. 无需网络，直接链接到 CMCC 项目路径

---

## 七、常见问题

### Q: 为什么不用 conda-pack？

**A**: conda-pack 不支持可编辑包，会报错：
```
CondaPackError: Cannot pack an environment with editable packages
```

方案 B（直接 tar）完整保留所有包，包括可编辑安装。

### Q: 压缩包为什么这么大？

**A**: 包含完整的依赖栈：
- PyTorch 2.12.0 + CUDA 13.0 (~2GB)
- 第三方依赖 ~5GB（ffmpeg、opencv、numpy 等）
- 总计 ~8GB 未压缩，gzip 后 ~3.7GB

### Q: CMCC 端 CUDA 版本不匹配怎么办？

**A**: 本地 CUDA 13.0，CMCC CUDA 12.4。PyTorch 2.12.0 支持 12.4+，兼容无问题。

如遇 JIT 编译问题，设置：
```bash
export TORCH_CUDA_ARCH_LIST="9.0"  # H100 架构
```

---

## 八、下一步行动

### 本地端（待完成）

- [ ] 执行 `bash upload_to_modelscope.sh` 上传环境到 ModelScope
- [ ] 验证上传成功（访问数据集页面确认文件）
- [ ] 同时上传 `CMCC_DEPLOYMENT_GUIDE.sh` 到 ModelScope

### CMCC 端（待执行）

- [ ] 下载部署脚本或文档
- [ ] 执行部署（方案 A 或 B）
- [ ] 验证环境可用性
- [ ] 运行冒烟测试
- [ ] 记录部署日志（如遇问题）

---

## 九、联系与支持

**项目路径**: `/mnt/afs/davidwang/workspace/sana_wm_pipeline`

**参考文档**:
- 打包方案: `PACKAGING_PLAN.md`
- 自动部署: `CMCC_DEPLOYMENT_GUIDE.sh`
- 手动部署: `CMCC_MANUAL_DEPLOYMENT.md`
- CMCC 适配: `CMCC_ADAPTATION_PLAN.md`
- 排查记录: `task_progress_cmcc_refine.md`
- 冒烟测试: `task_plan_spatialvid_smoke.md`

**关键命令速查**:
```bash
# 本地上传
bash upload_to_modelscope.sh

# CMCC 自动部署
bash CMCC_DEPLOYMENT_GUIDE.sh

# CMCC 快捷激活
source /root/work/david_work/conda_envs/activate_wm.sh

# 运行冒烟测试
bash experiments/data_production_smoke/smoke_cmcc_pass.sh
```
