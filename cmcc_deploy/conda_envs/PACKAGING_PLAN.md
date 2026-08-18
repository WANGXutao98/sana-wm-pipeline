# Conda 环境打包方案

**日期**: 2026-08-17  
**目标**: 打包 sana_qc 和 sana_wm 环境，上传到 ModelScope  
**原则**: 不影响本机原有环境

---

## 一、问题发现

### 1.1 可编辑包冲突
```
CondaPackError: Cannot pack an environment with editable packages
- nvidia_vipe
- sana_wm_pipeline
```

### 1.2 解决方案对比

| 方案 | 优点 | 缺点 | 风险 |
|------|------|------|------|
| `--ignore-editable-packages` | 简单快速 | 目标环境缺少这两个包 | ⚠️ CMCC 无法运行 |
| 克隆环境后打包 | 不影响原环境 | 需要额外磁盘空间 | ✓ 安全 |
| 直接 tar 环境目录 | 完整保留 | 体积大，不压缩 | ✓ 安全 |

**选定方案**: 克隆环境 + 安装非可编辑版本 + 打包

---

## 二、执行计划

### 2.1 环境克隆与清理

**步骤**:
1. 克隆原始环境（只复制非可编辑包）
2. 在新环境中安装 vipe 和 sana_wm_pipeline 的wheel/源码
3. 打包新环境
4. 删除临时克隆环境

**命令序列**:
```bash
# 1. 克隆 sana_qc（排除可编辑包）
conda create -n sana_qc_clean --clone sana_qc --copy

# 2. 激活并安装缺失包
conda activate sana_qc_clean
# 安装 vipe（从 third_party/vipe）
pip install /mnt/afs/davidwang/workspace/sana_wm_pipeline/third_party/vipe
# 安装 sana_wm_pipeline（从项目根）
pip install /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 3. 打包
conda-pack -n sana_qc_clean -o sana_qc_clean.tar.gz --compress-level 6

# 4. 清理
conda deactivate
conda env remove -n sana_qc_clean
```

**磁盘占用估算**:
- 原环境: ~5GB × 2 = 10GB
- 克隆环境: ~5GB × 2 = 10GB（临时）
- 压缩包: ~2GB × 2 = 4GB
- **峰值需求**: 14GB（克隆时）

### 2.2 环境安全检查清单

**执行前验证**:
- [ ] 原环境 sana_qc 可激活
- [ ] 原环境 sana_wm 可激活
- [ ] 磁盘空间 > 20GB
- [ ] 输出目录存在且可写

**执行后验证**:
- [ ] 压缩包文件存在
- [ ] 压缩包大小 > 500MB
- [ ] 原环境未被修改（检查 mtime）
- [ ] 临时环境已删除

---

## 三、方案 B: 直接 tar（备选）

**适用场景**: 如果克隆失败或磁盘空间不足

```bash
cd /mnt/afs/davidwang/miniconda3/envs

# 打包 sana_qc（保留所有可编辑包）
tar -czf /mnt/afs/davidwang/workspace/sana_wm_pipeline/cmcc_deploy/conda_envs/sana_qc_clean.tar.gz \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    sana_qc

# 打包 sana_wm
tar -czf /mnt/afs/davidwang/workspace/sana_wm_pipeline/cmcc_deploy/conda_envs/sana_wm_clean.tar.gz \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    sana_wm
```

**优点**: 
- 完整保留可编辑包
- 不创建临时环境

**缺点**:
- 压缩包更大（~3-4GB）
- 包含绝对路径，需解压到相同路径

---

## 四、CMCC 端解压方案

### 4.1 方案 A 解压（conda-pack）
```bash
# 创建目标目录
mkdir -p /root/work/david_work/envs/sana_qc_clean
mkdir -p /root/work/david_work/envs/sana_wm_clean

# 解压
tar -xzf sana_qc_clean.tar.gz -C /root/work/david_work/envs/sana_qc_clean
tar -xzf sana_wm_clean.tar.gz -C /root/work/david_work/envs/sana_wm_clean

# 激活
source /root/work/david_work/envs/sana_qc_clean/bin/activate
```

### 4.2 方案 B 解压（tar）
```bash
# 必须解压到 miniconda3/envs 下
cd /opt/conda  # CMCC 的 conda 路径
tar -xzf sana_qc_clean.tar.gz -C envs/
tar -xzf sana_wm_clean.tar.gz -C envs/

# 激活
conda activate sana_qc_clean
```

---

## 五、ModelScope 上传

```bash
# 安装 modelscope CLI
pip install modelscope

# 登录（需要 token）
modelscope login --token <YOUR_TOKEN>

# 上传文件
modelscope upload \
    --dataset davidxwang/sana_spatialvid_smoke_data \
    --local_path sana_qc_clean.tar.gz \
    --remote_path envs/sana_qc_clean.tar.gz

modelscope upload \
    --dataset davidxwang/sana_spatialvid_smoke_data \
    --local_path sana_wm_clean.tar.gz \
    --remote_path envs/sana_wm_clean.tar.gz
```

---

## 六、回滚方案

### 6.1 如果克隆环境失败
```bash
# 删除失败的克隆
conda env remove -n sana_qc_clean -y
conda env remove -n sana_wm_clean -y

# 切换到方案 B（直接 tar）
```

### 6.2 如果原环境被意外修改
```bash
# 检查修改时间
ls -la /mnt/afs/davidwang/miniconda3/envs/sana_qc/
ls -la /mnt/afs/davidwang/miniconda3/envs/sana_wm/

# 如果有备份，恢复：
# （建议执行前先备份）
tar -xzf sana_qc_backup.tar.gz -C /mnt/afs/davidwang/miniconda3/envs/
```

---

## 七、执行检查点

### Checkpoint 1: 开始前
- [ ] 已阅读完整方案
- [ ] 已确认磁盘空间充足
- [ ] 已备份关键环境（可选）
- [ ] 已创建输出目录

### Checkpoint 2: 克隆后
- [ ] 克隆环境可激活
- [ ] 可编辑包已正确安装
- [ ] 原环境未被修改

### Checkpoint 3: 打包后
- [ ] 压缩包文件存在
- [ ] 压缩包完整性校验通过
- [ ] 临时环境已清理

### Checkpoint 4: 上传后
- [ ] ModelScope 可访问文件
- [ ] 文件 MD5 校验一致

---

## 八、推荐执行方案

**最终建议**: 使用方案 B（直接 tar）

**理由**:
1. **简单可靠**: 一条命令完成
2. **完整保留**: 可编辑包不丢失
3. **无风险**: 不创建/修改任何环境
4. **易回溯**: 原样打包，原样恢复

**唯一代价**: 压缩包稍大（~4GB vs ~2GB）

---

**下一步**: 等待用户确认方案后执行
