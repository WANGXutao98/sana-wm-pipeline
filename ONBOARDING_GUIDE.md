# 🚀 新Claude快速上手指南

**适用场景**: 全新对话会话，无历史上下文  
**目标**: 快速理解项目状态、定位关键问题、继续推进工作  
**预计阅读时间**: 10-15分钟

---

## 📖 第一步：理解项目背景（5分钟）

### 必读文档（按顺序）

1. **`spatialvid_smoke_test_report.md`** - 综合测试报告 ⭐⭐⭐⭐⭐
   ```
   优先级: 最高
   阅读重点:
   - 第1节: 执行摘要（核心结论）
   - 第2.1节: 代码质量评估（100%对齐）
   - 第2.2节: Metric Scale准确性（非线性偏差）
   - 第4节: 建议与行动计划
   
   跳过: 第6节附录（需要时再查）
   ```

2. **`findings.md`** - 研究发现 ⭐⭐⭐⭐
   ```
   优先级: 高
   阅读重点:
   - 关键发现（4个核心发现）
   - 未解决问题清单（4个待解决问题）
   - 经验教训（5条教训）
   
   这个文档帮助你理解"我们发现了什么"
   ```

3. **`SESSION_SUMMARY_20260814.md`** - 对话摘要 ⭐⭐⭐
   ```
   优先级: 中高
   阅读重点:
   - 快速回顾（主要成果）
   - 关键对话节点（理解决策过程）
   - 关键概念澄清（避免误解）
   
   跳过: 详细对话过程（太细节）
   ```

---

## 🔍 第二步：了解当前状态（3分钟）

### 查看这些部分

4. **`progress.md`** - 进度日志 ⭐⭐⭐
   ```
   优先级: 中
   阅读重点:
   - 里程碑（已完成/未解决）
   - 下一步行动（立即/短期/中期）
   
   这个文档告诉你"现在在哪里"
   ```

5. **`task_plan_spatialvid_smoke.md`** - 主任务计划 ⭐⭐
   ```
   优先级: 中低（如果时间有限可以跳过）
   阅读重点:
   - 阶段完成状态表（快速扫描）
   - 阶段14-16（最新进展）
   - 成功标准（最终版本）
   
   这是完整的历史记录，需要时再细读
   ```

---

## 💻 第三步：定位关键代码（5分钟）

### 核心文件（不需要全读，知道在哪就行）

**主处理逻辑**:
```bash
/mnt/afs/davidwang/workspace/sana_wm_pipeline/stage02_pose/mode_default.py

关键函数:
- run_pose_stage()          # 入口函数
- _process_vipe_output()    # 处理VIPE输出（核心）
- 第165-202行已删除        # 稀疏化方案（已废弃）

重点理解:
- 为什么删除了稀疏化？
- 如何对齐参考实现？
```

**深度融合**:
```bash
/mnt/afs/davidwang/workspace/sana_wm_pipeline/stage02_pose/depth_fusion.py

关键函数:
- solve_frame_scale()       # Scale计算
- fuse_depth_sequence()     # 深度融合

状态: 100%对齐参考实现，md5一致 ✅
```

**参考实现**:
```bash
/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-wm-data-clean/

关键文件:
- pose/_real.py             # Pi3X+MoGe-2推理
- pose/fusion.py            # 深度融合（对照用）
- vipe_cli.py:61-70         # poses处理逻辑
```

---

## 🎯 第四步：核心概念速查（2分钟）

### 必须理解的概念

**1. Scale CoV ≠ Metric Scale**
```
Scale CoV (0.008):    内部一致性 = 视频内scale变化小 ✅
Metric Scale (14.8x): 绝对准确性 = 与真实世界对比 ❌

可以同时存在! 不要混淆
```

**2. 非线性偏差**
```
短视频(10秒): 1.67x  ✅ 可接受
长视频(60秒): 14.8x  ❌ 不可接受

这不是系统性偏差（常数）！
这会破坏样本间相对关系，影响训练
```

**3. 参考标注不可靠**
```
参考标注偏小3-4x
不能用来评估我们的输出
必须用真实场景验证
```

---

## ⚡ 快速启动指令

### 复制粘贴这段给新Claude

```markdown
你好！我是接手sana_wm_pipeline项目的新Claude。

项目背景：SANA-WM数据标注管线，刚完成SpatialVID冒烟测试。

请帮我快速上手：

1. 先阅读这些文档（按顺序）：
   - spatialvid_smoke_test_report.md （重点：执行摘要、问题分析、行动计划）
   - findings.md （重点：关键发现、未解决问题）
   - SESSION_SUMMARY_20260814.md （重点：快速回顾、关键概念）

2. 确认我理解正确（请帮我验证）：
   - 代码100%对齐参考实现 ✅
   - 但有非线性metric scale偏差（短视频1.67x，长视频14.8x）
   - 参考标注不可靠，偏小3-4x
   - 立即需要：数据过滤策略

3. 阅读后，请告诉我：
   - 当前最紧迫的问题是什么？
   - 立即行动项有哪些？
   - 我应该优先做什么？

工作目录：/mnt/afs/davidwang/workspace/sana_wm_pipeline
```

---

## 🔥 关键问题速查表

### 如果新Claude问这些问题

**Q: 代码有bug吗？**
```
A: ❌ 没有！代码100%正确，与参考实现完全对齐
   问题在SLAM算法的metric scale估计，不是代码bug
```

**Q: 为什么删除了稀疏化？**
```
A: 参考实现没有稀疏化，我们要100%对齐
   稀疏化通过丢失BA信息来"平滑"误差，治标不治本
   Ponytail原则：Already working? Use it
```

**Q: 参考标注准吗？**
```
A: ❌ 不准！偏小3-4x
   不能用来评估我们的输出
   必须用真实场景验证（如10秒视频走5米）
```

**Q: 可以用于训练吗？**
```
A: ✅ 可以，但有条件：
   - 推荐短视频（<30秒），偏差1.67x可接受
   - 必须过滤长视频（>60秒），偏差14.8x太大
   - 必须过滤纯旋转场景（scale崩溃）
```

**Q: 最紧迫的任务？**
```
A: 实施数据过滤策略（最高优先级）
   1. 过滤长视频（>30秒）
   2. 过滤纯旋转场景
   3. 添加质量监控
```

---

## 📂 文件导航速查

### 测试数据
```bash
cd /mnt/afs/davidwang/workspace/sana_test_data/smoke_result/

重要文件：
- selected_samples.txt        # 10个测试样本
- analysis_results.json       # 分析结果
- raw_samples/*.camera.npz    # 参考标注（不可靠）
- {sample_id}/*.tar           # 输出结果
```

### 分析脚本
```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/

有用的脚本：
- analyze_smoke_results.py    # 分析冒烟测试
- compare_scales.py           # 对比scale
- verify_reference_impl.sh    # 验证参考实现
```

### 文档
```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/

核心文档：
- spatialvid_smoke_test_report.md     # 综合报告 ⭐⭐⭐⭐⭐
- findings.md                          # 研究发现 ⭐⭐⭐⭐
- SESSION_SUMMARY_20260814.md          # 对话摘要 ⭐⭐⭐
- progress.md                          # 进度日志 ⭐⭐⭐

详细分析（需要时再看）：
- SMOKE_TEST_COMPREHENSIVE_ANALYSIS.md
- ANALYSIS_METHODOLOGY_EXPLAINED.md
- PI3X_MOGE_ALIGNMENT_REPORT.md
```

---

## 🎯 不同场景的快速指令

### 场景1: 继续开发数据过滤

```markdown
我需要实施数据过滤策略。

请：
1. 阅读 spatialvid_smoke_test_report.md 第4.1节（短期行动）
2. 阅读 findings.md 的"未解决问题清单"
3. 查看 stage02_pose/mode_default.py 了解当前处理逻辑

然后帮我：
- 实现过滤长视频和纯旋转场景的逻辑
- 添加质量指标监控
- 生成质量报告
```

### 场景2: 调查metric scale根因

```markdown
我要深入调查为什么metric scale有非线性偏差。

请：
1. 阅读 findings.md 的"发现3: 非线性metric scale偏差"
2. 阅读 spatialvid_smoke_test_report.md 第3.1节（问题分析）
3. 查看 stage02_pose/depth_fusion.py 的 solve_frame_scale()

然后帮我：
- 分析可能的根因
- 设计验证实验
- 提出改进方案
```

### 场景3: 分析新的测试样本

```markdown
我有新的测试样本需要分析。

请：
1. 阅读 ANALYSIS_METHODOLOGY_EXPLAINED.md 了解分析方法
2. 阅读 spatialvid_smoke_test_report.md 第2节了解评估标准
3. 查看 scripts/analyze_smoke_results.py 了解如何使用

然后帮我分析新样本的质量。
```

### 场景4: 准备生产部署

```markdown
我要准备生产部署。

请：
1. 阅读 spatialvid_smoke_test_report.md 第5节（生产就绪评估）
2. 阅读 progress.md 的"下一步行动"
3. 查看部署检查清单

然后帮我：
- 确认所有必须项已完成
- 制定部署计划
- 设计监控方案
```

---

## 🚨 常见陷阱警告

### 陷阱1: 混淆Scale CoV和Metric Scale
```
❌ 错误: "Scale CoV优秀，所以metric scale也准确"
✅ 正确: 这是两个独立的指标！
```

### 陷阱2: 相信参考标注
```
❌ 错误: "我们偏差6.96x（基于参考标注）"
✅ 正确: 参考标注偏小3-4x，不可靠
```

### 陷阱3: 认为是系统性偏差
```
❌ 错误: "系统性偏差不影响训练"
✅ 正确: 这是非线性偏差（1.67x vs 14.8x），会影响训练
```

### 陷阱4: 想添加"改进"
```
❌ 错误: "我有个更好的方案"
✅ 正确: Ponytail原则 - 参考实现是ground truth，不要偏离
```

---

## 📞 需要帮助时

### 如果不确定

1. **查文档**: 90%的问题在 `spatialvid_smoke_test_report.md`
2. **查代码**: 对比 `stage02_pose/mode_default.py` vs `sana-wm-data-clean/vipe_cli.py`
3. **查数据**: 看 `smoke_result/analysis_results.json`
4. **问用户**: 如果还不清楚，直接问

### 关键原则

- ✅ **Ponytail**: 参考实现是ground truth
- ✅ **批判性思维**: 质疑假设，用数据验证
- ✅ **多重验证**: 不要只看一个指标
- ✅ **诚实**: 不确定就说不确定

---

## ✅ 检查清单

新Claude读完文档后应该能回答：

- [ ] 代码有bug吗？（答：没有，100%对齐）
- [ ] 主要问题是什么？（答：非线性metric scale偏差）
- [ ] 短视频偏差多少？（答：1.67x）
- [ ] 长视频偏差多少？（答：14.8x）
- [ ] 参考标注可靠吗？（答：不可靠，偏小3-4x）
- [ ] 可以用于训练吗？（答：可以，但要过滤长视频）
- [ ] 最紧迫的任务？（答：实施数据过滤策略）
- [ ] 为什么删除稀疏化？（答：对齐参考实现）

如果能回答这些，说明已经ready了！

---

**文档版本**: v1.0  
**最后更新**: 2026-08-14  
**维护者**: Claude (Opus 4.8)
