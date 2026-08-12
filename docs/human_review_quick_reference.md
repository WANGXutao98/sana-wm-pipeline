# 人工审查快速参考卡

## 必填字段

1. **human_verdict**: `pass` / `fail` / 留空
2. **primary_issue**: 从11个选项中选择

## Primary Issue 速查

- `trajectory_minor_jump` - 小跳变
- `trajectory_major_jump` - 大跳变  
- `video_blurry` - 模糊
- `video_artifacts` - 伪影
- `caption_mismatch` - 字幕不符
- `caption_too_vague` - 字幕过泛
- `black_frames` - 黑屏
- `scene_cut_abrupt` - 场景切换
- `multiple_issues` - 多问题
- `no_issue` - 无问题
- `other` - 其他（需notes说明）

## 判断原则

**Pass**: 小瑕疵、可训练、人眼看平滑
**Fail**: 严重模糊、字幕不符、内容不适合
**留空**: 不确定时使用自动判断

## VLC快捷键

- 空格：播放/暂停
- N：下一个
- ]：加速
- [：减速

## 审查重点

1. **轨迹质量** - 相机运动是否平滑？
2. **视频质量** - 画面是否清晰？
3. **内容连贯** - 是否有突兀切换？
4. **字幕匹配** - 描述是否准确？

## 工作流程

1. 打开 `decisions_template.csv` (Excel)
2. 参考 `review_list.csv` 查看指标
3. 在 `videos/` 目录播放视频
4. 填写 `human_verdict` 和 `primary_issue`
5. 每50个样本保存一次
6. 完成后另存为 `decisions_filled.csv`

## 常见场景

| 场景 | auto_verdict | 你的判断 | human_verdict | primary_issue |
|------|--------------|---------|---------------|---------------|
| 小跳变，整体平滑 | fail | 可接受 | pass | trajectory_minor_jump |
| 严重模糊 | pass | 不可用 | fail | video_blurry |
| 字幕短但准确 | fail | 足够描述 | pass | no_issue |
| 完全不确定 | fail/pass | 不知道 | (留空) | (留空) |

## 质量自查

- [ ] 至少90%样本已填写
- [ ] verdict只有pass/fail或留空
- [ ] 已填写verdict的都有primary_issue
- [ ] 文件另存为 decisions_filled.csv
- [ ] 编码为UTF-8

## 联系支持

技术问题 / 判断标准不明 / 发现系统问题
→ 联系技术负责人
