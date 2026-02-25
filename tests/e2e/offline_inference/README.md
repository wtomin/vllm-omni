# vLLM-Omni 兼容性测试框架

基于批量处理的特性兼容性测试和性能评估框架。

## 📚 文档导航

| 文档 | 用途 | 适合人群 |
|------|------|----------|
| **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** | 快速参考卡片 | ⭐ 所有用户 |
| **[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)** | 完整评估指南 | 深度使用者 |
| **[BATCH_INTEGRATION_SUMMARY.md](BATCH_INTEGRATION_SUMMARY.md)** | 技术实现细节 | 开发者 |

## 🚀 快速开始

### 1. 最简单的测试（3 个提示词）

```bash
cd tests/e2e/offline_inference

python run_compat_test.py \
    --baseline-feature cfg_parallel \
    --addons teacache \
    --num-prompts 3 \
    --steps 10
```

### 2. 分析结果

```bash
python analyze_compat_results.py \
    --results-dir ./compat_results/cfg_parallel \
    --charts
```

### 3. 查看报告

```bash
# 查看 JSON 报告
cat ./compat_results/cfg_parallel/report.json

# 查看图表
open ./compat_results/cfg_parallel/chart_quality.png
open ./compat_results/cfg_parallel/chart_speedgain.png
```

就这么简单！🎉

## 📖 使用场景

### 场景 1: 新特性开发

在开发新特性后，快速验证与现有特性的兼容性：

```bash
python run_compat_test.py \
    --baseline-feature <your_new_feature> \
    --addons cfg_parallel teacache ulysses \
    --num-prompts 20 \
    --steps 30
```

### 场景 2: 性能优化

对比优化前后的性能变化：

```bash
# 优化前
python run_compat_test.py --baseline-feature cfg_parallel \
    --output-dir ./before_optimization

# 优化后
python run_compat_test.py --baseline-feature cfg_parallel \
    --output-dir ./after_optimization

# 对比
python compare_results.py \
    ./before_optimization/cfg_parallel/report.json \
    ./after_optimization/cfg_parallel/report.json \
    --best
```

### 场景 3: CI/CD 集成

自动化测试流程：

```bash
# 在 PR 合并前运行
python run_compat_test.py \
    --baseline-feature cfg_parallel \
    --addons teacache cache_dit \
    --num-prompts 10 \
    --steps 20 \
    --output-dir ./ci_test

# 检查是否有失败
python analyze_compat_results.py --results-dir ./ci_test/cfg_parallel
```

## 🔧 工具说明

### 核心工具

| 工具 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `batch_text_to_image.py` | 批量图像生成 | 提示词文件 | 图片 + 时间统计 |
| `run_compat_test.py` | 兼容性测试执行 | 特性配置 | 测试结果目录 |
| `analyze_compat_results.py` | 结果分析 | 测试结果目录 | JSON报告 + 图表 |
| `compare_results.py` | 多结果对比 | 多个 JSON 报告 | 对比分析 |

### 辅助脚本

| 脚本 | 用途 |
|------|------|
| `quick_eval.sh` | 快速评估（少量提示词） |
| `example_batch_usage.sh` | 批量脚本使用示例 |

## 📊 输出说明

### 测试结果结构

```
compat_results/
└── cfg_parallel/                    # 基线特性目录
    ├── manifest.json                # 测试元数据
    ├── report.json                  # 分析报告（运行 analyze 后生成）
    ├── chart_quality.png            # 质量对比图表
    ├── chart_speedgain.png          # 性能对比图表
    ├── diff_report.html             # HTML 图像对比报告
    ├── baseline/                    # 纯基线配置
    │   ├── batch_generation.log     # 批量生成日志（所有详细信息和时间）
    │   ├── batch_generation.exitcode # 批量退出码
    │   ├── prompt_00.png            # 生成的图片
    │   ├── prompt_00.exitcode       # 退出码（0=成功）
    │   └── ...
    ├── cfg_parallel/                # 基线特性单独运行
    └── cfg_parallel+teacache/       # 组合特性运行
```

### 关键指标

- **Speedup**: 相对于纯基线的加速比
- **MeanDiff**: 平均像素差异（0-1 范围）
- **MaxDiff**: 最大像素差异
- **Status**: PASS ✅ / WARN ⚠️ / FAIL ❌ / ERROR 💥

## 🎯 支持的特性

| 特性 ID | 说明 | GPU 需求 | 有损? | 典型加速比 |
|---------|------|----------|-------|------------|
| `cfg_parallel` | CFG 并行 | ×2 | ❌ | 1.8x |
| `teacache` | TeaCache 缓存 | ×1 | ✅ | 1.5x |
| `cache_dit` | Cache-DiT 缓存 | ×1 | ✅ | 1.7x |
| `ulysses` | Ulysses 序列并行 | ×2 | ❌ | 1.6x |
| `ring` | Ring 序列并行 | ×2 | ❌ | 1.5x |
| `tp` | 张量并行 | ×2 | ❌ | 1.4x |

*注: 加速比会根据配置和硬件有所不同*

## 🔍 常见问题

### Q: 为什么配置被跳过？

```
SKIP 'cfg_parallel+ulysses' — requires 4 GPUs, only 2 available
```

**A**: GPU 数量不足。`cfg_parallel` 需要 2 GPU，`ulysses` 也需要 2 GPU，组合起来需要 4 GPU。

**解决方案**: 减少特性组合或使用更多 GPU。

### Q: 如何加快测试速度？

**A**: 使用以下参数：
- `--num-prompts 3` - 减少提示词数量
- `--steps 10` - 减少推理步数
- `--height 512 --width 512` - 减小图像尺寸

### Q: WARN 状态是否需要关注？

**A**: WARN 通常出现在有损特性（如 TeaCache）上，质量损失在可接受范围内。如果差异太大可以调整阈值。

### Q: 如何添加新特性？

**A**: 在 `run_compat_test.py` 的 `FEATURE_REGISTRY` 中添加：

```python
FEATURE_REGISTRY = {
    "my_feature": {
        "args": ["--my-arg", "value"],
        "gpu_multiplier": 1,  # GPU 需求
        "lossy": False,       # 是否有损
        "label": "My Feature",
    },
}
```

### Q: 如何查看详细日志？

**A**: 查看批量生成日志：

```bash
cat ./compat_results/cfg_parallel/cfg_parallel+teacache/batch_generation.log
```

## 📈 性能基准

基于 20 个提示词，30 推理步数，1024×1024 图像：

| 配置 | 平均时间 | 加速比 | 质量损失 |
|------|----------|--------|----------|
| 纯基线 | 10.2s | 1.0x | — |
| CFG Parallel | 5.6s | 1.82x | 0.0000 |
| CFG + TeaCache | 2.9s | 3.52x | 0.0823 |
| CFG + Cache-DiT | 2.7s | 3.78x | 0.1124 |

*实际性能因硬件和配置而异*

## 🤝 贡献

欢迎提交问题和改进建议！

### 添加新特性

1. 在 `FEATURE_REGISTRY` 中注册特性
2. 运行测试验证
3. 提交 PR

### 改进文档

1. 更新相关 Markdown 文件
2. 确保示例代码可运行
3. 提交 PR

## 📝 更新日志

### v2.0 (2024-02) - 批量处理版本

- ✅ 集成 `batch_text_to_image.py` 实现批量处理
- ✅ 单次模型加载处理多个提示词
- ✅ 自动计算平均生成时间
- ✅ 优化临时文件管理
- ✅ 完善文档和示例

### v1.0 (2024-01) - 初始版本

- ✅ 基本兼容性测试框架
- ✅ 结果分析和可视化
- ✅ 支持多种特性组合

## 📞 联系方式

- 问题反馈: [GitHub Issues](https://github.com/your-repo/issues)
- 文档: [项目 Wiki](https://github.com/your-repo/wiki)

---

**开始你的第一个测试吧！** 🚀

```bash
cd tests/e2e/offline_inference
bash quick_eval.sh
```
