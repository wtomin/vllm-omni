# vLLM-Omni 兼容性测试框架

基于批量处理的特性兼容性测试和性能评估框架，覆盖全部 13 种 diffusion 加速/优化特性。

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

### 3. 诊断图像差异

```bash
# 诊断单个配置
python diagnose_diff.py \
    --results-dir ./compat_results/cfg_parallel \
    --config cfg_parallel+teacache

# 一次诊断所有配置，并生成 HTML 报告
python diagnose_diff.py \
    --results-dir ./compat_results/cfg_parallel \
    --all --html --save-json
```

就这么简单！🎉

## 🎯 支持的特性（13 种）

### 加速：缓存方法（有损）

| 特性 ID | 说明 | GPU 需求 | 有损? | 典型加速比 |
|---------|------|----------|-------|------------|
| `teacache` | TeaCache 自适应缓存 | ×1 | ✅ | ~1.5× |
| `cache_dit` | Cache-DiT（DBCache + TaylorSeer + SCM） | ×1 | ✅ | ~1.7× |

> `teacache` 和 `cache_dit` **不兼容**，不可同时使用。

### 加速：并行方法（无损）

| 特性 ID | 说明 | GPU 需求 | 有损? | 典型加速比 |
|---------|------|----------|-------|------------|
| `cfg_parallel` | CFG 正/负分支分发到 2 个 GPU | ×2 | ❌ | ~1.8× |
| `ulysses` | Ulysses 序列并行（all-to-all） | ×2 | ❌ | ~1.6× |
| `ring` | Ring 序列并行（ring 通信） | ×2 | ❌ | ~1.5× |
| `tp` | 张量并行（权重分片） | ×2 | ❌ | ~1.4× |
| `hsdp` | HSDP（FSDP2 权重分片，运行时重组） | ×2 | ❌ | ~1.3× |

> `tp` 和 `hsdp` **不兼容**，不可同时使用。

### 内存优化

| 特性 ID | 说明 | GPU 需求 | 有损? | 备注 |
|---------|------|----------|-------|------|
| `cpu_offload` | 模块级 CPU 卸载（DiT + 文本编码器） | ×1 | ❌ | 仅单卡 |
| `layerwise_offload` | 逐层 CPU 卸载（每次仅保留 1 个 block 在 GPU） | ×1 | ❌ | 仅单卡 |
| `vae_patch_parallel` ⚠️ | VAE patch 并行解码 | —（复用并行基线的 GPU） | ❌ | **addon-only**，见下方说明 |
| `fp8` | FP8 量化（Ada/Hopper W8A8） | ×1 | ✅(轻微) | 不兼容 gguf |
| `gguf` | GGUF 量化（Q4/Q8 等） | ×1 | ✅(轻微) | 不兼容 fp8 |

> `cpu_offload`（逐层）和 `vae_patch_parallel` **不兼容**。  
> `fp8` 和 `gguf` **不兼容**。

#### vae_patch_parallel 特殊说明

`vae_patch_parallel` 是 **addon-only** 特性，**不能**单独作为 `--baseline-feature` 使用。  
`--vae-patch-parallel-size` 必须等于基线并行方法的 parallel size 乘积：

```bash
# ✅ 正确：vae_patch_parallel 作为 addon，叠加在 tp (×2) 之上
python run_compat_test.py \
    --baseline-feature tp \
    --addons vae_patch_parallel \
    --model Qwen/Qwen-Image

# ✅ 正确：叠加在 cfg_parallel (×2) 之上
python run_compat_test.py \
    --baseline-feature cfg_parallel \
    --addons vae_patch_parallel \
    --model Qwen/Qwen-Image

# ❌ 错误：不能单独作为 baseline
python run_compat_test.py --baseline-feature vae_patch_parallel  # 报错
```

### 扩展功能

| 特性 ID | 说明 | GPU 需求 | 有损? | 备注 |
|---------|------|----------|-------|------|
| `lora` | LoRA 推理适配器 | ×1 | ❌ | 需要 `--lora-path` |

## ⛔ 冲突规则

以下特性组合**不兼容**。当测试矩阵中出现冲突组合时，该 test case 会自动标记为 `SKIP (conflict)` 并跳过，**不计入失败**。

| 特性 A | 特性 B | 原因 |
|--------|--------|------|
| `tp` | `hsdp` | Tensor Parallel 和 HSDP 不兼容 |
| `teacache` | `cache_dit` | 两种缓存方法不可同时开启 |
| `layerwise_offload` | `cpu_offload` | 两种 CPU 卸载方法不兼容 |
| `fp8` | `gguf` | 两种量化方法不兼容 |
| `layerwise_offload` | 任何多卡特性 | 逐层卸载目前仅支持单卡（`gpu_multiplier > 1` 的特性均冲突） |

多卡特性（`gpu_multiplier > 1`）包括：`cfg_parallel`、`ulysses`、`ring`、`tp`、`hsdp`。

### 冲突检测示例

```
# 运行时终端输出：
[WARN]  SKIP 'tp+hsdp'             — Tensor Parallel and HSDP are not compatible
[WARN]  SKIP 'teacache+cache_dit'  — TeaCache and Cache-DiT are not compatible
[WARN]  SKIP 'fp8+gguf'            — FP8 quantization and GGUF quantization are not compatible
[WARN]  SKIP 'layerwise_offload+tp'— 'layerwise_offload' supports single-card only and cannot
                                      be combined with multi-GPU feature(s): ['tp']

# 最终汇总区分冲突跳过 vs. GPU 不足跳过：
  SKIP (conflict)  : 2  (incompatible feature pairs)
  SKIP (GPU)       : 1  (insufficient GPUs)
  SKIP total       : 3  (configs, not prompts)
```

### 扩展冲突规则

在 `run_compat_test.py` 中的 `CONFLICT_RULES` 列表添加条目即可：

```python
CONFLICT_RULES: list[tuple[str, str, str]] = [
    ("tp",               "hsdp",          "Tensor Parallel and HSDP are not compatible"),
    ("teacache",         "cache_dit",     "TeaCache and Cache-DiT are not compatible"),
    ("layerwise_offload","cpu_offload",   "Layerwise and module-level CPU offloading are not compatible"),
    ("fp8",              "gguf",          "FP8 and GGUF quantization are not compatible"),
    # 新增：
    ("my_feature_a",     "my_feature_b",  "描述原因"),
]
```

如果某特性仅支持单卡，将其加入 `SINGLE_CARD_ONLY`：

```python
SINGLE_CARD_ONLY: frozenset[str] = frozenset({"layerwise_offload", "my_single_card_feature"})
```

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

### 场景 2: 内存优化验证

```bash
# 验证 FP8 量化不影响图像质量
python run_compat_test.py \
    --baseline-feature fp8 \
    --addons cfg_parallel \
    --model Qwen/Qwen-Image-2512 \
    --num-prompts 10 --steps 20

# 验证逐层 CPU 卸载
python run_compat_test.py \
    --baseline-feature layerwise_offload \
    --num-prompts 5 --steps 10
```

### 场景 3: LoRA 推理验证

```bash
python run_compat_test.py \
    --baseline-feature lora \
    --lora-path /path/to/my/lora_adapter \
    --model Tongyi-MAI/Z-Image-Turbo \
    --num-prompts 10 --steps 20
```

### 场景 4: 性能优化对比

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

### 场景 5: 验证冲突跳过

当不兼容的特性组合出现时，test case 会自动跳过（不计为失败）：

```bash
# teacache 和 cache_dit 不兼容 → cfg+teacache+cache_dit 被自动跳过
python run_compat_test.py \
    --baseline-feature teacache \
    --addons cache_dit \
    --model Qwen/Qwen-Image-2512

# 终端输出示例：
# [WARN]  SKIP 'teacache+cache_dit' — TeaCache and Cache-DiT are not compatible
```

### 场景 7: CI/CD 集成

```bash
python run_compat_test.py \
    --baseline-feature cfg_parallel \
    --addons teacache cache_dit fp8 \
    --num-prompts 10 \
    --steps 20 \
    --output-dir ./ci_test

python analyze_compat_results.py --results-dir ./ci_test/cfg_parallel
```

## 🔧 工具说明

### 核心工具

| 工具 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `batch_text_to_image.py` | 批量图像生成 | 提示词文件 | 图片 + 时间统计 |
| `run_compat_test.py` | 兼容性测试执行 | 特性配置 | 测试结果目录 |
| `analyze_compat_results.py` | 结果分析 | 测试结果目录 | JSON 报告 + 图表 |
| `diagnose_diff.py` | 图像差异诊断 | 测试结果目录 | 差异报告 + HTML |
| `compare_results.py` | 多结果对比 | 多个 JSON 报告 | 对比分析 |

### diagnose_diff.py 参数

| 参数 | 说明 |
|------|------|
| `--results-dir PATH` | 结果目录（含 `baseline/` 和各配置子目录） |
| `--config NAME...` | 要诊断的一个或多个配置名称 |
| `--all` | 自动发现并诊断目录下所有非参考配置 |
| `--reference NAME` | 参考配置名称（默认 `baseline`） |
| `--top N` | 每个配置最多展示 N 张最差图片（默认 10） |
| `--html` | 生成内联图片的 HTML 对比报告 |
| `--save-json` | 每个配置保存一份 JSON 报告 |

> SSIM 指标需要 `pip install scikit-image`，未安装时自动降级为 MeanDiff/MaxDiff。

## 📊 输出说明

### 测试结果结构

```
compat_results/
└── cfg_parallel/                    # 基线特性目录
    ├── manifest.json                # 测试元数据
    ├── report.json                  # 分析报告（运行 analyze 后生成）
    ├── diff_report.html             # HTML 图像对比报告（--html）
    ├── chart_quality.png            # 质量对比图表
    ├── chart_speedgain.png          # 性能对比图表
    ├── baseline/                    # 纯基线配置
    │   ├── batch_generation.log     # 批量生成日志
    │   ├── batch_generation.exitcode
    │   ├── prompt_00.png
    │   ├── prompt_00.exitcode
    │   └── ...
    ├── cfg_parallel/                # 基线特性单独运行
    └── cfg_parallel+teacache/       # 组合特性运行
```

### 关键指标

- **Speedup**: 相对于纯基线的加速比
- **MeanDiff**: 平均像素差异（0–1，越小越好）
- **MaxDiff**: 最大像素差异（0–1）
- **SSIM**: 结构相似性（0–1，越大越好；需 scikit-image）
- **Status**: OK ✅ / WARN ⚠️ / LARGE ❌

## 🔍 常见问题

### Q: 为什么配置被跳过？

```
SKIP 'cfg_parallel+ulysses' — requires 4 GPUs, only 2 available
```

**A**: GPU 数量不足。减少特性组合或使用更多 GPU。

### Q: LoRA 测试怎么运行？

**A**: 提供 `--lora-path` 参数：

```bash
python run_compat_test.py \
    --baseline-feature lora \
    --lora-path /path/to/adapter \
    --model Tongyi-MAI/Z-Image-Turbo
```

### Q: HSDP 和 TP 不兼容怎么办？

**A**: HSDP 不能与 `--tensor-parallel-size > 1` 或 data parallelism 同时使用。单独测试：

```bash
python run_compat_test.py \
    --baseline-feature hsdp \
    --model black-forest-labs/FLUX.1-dev \
    --num-prompts 5 --steps 10
```

### Q: 如何加快测试速度？

**A**: 使用以下参数：
- `--num-prompts 3` — 减少提示词数量
- `--steps 10` — 减少推理步数
- `--height 512 --width 512` — 减小图像尺寸

### Q: WARN 状态是否需要关注？

**A**: WARN 通常出现在有损特性（TeaCache、Cache-DiT、FP8、GGUF）上，质量损失在可接受范围内。

### Q: 如何添加新特性？

**A**: 在 `run_compat_test.py` 的 `FEATURE_REGISTRY` 中添加条目：

```python
"my_feature": {
    "args": ["--my-arg", "value"],
    "gpu_multiplier": 1,    # GPU 倍增系数
    "lossy": False,         # 是否有损
    "label": "My Feature",  # 显示名称
    "category": "parallelism",  # 类别
    "note": "Brief description.",
},
```

## 📈 性能基准参考

基于 20 个提示词，30 推理步数，1024×1024，Qwen/Qwen-Image-2512：

| 配置 | 平均时间 | 加速比 | 质量损失（MeanDiff） |
|------|----------|--------|----------------------|
| 纯基线 | 10.2s | 1.0× | — |
| CFG Parallel | 5.6s | 1.82× | 0.0000 |
| CFG + TeaCache | 2.9s | 3.52× | 0.0823 |
| CFG + Cache-DiT | 2.7s | 3.78× | 0.1124 |
| FP8 | 7.1s | 1.44× | ~0.01 |

*实际性能因硬件和配置而异*

## 🤝 贡献

### 添加新特性

1. 在 `run_compat_test.py` 的 `FEATURE_REGISTRY` 中注册
2. 如需新 CLI 参数，在 `batch_text_to_image.py` 中添加
3. 运行测试验证，提交 PR

### 改进文档

1. 更新本 README 和相关 Markdown 文件
2. 确保示例代码可运行
3. 提交 PR

## 📝 更新日志

### v3.0 (2026-03) — 全特性覆盖版本

- ✅ 新增 7 种缺失特性：`hsdp`, `cpu_offload`, `layerwise_offload`, `vae_patch_parallel`, `fp8`, `gguf`, `lora`
- ✅ `run_compat_test.py` 现覆盖全部 13 种 diffusion 特性
- ✅ `vae_patch_parallel` 正确标记为 addon-only，并在 `build_configs()` 中强制校验
- ✅ 新增 `CONFLICT_RULES` + `SINGLE_CARD_ONLY` 冲突检测：5 类不兼容组合自动跳过，汇总区分 `SKIP (conflict)` 与 `SKIP (GPU)`
- ✅ `batch_text_to_image.py` 新增 `--use-hsdp`, `--hsdp-shard-size`, `--hsdp-replicate-size`, `--enable-expert-parallel`, `--lora-path`, `--lora-scale`, gguf 量化支持
- ✅ `diagnose_diff.py` 全面重写：`--all` 批量模式、SSIM 指标、HTML 报告、彩色终端输出、多配置汇总表

### v2.0 (2024-02) — 批量处理版本

- ✅ 集成 `batch_text_to_image.py` 实现批量处理
- ✅ 单次模型加载处理多个提示词
- ✅ 自动计算平均生成时间

### v1.0 (2024-01) — 初始版本

- ✅ 基本兼容性测试框架
- ✅ 结果分析和可视化

---

**开始你的第一个测试吧！** 🚀

```bash
cd tests/e2e/offline_inference

python run_compat_test.py \
    --baseline-feature cfg_parallel \
    --addons teacache fp8 \
    --model Qwen/Qwen-Image-2512 \
    --num-prompts 5 --steps 10
```
