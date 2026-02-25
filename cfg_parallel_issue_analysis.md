# CFG-Parallel 精度问题分析

## 问题描述

在测试 `riverclouds/qwen_image_random` 模型时发现：
- **num_inference_steps=4**: 没有误差，pixel 差异为零 ✅
- **num_inference_steps=8**: 30% 的像素不一致 ❌

## 根本原因分析

### 1. Scheduler 内部状态不同步

FlowUniPCMultistepScheduler 是一个 **multistep solver**，维护以下内部状态：

```python
# 在 vllm_omni/diffusion/models/schedulers/scheduling_flow_unipc_multistep.py 第 116-125 行
self.model_outputs: list[torch.Tensor | None] = [None] * solver_order  # 历史模型输出
self.timestep_list: list[Any | None] = [None] * solver_order            # 历史时间步
self.lower_order_nums = 0                                                # Warmup 计数器
self.last_sample: torch.Tensor | None = None                            # 上一步的样本
self._step_index: int | None = None                                      # 当前步数索引
self.this_order: int = 1                                                 # 当前 solver order
```

### 2. CFG-Parallel 的执行流程

在 `scheduler_step_maybe_with_cfg()` 中（`vllm_omni/diffusion/distributed/cfg_parallel.py` 第 199-235 行）：

```python
if cfg_parallel_ready:
    cfg_rank = get_classifier_free_guidance_rank()
    
    # 🔴 关键问题：只有 rank 0 计算 scheduler step
    if cfg_rank == 0:
        latents = self.scheduler_step(noise_pred, t, latents)
    
    # 只同步 latents，scheduler 内部状态未同步
    latents = latents.contiguous()
    cfg_group.broadcast(latents, src=0)
```

**执行差异：**

| 操作 | CFG-Parallel (cfg_parallel_size=2) | Sequential (cfg_parallel_size=1) |
|------|-------------------------------------|----------------------------------|
| Predict Noise | Rank 0: positive<br>Rank 1: negative | 单卡：先 positive，后 negative |
| Scheduler Step | **仅 Rank 0 执行** | 单卡：执行一次 |
| Scheduler 状态更新 | **仅 Rank 0 更新** | 单卡：正常更新 |
| Latents 同步 | Broadcast 从 rank 0 到所有 ranks | 不需要 |

### 3. 问题的关键

**Rank 1 的 scheduler 内部状态从未被更新！**

虽然 latents 在每一步后通过 `broadcast` 同步，但 scheduler 的内部状态（历史信息）**没有同步**。这导致：

1. Rank 1 的 scheduler 保持初始状态
2. 在下一次循环中，虽然 latents 是同步的，但 rank 1 的 scheduler 仍然认为自己处于初始状态
3. **实际上这不应该影响 predict_noise，因为 predict_noise 不依赖 scheduler 状态**

### 4. 真正的问题来源

等等，让我重新审视...如果 predict_noise 不依赖 scheduler 状态，那为什么会有差异？

**可能的原因：**

#### 假设 A: 数值精度累积误差
- CFG parallel 和 sequential 模式下的计算顺序不同
- Floating point 运算的舍入误差可能累积
- 但这不应该导致 30% 的像素差异...

#### 假设 B: Scheduler 的 multistep 历史影响下一步的计算
让我检查 scheduler 是否在某处被引用...

实际上，我意识到一个关键点：在每次循环开始时，两个 ranks 都会使用相同的 `latents`（通过上一步的 broadcast 同步），但是...

**啊哈！我找到了！**

问题可能在于 **scheduler 的 multistep 算法依赖历史信息来计算当前步**。

在 `scheduler.step()` 第 670-674 行：

```python
prev_sample = self.multistep_uni_p_bh_update(
    model_output=model_output,  # 🔴 当前输出
    sample=sample,               # 🔴 当前样本
    order=self.this_order,       # 🔴 依赖 this_order（基于 lower_order_nums）
)
```

`multistep_uni_p_bh_update` 使用 `self.model_outputs` 历史来进行高阶预测。但这个函数只在 rank 0 上执行，rank 1 的 scheduler 从未更新其历史。

**但关键是：rank 1 在下一步循环中不会调用 scheduler.step()，它只参与 predict_noise。**

所以理论上不应该有问题...除非...

#### 假设 C: 某些共享的全局状态或 cache

让我检查是否有任何 cache 或全局状态在两个模式下表现不同。

## 核心问题

### Scheduler 状态在 Rank 1 未更新

在 `scheduler_step_maybe_with_cfg()` 中：

```python
if cfg_parallel_ready:
    cfg_rank = get_classifier_free_guidance_rank()
    
    if cfg_rank == 0:
        latents = self.scheduler_step(noise_pred, t, latents)  # ✅ 更新 scheduler 状态
    # else: rank 1 什么都不做！❌
    
    latents = latents.contiguous()
    cfg_group.broadcast(latents, src=0)  # 只同步 latents，不同步 scheduler 状态
```

**结果：**
- Rank 0 的 scheduler 状态正常更新（model_outputs, timestep_list, last_sample, lower_order_nums, step_index）
- Rank 1 的 scheduler 状态**保持初始值**，从未更新

### 为什么这会导致问题？

虽然 `predict_noise` 不直接依赖 scheduler 状态，但**可能的原因包括：**

1. **Scheduler Multistep 历史依赖**：
   - FlowUniPCMultistepScheduler 使用历史 model_outputs 来计算高阶预测
   - 虽然只有 rank 0 执行 scheduler.step()，但如果有任何共享状态或全局变量...

2. **Cache 机制的状态追踪**：
   - TeaCache 使用 `do_true_cfg` 和 cfg_rank 来区分 positive/negative 分支
   - 如果启用了 cache，可能会因为 scheduler 状态不同而产生不一致

3. **数值精度累积误差**：
   - CFG-parallel: positive 和 negative 并行计算，然后组合
   - Sequential: positive 先计算，然后 negative，最后组合
   - 浮点运算顺序不同可能导致舍入误差累积

4. **Transformer 内部状态**：
   - 如果 transformer 有任何依赖于"当前是第几步"的逻辑
   - 但这不太可能，因为 transformer 应该是无状态的

### 为什么 steps=4 无损，steps=8 有损？

Scheduler 的 warmup 阶段：

```python
# scheduling_flow_unipc_multistep.py
if self.lower_order_nums < self.config.solver_order:
    self.lower_order_nums += 1
```

- **Steps 1-3**: `lower_order_nums < solver_order`（通常 solver_order=2），使用低阶方法
- **Steps 4+**: 进入完整的 multistep 模式，开始使用历史 model_outputs

推测：
- 当 steps ≤ 4 时，scheduler 还在 warmup，不依赖或很少依赖历史
- 当 steps > 4 时，scheduler 开始严重依赖历史 model_outputs，而 rank 1 的历史是空的或过时的

## 诊断步骤

### 1. 运行诊断脚本

```bash
python diagnose_cfg_parallel.py
```

这个脚本会测试不同推理步数（2, 4, 6, 8, 10, 12）下的差异，帮助确认问题何时开始出现。

### 2. 检查是否启用了 cache

在用户的测试中，确认是否传递了 `cache_backend` 参数。如果启用了 cache，这可能导致额外的状态不一致。

### 3. 添加 scheduler 状态日志

修改 `scheduler_step_maybe_with_cfg()` 来记录 scheduler 的内部状态，对比两个 ranks 的状态差异。

## 可能的解决方案

### 方案 A: 同步 Scheduler 状态到所有 Ranks

修改 `scheduler_step_maybe_with_cfg()` 来同步 scheduler 的内部状态：

```python
def scheduler_step_maybe_with_cfg(
    self, noise_pred: torch.Tensor, t: torch.Tensor, latents: torch.Tensor, do_true_cfg: bool
) -> torch.Tensor:
    cfg_parallel_ready = do_true_cfg and get_classifier_free_guidance_world_size() > 1
    
    if cfg_parallel_ready:
        cfg_group = get_cfg_group()
        cfg_rank = get_classifier_free_guidance_rank()
        
        if cfg_rank == 0:
            latents = self.scheduler_step(noise_pred, t, latents)
        
        latents = latents.contiguous()
        cfg_group.broadcast(latents, src=0)
        
        # 🔧 新增：同步 scheduler 状态
        # 需要同步：model_outputs, timestep_list, last_sample, lower_order_nums, step_index, this_order
        if cfg_rank == 0:
            # Rank 0 准备状态数据
            state_dict = {
                'lower_order_nums': self.scheduler.lower_order_nums,
                '_step_index': self.scheduler._step_index,
                'this_order': self.scheduler.this_order,
                # model_outputs 和 timestep_list 需要特殊处理
            }
        else:
            state_dict = None
        
        # Broadcast 状态到所有 ranks
        # ... 需要实现状态序列化和反序列化 ...
```

**问题**: 这个方案复杂，需要序列化/反序列化 scheduler 的内部状态。

### 方案 B: 让所有 Ranks 都执行 Scheduler Step（推荐）

更简单的方案：让 rank 1 也执行 scheduler.step()，即使它的 noise_pred 不会被使用。

```python
def scheduler_step_maybe_with_cfg(
    self, noise_pred: torch.Tensor, t: torch.Tensor, latents: torch.Tensor, do_true_cfg: bool
) -> torch.Tensor:
    cfg_parallel_ready = do_true_cfg and get_classifier_free_guidance_world_size() > 1
    
    if cfg_parallel_ready:
        cfg_group = get_cfg_group()
        cfg_rank = get_classifier_free_guidance_rank()
        
        # 🔧 修改：所有 ranks 都执行 scheduler step（保持状态一致）
        if cfg_rank == 0:
            latents = self.scheduler_step(noise_pred, t, latents)
        else:
            # Rank 1 也执行 step 来更新内部状态，但使用 rank 0 broadcast 的 latents
            # 注意：rank 1 的 noise_pred 是 None，需要创建一个 dummy 值或使用 latents
            dummy_noise = torch.zeros_like(latents)
            _ = self.scheduler_step(dummy_noise, t, latents)
        
        # 同步最终的 latents（只使用 rank 0 的结果）
        latents = latents.contiguous()
        cfg_group.broadcast(latents, src=0)
```

**优点**: 简单，保持所有 ranks 的 scheduler 状态一致
**缺点**: rank 1 做了额外的计算（但很小）

### 方案 C: 禁用 Multistep 方法

使用更简单的 Euler scheduler，避免依赖历史状态：

```python
# 在 pipeline 初始化时
if od_config.parallel_config.cfg_parallel_size > 1:
    # 使用 Euler scheduler 而不是 UniPC multistep
    self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(...)
```

**优点**: 避免状态同步问题
**缺点**: 可能需要更多的推理步数才能达到相同质量

## 下一步行动

1. 运行 `diagnose_cfg_parallel.py` 确认问题模式
2. 检查是否有默认启用的 cache
3. 实施方案 B（最简单有效）
4. 添加集成测试验证多步推理的无损性

</thinking>

<function_calls>
<invoke name="SemanticSearch">
<parameter name="query">Does the transformer or attention mechanism have any internal state or use random numbers during inference?