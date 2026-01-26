# MoE (Mixture of Experts) Implementation for CL-Agent

## 概述

MoE方法为每个任务训练独立的LoRA模块（expert），使用Mix采样方式但保持任务完全独立。每个任务有自己的actor和critic worker，在训练时串行采样和更新以避免显存问题。

## 核心特性

1. **9个独立的LoRA模块**：每个任务有自己的expert
2. **9个独立的actor-critic pairs**：完全的任务隔离
3. **串行采样**：依次从每个任务采样，使用对应的LoRA
4. **串行更新**：依次更新每个任务的LoRA，避免显存溢出
5. **完全独立训练**：等同于9个任务各自独立训练

## 已完成的实现

### 1. MoE CL方法 (`ragen/cl_methods/moe.py`)

**功能**：
- 管理9个任务的状态和配置
- 跟踪每个任务的训练步数
- 保存和加载MoE方法状态

**关键类**：
- `MoEConfig`: 配置类，包含任务数量、任务名称、LoRA配置
- `MoECLMethod`: MoE方法类，继承自`BaseCLMethod`

### 2. MoE训练器 (`ragen/trainer/moe_trainer.py`)

**功能**：
- 创建和管理9个独立的actor-critic pairs
- 实现串行采样和更新逻辑
- 处理任务切换和worker管理
- 实现validation和checkpoint保存

**关键方法**：
- `init_moe_workers()`: 初始化9个独立的worker pairs
- `_create_actor_worker()`: 创建actor worker
- `_create_critic_worker()`: 创建critic worker
- `switch_to_task()`: 切换当前活跃任务
- `_serial_sampling()`: 串行采样所有任务
- `_serial_updates()`: 串行更新所有任务
- `_validate_all_tasks()`: 验证所有任务
- `_save_checkpoint()`: 保存所有任务的checkpoint

### 3. 训练脚本 (`train_moe.py`)

**功能**：
- 加载配置和初始化环境
- 创建MoE方法和训练器
- 启动训练流程

**关键函数**：
- `build_task_configs()`: 从配置构建任务配置
- `create_moe_method()`: 创建MoE方法实例
- `main()`: 主入口函数

### 4. 配置文件 (`config/moe_training_9tasks.yaml`)

**配置内容**：
- 9个任务的训练和验证配置
- 每个任务的环境标签和组数
- 训练参数（总步数、验证频率等）

### 5. 训练脚本 (`run_moe_9tasks.sh`)

**功能**：
- 提供便捷的训练启动脚本
- 支持环境变量配置
- 自动处理GPU设置

## 训练流程

### 每个训练步骤

1. **Phase 1: 串行采样**
   ```
   For each task (0-8):
     - Switch to task's actor worker
     - Sample trajectories using task's LoRA
     - Store batch for this task
   ```

2. **Phase 2: 串行更新**
   ```
   For each task (0-8):
     - Switch to task's actor and critic workers
     - Compute advantages using task's advantage estimator
     - Update task's actor with mini-batches
     - Update task's critic
     - Record metrics
   ```

### Validation

- 在指定的validation step验证所有9个任务
- 每个任务使用自己的LoRA进行验证
- 记录每个任务的独立指标

### Checkpoint保存

```
checkpoints/moe_9tasks/{timestamp}/global_step_{N}/
  ├── task_0_bandit_low/
  │   ├── actor/
  │   │   └── model.pt
  │   └── critic/
  │       └── model.pt
  ├── task_1_bandit_medium/
  │   ├── actor/
  │   └── critic/
  ├── ...
  ├── task_8_frozen_lake_hard/
  │   ├── actor/
  │   └── critic/
  └── cl_method_state.pt
```

## 使用方法

### 基本使用

```bash
# 使用默认配置
bash run_moe_9tasks.sh

# 使用自定义GPU
CUDA_VISIBLE_DEVICES=0 bash run_moe_9tasks.sh

# 使用自定义参数
TOTAL_STEPS=900 TEST_FREQ=20 SAVE_FREQ=100 bash run_moe_9tasks.sh
```

### Python API

```python
from ragen.cl_methods.moe import MoECLMethod, MoEConfig

# 创建配置
config = MoEConfig(
    num_tasks=9,
    task_names=['bandit_low', 'bandit_medium', ...],
    lora_rank=64,
    lora_alpha=64,
)

# 创建方法
method = MoECLMethod(config)

# 使用在训练中
method.on_task_start(0, "moe_training", None)
# ... training ...
method.on_task_end(0, "moe_training", checkpoint_path)
```

## 测试结果

### 基本功能测试

```bash
# 测试MoE方法注册
✓ MoE registered: MoECLMethod

# 测试配置创建
✓ Config created: num_tasks=9

# 测试方法实例化
✓ Method instantiated

# 测试配置文件加载
✓ Config loaded successfully
✓ Configuration file is valid
```

所有基本测试通过！

## 关键设计决策

### 1. 串行 vs 并行

**选择**：串行采样和更新

**原因**：
- 避免显存溢出（9个任务同时更新需要大量显存）
- 保证任务完全独立
- 实现简单，易于调试

### 2. Worker管理

**设计**：每个任务独立的actor-critic pair

**实现**：
- 使用Ray remote创建独立workers
- 通过`switch_to_task()`切换活跃worker
- 每个worker有自己的LoRA模块

### 3. Advantage Estimator

**设计**：每个任务独立的advantage estimator

**原因**：
- 保证任务间完全独立
- 避免任务间的统计信息混淆

### 4. ES Manager

**设计**：每个任务独立的训练和验证ES manager

**原因**：
- 每个任务有不同的环境配置
- 支持任务特定的环境设置

## 与其他方法的对比

| 方法 | LoRA共享 | 采样方式 | 更新方式 | 任务独立性 |
|------|---------|---------|---------|-----------|
| Baseline | 共享 | 顺序 | 共享参数 | 无 |
| O-LoRA | 独立 | 顺序 | 正交约束 | 部分 |
| SD-LoRA | 独立 | 顺序 | Scaling factors | 部分 |
| Mix | 共享 | 混合 | 共享参数 | 无 |
| **MoE** | **独立** | **混合** | **完全独立** | **完全** |

## 预期性能

- **训练时间**：比Mix慢（串行采样和更新）
- **显存占用**：与单任务训练相当（串行更新）
- **任务性能**：接近独立训练的结果
- **遗忘程度**：无遗忘（任务完全独立）

## 文件清单

### 新增文件

1. `ragen/cl_methods/moe.py` - MoE CL方法
2. `ragen/trainer/moe_trainer.py` - MoE训练器
3. `train_moe.py` - 训练脚本
4. `config/moe_training_9tasks.yaml` - 配置文件
5. `run_moe_9tasks.sh` - Shell脚本
6. `docs/MOE_IMPLEMENTATION.md` - 本文档

### 修改文件

1. `ragen/cl_methods/__init__.py` - 注册MoE方法

## 下一步

1. **运行短期测试**：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python train_moe.py --config-name moe_training_9tasks trainer.total_training_steps=10
   ```

2. **运行完整训练**：
   ```bash
   CUDA_VISIBLE_DEVICES=0 bash run_moe_9tasks.sh
   ```

3. **分析结果**：
   - 检查每个任务的checkpoint
   - 验证任务独立性
   - 比较与其他方法的性能

## 注意事项

1. **显存管理**：串行更新确保显存可控
2. **训练时间**：串行操作会增加训练时间
3. **Checkpoint大小**：9个任务的checkpoint较大
4. **Worker数量**：确保Ray有足够资源创建18个workers（9个actor + 9个critic）

## 故障排除

### 问题1：Ray worker创建失败

**解决方案**：检查Ray资源配置，确保有足够的GPU和内存

### 问题2：显存溢出

**解决方案**：
- 确认串行更新正在工作
- 减小batch size
- 降低GPU memory utilization

### 问题3：任务切换失败

**解决方案**：检查`switch_to_task()`逻辑，确保task_idx有效

## 总结

MoE方法已完整实现，包括：
- ✅ 完整的MoE CL方法类
- ✅ 完整的MoE训练器
- ✅ 训练脚本和配置
- ✅ Shell脚本
- ✅ 基本功能测试通过

现在可以开始训练了！
