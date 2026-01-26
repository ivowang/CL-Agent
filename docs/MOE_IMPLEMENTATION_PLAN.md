# MoE (Mixture of Experts) Implementation Plan

## 概述

MoE方法为每个任务训练独立的LoRA模块，使用Mix采样方式但保持任务完全独立。

## 已完成的工作

1. ✅ 创建 `ragen/cl_methods/moe.py` - MoE CL方法类
2. ✅ 更新 `ragen/cl_methods/__init__.py` - 注册MoE方法
3. ✅ 创建 `ragen/trainer/moe_trainer.py` - MoE训练器框架
4. ✅ 创建 `config/moe_training_9tasks.yaml` - 配置文件

## 需要完成的工作

### 1. 完善 MoEAgentTrainer (ragen/trainer/moe_trainer.py)

当前的moe_trainer.py提供了框架，但需要完善以下部分：

#### 问题1: Worker创建方法不完整

当前的`_create_actor_worker`和`_create_critic_worker`只是简单复制了worker引用，需要真正创建新的worker实例。

**解决方案**：
```python
def _create_actor_worker(self, task_idx: int):
    """Create an actor worker for a specific task."""
    # 需要从RayWorkerGroup创建新的worker
    # 参考base trainer的worker创建逻辑
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.ppo.utils import Role

    # 创建新的worker group
    actor_wref = self.resource_pool_manager.create_worker(
        role=Role.ActorRollout,
        worker_cls=self.role_worker_mapping[Role.ActorRollout],
        config=self.config,
    )
    return actor_wref
```

#### 问题2: 缺少advantage estimator

每个task需要自己的advantage estimator。

**解决方案**：
```python
def __init__(self, *args, **kwargs):
    # ... existing code ...

    # Create advantage estimators for each task
    self.task_advantage_estimators = []
    for task_idx in range(self.num_tasks):
        adv_estimator = AdvantageEstimator(
            self.config.algorithm.adv_estimator,
            self.config.algorithm.gamma,
            self.config.algorithm.lam
        )
        self.task_advantage_estimators.append(adv_estimator)
```

#### 问题3: 缺少完整的update逻辑

`_update_single_task`需要完整的PPO更新逻辑。

**解决方案**：参考`RayAgentTrainer.fit()`中的update逻辑，包括：
- Compute advantages
- Apply KL penalty
- Update actor (multiple mini-batches)
- Update critic
- Collect metrics

### 2. 创建 train_moe.py

基于train_mix.py创建，主要修改：

```python
from ragen.trainer.moe_trainer import MoEAgentTrainer
from ragen.cl_methods.moe import MoECLMethod, MoEConfig

def build_task_configs(config) -> List[Dict[str, Any]]:
    """Build task configurations from moe_training config."""
    moe_config = config.moe_training
    task_configs = []

    for task in moe_config.tasks:
        task_config = {
            'name': task.name,
            'train_tags': list(task.train_tags),
            'train_n_groups': list(task.train_n_groups),
            'val_tags': list(task.val_tags),
            'val_n_groups': list(task.val_n_groups),
        }
        task_configs.append(task_config)

    return task_configs

def create_moe_method(config, task_configs: List[Dict]) -> MoECLMethod:
    """Create and initialize the MoECLMethod instance."""
    moe_config = MoEConfig(
        name="moe",
        num_tasks=len(task_configs),
        task_names=[t['name'] for t in task_configs],
        lora_rank=config.lora.rank,
        lora_alpha=config.lora.alpha,
    )

    moe_method = MoECLMethod(moe_config)
    return moe_method

@hydra.main(version_base=None, config_path="config", config_name="moe_training_9tasks")
def main(config: DictConfig):
    # ... setup code ...

    # Build task configs
    task_configs = build_task_configs(config)

    # Create MoE method
    moe_method = create_moe_method(config, task_configs)

    # Create MoE trainer
    trainer = MoEAgentTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        moe_method=moe_method,
        task_configs=task_configs,
    )

    # Initialize MoE workers
    trainer.init_moe_workers()

    # Start training
    trainer.fit()
```

### 3. 创建训练脚本 run_moe_9tasks.sh

```bash
#!/bin/bash
# Training script for 9-task MoE

set -e

# Default parameters
TOTAL_STEPS=${TOTAL_STEPS:-900}
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-100}

# Build command
CMD="python train_moe.py --config-name moe_training_9tasks"

# Override GPU settings if CUDA_VISIBLE_DEVICES is set
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#GPUS[@]}
    CMD="$CMD system.CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    CMD="$CMD trainer.n_gpus_per_node=$NUM_GPUS"
fi

# Add training parameters
CMD="$CMD trainer.total_training_steps=$TOTAL_STEPS"
CMD="$CMD trainer.test_freq=$TEST_FREQ"
CMD="$CMD trainer.save_freq=$SAVE_FREQ"

# Print configuration
echo "=========================================="
echo "MoE 9-Task Training"
echo "=========================================="
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (${NUM_GPUS} GPU(s))"
fi
echo "Total Steps: $TOTAL_STEPS"
echo "Test Frequency: $TEST_FREQ"
echo "Save Frequency: $SAVE_FREQ"
echo "=========================================="
echo ""

# Run training
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD
```

## 关键实现要点

### 1. Worker管理

- 每个task需要独立的actor和critic worker
- Workers通过Ray remote创建
- 使用`switch_to_task()`切换当前活跃的worker

### 2. 串行采样

```python
def _serial_sampling(self) -> List[DataProto]:
    task_batches = []
    for task_idx in range(self.num_tasks):
        self.switch_to_task(task_idx)  # 切换到该任务的worker
        batch = self._sample_single_task()  # 使用该任务的LoRA采样
        task_batches.append(batch)
    return task_batches
```

### 3. 串行更新

```python
def _serial_updates(self, task_batches: List[DataProto]):
    for task_idx in range(self.num_tasks):
        self.switch_to_task(task_idx)  # 切换到该任务的worker
        batch = task_batches[task_idx]
        self._update_single_task(batch, task_idx)  # 更新该任务的LoRA
```

### 4. Checkpoint保存

```python
checkpoint_dir/
  task_0_bandit_low/
    actor/
      model.pt
    critic/
      model.pt
  task_1_bandit_medium/
    actor/
      model.pt
    critic/
      model.pt
  ...
  task_8_frozen_lake_hard/
    actor/
      model.pt
    critic/
      model.pt
  cl_method_state.pt
```

## 测试步骤

1. 测试MoE方法注册：
```bash
python -c "from ragen.cl_methods import get_cl_method; print(get_cl_method('moe'))"
```

2. 测试配置加载：
```bash
python -c "from omegaconf import OmegaConf; config = OmegaConf.load('config/moe_training_9tasks.yaml'); print(config.moe_training.tasks[0])"
```

3. 运行短期训练测试：
```bash
CUDA_VISIBLE_DEVICES=0 python train_moe.py --config-name moe_training_9tasks trainer.total_training_steps=10
```

## 预期行为

- 每个训练步骤：
  1. 串行采样9个任务（每个任务用自己的LoRA）
  2. 串行更新9个任务（每个任务更新自己的LoRA）

- Validation时：
  - 验证所有9个任务
  - 每个任务使用自己的LoRA

- 最终结果：
  - 9个独立训练的LoRA模块
  - 每个任务的性能应该接近独立训练的结果

## 注意事项

1. **显存管理**：串行更新确保同时只有一个任务在更新，避免OOM
2. **Worker隔离**：每个worker有独立的LoRA，确保任务间完全独立
3. **Checkpoint大小**：9个任务的checkpoint会比较大，注意磁盘空间
4. **训练时间**：串行采样和更新会比并行慢，但保证了任务独立性

## 下一步

1. 完善`moe_trainer.py`中的worker创建和更新逻辑
2. 创建完整的`train_moe.py`
3. 创建`run_moe_9tasks.sh`
4. 测试短期训练
5. 运行完整训练并验证结果
