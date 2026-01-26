# MoE Checkpoint-Based Implementation

## 实现方式

MoE 现在使用 **checkpoint 机制** 来实现 LoRA 切换，而不是内存中的参数传递。

## 工作原理

### 1. 临时 Checkpoint 目录

```python
# 在初始化时创建临时目录
self.temp_checkpoint_dir = tempfile.mkdtemp(prefix="moe_lora_")

# 目录结构：
/tmp/moe_lora_XXXXXX/
  ├── task_0/
  │   └── actor/
  │       └── model.pt
  ├── task_1/
  │   └── actor/
  │       └── model.pt
  ...
  └── task_8/
      └── actor/
          └── model.pt
```

### 2. LoRA 保存

```python
def _save_current_lora_checkpoint(self, task_idx: int):
    """保存当前 LoRA 到 checkpoint"""
    task_checkpoint_dir = os.path.join(self.temp_checkpoint_dir, f"task_{task_idx}")
    actor_dir = os.path.join(task_checkpoint_dir, "actor")

    # 使用现有的 save_checkpoint 方法
    ray.get(self.actor_rollout_wg.save_checkpoint.remote(actor_dir, None, 0))
```

### 3. LoRA 加载

```python
def _load_lora_checkpoint(self, task_idx: int):
    """从 checkpoint 加载 LoRA"""
    task_checkpoint_dir = os.path.join(self.temp_checkpoint_dir, f"task_{task_idx}")
    actor_dir = os.path.join(task_checkpoint_dir, "actor")

    # 使用现有的 load_checkpoint 方法
    ray.get(self.actor_rollout_wg.load_checkpoint.remote(actor_dir))
```

### 4. 任务切换

```python
def switch_to_task(self, task_idx: int):
    """切换到不同的任务"""
    # 1. 保存当前任务的 LoRA checkpoint
    self._save_current_lora_checkpoint(self.current_task_idx)

    # 2. 加载目标任务的 LoRA checkpoint
    self._load_lora_checkpoint(task_idx)

    # 3. 切换 ES managers
    self.agent_proxy.train_es_manager = self.task_train_es_managers[task_idx]
    # ...
```

### 5. 清理

```python
def _cleanup_temp_checkpoints(self):
    """训练结束后清理临时 checkpoint"""
    shutil.rmtree(self.temp_checkpoint_dir)
```

## 优点

1. **可靠性**：使用经过测试的 checkpoint 机制
2. **不依赖 Ray**：不需要自定义 Ray 方法
3. **持久化**：checkpoint 可以保存到磁盘
4. **兼容性**：与现有代码完全兼容

## 缺点

1. **速度**：每次切换需要磁盘 I/O（约 5-10 秒）
2. **磁盘空间**：需要临时存储 9 个 checkpoint

## 性能预期

### 任务切换时间

- **保存 checkpoint**：约 2-5 秒
- **加载 checkpoint**：约 2-5 秒
- **总切换时间**：约 5-10 秒

### 训练时间估算

```
每个训练步骤：
- 串行采样：9 个任务 × (切换 5秒 + 采样 30秒) = 315秒
- 串行更新：9 个任务 × (切换 5秒 + 更新 20秒) = 225秒
- 总计：约 540秒/步 = 9分钟/步

900 步训练：
- 总时间：900 × 9分钟 = 8100分钟 ≈ 135小时 ≈ 5.6天
```

### 显存占用

- **训练时**：与单任务相当（只有一个 LoRA 在 GPU）
- **临时存储**：9 × checkpoint 大小 ≈ 5-10 GB

## 与内存切换的对比

| 特性 | Checkpoint 机制 | 内存切换 |
|------|----------------|----------|
| 可靠性 | ✅ 高 | ⚠️ 需要调试 |
| 实现复杂度 | ✅ 简单 | ⚠️ 复杂 |
| 切换速度 | ⚠️ 5-10秒 | ✅ 1-2秒 |
| 显存占用 | ✅ 低 | ✅ 低 |
| 磁盘占用 | ⚠️ 5-10 GB | ✅ 无 |
| Ray 依赖 | ✅ 无 | ❌ 需要自定义方法 |

## 使用方法

### 基本训练

```bash
# 使用默认设置
bash run_moe_9tasks.sh

# 自定义参数
CUDA_VISIBLE_DEVICES=0 TOTAL_STEPS=900 bash run_moe_9tasks.sh
```

### 训练输出

```
[MoE] Created temporary checkpoint directory: /tmp/moe_lora_abc123
[MoE] Saving initial LoRA checkpoint for task 0
[MoE] Saved LoRA checkpoint for task 0 to /tmp/moe_lora_abc123/task_0/actor

[MoE] Training Step 1/900
[MoE] Phase 1: Serial sampling from 9 tasks...
[MoE] Sampling from task 0 (bandit_low)...
[MoE] Switching from task 0 to task 1
[MoE] Saving LoRA checkpoint for task 0
[MoE] Loading LoRA checkpoint for task 1
[MoE] Switched to task 1: bandit_medium
...

[MoE] Training completed!
[MoE] Cleaning up temporary checkpoints: /tmp/moe_lora_abc123
```

## 未来优化

如果需要更快的切换速度，可以：

1. **解决 Ray 调用问题**：实现内存中的 LoRA 切换
2. **使用 SSD**：将临时目录放在 SSD 上
3. **并行保存**：在后台异步保存 checkpoint
4. **缓存优化**：只在必要时保存/加载

## 总结

Checkpoint 机制是一个**可靠且实用**的解决方案：

- ✅ 立即可用
- ✅ 不需要调试 Ray
- ✅ 完全的任务独立性
- ⚠️ 速度较慢但可接受

对于研究和实验来说，这是一个很好的起点。如果需要更快的速度，可以在未来优化为内存切换。
