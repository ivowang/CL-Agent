# MoE (Mixture of Experts) for CL-Agent

## Quick Start

### Basic Training

```bash
# Train with default settings (GPU 0, 900 steps)
bash run_moe_9tasks.sh

# Use specific GPU
CUDA_VISIBLE_DEVICES=3 bash run_moe_9tasks.sh

# Custom training steps
TOTAL_STEPS=100 bash run_moe_9tasks.sh

# Full customization
CUDA_VISIBLE_DEVICES=0 TOTAL_STEPS=900 TEST_FREQ=20 SAVE_FREQ=100 bash run_moe_9tasks.sh
```

### Direct Python Usage

```bash
# Basic training
python train_moe.py --config-name moe_training_9tasks

# Short test run (10 steps)
python train_moe.py --config-name moe_training_9tasks trainer.total_training_steps=10

# Custom parameters
python train_moe.py --config-name moe_training_9tasks \
  trainer.total_training_steps=900 \
  trainer.test_freq=20 \
  trainer.save_freq=100
```

## What is MoE?

MoE (Mixture of Experts) is a continual learning method where:
- **Each task has its own independent LoRA module** (expert)
- **Each task has its own actor-critic pair** (complete isolation)
- **Training uses mixed sampling** (all tasks in each step, like Mix method)
- **Updates are serial** (one task at a time to avoid memory issues)

### Key Differences from Other Methods

| Method | LoRA Sharing | Sampling | Updates | Task Independence |
|--------|-------------|----------|---------|-------------------|
| Baseline | Shared | Sequential | Shared params | None |
| O-LoRA | Independent | Sequential | Orthogonal constraint | Partial |
| SD-LoRA | Independent | Sequential | Scaling factors | Partial |
| Mix | Shared | Mixed | Shared params | None |
| **MoE** | **Independent** | **Mixed** | **Fully independent** | **Complete** |

## Architecture

```
MoE Training Architecture:
┌─────────────────────────────────────────────────────────┐
│                    MoEAgentTrainer                      │
├─────────────────────────────────────────────────────────┤
│  Task 0: Actor₀ + Critic₀ + LoRA₀ + ES_Manager₀       │
│  Task 1: Actor₁ + Critic₁ + LoRA₁ + ES_Manager₁       │
│  Task 2: Actor₂ + Critic₂ + LoRA₂ + ES_Manager₂       │
│  ...                                                    │
│  Task 8: Actor₈ + Critic₈ + LoRA₈ + ES_Manager₈       │
└─────────────────────────────────────────────────────────┘

Training Flow (each step):
1. Serial Sampling:
   Task 0 → Sample with LoRA₀ → Batch₀
   Task 1 → Sample with LoRA₁ → Batch₁
   ...
   Task 8 → Sample with LoRA₈ → Batch₈

2. Serial Updates:
   Task 0 → Update LoRA₀ with Batch₀
   Task 1 → Update LoRA₁ with Batch₁
   ...
   Task 8 → Update LoRA₈ with Batch₈
```

## Training Process

### Each Training Step

**Phase 1: Serial Sampling**
```
For each task (0-8):
  - Switch to task's actor worker
  - Sample trajectories using task's LoRA
  - Store batch for this task
```

**Phase 2: Serial Updates**
```
For each task (0-8):
  - Switch to task's actor and critic workers
  - Compute advantages using task's advantage estimator
  - Update task's actor with mini-batches
  - Update task's critic
  - Record metrics
```

### Validation

- Validates all 9 tasks at specified intervals
- Each task uses its own LoRA for validation
- Records independent metrics per task

### Checkpoint Structure

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

## Configuration

### Main Config File: `config/moe_training_9tasks.yaml`

Key sections:
- `moe_training.tasks`: List of 9 tasks with train/val configurations
- `trainer.total_training_steps`: Total training steps (default: 900)
- `trainer.test_freq`: Validation frequency (default: 20)
- `trainer.save_freq`: Checkpoint save frequency (default: 100)
- `lora.rank`: LoRA rank (default: 64)
- `lora.alpha`: LoRA alpha (default: 64)

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU selection | 0 |
| `TOTAL_STEPS` | Total training steps | 900 |
| `TEST_FREQ` | Validation frequency | 20 |
| `SAVE_FREQ` | Checkpoint save frequency | 100 |

## Performance Characteristics

### Memory Usage
- **Sampling phase**: Single task memory (serial sampling)
- **Update phase**: Single task memory (serial updates)
- **Total**: Approximately equal to single-task training

### Training Time
- **Per step**: ~9× single-task training (serial execution)
- **Total time**: 900 steps × 9× ≈ 8100 single-task steps

### Expected Results
- **Task performance**: Should approach independent training results
- **Catastrophic forgetting**: None (tasks are completely independent)
- **Scalability**: Linear with number of tasks

## Implementation Files

### Core Implementation
- `ragen/cl_methods/moe.py` (170 lines) - MoE CL method class
- `ragen/trainer/moe_trainer.py` (549 lines) - MoE trainer
- `train_moe.py` (300+ lines) - Training script

### Configuration and Scripts
- `config/moe_training_9tasks.yaml` - 9-task configuration
- `run_moe_9tasks.sh` - Training shell script

### Documentation
- `docs/MOE_IMPLEMENTATION.md` - Implementation details
- `docs/MOE_SUMMARY.md` - Complete summary
- `docs/MOE_README.md` - This file

## Python API

### Creating MoE Method

```python
from ragen.cl_methods.moe import MoECLMethod, MoEConfig

# Create configuration
config = MoEConfig(
    num_tasks=9,
    task_names=['bandit_low', 'bandit_medium', 'bandit_hard',
                'sokoban_low', 'sokoban_medium', 'sokoban_hard',
                'frozen_lake_low', 'frozen_lake_medium', 'frozen_lake_hard'],
    lora_rank=64,
    lora_alpha=64,
)

# Create method
method = MoECLMethod(config)

# Use in training
method.on_task_start(0, "moe_training", None)
# ... training ...
method.on_task_end(0, "moe_training", checkpoint_path)
```

### Using MoE Trainer

```python
from ragen.trainer.moe_trainer import MoEAgentTrainer

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

# Initialize workers
trainer.init_moe_workers()

# Start training
trainer.fit()
```

## Troubleshooting

### Issue: Ray worker creation fails

**Solution**: Check Ray resource configuration
```bash
ray status  # Check available resources
```

Ensure sufficient GPU and memory for 18 workers (9 actors + 9 critics).

### Issue: Out of memory

**Possible causes**:
1. Serial updates not working correctly
2. Batch size too large
3. Too many concurrent workers

**Solutions**:
```yaml
# Reduce batch size in config
data:
  train_batch_size: 64  # Reduce from 128

# Reduce GPU memory utilization
trainer:
  gpu_memory_utilization: 0.8  # Reduce from 0.9
```

### Issue: Task switching fails

**Solution**: Check task index validity
```python
# Valid task indices: 0-8
assert 0 <= task_idx < 9
```

### Issue: Slow training

**Expected behavior**: MoE is ~9× slower than single-task due to serial execution.

**Optimization tips**:
- Reduce validation frequency: `TEST_FREQ=50`
- Reduce checkpoint frequency: `SAVE_FREQ=200`
- Use fewer training steps for testing: `TOTAL_STEPS=100`

## Testing

### Quick Functionality Test

```bash
# 10-step test run
CUDA_VISIBLE_DEVICES=0 python train_moe.py \
  --config-name moe_training_9tasks \
  trainer.total_training_steps=10
```

Expected output:
```
[MoE] Training Step 1/10
[MoE] Phase 1: Serial sampling from 9 tasks...
[MoE] Sampling from task 0 (bandit_low)...
[MoE] Task 0 sampled 128 trajectories
...
[MoE] Phase 2: Serial updates for 9 tasks...
[MoE] Updating task 0 (bandit_low)...
[MoE] Task 0 updated
...
```

### Validation Test

```bash
# Run with frequent validation
TEST_FREQ=5 TOTAL_STEPS=20 bash run_moe_9tasks.sh
```

Should see validation results for all 9 tasks every 5 steps.

### Checkpoint Test

```bash
# Run with frequent checkpointing
SAVE_FREQ=10 TOTAL_STEPS=30 bash run_moe_9tasks.sh
```

Check that checkpoints are saved at steps 10, 20, 30 with all 9 task directories.

## Comparison with Other Methods

### When to Use MoE

**Use MoE when**:
- You want complete task independence
- You have sufficient memory for multiple workers
- Training time is not critical
- You want to avoid catastrophic forgetting entirely

**Use other methods when**:
- Training time is critical → Use Mix or Baseline
- Memory is limited → Use O-LoRA or SD-LoRA
- You want parameter efficiency → Use O-LoRA or SD-LoRA

### Performance Trade-offs

| Aspect | MoE | O-LoRA | SD-LoRA | Mix | Baseline |
|--------|-----|--------|---------|-----|----------|
| Task independence | ✓✓✓ | ✓✓ | ✓✓ | ✗ | ✗ |
| Training speed | ✗ | ✓✓ | ✓✓ | ✓✓✓ | ✓✓✓ |
| Memory efficiency | ✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| No forgetting | ✓✓✓ | ✓✓ | ✓✓ | ✗ | ✗ |
| Implementation complexity | ✓✓ | ✓✓ | ✓ | ✓✓✓ | ✓✓✓ |

## Advanced Usage

### Custom Task Configuration

Edit `config/moe_training_9tasks.yaml`:

```yaml
moe_training:
  tasks:
    - name: my_task_1
      train_tags: [MyEnv]
      train_n_groups: [4]
      val_tags: [MyEnv]
      val_n_groups: [2]
    - name: my_task_2
      # ...
```

### Resume from Checkpoint

```bash
python train_moe.py \
  --config-name moe_training_9tasks \
  trainer.resume_checkpoint=/path/to/checkpoint
```

### Custom LoRA Configuration

```bash
python train_moe.py \
  --config-name moe_training_9tasks \
  lora.rank=128 \
  lora.alpha=128
```

## FAQ

**Q: Why is MoE slower than other methods?**
A: MoE uses serial sampling and updates to maintain complete task independence and avoid memory issues. This makes it ~9× slower than single-task training.

**Q: Can I run MoE with fewer tasks?**
A: Yes, modify the `moe_training.tasks` list in the config file to include only the tasks you want.

**Q: How much memory does MoE need?**
A: Approximately the same as single-task training, since updates are serial. However, you need Ray resources for 18 workers.

**Q: Can I parallelize MoE training?**
A: The current implementation uses serial updates to avoid memory issues. Parallelization would require careful memory management and is left for future work.

**Q: How do I evaluate a trained MoE model?**
A: Use the validation functionality built into the trainer, or load individual task checkpoints for evaluation.

## Citation

If you use MoE in your research, please cite:

```bibtex
@software{cl_agent_moe,
  title = {MoE Implementation for CL-Agent},
  author = {CL-Agent Team},
  year = {2026},
  url = {https://github.com/your-repo/CL-Agent}
}
```

## Support

For issues and questions:
1. Check this README and other documentation in `docs/`
2. Review the troubleshooting section
3. Open an issue on GitHub

## License

Same as CL-Agent framework.
