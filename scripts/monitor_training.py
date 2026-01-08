#!/usr/bin/env python3
"""
监控RAGEN训练进度的脚本

使用方法:
    python scripts/monitor_training.py --log_file wandb/latest-run/files/output.log
    python scripts/monitor_training.py --log_file wandb/latest-run/files/output.log --plot
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="监控RAGEN训练进度")
    parser.add_argument(
        "--log_file",
        type=str,
        default="wandb/latest-run/files/output.log",
        help="日志文件路径",
    )
    parser.add_argument(
        "--plot", action="store_true", help="是否绘制图表"
    )
    parser.add_argument(
        "--tail", type=int, default=20, help="显示最后N个步骤"
    )
    return parser.parse_args()


def extract_metrics(log_file: Path) -> Dict[str, List]:
    """从日志文件中提取训练指标"""
    metrics = {
        "step": [],
        "train_success": [],
        "train_reward": [],
        "val_success": [],
        "val_reward": [],
        "actor_lr": [],
        "critic_lr": [],
        "actor_loss": [],
        "critic_loss": [],
    }

    if not log_file.exists():
        print(f"错误: 日志文件不存在: {log_file}")
        return metrics

    with open(log_file, "r") as f:
        for line in f:
            # 提取step
            step_match = re.search(r"step:(\d+)", line)
            if not step_match:
                continue

            step = int(step_match.group(1))
            metrics["step"].append(step)

            # 提取训练success rate
            train_success_match = re.search(r"train/CoordFrozenLake/success:(\d+\.\d+)", line)
            if train_success_match:
                metrics["train_success"].append(float(train_success_match.group(1)))
            else:
                metrics["train_success"].append(None)

            # 提取验证success rate
            val_success_match = re.search(r"val-env/CoordFrozenLake/success:(\d+\.\d+)", line)
            if val_success_match:
                metrics["val_success"].append(float(val_success_match.group(1)))
            else:
                metrics["val_success"].append(None)

            # 提取actor loss
            actor_loss_match = re.search(r"actor/pg_loss:(-?\d+\.\d+)", line)
            if actor_loss_match:
                metrics["actor_loss"].append(float(actor_loss_match.group(1)))
            else:
                metrics["actor_loss"].append(None)

            # 提取critic loss
            critic_loss_match = re.search(r"critic/vf_loss:(\d+\.\d+)", line)
            if critic_loss_match:
                metrics["critic_loss"].append(float(critic_loss_match.group(1)))
            else:
                metrics["critic_loss"].append(None)

            # 提取learning rate
            actor_lr_match = re.search(r"actor/lr:(\d+\.?\d*[eE]?-?\d*)", line)
            if actor_lr_match:
                metrics["actor_lr"].append(float(actor_lr_match.group(1)))
            else:
                metrics["actor_lr"].append(None)

            critic_lr_match = re.search(r"critic/lr:(\d+\.?\d*[eE]?-?\d*)", line)
            if critic_lr_match:
                metrics["critic_lr"].append(float(critic_lr_match.group(1)))
            else:
                metrics["critic_lr"].append(None)

    return metrics


def print_summary(metrics: Dict[str, List], tail: int = 20):
    """打印训练摘要"""
    if not metrics["step"]:
        print("未找到训练数据")
        return

    print(f"\n{'='*80}")
    print(f"训练进度摘要 (最后 {tail} 步)")
    print(f"{'='*80}\n")

    # 获取最后N个步骤
    start_idx = max(0, len(metrics["step"]) - tail)

    print(f"{'Step':<8} {'Train Success':<15} {'Val Success':<15} {'Actor Loss':<15} {'Critic Loss':<15}")
    print("-" * 80)

    for i in range(start_idx, len(metrics["step"])):
        step = metrics["step"][i]
        train_succ = metrics["train_success"][i]
        val_succ = metrics["val_success"][i]
        actor_loss = metrics["actor_loss"][i]
        critic_loss = metrics["critic_loss"][i]

        train_succ_str = f"{train_succ:.4f}" if train_succ is not None else "N/A"
        val_succ_str = f"{val_succ:.4f}" if val_succ is not None else "N/A"
        actor_loss_str = f"{actor_loss:.4f}" if actor_loss is not None else "N/A"
        critic_loss_str = f"{critic_loss:.4f}" if critic_loss is not None else "N/A"

        print(f"{step:<8} {train_succ_str:<15} {val_succ_str:<15} {actor_loss_str:<15} {critic_loss_str:<15}")

    # 打印统计信息
    train_successes = [s for s in metrics["train_success"] if s is not None]
    if train_successes:
        print(f"\n{'='*80}")
        print("训练 Success Rate 统计:")
        print(f"  最小值: {min(train_successes):.4f}")
        print(f"  最大值: {max(train_successes):.4f}")
        print(f"  平均值: {sum(train_successes)/len(train_successes):.4f}")
        print(f"  当前值: {train_successes[-1]:.4f}")

    val_successes = [s for s in metrics["val_success"] if s is not None]
    if val_successes:
        print(f"\n验证 Success Rate 统计:")
        print(f"  最小值: {min(val_successes):.4f}")
        print(f"  最大值: {max(val_successes):.4f}")
        print(f"  平均值: {sum(val_successes)/len(val_successes):.4f}")
        print(f"  当前值: {val_successes[-1]:.4f}")

    print(f"{'='*80}\n")


def plot_metrics(metrics: Dict[str, List]):
    """绘制训练曲线"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("错误: 需要安装 matplotlib 才能绘图")
        print("运行: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("RAGEN 训练监控", fontsize=16)

    # Success Rate
    ax = axes[0, 0]
    train_succ = [s if s is not None else np.nan for s in metrics["train_success"]]
    val_succ = [s if s is not None else np.nan for s in metrics["val_success"]]
    ax.plot(metrics["step"], train_succ, label="Train Success", marker="o", markersize=3)
    ax.plot(metrics["step"], val_succ, label="Val Success", marker="s", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate 变化")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Actor Loss
    ax = axes[0, 1]
    actor_loss = [l if l is not None else np.nan for l in metrics["actor_loss"]]
    ax.plot(metrics["step"], actor_loss, label="Actor Loss", marker="o", markersize=3, color="orange")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Actor Loss 变化")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Critic Loss
    ax = axes[1, 0]
    critic_loss = [l if l is not None else np.nan for l in metrics["critic_loss"]]
    ax.plot(metrics["step"], critic_loss, label="Critic Loss", marker="o", markersize=3, color="green")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Critic Loss 变化")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning Rate
    ax = axes[1, 1]
    actor_lr = [lr if lr is not None else np.nan for lr in metrics["actor_lr"]]
    critic_lr = [lr if lr is not None else np.nan for lr in metrics["critic_lr"]]
    ax.plot(metrics["step"], actor_lr, label="Actor LR", marker="o", markersize=3)
    ax.plot(metrics["step"], critic_lr, label="Critic LR", marker="s", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate 变化")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = "training_progress.png"
    plt.savefig(output_file, dpi=150)
    print(f"图表已保存到: {output_file}")


def main():
    args = parse_args()
    log_file = Path(args.log_file)

    print(f"正在读取日志文件: {log_file}")
    metrics = extract_metrics(log_file)

    if not metrics["step"]:
        print("未找到训练数据!")
        sys.exit(1)

    print_summary(metrics, tail=args.tail)

    if args.plot:
        plot_metrics(metrics)


if __name__ == "__main__":
    main()

