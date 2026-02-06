"""
Experience Replay (ER) for In-Context Lifelong Memory.

This ER variant stores successful trajectories as in-context examples
and never updates model parameters. Each experience is a full successful
rollout and is injected into the system prompt for future rollouts.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class Experience:
    """A single successful rollout (full interaction history)."""
    trajectory: List[Dict[str, Any]]
    env_tag: str
    total_reward: float
    num_turns: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory": self.trajectory,
            "env_tag": self.env_tag,
            "total_reward": self.total_reward,
            "num_turns": self.num_turns,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        return cls(
            trajectory=data["trajectory"],
            env_tag=data["env_tag"],
            total_reward=data["total_reward"],
            num_turns=data["num_turns"],
            timestamp=data["timestamp"],
        )


class ReplayBuffer:
    """A fixed-size buffer for successful experiences."""

    def __init__(self, max_size: int, env_tag: str):
        self.max_size = int(max_size)
        self.env_tag = env_tag
        self.buffer: List[Experience] = []
        self._total_collected = 0

    @property
    def size(self) -> int:
        return len(self.buffer)

    @property
    def is_full(self) -> bool:
        return self.size >= self.max_size

    @property
    def total_collected(self) -> int:
        return self._total_collected

    def add(self, experience: Experience) -> bool:
        """Add one experience; returns True if buffer becomes full."""
        if self.is_full:
            return True
        self.buffer.append(experience)
        self._total_collected += 1
        return self.is_full

    def get_all(self) -> List[Experience]:
        return list(self.buffer)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "max_size": self.max_size,
            "env_tag": self.env_tag,
            "total_collected": self._total_collected,
            "experiences": [exp.to_dict() for exp in self.buffer],
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "ReplayBuffer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        buffer = cls(max_size=data["max_size"], env_tag=data["env_tag"])
        buffer._total_collected = data.get("total_collected", 0)
        buffer.buffer = [Experience.from_dict(exp) for exp in data["experiences"]]
        return buffer


@dataclass
class ERConfig:
    """Configuration for ER memory."""
    buffer_size: int = 20
    env_tag: str = "BanditLow"
    output_dir: str = "results/experience_replay"


class ExperienceReplayMethod:
    """
    ER method: collect successful trajectories and reuse them in-context.
    """

    def __init__(self, config: ERConfig):
        self.config = config
        self.buffer = ReplayBuffer(max_size=config.buffer_size, env_tag=config.env_tag)

    def is_full(self) -> bool:
        return self.buffer.is_full

    def add_experience(self, experience: Experience) -> bool:
        return self.buffer.add(experience)

    def extract_successful_experiences(self, rollout_states: List[Dict[str, Any]]) -> List[Experience]:
        """Extract successful experiences from rollout states."""
        successful: List[Experience] = []
        for state in rollout_states:
            metrics = state.get("metrics", {})
            is_success = False
            for key, value in metrics.items():
                if key.endswith("/success") and float(value) == 1.0:
                    is_success = True
                    break
            if not is_success:
                continue

            history = list(state.get("history", []))
            if history and "llm_response" not in history[-1]:
                history = history[:-1]

            total_reward = 0.0
            num_turns = 0
            for turn in history:
                if "reward" in turn:
                    total_reward += float(turn["reward"])
                if "llm_response" in turn:
                    num_turns += 1

            env_tag = state.get("tag", self.config.env_tag)
            successful.append(
                Experience(
                    trajectory=history,
                    env_tag=env_tag,
                    total_reward=total_reward,
                    num_turns=num_turns,
                    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                )
            )
        return successful

    def format_experience_as_example(self, experience: Experience, index: int) -> str:
        """Format a single experience as a prompt example."""
        lines = [f"[Example {index}] (Reward: {experience.total_reward:.2f})"]
        for i, turn in enumerate(experience.trajectory, 1):
            if "state" in turn:
                lines.append(f"Turn {i} State:\n{turn['state']}")
            if "actions_left" in turn:
                lines.append(f"Turn {i} Actions left: {turn['actions_left']}")
            if "llm_response" in turn:
                lines.append(f"Turn {i} Response:\n{turn['llm_response']}")
            if "reward" in turn:
                lines.append(f"Turn {i} Reward: {turn['reward']}")
        lines.append(f"[End Example {index}]\n")
        return "\n".join(lines)

    def get_examples_text(self) -> str:
        """Get formatted examples text using all experiences in the buffer."""
        if self.buffer.size == 0:
            return ""
        parts = ["\n--- Previous Successful Trajectories (In-Context Memory) ---\n"]
        for idx, exp in enumerate(self.buffer.get_all(), 1):
            parts.append(self.format_experience_as_example(exp, idx))
        parts.append("--- End of Memory ---\n\n")
        return "\n".join(parts)

    def save(self, path: str) -> None:
        self.buffer.save(path)

    def load(self, path: str) -> None:
        self.buffer = ReplayBuffer.load(path)
