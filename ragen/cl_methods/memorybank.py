"""
MemoryBank: In-context long-term memory with retrieval and strength-based decay.

This implementation stores successful trajectories as memories, retrieves
relevant memories for each new state, and updates memory strength when recalled.
No model parameters are updated.
"""

from __future__ import annotations

import math
import os
import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np


@dataclass
class MemoryEntry:
    """A single memory entry (successful rollout)."""
    trajectory: List[Dict[str, Any]]
    env_tag: str
    total_reward: float
    num_turns: int
    timestamp: str
    memory_strength: float
    last_recall_step: int
    created_step: int
    memory_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory": self.trajectory,
            "env_tag": self.env_tag,
            "total_reward": self.total_reward,
            "num_turns": self.num_turns,
            "timestamp": self.timestamp,
            "memory_strength": self.memory_strength,
            "last_recall_step": self.last_recall_step,
            "created_step": self.created_step,
            "memory_text": self.memory_text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            trajectory=data["trajectory"],
            env_tag=data["env_tag"],
            total_reward=data["total_reward"],
            num_turns=data["num_turns"],
            timestamp=data["timestamp"],
            memory_strength=data.get("memory_strength", 1.0),
            last_recall_step=data.get("last_recall_step", 0),
            created_step=data.get("created_step", 0),
            memory_text=data.get("memory_text", ""),
        )


@dataclass
class MemoryBankConfig:
    """Configuration for MemoryBank."""
    buffer_size: int = 20
    env_tag: str = "BanditLow"
    top_k: int = 4
    decay_tau: float = 5.0
    output_dir: str = "results/memorybank"


class MemoryBank:
    """MemoryBank storage and retrieval."""

    def __init__(self, config: MemoryBankConfig):
        self.config = config
        self.entries: List[MemoryEntry] = []
        self._total_collected = 0

    @property
    def size(self) -> int:
        return len(self.entries)

    @property
    def is_full(self) -> bool:
        return self.size >= self.config.buffer_size

    @property
    def total_collected(self) -> int:
        return self._total_collected

    def add(self, entry: MemoryEntry) -> bool:
        if self.is_full:
            return True
        self.entries.append(entry)
        self._total_collected += 1
        return self.is_full

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+", text.lower())

    def _tfidf_similarity(self, query: str, docs: List[str]) -> np.ndarray:
        if not docs:
            return np.array([])
        tokenized_docs = [self._tokenize(doc) for doc in docs]
        tokenized_query = self._tokenize(query)

        vocab = {}
        for tokens in tokenized_docs + [tokenized_query]:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        if not vocab:
            return np.zeros(len(docs), dtype=float)

        def build_tf(tokens):
            vec = np.zeros(len(vocab), dtype=float)
            for tok in tokens:
                vec[vocab[tok]] += 1.0
            if tokens:
                vec /= max(len(tokens), 1)
            return vec

        tfs = [build_tf(tokens) for tokens in tokenized_docs]
        tf_query = build_tf(tokenized_query)

        df = np.zeros(len(vocab), dtype=float)
        for vec in tfs:
            df += (vec > 0).astype(float)
        df += (tf_query > 0).astype(float)
        idf = np.log((1.0 + len(docs) + 1.0) / (1.0 + df)) + 1.0

        doc_vecs = [tf * idf for tf in tfs]
        query_vec = tf_query * idf

        query_norm = np.linalg.norm(query_vec) + 1e-8
        sims = []
        for dv in doc_vecs:
            denom = (np.linalg.norm(dv) + 1e-8) * query_norm
            sims.append(float(np.dot(dv, query_vec) / denom))
        return np.array(sims, dtype=float)

    def _retention(self, step: int, entry: MemoryEntry) -> float:
        t = max(step - entry.last_recall_step, 0)
        s = max(entry.memory_strength, 1e-6)
        return math.exp(-t / (self.config.decay_tau * s))

    def retrieve(self, query: str, step: int) -> List[MemoryEntry]:
        if not self.entries:
            return []
        docs = [entry.memory_text for entry in self.entries]
        sims = self._tfidf_similarity(query, docs)
        if sims.size == 0:
            return []

        scores = []
        for sim, entry in zip(sims, self.entries):
            scores.append(sim * self._retention(step, entry))
        scores = np.array(scores, dtype=float)

        top_k = min(self.config.top_k, len(self.entries))
        if top_k <= 0:
            return []
        top_indices = np.argsort(scores)[::-1][:top_k]
        retrieved = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            entry = self.entries[idx]
            entry.memory_strength += 1.0
            entry.last_recall_step = step
            retrieved.append(entry)
        return retrieved

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "buffer_size": self.size,
            "buffer_max_size": self.config.buffer_size,
            "total_collected": self._total_collected,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._total_collected = state.get("total_collected", 0)
        entries = state.get("entries", [])
        self.entries = [MemoryEntry.from_dict(item) for item in entries]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.get_state_dict(), f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.load_state_dict(state)


class MemoryBankMethod:
    """MemoryBank method: extract successes and retrieve memories for prompts."""

    def __init__(self, config: MemoryBankConfig):
        self.config = config
        self.bank = MemoryBank(config)

    def is_full(self) -> bool:
        return self.bank.is_full

    def add_memory(self, entry: MemoryEntry) -> bool:
        return self.bank.add(entry)

    def _build_memory_text(self, trajectory: List[Dict[str, Any]]) -> str:
        parts = []
        for turn in trajectory:
            if "state" in turn:
                parts.append(f"State: {turn['state']}")
            if "llm_response" in turn:
                parts.append(f"Response: {turn['llm_response']}")
            if "reward" in turn:
                parts.append(f"Reward: {turn['reward']}")
        return "\n".join(parts)

    def extract_successful_memories(
        self, rollout_states: List[Dict[str, Any]], step: int
    ) -> List[MemoryEntry]:
        memories: List[MemoryEntry] = []
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
            memory_text = self._build_memory_text(history)
            memories.append(
                MemoryEntry(
                    trajectory=history,
                    env_tag=env_tag,
                    total_reward=total_reward,
                    num_turns=num_turns,
                    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                    memory_strength=1.0,
                    last_recall_step=step,
                    created_step=step,
                    memory_text=memory_text,
                )
            )
        return memories

    def format_memory_as_example(self, entry: MemoryEntry, index: int) -> str:
        lines = [f"[Memory {index}] (Reward: {entry.total_reward:.2f})"]
        for i, turn in enumerate(entry.trajectory, 1):
            if "state" in turn:
                lines.append(f"Turn {i} State:\n{turn['state']}")
            if "actions_left" in turn:
                lines.append(f"Turn {i} Actions left: {turn['actions_left']}")
            if "llm_response" in turn:
                lines.append(f"Turn {i} Response:\n{turn['llm_response']}")
            if "reward" in turn:
                lines.append(f"Turn {i} Reward: {turn['reward']}")
        lines.append(f"[End Memory {index}]\n")
        return "\n".join(lines)

    def get_examples_text(self, query: str, step: int) -> str:
        retrieved = self.bank.retrieve(query, step)
        if not retrieved:
            return ""
        parts = ["\n--- Retrieved Memories (MemoryBank) ---\n"]
        for idx, entry in enumerate(retrieved, 1):
            parts.append(self.format_memory_as_example(entry, idx))
        parts.append("--- End of Memories ---\n\n")
        return "\n".join(parts)

    def save(self, path: str) -> None:
        self.bank.save(path)

    def load(self, path: str) -> None:
        self.bank.load(path)
