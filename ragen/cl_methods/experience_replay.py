"""
Experience Replay (ER) Method for In-Context Lifelong Learning

This method collects successful trajectories and uses them as in-context examples
to help the LLM learn from past experiences without updating model parameters.

Key concepts:
- An "experience" is a complete successful trajectory (interaction history)
- Experiences are stored in a replay buffer
- During inference, buffer contents are prepended to prompts as examples
- No parameter updates - pure in-context learning
"""

import os
import json
import pickle
import random
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime


@dataclass
class Experience:
    """A single experience (successful trajectory)."""
    trajectory: List[Dict]  # The complete interaction history
    env_tag: str  # Environment type (e.g., "Bandit", "CoordSokoban")
    total_reward: float  # Total reward achieved
    num_turns: int  # Number of turns in the trajectory
    timestamp: str  # When the experience was collected
    
    def to_dict(self) -> Dict:
        return {
            'trajectory': self.trajectory,
            'env_tag': self.env_tag,
            'total_reward': self.total_reward,
            'num_turns': self.num_turns,
            'timestamp': self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Experience':
        return cls(
            trajectory=data['trajectory'],
            env_tag=data['env_tag'],
            total_reward=data['total_reward'],
            num_turns=data['num_turns'],
            timestamp=data['timestamp'],
        )


class ReplayBuffer:
    """
    Buffer to store successful experiences.
    
    Args:
        max_size: Maximum number of experiences to store (N)
        env_tag: Environment type this buffer is for
    """
    
    def __init__(self, max_size: int, env_tag: str):
        self.max_size = max_size
        self.env_tag = env_tag
        self.buffer: List[Experience] = []
        self._total_collected = 0  # Total experiences ever collected
    
    @property
    def size(self) -> int:
        """Current number of experiences in buffer."""
        return len(self.buffer)
    
    @property
    def is_full(self) -> bool:
        """Whether buffer has reached max_size."""
        return self.size >= self.max_size
    
    @property
    def total_collected(self) -> int:
        """Total number of experiences ever collected."""
        return self._total_collected
    
    def add(self, experience: Experience) -> bool:
        """
        Add an experience to the buffer.
        
        Args:
            experience: The experience to add
            
        Returns:
            True if buffer became full after adding
        """
        if not self.is_full:
            self.buffer.append(experience)
            self._total_collected += 1
            return self.is_full
        return True  # Already full
    
    def get_all(self) -> List[Experience]:
        """Get all experiences in the buffer."""
        return list(self.buffer)
    
    def get_random(self, n: int) -> List[Experience]:
        """Get n random experiences from the buffer."""
        if n >= self.size:
            return self.get_all()
        return random.sample(self.buffer, n)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []
    
    def save(self, path: str):
        """Save buffer to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        data = {
            'max_size': self.max_size,
            'env_tag': self.env_tag,
            'total_collected': self._total_collected,
            'experiences': [exp.to_dict() for exp in self.buffer]
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'ReplayBuffer':
        """Load buffer from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        buffer = cls(max_size=data['max_size'], env_tag=data['env_tag'])
        buffer._total_collected = data['total_collected']
        buffer.buffer = [Experience.from_dict(exp) for exp in data['experiences']]
        return buffer


@dataclass
class ERConfig:
    """Configuration for Experience Replay method."""
    # Buffer settings
    buffer_size: int = 20  # N: Maximum experiences to collect
    val_frequency: int = 5  # M: Validate every M new experiences
    
    # In-context example settings
    max_examples_in_prompt: int = 3  # Max examples to include in prompt
    example_selection: str = "random"  # "random", "recent", "best"
    
    # Environment settings
    env_tag: str = "Bandit"  # Which environment to train on
    
    # Output settings
    output_dir: str = "results/experience_replay"


class ExperienceReplayMethod:
    """
    Experience Replay method for in-context lifelong learning.
    
    This method:
    1. Runs rollouts without parameter updates
    2. Collects successful trajectories into a replay buffer
    3. Uses buffer contents as in-context examples for future prompts
    4. Triggers validation at specified intervals
    """
    
    def __init__(self, config: ERConfig):
        self.config = config
        self.buffer = ReplayBuffer(
            max_size=config.buffer_size,
            env_tag=config.env_tag
        )
        self._last_val_count = 0  # Buffer size at last validation
        
    def should_validate(self) -> bool:
        """Check if validation should be triggered."""
        current_size = self.buffer.size
        # Validate every M experiences
        if current_size > 0 and current_size > self._last_val_count:
            if current_size % self.config.val_frequency == 0:
                return True
        return False
    
    def mark_validated(self):
        """Mark that validation was just performed."""
        self._last_val_count = self.buffer.size
    
    def is_training_complete(self) -> bool:
        """Check if training is complete (buffer full)."""
        return self.buffer.is_full
    
    def extract_successful_experiences(
        self, 
        rollout_states: List[Dict]
    ) -> List[Experience]:
        """
        Extract successful experiences from rollout states.
        
        Args:
            rollout_states: List of rollout state dicts from es_manager.get_rollout_states()
            
        Returns:
            List of Experience objects for successful episodes
        """
        successful = []
        
        for state in rollout_states:
            # Check if this episode was successful
            metrics = state.get('metrics', {})
            
            # Find success metric (format: "{tag}/success")
            is_success = False
            for key, value in metrics.items():
                if key.endswith('/success') and value == 1.0:
                    is_success = True
                    break
            
            if is_success:
                # Extract the trajectory (history without the final pending state)
                history = state.get('history', [])
                # Remove last entry if it's just a state without response
                if history and 'llm_response' not in history[-1]:
                    history = history[:-1]
                
                if history:  # Only add if we have actual interactions
                    # Calculate total reward
                    total_reward = sum(
                        turn.get('reward', 0) 
                        for turn in history 
                        if 'reward' in turn
                    )
                    
                    experience = Experience(
                        trajectory=history,
                        env_tag=state.get('tag', self.config.env_tag),
                        total_reward=total_reward,
                        num_turns=len(history),
                        timestamp=datetime.now().isoformat(),
                    )
                    successful.append(experience)
        
        return successful
    
    def add_experiences(self, experiences: List[Experience]) -> Tuple[int, bool]:
        """
        Add experiences to the buffer.
        
        Args:
            experiences: List of experiences to add
            
        Returns:
            Tuple of (number added, whether buffer is now full)
        """
        added = 0
        buffer_full = False
        
        for exp in experiences:
            if not self.buffer.is_full:
                buffer_full = self.buffer.add(exp)
                added += 1
                if buffer_full:
                    break
        
        return added, buffer_full
    
    def get_examples_for_prompt(self) -> List[Experience]:
        """
        Get examples to include in prompt based on config.
        
        Returns:
            List of experiences to use as in-context examples
        """
        if self.buffer.size == 0:
            return []
        
        max_examples = min(self.config.max_examples_in_prompt, self.buffer.size)
        
        if self.config.example_selection == "random":
            return self.buffer.get_random(max_examples)
        elif self.config.example_selection == "recent":
            return self.buffer.buffer[-max_examples:]
        elif self.config.example_selection == "best":
            # Sort by total reward and take the best
            sorted_exps = sorted(
                self.buffer.buffer, 
                key=lambda x: x.total_reward, 
                reverse=True
            )
            return sorted_exps[:max_examples]
        else:
            return self.buffer.get_random(max_examples)
    
    def format_experience_as_example(self, experience: Experience) -> str:
        """
        Format an experience as an in-context example.
        
        Args:
            experience: The experience to format
            
        Returns:
            Formatted string to include in prompt
        """
        lines = [f"=== Successful Example (Reward: {experience.total_reward:.2f}) ==="]
        
        for i, turn in enumerate(experience.trajectory, 1):
            if 'state' in turn:
                lines.append(f"[Turn {i}] State: {turn['state']}")
            if 'llm_response' in turn:
                lines.append(f"[Turn {i}] Response: {turn['llm_response']}")
            if 'reward' in turn:
                lines.append(f"[Turn {i}] Reward: {turn['reward']}")
        
        lines.append("=== End of Example ===\n")
        return "\n".join(lines)
    
    def get_examples_text(self) -> str:
        """
        Get formatted examples text to prepend to prompts.
        
        Returns:
            Formatted string with all examples
        """
        examples = self.get_examples_for_prompt()
        if not examples:
            return ""
        
        parts = [
            "\n--- Previous Successful Examples for Reference ---\n"
        ]
        for exp in examples:
            parts.append(self.format_experience_as_example(exp))
        parts.append("--- Now it's your turn. Learn from the examples above! ---\n\n")
        
        return "\n".join(parts)
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get state dict for saving."""
        return {
            'buffer_size': self.buffer.size,
            'buffer_max_size': self.buffer.max_size,
            'total_collected': self.buffer.total_collected,
            'last_val_count': self._last_val_count,
            'experiences': [exp.to_dict() for exp in self.buffer.buffer],
        }
    
    def load_state_dict(self, state: Dict[str, Any]):
        """Load state from dict."""
        self._last_val_count = state.get('last_val_count', 0)
        experiences_data = state.get('experiences', [])
        self.buffer.buffer = [Experience.from_dict(exp) for exp in experiences_data]
        self.buffer._total_collected = state.get('total_collected', len(self.buffer.buffer))
    
    def save(self, path: str):
        """Save the method state."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.get_state_dict(), f)
    
    def load(self, path: str):
        """Load the method state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.load_state_dict(state)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            'buffer_size': self.buffer.size,
            'buffer_max_size': self.buffer.max_size,
            'buffer_fill_ratio': self.buffer.size / self.buffer.max_size,
            'total_collected': self.buffer.total_collected,
            'is_full': self.buffer.is_full,
        }
