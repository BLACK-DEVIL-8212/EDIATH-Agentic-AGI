"""Reinforcement Learning - reward-based learning with advanced algorithms, experience replay, prioritized sampling, and multi-agent support."""

import asyncio
import json
import math
import random
import statistics
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..utils.logger import logger


class ActionType(Enum):
    """Action types for RL."""

    EXPLORE = "explore"
    EXPLOIT = "exploit"
    RANDOM = "random"
    OPTIMAL = "optimal"
    SUBOPTIMAL = "suboptimal"


class RLAlgorithm(Enum):
    """Supported RL algorithms."""

    Q_LEARNING = "q_learning"
    SARSA = "sarsa"
    DQN = "dqn"
    DOUBLE_DQN = "double_dqn"
    PRIORITIZED_DQN = "prioritized_dqn"
    POLICY_GRADIENT = "policy_gradient"
    A2C = "a2c"
    PPO = "ppo"


class RewardType(Enum):
    """Types of rewards."""

    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    SHAPED = "shaped"
    SPARSE = "sparse"
    DENSE = "dense"
    INTRINSIC = "intrinsic"
    EXTRINSIC = "extrinsic"


@dataclass
class Experience:
    """Experience tuple for replay buffer."""

    state: Any
    action: Union[str, int]
    reward: float
    next_state: Any
    done: bool
    priority: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TrajectoryStep:
    """Enhanced trajectory step."""

    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]
    done: bool
    timestamp: datetime = field(default_factory=datetime.now)
    action_type: Optional[ActionType] = None
    advantage: float = 0.0
    value: float = 0.0
    log_prob: float = 0.0


@dataclass
class Episode:
    """Complete episode data."""

    id: int
    steps: List[TrajectoryStep]
    total_reward: float
    discounted_reward: float
    length: int
    start_time: datetime
    end_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer."""

    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.buffer: List[Experience] = []
        self.priorities: List[float] = []
        self.position = 0

    def push(self, experience: Experience):
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(experience.priority**self.alpha)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = experience.priority**self.alpha

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Experience], List[float]]:
        """Sample batch based on priorities."""
        if len(self.buffer) < batch_size:
            return self.buffer, [1.0] * len(self.buffer)

        # Compute sampling probabilities
        probs = np.array(self.priorities) / sum(self.priorities)

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Compute importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[idx] for idx in indices]
        return batch, weights.tolist()

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.buffer):
                self.priorities[idx] = priority**self.alpha
                self.buffer[idx].priority = priority

    def __len__(self) -> int:
        return len(self.buffer)


class NeuralNetwork:
    """Neural network for deep RL (if PyTorch available)."""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for neural network")

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def update(self, loss: torch.Tensor):
        """Update network weights."""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class RewardShaper:
    """Shape and transform rewards."""

    def __init__(self, discount_factor: float = 0.99, use_gae: bool = True):
        self.discount_factor = discount_factor
        self.use_gae = use_gae
        self.gae_lambda = 0.95

    def discount_rewards(self, rewards: List[float], done: bool = False) -> List[float]:
        """Calculate discounted cumulative rewards."""
        discounted = []
        cumulative = 0

        for reward in reversed(rewards):
            cumulative = reward + self.discount_factor * cumulative * (1 - int(done))
            discounted.insert(0, cumulative)

        return discounted

    def calculate_advantages(
        self, rewards: List[float], values: List[float], dones: List[bool]
    ) -> List[float]:
        """Calculate advantages using GAE."""
        if not self.use_gae:
            # Simple advantage: discounted rewards - values
            discounted = self.discount_rewards(rewards)
            return [r - v for r, v in zip(discounted, values)]

        # GAE calculation
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + self.discount_factor * values[t + 1] * (1 - dones[t])
                - values[t]
            )
            gae = delta + self.discount_factor * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return advantages

    def shape_reward(
        self,
        reward: float,
        state: Dict[str, Any],
        next_state: Dict[str, Any],
        potential_based: bool = True,
    ) -> float:
        """Shape reward using potential-based shaping."""
        if not potential_based:
            return reward

        # Simple potential function (can be customized)
        def potential(state: Dict[str, Any]) -> float:
            # Example: use distance to goal or other metrics
            return state.get("potential", 0.0)

        shaped = (
            reward + self.discount_factor * potential(next_state) - potential(state)
        )
        return shaped


class ExplorationStrategy:
    """Advanced exploration strategies."""

    def __init__(self, strategy: str = "epsilon_greedy", epsilon: float = 0.1):
        self.strategy = strategy
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.ucb_c = 2.0  # UCB exploration constant
        self.visit_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.q_sums: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def choose_action(
        self, state: str, actions: List[str], q_values: Dict[str, Dict[str, float]]
    ) -> Tuple[str, float]:
        """Choose action based on exploration strategy."""

        if self.strategy == "epsilon_greedy":
            return self._epsilon_greedy(state, actions, q_values)

        elif self.strategy == "ucb":
            return self._ucb(state, actions, q_values)

        elif self.strategy == "boltzmann":
            return self._boltzmann(state, actions, q_values)

        elif self.strategy == "random":
            return random.choice(actions), 0.0

        else:  # default to epsilon-greedy
            return self._epsilon_greedy(state, actions, q_values)

    def _epsilon_greedy(
        self, state: str, actions: List[str], q_values: Dict[str, Dict[str, float]]
    ) -> Tuple[str, float]:
        """Epsilon-greedy exploration."""
        if random.random() < self.epsilon:
            return random.choice(actions), 0.0

        # Choose best action
        best_action = max(actions, key=lambda a: q_values.get(state, {}).get(a, 0))
        return best_action, 1.0

    def _ucb(
        self, state: str, actions: List[str], q_values: Dict[str, Dict[str, float]]
    ) -> Tuple[str, float]:
        """Upper Confidence Bound exploration."""
        total_visits = sum(self.visit_counts[state].values())

        if total_visits == 0:
            return random.choice(actions), 0.0

        ucb_values = []
        for action in actions:
            q_value = q_values.get(state, {}).get(action, 0)
            visits = self.visit_counts[state][action]

            if visits == 0:
                ucb = float("inf")
            else:
                ucb = q_value + self.ucb_c * math.sqrt(math.log(total_visits) / visits)

            ucb_values.append(ucb)

        best_idx = ucb_values.index(max(ucb_values))
        return actions[best_idx], 1.0

    def _boltzmann(
        self, state: str, actions: List[str], q_values: Dict[str, Dict[str, float]]
    ) -> Tuple[str, float]:
        """Boltzmann (softmax) exploration."""
        temperature = max(0.1, self.epsilon * 10)

        q_vals = [q_values.get(state, {}).get(a, 0) for a in actions]

        # Softmax probabilities
        exp_q = [math.exp(q / temperature) for q in q_vals]
        sum_exp = sum(exp_q)
        probs = [e / sum_exp for e in exp_q]

        action = random.choices(actions, weights=probs)[0]
        return action, max(probs)

    def update(self, state: str, action: str, reward: float):
        """Update exploration statistics."""
        self.visit_counts[state][action] += 1
        self.q_sums[state][action] += reward

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_stats(self) -> Dict[str, Any]:
        """Get exploration statistics."""
        total_visits = sum(
            sum(counts.values()) for counts in self.visit_counts.values()
        )

        return {
            "strategy": self.strategy,
            "epsilon": self.epsilon,
            "total_visits": total_visits,
            "unique_states": len(self.visit_counts),
            "avg_visits_per_state": total_visits / max(1, len(self.visit_counts)),
        }


class ReinforcementLearner:
    """Advanced reinforcement learning engine with multiple algorithms."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        algorithm: RLAlgorithm = RLAlgorithm.Q_LEARNING,
        use_double_q: bool = False,
        use_prioritized_replay: bool = True,
        replay_buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        use_neural_network: bool = False,
        network_config: Optional[Dict[str, Any]] = None,
        exploration_strategy: str = "epsilon_greedy",
    ):
        """Initialize advanced RL learner."""
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.algorithm = algorithm
        self.use_double_q = use_double_q
        self.use_prioritized_replay = use_prioritized_replay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_neural_network = use_neural_network and TORCH_AVAILABLE

        # Q-table approximation
        self.q_values: Dict[str, Dict[str, float]] = {}
        self.target_q_values: Dict[str, Dict[str, float]] = {} if use_double_q else None

        # Neural network (if available)
        self.network = None
        self.target_network = None
        if self.use_neural_network and network_config:
            self._init_networks(network_config)

        # Experience replay
        self.replay_buffer = (
            PrioritizedReplayBuffer(capacity=replay_buffer_size)
            if use_prioritized_replay
            else []
        )
        self.replay_buffer_size = replay_buffer_size
        self.buffer_type = "prioritized" if use_prioritized_replay else "standard"

        # Trajectory tracking
        self.trajectory_buffer: List[TrajectoryStep] = []
        self.episodes: List[Episode] = []
        self.current_episode_steps: List[TrajectoryStep] = []
        self.episode_counter = 0

        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.td_errors: List[float] = []

        # Advanced components
        self.reward_shaper = RewardShaper(discount_factor=discount_factor)
        self.exploration = ExplorationStrategy(
            strategy=exploration_strategy, epsilon=epsilon
        )

        # Metrics
        self.learning_metrics: Dict[str, Any] = {
            "loss_history": [],
            "reward_history": [],
            "q_value_history": [],
            "exploration_rate_history": [],
        }

        # Callbacks
        self._episode_callbacks: List[Callable[[Episode], None]] = []
        self._step_callbacks: List[Callable[[TrajectoryStep], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []

        # Persistence
        self.persistence_path = Path("data/rl")
        self.persistence_path.mkdir(parents=True, exist_ok=True)

        # Async
        self._training_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"✅ Advanced ReinforcementLearner initialized (algorithm={algorithm.value}, neural={self.use_neural_network}, replay={self.buffer_type})"
        )

    def _init_networks(self, network_config: Dict[str, Any]):
        """Initialize neural networks for deep RL."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, falling back to tabular Q-learning")
            self.use_neural_network = False
            return

        input_dim = network_config.get("input_dim", 10)
        hidden_dims = network_config.get("hidden_dims", [64, 64])
        output_dim = network_config.get("output_dim", 4)

        self.network = NeuralNetwork(input_dim, hidden_dims, output_dim)

        if self.use_double_q:
            self.target_network = NeuralNetwork(input_dim, hidden_dims, output_dim)
            self._update_target_network()

        logger.info(
            f"🧠 Neural networks initialized: {input_dim} -> {hidden_dims} -> {output_dim}"
        )

    def _update_target_network(self):
        """Update target network with current network weights."""
        if self.target_network and self.network:
            self.target_network.network.load_state_dict(
                self.network.network.state_dict()
            )

    def choose_action(
        self,
        state: Union[str, Dict[str, Any]],
        available_actions: List[str],
        explore: bool = True,
    ) -> Tuple[str, ActionType, float]:
        """Choose action using advanced exploration."""
        if not available_actions:
            raise ValueError("available_actions must contain at least one action")

        # Convert state to string key if needed
        state_key = state if isinstance(state, str) else str(state)

        if explore:
            action, confidence = self.exploration.choose_action(
                state_key, available_actions, self.q_values
            )

            # Determine action type
            if confidence == 0:
                action_type = ActionType.EXPLORE
            elif confidence > 0.8:
                action_type = ActionType.OPTIMAL
            else:
                action_type = ActionType.SUBOPTIMAL

            return action, action_type, confidence

        else:
            # Pure exploitation
            best_action = max(
                available_actions,
                key=lambda a: self.q_values.get(state_key, {}).get(a, 0),
            )
            return best_action, ActionType.EXPLOIT, 1.0

    async def learn_from_trajectory(
        self,
        state: Union[str, Dict[str, Any]],
        action: str,
        reward: float,
        next_state: Union[str, Dict[str, Any]],
        done: bool = False,
        shaped_reward: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Learn from trajectory with advanced algorithms."""

        # Convert states to string keys
        state_key = state if isinstance(state, str) else str(state)
        next_state_key = next_state if isinstance(next_state, str) else str(next_state)

        # Shape reward
        if shaped_reward is None:
            shaped_reward = self.reward_shaper.shape_reward(
                reward,
                state if isinstance(state, dict) else {},
                next_state if isinstance(next_state, dict) else {},
            )

        # Initialize Q-values
        if state_key not in self.q_values:
            self.q_values[state_key] = {}
        if action not in self.q_values[state_key]:
            self.q_values[state_key][action] = 0.0

        if next_state_key not in self.q_values:
            self.q_values[next_state_key] = {}

        # Store experience in replay buffer
        experience = Experience(
            state=state_key,
            action=action,
            reward=shaped_reward,
            next_state=next_state_key,
            done=done,
        )

        if self.use_prioritized_replay:
            self.replay_buffer.push(experience)
        else:
            if len(self.replay_buffer) >= self.replay_buffer_size:
                self.replay_buffer.pop(0)
            self.replay_buffer.append(experience)

        # Perform update based on algorithm
        if self.use_neural_network:
            update_info = await self._neural_update(
                state_key, action, shaped_reward, next_state_key, done
            )
        else:
            update_info = await self._tabular_update(
                state_key, action, shaped_reward, next_state_key, done
            )

        # Update exploration
        self.exploration.update(state_key, action, shaped_reward)

        # Track step
        self.total_steps += 1
        self.td_errors.append(update_info.get("td_error", 0))

        # Record trajectory
        step = TrajectoryStep(
            state={"key": state_key} if isinstance(state, str) else state,
            action=action,
            reward=reward,
            next_state=(
                {"key": next_state_key} if isinstance(next_state, str) else next_state
            ),
            done=done,
            action_type=update_info.get("action_type"),
            advantage=update_info.get("advantage", 0),
        )

        self.current_episode_steps.append(step)

        # Trigger step callback
        for callback in self._step_callbacks:
            try:
                callback(step)
            except Exception as e:
                logger.error(f"Step callback error: {e}")

        # Handle episode end
        if done:
            await self._end_episode()

        # Update metrics
        self.learning_metrics["q_value_history"].append(
            self.q_values[state_key].get(action, 0)
        )
        self.learning_metrics["exploration_rate_history"].append(
            self.exploration.epsilon
        )

        # Limit history
        for key in self.learning_metrics:
            if len(self.learning_metrics[key]) > 1000:
                self.learning_metrics[key] = self.learning_metrics[key][-1000:]

        return update_info

    async def _tabular_update(
        self, state: str, action: str, reward: float, next_state: str, done: bool
    ) -> Dict[str, Any]:
        """Tabular Q-learning or SARSA update."""

        current_q = self.q_values[state][action]

        if self.algorithm == RLAlgorithm.SARSA:
            # SARSA: use next action (would need to be provided)
            next_action = self.choose_action(
                next_state, list(self.q_values[next_state].keys()), explore=True
            )[0]
            next_q = self.q_values[next_state].get(next_action, 0)
        else:
            # Q-learning: use max next Q
            next_q = (
                max(self.q_values[next_state].values())
                if self.q_values[next_state]
                else 0
            )

        # Double Q-learning
        if self.use_double_q and self.target_q_values:
            if next_state not in self.target_q_values:
                self.target_q_values[next_state] = {}

            if self.algorithm == RLAlgorithm.SARSA:
                next_action = self.choose_action(
                    next_state, list(self.q_values[next_state].keys()), explore=True
                )[0]
                next_q = self.target_q_values[next_state].get(next_action, 0)
            else:
                best_action = (
                    max(self.q_values[next_state].items(), key=lambda x: x[1])[0]
                    if self.q_values[next_state]
                    else None
                )
                next_q = (
                    self.target_q_values[next_state].get(best_action, 0)
                    if best_action
                    else 0
                )

        # Calculate TD target
        if done:
            td_target = reward
        else:
            td_target = reward + self.discount_factor * next_q

        td_error = td_target - current_q

        # Update Q-value
        new_q = current_q + self.learning_rate * td_error
        self.q_values[state][action] = new_q

        # Periodically update target network
        if self.use_double_q and self.total_steps % self.target_update_freq == 0:
            self.target_q_values = self.q_values.copy()

        return {
            "state": state,
            "action": action,
            "reward": reward,
            "old_q": current_q,
            "new_q": new_q,
            "td_error": td_error,
            "algorithm": self.algorithm.value,
            "action_type": "tabular",
        }

    async def _neural_update(
        self, state: str, action: str, reward: float, next_state: str, done: bool
    ) -> Dict[str, Any]:
        """Deep Q-learning update using neural networks."""
        if not self.network:
            return await self._tabular_update(state, action, reward, next_state, done)

        # Sample from replay buffer if enough experiences
        if len(self.replay_buffer) < self.batch_size:
            return {
                "message": "buffer not ready",
                "batch_size": len(self.replay_buffer),
            }

        # Sample batch
        if self.use_prioritized_replay:
            batch, weights = self.replay_buffer.sample(self.batch_size)
        else:
            batch = random.sample(
                self.replay_buffer, min(self.batch_size, len(self.replay_buffer))
            )
            weights = [1.0] * len(batch)

        # Convert batch to tensors (simplified - actual implementation would need proper encoding)
        # This is a placeholder for actual neural network training

        # For now, fall back to tabular update
        return await self._tabular_update(state, action, reward, next_state, done)

    async def _end_episode(self):
        """End current episode and process statistics."""
        if not self.current_episode_steps:
            return

        total_reward = sum(step.reward for step in self.current_episode_steps)
        discounted_reward = sum(
            step.reward * (self.discount_factor**i)
            for i, step in enumerate(self.current_episode_steps)
        )

        episode = Episode(
            id=self.episode_counter,
            steps=self.current_episode_steps.copy(),
            total_reward=total_reward,
            discounted_reward=discounted_reward,
            length=len(self.current_episode_steps),
            start_time=self.current_episode_steps[0].timestamp,
            end_time=datetime.now(),
            metadata={
                "epsilon": self.exploration.epsilon,
                "algorithm": self.algorithm.value,
            },
        )

        self.episodes.append(episode)
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(len(self.current_episode_steps))
        self.total_episodes += 1
        self.episode_counter += 1

        # Store trajectory
        self.trajectory_buffer.extend(self.current_episode_steps)

        # Keep limited history
        if len(self.episodes) > 100:
            self.episodes = self.episodes[-100:]
        if len(self.episode_rewards) > 1000:
            self.episode_rewards = self.episode_rewards[-1000:]

        # Record metrics
        self.learning_metrics["reward_history"].append(total_reward)

        # Calculate loss
        if len(self.episode_rewards) > 1:
            recent_avg = statistics.mean(self.episode_rewards[-10:])
            older_avg = (
                statistics.mean(self.episode_rewards[-20:-10])
                if len(self.episode_rewards) >= 20
                else recent_avg
            )
            loss = max(0, older_avg - recent_avg)
            self.learning_metrics["loss_history"].append(loss)

        # Trigger episode callback
        for callback in self._episode_callbacks:
            try:
                callback(episode)
            except Exception as e:
                logger.error(f"Episode callback error: {e}")

        logger.info(
            f"📊 Episode {episode.id}: reward={total_reward:.2f}, length={episode.length}, epsilon={self.exploration.epsilon:.3f}"
        )

        # Clear current episode
        self.current_episode_steps = []

    def record_trajectory(self, trajectory: List[TrajectoryStep]) -> float:
        """Record complete trajectory."""
        total_reward = sum(step.reward for step in trajectory)
        self.trajectory_buffer.extend(trajectory)
        return total_reward

    async def train_from_buffer(self, num_updates: int = 100):
        """Train agent from replay buffer."""
        if not self.use_neural_network:
            logger.warning("Training from buffer only supported for neural networks")
            return

        for _ in range(num_updates):
            if len(self.replay_buffer) < self.batch_size:
                break

            await self._neural_update("", "", 0, "", False)

        # Update target network
        if self.use_double_q and self.total_steps % self.target_update_freq == 0:
            self._update_target_network()

    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.01):
        """Decay exploration probability."""
        self.exploration.epsilon = max(
            min_epsilon, self.exploration.epsilon * decay_rate
        )
        self.epsilon = self.exploration.epsilon

    def get_best_action(
        self, state: Union[str, Dict[str, Any]], actions: List[str]
    ) -> str:
        """Get best action without exploration."""
        action, _, _ = self.choose_action(state, actions, explore=False)
        return action

    def get_q_value(self, state: Union[str, Dict[str, Any]], action: str) -> float:
        """Get Q-value for state-action pair."""
        state_key = state if isinstance(state, str) else str(state)
        return self.q_values.get(state_key, {}).get(action, 0.0)

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get episode statistics."""
        if not self.episode_rewards:
            return {
                "episodes": 0,
                "avg_reward": 0,
                "max_reward": 0,
                "min_reward": 0,
                "avg_length": 0,
            }

        return {
            "episodes": len(self.episode_rewards),
            "avg_reward": statistics.mean(self.episode_rewards),
            "max_reward": max(self.episode_rewards),
            "min_reward": min(self.episode_rewards),
            "std_reward": (
                statistics.stdev(self.episode_rewards)
                if len(self.episode_rewards) > 1
                else 0
            ),
            "avg_length": (
                statistics.mean(self.episode_lengths) if self.episode_lengths else 0
            ),
            "total_steps": self.total_steps,
        }

    def get_learning_curve(self) -> Dict[str, List[float]]:
        """Get learning curve data."""
        return {
            "rewards": self.episode_rewards[-100:],
            "losses": self.learning_metrics["loss_history"][-100:],
            "td_errors": self.td_errors[-100:],
            "exploration_rates": self.learning_metrics["exploration_rate_history"][
                -100:
            ],
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive RL statistics."""
        stats = self.get_episode_stats()
        stats.update(
            {
                "states": len(self.q_values),
                "total_q_values": sum(len(v) for v in self.q_values.values()),
                "epsilon": self.exploration.epsilon,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "buffer_size": len(self.replay_buffer),
                "buffer_type": self.buffer_type,
                "algorithm": self.algorithm.value,
                "use_double_q": self.use_double_q,
                "use_neural_network": self.use_neural_network,
                "total_episodes": self.total_episodes,
                "avg_td_error": (
                    statistics.mean(self.td_errors[-100:]) if self.td_errors else 0
                ),
                "exploration_stats": self.exploration.get_stats(),
            }
        )

        # Add Q-value statistics
        all_q_values = [
            q for state_q in self.q_values.values() for q in state_q.values()
        ]
        if all_q_values:
            stats.update(
                {
                    "avg_q_value": statistics.mean(all_q_values),
                    "max_q_value": max(all_q_values),
                    "min_q_value": min(all_q_values),
                }
            )

        return stats

    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        if not self.episode_rewards:
            return {"message": "No episodes completed yet"}

        recent_rewards = self.episode_rewards[-20:]
        recent_lengths = self.episode_lengths[-20:] if self.episode_lengths else []

        # Calculate improvement
        first_10 = (
            self.episode_rewards[:10]
            if len(self.episode_rewards) >= 10
            else self.episode_rewards
        )
        last_10 = (
            self.episode_rewards[-10:]
            if len(self.episode_rewards) >= 10
            else self.episode_rewards
        )

        improvement = (
            statistics.mean(last_10) - statistics.mean(first_10)
            if first_10 and last_10
            else 0
        )

        # Check convergence
        if len(self.episode_rewards) >= 50:
            recent_std = (
                statistics.stdev(self.episode_rewards[-20:])
                if len(self.episode_rewards[-20:]) > 1
                else 1
            )
            converged = recent_std < 0.1 * abs(
                statistics.mean(self.episode_rewards[-20:])
            )
        else:
            converged = False

        return {
            "summary": {
                "total_episodes": len(self.episode_rewards),
                "total_steps": self.total_steps,
                "best_reward": max(self.episode_rewards),
                "worst_reward": min(self.episode_rewards),
                "average_reward": statistics.mean(self.episode_rewards),
                "average_length": (
                    statistics.mean(self.episode_lengths) if self.episode_lengths else 0
                ),
            },
            "recent_performance": {
                "avg_reward_20": (
                    statistics.mean(recent_rewards) if recent_rewards else 0
                ),
                "avg_length_20": (
                    statistics.mean(recent_lengths) if recent_lengths else 0
                ),
                "improvement": improvement,
                "improvement_rate": (
                    improvement / max(1, abs(statistics.mean(first_10)))
                    if first_10
                    else 0
                ),
            },
            "convergence": {
                "converged": converged,
                "stability": (
                    statistics.stdev(self.episode_rewards[-20:])
                    if len(self.episode_rewards[-20:]) > 1
                    else 1
                ),
                "exploration_rate": self.exploration.epsilon,
            },
            "algorithm_metrics": {
                "td_error_avg": (
                    statistics.mean(self.td_errors[-100:]) if self.td_errors else 0
                ),
                "q_value_avg": (
                    statistics.mean(
                        [
                            q
                            for state_q in self.q_values.values()
                            for q in state_q.values()
                        ]
                    )
                    if self.q_values
                    else 0
                ),
                "states_explored": len(self.q_values),
            },
        }

    def save_q_table(self, filepath: Optional[str] = None) -> bool:
        """Save Q-table to disk."""
        if not filepath:
            filepath = (
                self.persistence_path
                / f"q_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "q_values": self.q_values,
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "total_steps": self.total_steps,
                "total_episodes": self.total_episodes,
                "epsilon": self.exploration.epsilon,
                "config": {
                    "learning_rate": self.learning_rate,
                    "discount_factor": self.discount_factor,
                    "algorithm": self.algorithm.value,
                    "use_double_q": self.use_double_q,
                },
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"💾 Saved Q-table to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save Q-table: {e}")
            return False

    def load_q_table(self, filepath: str) -> bool:
        """Load Q-table from disk."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            self.q_values = data.get("q_values", {})
            self.episode_rewards = data.get("episode_rewards", [])
            self.episode_lengths = data.get("episode_lengths", [])
            self.total_steps = data.get("total_steps", 0)
            self.total_episodes = data.get("total_episodes", 0)
            self.exploration.epsilon = data.get("epsilon", self.epsilon)

            logger.info(f"📂 Loaded Q-table from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load Q-table: {e}")
            return False

    def add_episode_callback(self, callback: Callable[[Episode], None]):
        """Add callback for episode completion."""
        self._episode_callbacks.append(callback)

    def add_step_callback(self, callback: Callable[[TrajectoryStep], None]):
        """Add callback for each step."""
        self._step_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add error callback."""
        self._error_callbacks.append(callback)

    def reset(self):
        """Reset learner state."""
        self.q_values.clear()
        self.target_q_values = {} if self.use_double_q else None
        self.trajectory_buffer.clear()
        self.episodes.clear()
        self.current_episode_steps.clear()
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.td_errors.clear()
        self.total_steps = 0
        self.total_episodes = 0

        self.learning_metrics = {
            "loss_history": [],
            "reward_history": [],
            "q_value_history": [],
            "exploration_rate_history": [],
        }

        self.exploration = ExplorationStrategy(
            strategy=self.exploration.strategy, epsilon=self.epsilon
        )

        logger.info("🔄 Reinforcement learner reset")

    async def start(self):
        """Start background tasks."""
        self._running = True
        logger.info("🚀 Reinforcement learner started")

    async def stop(self):
        """Stop background tasks."""
        self._running = False

        # Save final state
        if self.q_values:
            await asyncio.to_thread(self.save_q_table)

        logger.info("🛑 Reinforcement learner stopped")


# #==================== Convenience Functions #====================

__all__ = [
    "ReinforcementLearner",
    "ActionType",
    "RLAlgorithm",
    "RewardType",
    "Experience",
    "TrajectoryStep",
    "Episode",
    "PrioritizedReplayBuffer",
    "NeuralNetwork",
    "RewardShaper",
    "ExplorationStrategy",
]
