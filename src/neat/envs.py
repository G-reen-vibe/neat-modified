"""
RL environment wrappers for NEAT evaluation.

Each wrapper exposes:
    - n_inputs, n_outputs (for Config)
    - evaluate(genome, max_steps, seed) -> (total_reward, n_steps, episode_info)
    - rollout(genome, ...) -> dict with frames, obs, actions, rewards

Supported envs:
    - CartPole-v1     (4 obs, 2 actions)
    - Acrobot-v1      (6 obs, 3 actions)
    - MountainCar-v0  (2 obs, 3 actions)
    - LunarLander-v3  (8 obs, 4 actions)
    - Blackjack-v1    (3 obs Tuple, 2 actions, stochastic)

Features:
    - Observation normalization (per-env mean/std scaling)
    - Optional reward shaping (helps MountainCar)
    - Multi-episode evaluation for stochastic envs (Blackjack)
    - Tuple observation flattening (Blackjack)
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import gymnasium as gym

from . import Genome, Config


# Per-env observation normalization ranges (clip values)
ENV_OBS_CLIP = {
    "CartPole-v1":      (-10.0, 10.0),
    "Acrobot-v1":       (-np.pi*2, np.pi*2),
    "MountainCar-v0":   (-1.2, 0.6),
    "LunarLander-v3":   (-10.0, 10.0),
    "Blackjack-v1":     (0.0, 32.0),
}

# Per-env observation scaling (divide by these to normalize to ~[-1, 1])
ENV_OBS_SCALE = {
    "CartPole-v1":      np.array([2.4, 5.0, 0.25, 5.0]),
    "Acrobot-v1":       np.array([1.0, 1.0, 1.0, 1.0, 12.0, 28.0]),
    "MountainCar-v0":   np.array([1.8, 0.14]),
    "LunarLander-v3":   np.array([2.5, 2.5, 10.0, 10.0, 6.3, 10.0, 1.0, 1.0]),
    "Blackjack-v1":     np.array([31.0, 10.0, 1.0]),
}


def _flatten_obs(obs, env_name: str) -> np.ndarray:
    """Flatten observation to a 1-D float vector."""
    if isinstance(obs, tuple):
        # Blackjack: (player_sum, dealer_card, usable_ace)
        arr = np.array(obs, dtype=np.float64)
    else:
        arr = np.asarray(obs, dtype=np.float64).flatten()
    # Apply per-env scaling
    if env_name in ENV_OBS_SCALE:
        scale = ENV_OBS_SCALE[env_name]
        if len(arr) == len(scale):
            arr = arr / scale
    return arr


class DiscreteEnvWrapper:
    """Wraps a Gymnasium discrete-action env for NEAT evaluation."""

    def __init__(self, env_name: str, seed: int = 0,
                 max_steps: int = 1000,
                 n_eval_episodes: int = 1,
                 render_mode: Optional[str] = None,
                 reward_shaping: Optional[str] = None):
        self.env_name = env_name
        self.seed = seed
        self.max_steps = max_steps
        self.n_eval_episodes = n_eval_episodes
        self.render_mode = render_mode
        self.reward_shaping = reward_shaping

        # Probe the env to get shapes
        env = gym.make(env_name, render_mode=render_mode)
        # Handle Tuple observation space (Blackjack)
        if isinstance(env.observation_space, gym.spaces.Tuple):
            self.n_inputs = sum(int(s.n) if hasattr(s, 'n') else int(np.prod(s.shape))
                                for s in env.observation_space)
            # For Blackjack: encode (player, dealer, ace) as 3 separate inputs
            self.n_inputs = len(env.observation_space)
        else:
            self.n_inputs = int(np.prod(env.observation_space.shape))
        self.n_outputs = env.action_space.n
        env.close()
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def _make_env(self):
        return gym.make(self.env_name, render_mode=self.render_mode)

    def _action_from_output(self, output: np.ndarray) -> int:
        return int(np.argmax(output))

    def _shape_reward(self, obs, raw_reward: float, env_reward: float,
                      steps: int) -> float:
        """Apply reward shaping (per-env).  Returns the shaped reward delta
        to add to the cumulative reward."""
        if self.reward_shaping is None:
            return env_reward
        if self.reward_shaping == "mountaincar_position":
            # Reward proportional to position (higher = closer to flag)
            # obs = [position, velocity], goal is position >= 0.5
            position = obs[0] if not isinstance(obs, tuple) else 0
            # Bonus for being close to the goal
            shaped = env_reward + (position + 0.3) * 0.01
            return shaped
        if self.reward_shaping == "mountaincar_energy":
            # Reward for kinetic + potential energy (encourage exploration)
            if isinstance(obs, tuple):
                return env_reward
            position, velocity = obs[0], obs[1]
            # Potential: height = sin(3*position) * 0.45 + 0.55 (track shape)
            # Just use position+0.5 as a proxy
            potential = position + 0.5
            kinetic = velocity ** 2
            shaped = env_reward + potential * 0.01 + kinetic * 0.1
            return shaped
        if self.reward_shaping == "mountaincar_aggressive":
            # Strong shaping: big bonus for position progress + velocity magnitude
            # This is what actually works for NEAT on MountainCar
            if isinstance(obs, tuple):
                return env_reward
            position, velocity = obs[0], obs[1]
            # Bonus for position (goal at 0.5); start at -0.5
            position_bonus = max(0, position + 0.5) * 1.0  # +0 to +1.0 per step
            # Bonus for high |velocity| (encourages swinging)
            velocity_bonus = abs(velocity) * 5.0
            # Big bonus for reaching the goal
            goal_bonus = 100.0 if position >= 0.5 else 0.0
            shaped = env_reward + position_bonus + velocity_bonus + goal_bonus
            return shaped
        if self.reward_shaping == "lunarlander_closer":
            # Reward for being close to landing pad (x~0, y~0)
            if isinstance(obs, tuple):
                return env_reward
            x, y = obs[0], obs[1]
            shaped = env_reward - (abs(x) * 0.1) - (abs(y) * 0.05)
            return shaped
        if self.reward_shaping == "lunarlander_aggressive":
            # Strong shaping: reward for being upright, low, near center
            if isinstance(obs, tuple):
                return env_reward
            x, y, vx, vy, angle, angular_vel, leg1, leg2 = obs
            # Reward legs touching
            leg_bonus = (leg1 + leg2) * 5.0
            # Penalty for being far from center
            center_penalty = abs(x) * 0.5
            # Penalty for tilting
            tilt_penalty = abs(angle) * 2.0
            # Penalty for high velocity (encourage gentle landing)
            vel_penalty = (abs(vx) + abs(vy)) * 0.2
            # Reward for being low (closer to ground)
            low_bonus = max(0, 1.0 - y) * 0.5 if y < 1.0 else 0
            shaped = env_reward + leg_bonus + low_bonus - center_penalty - tilt_penalty - vel_penalty
            return shaped
        return env_reward

    # ------------------------------------------------------------------
    def evaluate(self, g: Genome, episode_seed: Optional[int] = None) -> Tuple[float, int, Dict]:
        """Run `n_eval_episodes` episodes and return mean reward."""
        env = self._make_env()
        total_rewards = []
        total_steps = 0
        info: Dict[str, Any] = {"rewards": []}
        for ep in range(self.n_eval_episodes):
            s = episode_seed if episode_seed is not None else int(self._rng.integers(0, 2**31 - 1))
            obs, _ = env.reset(seed=s)
            ep_reward = 0.0
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated) and steps < self.max_steps:
                obs_flat = _flatten_obs(obs, self.env_name)
                # Clip to a reasonable range
                obs_clipped = np.clip(obs_flat, -10.0, 10.0)
                out = g.forward(obs_clipped)
                action = self._action_from_output(out)
                obs, r, terminated, truncated, _ = env.step(action)
                # Apply reward shaping
                r_shaped = self._shape_reward(obs, r, r, steps)
                ep_reward += r_shaped
                steps += 1
            total_rewards.append(ep_reward)
            total_steps += steps
        env.close()
        mean_reward = float(np.mean(total_rewards))
        info["rewards"] = total_rewards
        return mean_reward, total_steps, info

    # ------------------------------------------------------------------
    def evaluate_raw(self, g: Genome, episode_seed: Optional[int] = None) -> Tuple[float, int, Dict]:
        """Like evaluate() but uses RAW (unshaped) rewards - for final eval."""
        original_shaping = self.reward_shaping
        self.reward_shaping = None
        try:
            return self.evaluate(g, episode_seed)
        finally:
            self.reward_shaping = original_shaping

    # ------------------------------------------------------------------
    def rollout(self, g: Genome, episode_seed: Optional[int] = None,
                render: bool = False) -> Dict[str, Any]:
        """Run a single episode and return trajectory data + frames."""
        env = self._make_env()
        s = episode_seed if episode_seed is not None else int(self._rng.integers(0, 2**31 - 1))
        obs, _ = env.reset(seed=s)
        obs_history = [_flatten_obs(obs, self.env_name).tolist()]
        action_history = []
        reward_history = []
        frames = []
        terminated = truncated = False
        steps = 0
        total_reward = 0.0
        while not (terminated or truncated) and steps < self.max_steps:
            if render:
                frames.append(env.render())
            obs_flat = _flatten_obs(obs, self.env_name)
            obs_clipped = np.clip(obs_flat, -10.0, 10.0)
            out = g.forward(obs_clipped)
            action = self._action_from_output(out)
            obs, r, terminated, truncated, _ = env.step(action)
            obs_history.append(_flatten_obs(obs, self.env_name).tolist())
            action_history.append(action)
            reward_history.append(r)
            total_reward += r
            steps += 1
        if render:
            frames.append(env.render())
        env.close()
        return {
            "obs": obs_history,
            "actions": action_history,
            "rewards": reward_history,
            "total_reward": total_reward,
            "steps": steps,
            "frames": frames,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_env(env_name: str, **kwargs) -> DiscreteEnvWrapper:
    """Factory for env wrappers.  Currently all supported envs are discrete."""
    return DiscreteEnvWrapper(env_name, **kwargs)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for env_name in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0",
                     "LunarLander-v3", "Blackjack-v1"]:
        env = make_env(env_name, max_steps=200, n_eval_episodes=1, seed=42)
        print(f"{env_name}: n_inputs={env.n_inputs}, n_outputs={env.n_outputs}")
