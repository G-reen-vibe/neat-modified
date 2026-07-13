"""
RL environment wrappers for NEAT evaluation.

Each wrapper exposes:
    - n_inputs, n_outputs (for Config)
    - evaluate(genome, max_steps, seed) -> (total_reward, n_steps, episode_info)
    - render(genome, max_steps, seed) -> list of frames (for visualization)

We support:
    - CartPole-v1   (discrete 2, obs 4)
    - LunarLander-v3 (discrete 4, obs 8)
    - Acrobot-v1    (discrete 3, obs 6)
    - MountainCar-v0 (discrete 3, obs 2)

For discrete-action envs, the genome's output is argmax'd to pick the action.
We could also support continuous via a Gaussian policy, but for simplicity we
focus on discrete here.
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import gymnasium as gym

from . import Genome, Config


class DiscreteEnvWrapper:
    """Wraps a Gymnasium discrete-action env for NEAT evaluation."""

    def __init__(self, env_name: str, seed: int = 0,
                 max_steps: int = 1000,
                 n_eval_episodes: int = 1,
                 render_mode: Optional[str] = None):
        self.env_name = env_name
        self.seed = seed
        self.max_steps = max_steps
        self.n_eval_episodes = n_eval_episodes
        self.render_mode = render_mode

        # Probe the env to get shapes
        env = gym.make(env_name, render_mode=render_mode)
        self.n_inputs = int(np.prod(env.observation_space.shape))
        self.n_outputs = env.action_space.n
        env.close()
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def _make_env(self):
        return gym.make(self.env_name, render_mode=self.render_mode)

    def _action_from_output(self, output: np.ndarray) -> int:
        return int(np.argmax(output))

    # ------------------------------------------------------------------
    def evaluate(self, g: Genome, episode_seed: Optional[int] = None) -> Tuple[float, int, Dict]:
        """Run `n_eval_episodes` episodes and return mean reward."""
        env = self._make_env()
        total_rewards = []
        total_steps = 0
        info = {}
        for ep in range(self.n_eval_episodes):
            s = episode_seed if episode_seed is not None else int(self._rng.integers(0, 2**31 - 1))
            obs, _ = env.reset(seed=s)
            ep_reward = 0.0
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated) and steps < self.max_steps:
                # Normalize obs (clip to reasonable range for stability)
                obs_clipped = np.clip(obs, -10.0, 10.0)
                out = g.forward(obs_clipped)
                action = self._action_from_output(out)
                obs, r, terminated, truncated, _ = env.step(action)
                ep_reward += r
                steps += 1
            total_rewards.append(ep_reward)
            total_steps += steps
        env.close()
        mean_reward = float(np.mean(total_rewards))
        return mean_reward, total_steps, {"rewards": total_rewards}

    # ------------------------------------------------------------------
    def rollout(self, g: Genome, episode_seed: Optional[int] = None,
                render: bool = False) -> Dict[str, Any]:
        """Run a single episode and return trajectory data + frames."""
        env = self._make_env()
        s = episode_seed if episode_seed is not None else int(self._rng.integers(0, 2**31 - 1))
        obs, _ = env.reset(seed=s)
        obs_history = [obs.tolist()]
        action_history = []
        reward_history = []
        frames = []
        terminated = truncated = False
        steps = 0
        total_reward = 0.0
        while not (terminated or truncated) and steps < self.max_steps:
            if render:
                frames.append(env.render())
            obs_clipped = np.clip(obs, -10.0, 10.0)
            out = g.forward(obs_clipped)
            action = self._action_from_output(out)
            obs, r, terminated, truncated, _ = env.step(action)
            obs_history.append(obs.tolist())
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
    env = make_env("CartPole-v1", max_steps=200, n_eval_episodes=1, seed=42)
    print(f"CartPole-v1: n_inputs={env.n_inputs}, n_outputs={env.n_outputs}")
    # Build a random genome and evaluate
    from neat import Config, GlobalIndex, Genome
    import numpy as np
    cfg = Config(n_inputs=env.n_inputs, n_outputs=env.n_outputs, bias_enabled=True)
    cfg.output_activation = "tanh"
    idx = GlobalIndex(env.n_inputs, env.n_outputs, bias_enabled=True)
    g = Genome(cfg, idx)
    # Connect each input to each output
    for i in range(env.n_inputs):
        for j in range(env.n_outputs):
            g.add_conn(i, env.n_inputs + j, 0.1)
    g.add_conn(env.n_inputs + env.n_outputs, env.n_inputs, 0.1)  # bias -> out 0
    g.add_conn(env.n_inputs + env.n_outputs, env.n_inputs + 1, 0.1)  # bias -> out 1
    reward, steps, info = env.evaluate(g, episode_seed=0)
    print(f"Random genome: reward={reward:.2f}, steps={steps}")
