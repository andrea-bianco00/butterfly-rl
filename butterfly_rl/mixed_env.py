"""Seeded curriculum environment: sample past and current conditions each episode.

Gymnasium wrapper creates a fresh FlappyButterflyEnv at every reset. The PPO
policy always sees the same eight-dimensional observation and action spaces.
"""
import numpy as np
import gymnasium as gym
from .flappy_env import FlappyButterflyEnv, LEVELS

class MixedFlappyEnv(gym.Env):
    metadata={"render_modes":[]}

    def __init__(self, stage="calm", seed=1729):
        super().__init__()
        if stage not in LEVELS:
            raise ValueError(stage)
        self.stage=stage
        self.rng=np.random.default_rng(seed)
        self.current=None
        self.action_space=FlappyButterflyEnv().action_space
        self.observation_space=FlappyButterflyEnv().observation_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng=np.random.default_rng(seed)
        stage_idx=LEVELS.index(self.stage)
        # 50% current level; 50% spread over previous levels.
        if stage_idx==0:
            level="calm"
        elif self.rng.random()<0.5:
            level=self.stage
        else:
            level=LEVELS[int(self.rng.integers(0,stage_idx))]
        if self.current is not None:
            self.current.close()
        self.current=FlappyButterflyEnv(level=level)
        episode_seed=int(self.rng.integers(0,2**31-1))
        return self.current.reset(seed=episode_seed)

    def step(self, action):
        if self.current is None:
            raise RuntimeError("Call reset before step")
        return self.current.step(action)

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current=None
        super().close()
