"""V5: frozen swept-collision task with correctly sampled gate spacing.

This intentionally defines a NEW environment version; old models are not
silently transferred. This is a stylized 2D Flappy-like control problem.
"""
import numpy as np
import gymnasium as gym
from .curriculum_v3 import STAGES,SPECS,CHALLENGES
from .curriculum_v4 import CurriculumFlappyEnvV4

class CurriculumFlappyEnvV5(CurriculumFlappyEnvV4):
    def _new_gate(self,x):
        r=self.np_random
        center=float(r.uniform(1.05,2.95))
        half=float(r.uniform(*self.spec["gap"]))
        self.gates.append((float(x),center,half))

class ReplayCurriculumEnvV5(gym.Env):
    metadata={"render_modes":[]}
    def __init__(self,stage="wide",seed=1729):
        super().__init__()
        if stage not in STAGES:raise ValueError(stage)
        self.stage=stage
        self.rng=np.random.default_rng(seed)
        self.current=None
        proto=CurriculumFlappyEnvV5()
        self.action_space=proto.action_space
        self.observation_space=proto.observation_space
        proto.close()
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:self.rng=np.random.default_rng(seed)
        i=STAGES.index(self.stage)
        scenario=self.stage if i==0 or self.rng.random()<0.5 else STAGES[int(self.rng.integers(i))]
        if self.current is not None:self.current.close()
        self.current=CurriculumFlappyEnvV5(scenario)
        return self.current.reset(seed=int(self.rng.integers(0,2**31-1)))
    def step(self,action):
        if self.current is None:raise RuntimeError("reset first")
        return self.current.step(action)
    def close(self):
        if self.current is not None:self.current.close()
        self.current=None
        super().close()
