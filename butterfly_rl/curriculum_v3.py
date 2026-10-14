"""Versioned PPO curriculum. All obstacle geometry is sampled from the simulator RNG.

This is a simplified Flappy-like game, NOT physical butterfly flight.
The oracle reference controller is never used for training.
"""
import numpy as np
import gymnasium as gym
from .flappy_env import FlappyButterflyEnv

STAGES=("wide","normal","dense","windy","storm")
SPECS={
 "wide":dict(gap=(1.00,1.20),spacing=(3.0,3.5),wind=0.0,rain=0.0),
 "normal":dict(gap=(0.72,0.92),spacing=(2.6,3.2),wind=0.5,rain=0.0),
 "dense":dict(gap=(0.60,0.80),spacing=(2.0,2.6),wind=0.7,rain=0.0),
 "windy":dict(gap=(0.60,0.80),spacing=(2.0,2.6),wind=1.3,rain=0.2),
 "storm":dict(gap=(0.53,0.72),spacing=(1.8,2.4),wind=1.7,rain=0.4),
}
# Strictly held-out challenge distributions: never sampled during training.
CHALLENGES={
 "narrow_ood":dict(gap=(0.32,0.46),spacing=(2.6,3.2),wind=0.7,rain=0.0),
 "dense_ood":dict(gap=(0.60,0.80),spacing=(1.25,1.65),wind=0.7,rain=0.0),
 "wind_ood":dict(gap=(0.60,0.80),spacing=(2.0,2.6),wind=2.6,rain=0.4),
 "combined_ood":dict(gap=(0.32,0.46),spacing=(1.25,1.65),wind=2.6,rain=0.4),
}

class CurriculumFlappyEnv(FlappyButterflyEnv):
    def __init__(self,scenario="wide",max_steps=1200):
        if scenario not in SPECS and scenario not in CHALLENGES:
            raise ValueError(scenario)
        self.scenario=scenario
        self.spec=(SPECS|CHALLENGES)[scenario]
        super().__init__(level="obstacles",max_steps=max_steps)

    def _new_gate(self,x):
        r=self.np_random
        # Use the same distribution for initial and replacement gates.
        if self.gates and self.steps>0:
            x=self.gates[-1][0]+float(r.uniform(*self.spec["spacing"]))
        center=float(r.uniform(1.05,2.95))
        half=float(r.uniform(*self.spec["gap"]))
        self.gates.append((float(x),center,half))

    def reset(self,*,seed=None,options=None):
        obs,info=super().reset(seed=seed,options=options)
        self.wind_scale=self.spec["wind"]*float(self.np_random.uniform(0.75,1.25))
        self.rain_drag=self.spec["rain"]*float(self.np_random.uniform(0.75,1.25))
        # Base reset uses 2.8 spacing; rebuild all initial gates consistently.
        self.gates=[]
        self._credited_gates=set()
        x=3.0
        for _ in range(4):
            self._new_gate(x)
            x+=float(self.np_random.uniform(*self.spec["spacing"]))
        return self._obs(),self._info(False)

class ReplayCurriculumEnv(gym.Env):
    """50% latest stage, 50% uniformly replayed earlier stages."""
    metadata={"render_modes":[]}
    def __init__(self,stage="wide",seed=1729):
        super().__init__()
        if stage not in STAGES:raise ValueError(stage)
        self.stage=stage
        self.rng=np.random.default_rng(seed)
        self.current=None
        prototype=CurriculumFlappyEnv()
        self.action_space=prototype.action_space
        self.observation_space=prototype.observation_space
        prototype.close()

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:self.rng=np.random.default_rng(seed)
        index=STAGES.index(self.stage)
        if index==0 or self.rng.random()<0.5:
            scenario=self.stage
        else:
            scenario=STAGES[int(self.rng.integers(index))]
        if self.current is not None:self.current.close()
        self.current=CurriculumFlappyEnv(scenario)
        s=int(self.rng.integers(0,2**31-1))
        return self.current.reset(seed=s)

    def step(self,action):
        if self.current is None:raise RuntimeError("reset required")
        return self.current.step(action)

    def close(self):
        if self.current is not None:self.current.close()
        self.current=None
        super().close()
