"""Stylized 2D flight control benchmark; NOT validated butterfly aerodynamics.

Weather and obstacle distributions are randomized by Gymnasium's seeded RNG.
A terminal collision always overrides reaching the target.
"""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

DT = 0.04
LEVELS = ("calm", "wind", "rain", "obstacles", "storm")
CONFIG = {
    "calm": (0.0, 0.0, False),
    "wind": (0.7, 0.0, False),
    "rain": (0.9, 0.35, False),
    "obstacles": (0.9, 0.0, True),
    "storm": (1.8, 0.65, True),
}

class RobustButterflyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, level="calm", max_steps=400):
        super().__init__()
        if level not in LEVELS:
            raise ValueError(f"Unknown level: {level}")
        self.level = level
        self.max_steps = int(max_steps)
        self.action_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.x = float(rng.uniform(-0.3, 0.3))
        self.z = float(rng.uniform(0.85, 1.65))
        self.vx = float(rng.uniform(-0.3, 0.3))
        self.vz = float(rng.uniform(-0.3, 0.3))
        self.theta = float(rng.uniform(-0.3, 0.3))
        self.omega = float(rng.uniform(-0.3, 0.3))
        self.wings = np.zeros(2, dtype=np.float64)
        self.wind = np.zeros(2, dtype=np.float64)
        self.target = np.array([5.0, 1.8], dtype=float)
        self.steps = 0
        self.collision = False
        base_wind, base_rain, use_obstacles = CONFIG[self.level]
        self.wind_intensity = float(base_wind * rng.uniform(0.65, 1.45))
        self.rain_drag = float(base_rain * rng.uniform(0.7, 1.3))
        self.obstacles = []
        if use_obstacles:
            for px in sorted(rng.uniform(1.0, 4.5, size=int(rng.integers(2, 5)))):
                self.obstacles.append((float(px), float(rng.uniform(0.65, 3.0)),
                                       float(rng.uniform(0.22, 0.52))))
        self.mass_factor = float(rng.uniform(0.85, 1.15))
        self.lift_factor = float(rng.uniform(0.85, 1.15))
        return self._obs(), self._info(False, False)

    def _nearest(self):
        if not self.obstacles:
            return (3.0, 3.0, 0.0)
        return min(((ox-self.x, oz-self.z, r) for ox, oz, r in self.obstacles),
                   key=lambda o: o[0]**2 + o[1]**2)

    def _obs(self):
        dx, dz, r = self._nearest()
        values = [(self.target[0]-self.x)/6, (self.target[1]-self.z)/3,
                  self.vx/5, self.vz/5, math.sin(self.theta), math.cos(self.theta),
                  self.omega/8, self.wings[0], self.wings[1],
                  self.wind[0]/5, self.wind[1]/5, dx/5, dz/3, r]
        return np.clip(np.asarray(values, dtype=np.float32), -1, 1)

    def _info(self, crashed, reached):
        return {"level": self.level, "crashed": bool(crashed),
                "reached_target": bool(reached), "x": float(self.x),
                "z": float(self.z), "theta": float(self.theta),
                "wind_x": float(self.wind[0]), "wind_z": float(self.wind[1]),
                "wind_intensity": self.wind_intensity,
                "rain_drag": self.rain_drag, "obstacles": self.obstacles}

    def step(self, action):
        a = np.asarray(action, dtype=float)
        if a.shape != (2,) or not np.isfinite(a).all():
            raise ValueError("Expected two finite wing commands")
        a = np.clip(a, 0, 1)
        old_distance = float(np.linalg.norm(self.target - [self.x, self.z]))
        self.steps += 1
        self.wings += 0.2 * (a - self.wings)
        if self.wind_intensity:
            self.wind = 0.95*self.wind + self.np_random.normal(
                0, 0.15*self.wind_intensity, 2)
            self.wind = np.clip(self.wind, -3.5, 3.5)
        else:
            self.wind[:] = 0
        lift = 18.0*self.lift_factor*float(self.wings.sum())/self.mass_factor
        ax = lift*math.sin(self.theta) - (0.35+self.rain_drag)*(self.vx-self.wind[0])
        az = lift*math.cos(self.theta) - 9.81 - (0.4+self.rain_drag)*(self.vz-self.wind[1])
        angular_acc = 16*(self.wings[1]-self.wings[0]) - 2*self.omega + 0.3*self.wind[0]
        self.vx = float(np.clip(self.vx+DT*ax, -9, 9))
        self.vz = float(np.clip(self.vz+DT*az, -9, 9))
        self.omega = float(np.clip(self.omega+DT*angular_acc, -12, 12))
        self.theta = float(np.clip(self.theta+DT*self.omega, -math.pi, math.pi))
        self.x += DT*self.vx
        self.z += DT*self.vz
        self.collision = any((self.x-ox)**2+(self.z-oz)**2 <= (r+0.12)**2
                             for ox, oz, r in self.obstacles)
        crashed = bool(self.z <= 0 or self.z >= 5 or abs(self.theta) > 2
                       or self.collision or self.x < -2)
        reached = bool(not crashed and np.linalg.norm(self.target-[self.x,self.z]) < 0.35)
        terminated = crashed or reached
        truncated = bool(self.steps >= self.max_steps and not terminated)
        progress = old_distance - float(np.linalg.norm(self.target-[self.x,self.z]))
        # Dense progress reward + mild survival incentive, explicit safety and control costs.
        reward = (4*progress + 0.025 - 0.02*abs(self.theta)
                  - 0.005*float(np.square(a).sum()))
        if crashed:
            reward -= 15
        elif reached:
            reward += 30
        return self._obs(), float(reward), terminated, truncated, self._info(crashed, reached)
