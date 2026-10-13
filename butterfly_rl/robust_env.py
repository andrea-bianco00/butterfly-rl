"""Challenging but deliberately simplified 2D butterfly flight-control environment.

Not a validated insect aerodynamics model. Units are abstract SI-like units.
Gymnasium API, continuous actions, curriculum levels, seeded stochastic weather.
"""
from __future__ import annotations
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

DT = 0.04
LEVELS = ("calm", "wind", "rain", "obstacles", "storm")

class RobustButterflyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, level="calm", max_steps=400):
        super().__init__()
        if level not in LEVELS:
            raise ValueError(f"Unknown level {level!r}; choose from {LEVELS}")
        self.level = level
        self.max_steps = int(max_steps)
        # Left/right wing activation: [0,1]. Lift depends on wing response and body attitude.
        self.action_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        # Relative target, velocity, sin/cos attitude, angular velocity, wing response,
        # wind, nearest obstacle relative position and radius; all clipped to [-1,1].
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.x = 0.0
        self.z = float(rng.uniform(0.8, 1.5))
        self.vx = float(rng.uniform(-0.12, 0.12))
        self.vz = float(rng.uniform(-0.2, 0.2))
        self.theta = float(rng.uniform(-0.20, 0.20))
        self.omega = float(rng.uniform(-0.15, 0.15))
        self.wings = np.zeros(2, dtype=np.float64)
        self.wind = np.zeros(2, dtype=np.float64)
        self.target = np.array([5.0, 1.8])
        self.steps = 0
        self.collision = False
        self.obstacles = []
        if self.level in ("obstacles", "storm"):
            self.obstacles = [(float(px), float(rng.uniform(0.65,2.5)),0.34)
                              for px in (1.8,3.5)]
        self.mass_factor = float(rng.uniform(0.85, 1.15))
        self.lift_factor = float(rng.uniform(0.9, 1.1))
        return self._obs(), self._info()

    def _nearest(self):
        if not self.obstacles:
            return (3.0, 3.0, 0.0)
        return min(((ox-self.x, oz-self.z, radius) for ox,oz,radius in self.obstacles),
                   key=lambda item:item[0]**2+item[1]**2)

    def _obs(self):
        dx,dz,r = self._nearest()
        values = [
            (self.target[0]-self.x)/6, (self.target[1]-self.z)/3,
            self.vx/5, self.vz/5,
            math.sin(self.theta), math.cos(self.theta),
            self.omega/8, self.wings[0], self.wings[1],
            self.wind[0]/5, self.wind[1]/5,
            dx/5, dz/3, r,
        ]
        return np.clip(np.array(values,dtype=np.float32),-1,1)

    def _info(self):
        return {"level":self.level, "crashed":self.collision or self.z<=0,
                "reached_target":bool(np.linalg.norm(self.target-[self.x,self.z])<0.35),
                "x":float(self.x),"z":float(self.z),
                "theta":float(self.theta),"wind_x":float(self.wind[0]),
                "wind_z":float(self.wind[1])}

    def step(self, action):
        a = np.asarray(action,dtype=np.float64)
        if a.shape!=(2,) or not np.all(np.isfinite(a)):
            raise ValueError("Action must contain two finite wing controls")
        a=np.clip(a,0,1)
        previous_distance=float(np.linalg.norm(self.target-[self.x,self.z]))
        self.steps+=1
        # Wing actuators have inertia; action cannot create instantaneous lift.
        self.wings += (a-self.wings)*0.20
        wind_scale = {"calm":0,"wind":1.0,"rain":1.2,"obstacles":1.2,"storm":2.1}[self.level]
        if wind_scale:
            self.wind = 0.95*self.wind + self.np_random.normal(0,0.15*wind_scale,2)
            self.wind = np.clip(self.wind,-3.5,3.5)
        else:
            self.wind[:]=0
        raining = self.level in ("rain","storm")
        rain_drag = 0.5 if raining else 0.0
        # Abstract forces: lift roughly proportional to total wing activation,
        # body pitch tilts the force; asymmetry creates torque.
        lift=18.0*self.lift_factor*float(self.wings.sum())/self.mass_factor
        ax = lift*math.sin(self.theta) - (0.35+rain_drag)*(self.vx-self.wind[0])
        az = lift*math.cos(self.theta) - 9.81 - (0.4+rain_drag)*(self.vz-self.wind[1])
        angular_acc = 16.0*(self.wings[1]-self.wings[0])-2.0*self.omega
        angular_acc += 0.3*self.wind[0]
        self.vx=np.clip(self.vx+DT*ax,-9,9)
        self.vz=np.clip(self.vz+DT*az,-9,9)
        self.omega=np.clip(self.omega+DT*angular_acc,-12,12)
        self.theta=float(np.clip(self.theta+DT*self.omega,-math.pi,math.pi))
        self.x+=DT*self.vx
        self.z+=DT*self.vz
        for ox,oz,r in self.obstacles:
            if (self.x-ox)**2+(self.z-oz)**2 < (r+0.12)**2:
                self.collision=True
        reached=np.linalg.norm(self.target-[self.x,self.z])<0.35
        crashed=self.z<=0 or self.z>=5 or abs(self.theta)>2.0 or self.collision or self.x < -2
        terminated=bool(crashed or reached)
        truncated=bool(self.steps>=self.max_steps and not terminated)
        new_distance=float(np.linalg.norm(self.target-[self.x,self.z]))
        progress=previous_distance-new_distance
        reward=4.0*progress + 0.025 - 0.02*abs(self.theta) - 0.005*float(np.square(a).sum())
        if crashed:
            reward-=15.0
        if reached:
            reward+=30.0
        return self._obs(),float(reward),terminated,truncated,self._info()
