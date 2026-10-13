"""Simplified 2D flight control, NOT validated butterfly aerodynamics."""
import numpy as np

DT = 0.05
G = 9.81
TARGET_Z = 2.0
MAX_STEPS = 240

class ButterflyEnv:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.z = TARGET_Z + self.rng.uniform(-0.65, 0.65)
        self.vz = self.rng.uniform(-0.35, 0.35)
        self.theta = self.rng.uniform(-0.32, 0.32)
        self.omega = self.rng.uniform(-0.35, 0.35)
        self.steps = 0
        return self.observe()

    def observe(self):
        return np.array([self.z-TARGET_Z, self.vz, self.theta, self.omega], dtype=np.float64)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        thrust = G * (1 + 0.85*a[0])
        acceleration = thrust*np.cos(self.theta) - G - 0.45*self.vz
        angular_acceleration = 10*a[1] - 2*self.omega
        self.vz += acceleration*DT
        self.z += self.vz*DT
        self.omega += angular_acceleration*DT
        self.theta += self.omega*DT
        self.steps += 1
        error = self.observe()
        reward = 1 - (1.4*error[0]**2 + 0.15*error[1]**2 + 2.5*error[2]**2 + 0.15*error[3]**2 + 0.025*np.dot(a,a))
        crashed = self.z <= 0 or self.z >= 4.5 or abs(self.theta) >= 1.3
        done = crashed or self.steps >= MAX_STEPS
        if crashed:
            reward -= 20
        return self.observe(), float(reward), done, {"crashed": crashed}
