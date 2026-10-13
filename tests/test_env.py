import numpy as np
from butterfly_rl.env import ButterflyEnv
from butterfly_rl.policy import act

def test_deterministic_initialization():
    assert np.allclose(ButterflyEnv(10).observe(),ButterflyEnv(10).observe())

def test_bounded_actions():
    assert np.all(np.abs(act(np.ones(8),np.ones(4)))<=1)

def test_finite_transition():
    env=ButterflyEnv(4)
    obs,reward,done,info=env.step([0,0])
    assert np.isfinite(obs).all() and np.isfinite(reward)
