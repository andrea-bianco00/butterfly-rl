import numpy as np
import pytest
gym = pytest.importorskip("gymnasium")
from butterfly_rl.robust_env import RobustButterflyEnv, LEVELS

@pytest.mark.parametrize("level",LEVELS)
def test_observation_and_step(level):
    env=RobustButterflyEnv(level=level)
    obs,info=env.reset(seed=42)
    assert env.observation_space.contains(obs)
    for _ in range(25):
        obs,reward,terminated,truncated,info=env.step(np.array([0.5,0.5]))
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        if terminated or truncated:
            break

def test_seeded_determinism():
    a,b=RobustButterflyEnv("storm"),RobustButterflyEnv("storm")
    np.testing.assert_allclose(a.reset(seed=15)[0],b.reset(seed=15)[0])
    for _ in range(15):
        xa=a.step([0.8,0.3]);xb=b.step([0.8,0.3])
        np.testing.assert_allclose(xa[0],xb[0])
        assert xa[1:4]==xb[1:4]
        if xa[2] or xa[3]:
            break
