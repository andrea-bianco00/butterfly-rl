import numpy as np
import pytest
pytest.importorskip("gymnasium")
from butterfly_rl.flappy_env import FlappyButterflyEnv,LEVELS

@pytest.mark.parametrize("level",LEVELS)
def test_valid_steps(level):
    env=FlappyButterflyEnv(level)
    obs,_=env.reset(seed=1729)
    assert env.observation_space.contains(obs)
    for _ in range(20):
        obs,reward,done,truncated,info=env.step(np.array([0.6]))
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        if done or truncated:break

def test_deterministic_weather_and_gates():
    a,b=FlappyButterflyEnv("storm"),FlappyButterflyEnv("storm")
    assert np.array_equal(a.reset(seed=1729)[0],b.reset(seed=1729)[0])
    for _ in range(20):
        ra,rb=a.step([0.7]),b.step([0.7])
        assert np.array_equal(ra[0],rb[0])
        assert ra[1:4]==rb[1:4]
        if ra[2] or ra[3]:break

def test_different_obstacles_for_different_seeds():
    a,b=FlappyButterflyEnv("storm"),FlappyButterflyEnv("storm")
    a.reset(seed=1729);b.reset(seed=1730)
    assert a.gates!=b.gates

def test_invalid_action():
    env=FlappyButterflyEnv()
    env.reset(seed=1729)
    with pytest.raises(ValueError):env.step([float("nan")])
