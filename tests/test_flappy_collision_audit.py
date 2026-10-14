"""Regression checks: do not give credit for touching a pipe."""
import numpy as np
from butterfly_rl.flappy_env import FlappyButterflyEnv

def _prepared():
    env=FlappyButterflyEnv("obstacles")
    env.reset(seed=1729)
    env.gates=[(3.0,2.0,0.75)]
    env._credited_gates=set()
    env.x=2.9
    env.y=2.0
    env.vy=0
    env.lift=9.81/17
    return env

def test_no_gate_credit_at_wall_center():
    env=_prepared()
    env.step(np.array([9.81/17],dtype=np.float32))
    assert env.passed==0

def test_collision_never_counts_as_gate_passed():
    env=_prepared()
    env.y=3.5
    _,_,done,_,info=env.step(np.array([9.81/17],dtype=np.float32))
    assert done and info["crashed"] and info["passed"]==0

def test_gate_credit_only_after_full_clearance():
    env=_prepared()
    env.x=3.29
    _,_,done,_,info=env.step(np.array([9.81/17],dtype=np.float32))
    assert not done
    assert info["passed"]==1
    for _ in range(3):
        _,_,done,_,info=env.step(np.array([9.81/17],dtype=np.float32))
        assert info["passed"]==1
        if done:break

def test_observation_not_repeated_distance():
    env=_prepared()
    obs=env._obs()
    assert obs.shape==(8,)
    assert not np.isclose(obs[4],obs[7])
