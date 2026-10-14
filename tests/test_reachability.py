"""Deterministic, falsifiable checks for reachable and unreachable gates."""
import numpy as np
from butterfly_rl.flappy_env import FlappyButterflyEnv
from butterfly_rl.reference_controller import reference_action,run_episode

def test_straight_gate_can_be_crossed_without_collision():
    # Construct a physically possible path through a fixed gap.
    env=FlappyButterflyEnv("obstacles",max_steps=120)
    env.reset(seed=1729)
    env.wind_scale=0.0
    env.wind=0.0
    env.x=0.0
    env.y=2.0
    env.vy=0.0
    env.lift=9.81/17
    env.gates=[(3.0,2.0,0.85)]
    env._credited_gates=set()
    for _ in range(80):
        _,_,done,truncated,info=env.step([9.81/17])
        assert not done, "A straight centerline through a wide gap must be feasible"
        if info["passed"]==1:break
    assert info["passed"]==1
    env.close()

def test_off_center_gate_causes_collision():
    env=FlappyButterflyEnv("obstacles",max_steps=120)
    env.reset(seed=1729)
    env.wind_scale=0.0
    env.x=2.5
    env.y=3.5
    env.vy=0.0
    env.lift=9.81/17
    env.gates=[(3.0,2.0,0.85)]
    env._credited_gates=set()
    for _ in range(30):
        _,_,done,truncated,info=env.step([9.81/17])
        if done:break
    assert done and info["crashed"]
    assert info["passed"]==0
    env.close()

def test_reference_controller_valid_action():
    env=FlappyButterflyEnv("storm")
    env.reset(seed=1729)
    for _ in range(30):
        action=reference_action(env)
        assert action.shape==(1,) and np.isfinite(action).all()
        assert 0<=action[0]<=1
        _,_,done,truncated,_=env.step(action)
        if done or truncated:break
    env.close()

def test_baseline_determinism():
    a,_=run_episode("storm",1729,max_steps=200)
    b,_=run_episode("storm",1729,max_steps=200)
    assert a==b
