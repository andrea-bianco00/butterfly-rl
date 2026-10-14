"""Verify stress scenarios actually change the simulator."""
import numpy as np
from butterfly_rl.stress_audit import StressFlappyEnv,episode,CONDITIONS

def test_narrow_gaps_are_narrow():
    env=StressFlappyEnv("narrow_gaps")
    env.reset(seed=1729)
    assert all(0.30<=g[2]<=0.45 for g in env.gates)
    env.close()

def test_dense_initial_spacing():
    env=StressFlappyEnv("dense_obstacles")
    env.reset(seed=1729)
    distances=np.diff([g[0] for g in env.gates])
    assert all(1.3<=d<=1.8 for d in distances)
    env.close()

def test_wind_is_stronger():
    normal=StressFlappyEnv("standard_obstacles")
    strong=StressFlappyEnv("strong_wind")
    normal.reset(seed=1729)
    strong.reset(seed=1729)
    assert strong.wind_scale>normal.wind_scale
    normal.close();strong.close()

def test_repeatable_stress_episode():
    assert episode("combined_stress",1729,100)==episode("combined_stress",1729,100)

def test_no_success_inferred_from_execution():
    assert set(CONDITIONS)=={"standard_obstacles","narrow_gaps",
                            "dense_obstacles","strong_wind","combined_stress"}
