import numpy as np
from butterfly_rl.curriculum_v5 import CurriculumFlappyEnvV5

def test_gate_spacing_matches_scenario():
    env=CurriculumFlappyEnvV5("wide")
    env.reset(seed=1729)
    xs=[g[0] for g in env.gates]
    assert len(xs)==4
    assert all(3.0<=b-a<=3.5 for a,b in zip(xs,xs[1:]))
    env.close()

def test_v5_deterministic_seed():
    def trajectory():
        env=CurriculumFlappyEnvV5("storm")
        obs,_=env.reset(seed=2718)
        result=[]
        for _ in range(20):
            obs,r,term,trunc,info=env.step(np.array([0.58],dtype=np.float32))
            result.append((obs.tolist(),r,term,trunc,info["gates"]))
            if term or trunc:break
        env.close()
        return result
    assert trajectory()==trajectory()

def test_v5_initial_gate_gap_within_bounds():
    env=CurriculumFlappyEnvV5("dense")
    env.reset(seed=3141)
    assert all(.60<=half<=.80 for _,_,half in env.gates)
    env.close()
