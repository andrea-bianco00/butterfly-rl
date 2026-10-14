import numpy as np
from butterfly_rl.curriculum_v3 import CurriculumFlappyEnv,ReplayCurriculumEnv,STAGES,CHALLENGES

def test_all_initial_gaps_match_scenario():
    for scenario in list(STAGES)+list(CHALLENGES):
        env=CurriculumFlappyEnv(scenario)
        env.reset(seed=1729)
        lo,hi=env.spec["gap"]
        assert all(lo<=g[2]<=hi for g in env.gates)
        env.close()

def test_replacement_gates_respect_spacing_and_gap():
    env=CurriculumFlappyEnv("dense_ood")
    env.reset(seed=1729)
    env.steps=1
    old=env.gates[-1][0]
    env._new_gate(old+2.8)
    assert 1.25<=env.gates[-1][0]-old<=1.65
    assert 0.60<=env.gates[-1][2]<=0.80
    env.close()

def test_curriculum_reproducible():
    a=ReplayCurriculumEnv("storm")
    b=ReplayCurriculumEnv("storm")
    x,_=a.reset(seed=1729)
    y,_=b.reset(seed=1729)
    np.testing.assert_array_equal(x,y)
    for _ in range(30):
        ax=a.step(np.array([0.6],dtype=np.float32))
        bx=b.step(np.array([0.6],dtype=np.float32))
        np.testing.assert_array_equal(ax[0],bx[0])
        assert ax[1:4]==bx[1:4]
        if ax[2] or ax[3]:break
    a.close();b.close()

def test_held_out_parameters_disjoint_from_training():
    assert set(STAGES).isdisjoint(CHALLENGES)
    assert CHALLENGES["narrow_ood"]["gap"][1]<min(s["gap"][0] for s in
        __import__("butterfly_rl.curriculum_v3",fromlist=["SPECS"]).SPECS.values())
