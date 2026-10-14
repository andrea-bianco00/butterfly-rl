import numpy as np
from butterfly_rl.dynamic_audit import audit,search_episode
from butterfly_rl.curriculum_v3 import CurriculumFlappyEnv

def test_same_seed_same_physics():
    a=CurriculumFlappyEnv("storm")
    b=CurriculumFlappyEnv("storm")
    a.reset(seed=7301);b.reset(seed=7301)
    for _ in range(12):
        x=a.step(np.array([0.5],dtype=np.float32))
        y=b.step(np.array([0.5],dtype=np.float32))
        assert x[1:4]==y[1:4]
        assert x[4]==y[4]
        if x[2]:break

def test_search_reports_nonproof_on_failure():
    result=search_episode("combined_ood",7301,horizon=4,beam_width=2)
    assert result["status"] in {"found","not_found"}
    if result["status"]=="not_found":
        assert "reason" in result

def test_replay_witness_if_found():
    result=audit(["wide"],[7301],horizon=6,beam_width=3)
    episode=result["results"]["wide"]["episodes"][0]
    if episode["status"]=="found":
        assert episode["witness_replay_verified"]
