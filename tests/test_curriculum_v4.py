import numpy as np
from butterfly_rl.curriculum_v3 import CurriculumFlappyEnv
from butterfly_rl.curriculum_v4 import CurriculumFlappyEnvV4,swept_collision

def test_no_walls_and_safe_vertical_segment():
    assert not swept_collision(0,2,1,2,[])

def test_swept_collision_catches_endpoint_tunneling():
    gate=(2.,2.,0.5)
    # Endpoints are outside the expanded horizontal wall but
    # center passes through it above the safe opening.
    assert swept_collision(1.5,3.,2.5,3.,[gate])

def test_safe_crossing_through_opening():
    assert not swept_collision(1.5,2.,2.5,2.,[(2.,2.,0.5)])

def test_vertical_excursion_while_overlapping_wall():
    assert swept_collision(1.5,2.,2.5,3.,[(2.,2.,0.5)])

def test_boundary_crossing():
    assert swept_collision(0,2,0.1,4.,[])

def test_identical_seed_reproducible():
    for scenario in ["wide","storm","combined_ood"]:
        a=CurriculumFlappyEnvV4(scenario)
        b=CurriculumFlappyEnvV4(scenario)
        a.reset(seed=7301);b.reset(seed=7301)
        for _ in range(30):
            u=a.step(np.array([0.6],dtype=np.float32))
            v=b.step(np.array([0.6],dtype=np.float32))
            assert u[1:4]==v[1:4]
            assert u[4]==v[4]
            if u[2] or u[3]:break

def test_v3_unchanged_on_disk():
    assert CurriculumFlappyEnv is not CurriculumFlappyEnvV4

def test_invalid_action_rejected():
    e=CurriculumFlappyEnvV4("wide")
    e.reset(seed=7301)
    try:
        e.step(np.array([np.nan],dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("NaN action accepted")
