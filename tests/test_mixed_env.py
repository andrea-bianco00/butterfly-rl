import numpy as np
import pytest
pytest.importorskip("gymnasium")
from butterfly_rl.mixed_env import MixedFlappyEnv

def test_mixed_env_reproducible():
    a,b=MixedFlappyEnv("storm"),MixedFlappyEnv("storm")
    oa,ia=a.reset(seed=1729)
    ob,ib=b.reset(seed=1729)
    np.testing.assert_array_equal(oa,ob)
    assert ia["level"]==ib["level"]
    for _ in range(30):
        sa=a.step([0.65])
        sb=b.step([0.65])
        np.testing.assert_array_equal(sa[0],sb[0])
        assert sa[1:4]==sb[1:4]
        if sa[2] or sa[3]:break
    a.close();b.close()

def test_previous_levels_replayed():
    env=MixedFlappyEnv("storm")
    levels=set()
    for i in range(100):
        _,info=env.reset(seed=10000+i)
        levels.add(info["level"])
    assert "storm" in levels
    assert len(levels)>1
    env.close()
