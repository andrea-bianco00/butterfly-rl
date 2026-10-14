from butterfly_rl.compare_v3_v4 import compare

def test_paired_actions_and_reproducibility():
    a=compare(["wide","combined_ood"],[7301,7302],horizon=25)
    b=compare(["wide","combined_ood"],[7301,7302],horizon=25)
    assert a==b
    assert len(a["episodes"])==4
    for row in a["episodes"]:
        assert row["v3"]["steps_survived"]<=25
        assert row["v4"]["steps_survived"]<=25
        assert row["action_seed"]==row["seed"]+987654

def test_no_policy_claims():
    report=compare(["wide"],[7301],horizon=3)
    assert "not a learned policy" in report["limitations"][0]
