from butterfly_rl.physics_audit import safe_interval,adjacent_gate_diagnostic,audit

def test_safe_interval_has_radius_margin():
    assert safe_interval((3.,2.,.5))==(1.62,2.38)

def test_distant_aligned_gates_have_no_speed_violation():
    d=adjacent_gate_diagnostic((3.,2.,.6),(6.,2.,.6))
    assert d["minimum_vertical_displacement"]==0
    assert not d["violates_speed_bound"]

def test_unreachable_gate_pair_detected():
    d=adjacent_gate_diagnostic((3.,1.1,.32),(3.8,2.9,.32))
    assert d["violates_speed_bound"]

def test_audit_reproducible():
    a=audit("combined_ood",2,7301)
    b=audit("combined_ood",2,7301)
    assert a==b
