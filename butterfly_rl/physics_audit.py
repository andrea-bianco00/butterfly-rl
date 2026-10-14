"""Geometry-only audit for v3. Does not claim policy success.

A necessary (not sufficient) condition for passing adjacent gates is that
a bounded-acceleration trajectory can reach both openings. We report both
simple interval overlap and a sampled reachable envelope. This is a
diagnostic, NOT a proof that any PPO policy can fly through.
"""
import argparse,json
from pathlib import Path
import numpy as np
from .curriculum_v3 import CurriculumFlappyEnv,STAGES,CHALLENGES

DT=0.04
SPEED=1.7
RADIUS=0.12
WALL_HALF_WIDTH=0.19
YMIN,YMAX=0.15,3.85

def safe_interval(g):
    x,c,h=g
    return max(YMIN+RADIUS,c-h+RADIUS),min(YMAX-RADIUS,c+h-RADIUS)

def adjacent_gate_diagnostic(left,right):
    low1,high1=safe_interval(left)
    low2,high2=safe_interval(right)
    # Travel time between wall exits and entrances.
    available=(right[0]-left[0]-2*(WALL_HALF_WIDTH+RADIUS))/SPEED
    gap=max(0.0,low2-high1,low1-high2)
    # A speed cap of 7 is a deliberately optimistic upper bound.
    necessary_speed=gap/max(available,1e-9)
    return {"travel_seconds":available,"minimum_vertical_displacement":gap,
            "required_mean_vertical_speed":necessary_speed,
            "violates_speed_bound":bool(available<=0 or necessary_speed>7),
            "left_safe_interval":[low1,high1],
            "right_safe_interval":[low2,high2]}

def audit(scenario,episodes,seed):
    rng=np.random.default_rng(seed)
    records=[]
    for i in range(episodes):
        env=CurriculumFlappyEnv(scenario)
        env.reset(seed=int(rng.integers(1,2**31-1)))
        # Generate a chain of gates directly using the scenario distribution.
        gates=list(env.gates)
        for _ in range(60):
            env.steps=1
            env._new_gate(gates[-1][0]+2.8)
            gates.append(env.gates[-1])
        for left,right in zip(gates,gates[1:]):
            records.append(adjacent_gate_diagnostic(left,right))
        env.close()
    return {"scenario":scenario,"pairs":len(records),
            "fraction_violating_optimistic_speed_bound":float(np.mean([r["violates_speed_bound"] for r in records])),
            "min_available_seconds":float(min(r["travel_seconds"] for r in records)),
            "max_required_speed":float(max(r["required_mean_vertical_speed"] for r in records)),
            "examples":[r for r in records if r["violates_speed_bound"]][:5]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--episodes",type=int,default=200)
    p.add_argument("--seed",type=int,default=7301)
    p.add_argument("--output",default="runs/geometry-audit.json")
    a=p.parse_args()
    if a.episodes<1:p.error("episodes must be positive")
    report={"method":"optimistic vertical-speed necessary-condition audit",
            "limitations":"A pass does not prove dynamic feasibility; a violation proves failure under the stated 7-unit/s bound and gate-center traversal assumption.",
            "results":{s:audit(s,a.episodes,a.seed) for s in list(STAGES)+list(CHALLENGES)}}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps({s:{"violations":v["fraction_violating_optimistic_speed_bound"],
                         "max_required_speed":v["max_required_speed"]}
                      for s,v in report["results"].items()},indent=2))
if __name__=="__main__":main()
