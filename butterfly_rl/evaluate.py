"""Held-out seed evaluation; JSON trajectories are actual simulator rollouts."""
import argparse
import json
from pathlib import Path
import numpy as np
from .train import rollout

def evaluate(weights,seeds):
    outcomes=[rollout(weights,s,True) for s in seeds]
    returns=np.array([x[0] for x in outcomes])
    crashes=np.array([x[1] for x in outcomes])
    return {"episodes":len(seeds),"mean_return":float(returns.mean()),
        "std_return":float(returns.std(ddof=1)),"crash_rate":float(crashes.mean()),
        "mean_flight_seconds":float(np.mean([len(x[2])*0.05 for x in outcomes]))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results")
    a=p.parse_args()
    out=Path(a.output)
    weights=np.load(out/"policy.npy")
    seeds=list(range(10000,10100))
    report={"trained":evaluate(weights,seeds),
        "zero_action_baseline":evaluate(np.zeros(8),seeds),
        "random_fixed_policy_baseline":evaluate(np.random.default_rng(123).normal(size=8),seeds)}
    (out/"evaluation.json").write_text(json.dumps(report,indent=2))
    for label,w in [("trained",weights),("untrained",np.zeros(8))]:
        _,_,traj=rollout(w,2026,True)
        (out/f"{label}_trajectory.json").write_text(json.dumps(traj,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
