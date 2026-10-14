"""Bounded model-based search using the ACTUAL v3 environment transitions.

No trained policy or oracle is involved. Finding a path proves existence for
that specific seeded episode and the simulator's CURRENT discrete collision
semantics. Failure to find one is inconclusive, never proof of impossibility.
The simulator has known swept-collision limitations, reported separately.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from .curriculum_v3 import CurriculumFlappyEnv, STAGES, CHALLENGES

ACTIONS=(0.0,0.5,1.0)

def _score(env):
    gx,gc,gh=env._next_gate()
    distance=max(0.0,gx-env.x)
    # Prefer progress first, then centered and dynamically stable flight.
    return (env.passed*1000.0 + env.x*0.15
            - 1.6*abs(env.y-gc) / max(gh,0.1)
            - 0.18*abs(env.vy)
            - (0.3 if distance<0.8 else 0.0)*abs(env.vy))

def search_episode(scenario,seed,horizon=240,beam_width=32):
    if beam_width<1 or horizon<1:raise ValueError("positive horizon and beam_width required")
    root=CurriculumFlappyEnv(scenario,max_steps=horizon)
    root.reset(seed=seed)
    beam=[(root,[])]
    best_passed=0
    for step in range(horizon):
        candidates=[]
        for env,actions in beam:
            for action in ACTIONS:
                successor=copy.deepcopy(env)
                _,_,terminated,truncated,info=successor.step(np.array([action],dtype=np.float32))
                best_passed=max(best_passed,successor.passed)
                if not terminated:
                    path=actions+[action]
                    if truncated:
                        return {"status":"found","steps":step+1,"passed":successor.passed,
                                "actions":path}
                    candidates.append((_score(successor),successor,path))
        if not candidates:
            return {"status":"not_found","steps_explored":step+1,"best_passed":best_passed,
                    "reason":"all retained branches terminated"}
        # Quantize states to reduce near-duplicate trajectories. Distinct RNG
        # states are not guaranteed equivalent: this is a search heuristic.
        candidates.sort(key=lambda t:t[0],reverse=True)
        beam=[]
        seen=set()
        for _,env,path in candidates:
            key=(round(env.y,2),round(env.vy,2),round(env.lift,2),env.passed)
            if key in seen:continue
            seen.add(key)
            beam.append((env,path))
            if len(beam)>=beam_width:break
    return {"status":"not_found","steps_explored":horizon,
            "best_passed":best_passed,"reason":"search budget exhausted"}

def audit(scenarios,seeds,horizon=240,beam_width=32):
    results={}
    for scenario in scenarios:
        episodes=[]
        for seed in seeds:
            result=search_episode(scenario,int(seed),horizon,beam_width)
            # Re-execute the discovered controls independently to ensure the
            # reported witness is real and not a search bookkeeping error.
            if result["status"]=="found":
                env=CurriculumFlappyEnv(scenario,max_steps=horizon)
                env.reset(seed=int(seed))
                for i,action in enumerate(result["actions"]):
                    _,_,terminated,truncated,_=env.step(np.array([action],dtype=np.float32))
                    if terminated or (truncated != (i==horizon-1)):
                        raise AssertionError("witness replay mismatch")
                if not truncated:raise AssertionError("witness did not reach horizon")
                result["witness_replay_verified"]=True
            episodes.append({"seed":int(seed),**result})
        results[scenario]={"episodes":episodes,"witnesses_found":sum(e["status"]=="found" for e in episodes)}
    return {"method":"bounded beam search over actual v3 Gymnasium transitions",
            "actions":list(ACTIONS),"horizon_steps":horizon,"beam_width":beam_width,
            "limitations":[
                "Search failure does not establish infeasibility.",
                "A successful witness establishes feasibility only for that seed, horizon, action discretization, and the existing simulator.",
                "Current simulator uses discrete/end-step collision tests; this audit does not fix tunneling.",
                "Beam pruning and state quantization may discard valid trajectories.",
                "No PPO weights are loaded; this does not measure learning or policy performance."
            ],"results":results}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",nargs="+",type=int,default=[7301,7302,7303])
    p.add_argument("--horizon",type=int,default=240)
    p.add_argument("--beam-width",type=int,default=32)
    p.add_argument("--scenarios",nargs="+",choices=list(STAGES)+list(CHALLENGES),
                   default=list(STAGES)+list(CHALLENGES))
    p.add_argument("--output",default="runs/dynamic-audit.json")
    args=p.parse_args()
    report=audit(args.scenarios,args.seeds,args.horizon,args.beam_width)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2))
    for scenario,record in report["results"].items():
        print(f"{scenario}: {record['witnesses_found']}/{len(record['episodes'])} verified witnesses")
if __name__=="__main__":main()
