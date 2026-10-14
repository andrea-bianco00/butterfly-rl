"""Out-of-distribution stress audit for an UNTRAINED oracle PD baseline.

All results are measured on actual simulator episodes, including failures.
No claims about PPO performance. Does not change training distribution.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .flappy_env import FlappyButterflyEnv
from .reference_controller import reference_action

CONDITIONS = {
    "standard_obstacles": {"level":"obstacles","gap":None,"spacing":None,"wind_multiplier":1.0},
    "narrow_gaps": {"level":"obstacles","gap":(0.30,0.45),"spacing":None,"wind_multiplier":1.0},
    "dense_obstacles": {"level":"obstacles","gap":None,"spacing":(1.3,1.8),"wind_multiplier":1.0},
    "strong_wind": {"level":"storm","gap":None,"spacing":None,"wind_multiplier":2.5},
    "combined_stress": {"level":"storm","gap":(0.30,0.45),"spacing":(1.3,1.8),"wind_multiplier":2.5},
}

class StressFlappyEnv(FlappyButterflyEnv):
    """Deterministic gate generator; initial gates also use stress parameters."""
    def __init__(self,condition,max_steps=1200):
        self.condition=dict(CONDITIONS[condition])
        super().__init__(level=self.condition["level"],max_steps=max_steps)

    def _new_gate(self,x):
        r=self.np_random
        center=float(r.uniform(1.05,2.95))
        gap=self.condition["gap"]
        if gap is None:
            gap=(0.62,0.88) if self.level=="obstacles" else (0.55,0.78)
        half=float(r.uniform(*gap))
        if self.condition["spacing"] is not None and self.steps>0 and self.gates:
            x=self.gates[-1][0]+float(r.uniform(*self.condition["spacing"]))
        self.gates.append((float(x),center,half))

    def reset(self,*,seed=None,options=None):
        obs,info=super().reset(seed=seed,options=options)
        self.wind_scale*=self.condition["wind_multiplier"]
        # Rebuild initial gates with scenario-specific spacing.
        if self.condition["spacing"] is not None:
            self.gates=[]
            self._credited_gates=set()
            pos=3.0
            for _ in range(4):
                self._new_gate(pos)
                pos+=float(self.np_random.uniform(*self.condition["spacing"]))
        return self._obs(),self._info(False)

def episode(condition,seed,max_steps):
    env=StressFlappyEnv(condition,max_steps=max_steps)
    env.reset(seed=seed)
    last={}
    for step in range(max_steps):
        action=reference_action(env)
        _,reward,done,truncated,last=env.step(action)
        if done or truncated:break
    out={"seed":seed,"steps":step+1,"seconds":round((step+1)*0.04,3),
         "passed":int(last["passed"]),"crashed":bool(last["crashed"])}
    env.close()
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--episodes",type=int,default=100)
    p.add_argument("--seed-start",type=int,default=800000)
    p.add_argument("--max-steps",type=int,default=1200)
    p.add_argument("--output",default="runs/stress-audit.json")
    args=p.parse_args()
    if args.episodes<1 or args.max_steps<1:p.error("episodes and max-steps must be positive")
    results={}
    for condition in CONDITIONS:
        trials=[episode(condition,args.seed_start+i,args.max_steps) for i in range(args.episodes)]
        failures=sum(t["crashed"] for t in trials)
        results[condition]={
            "n":len(trials),"collisions":failures,
            "collision_rate":failures/len(trials),
            "mean_gates":float(np.mean([t["passed"] for t in trials])),
            "median_gates":float(np.median([t["passed"] for t in trials])),
            "min_gates":min(t["passed"] for t in trials),
            "mean_seconds":float(np.mean([t["seconds"] for t in trials])),
            "episodes":trials}
    report={"controller":"untrained_oracle_PD",
            "warning":"Not a trained PPO policy. Reads hidden simulator state.",
            "seed_start":args.seed_start,"max_steps":args.max_steps,
            "conditions":CONDITIONS,"results":results}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a!="episodes"}
                      for k,v in results.items()},indent=2))

if __name__=="__main__":main()
