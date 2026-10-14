"""Paired, pre-registered v3/v4 simulator comparison.

Identical seeds and open-loop actions. This is NOT a PPO evaluation: after
trajectories diverge, RNG draws and obstacles may also diverge. Report both
first disagreement and outcome; never interpret a paired delta as a causal
effect on trained policy performance.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .curriculum_v3 import CurriculumFlappyEnv,STAGES,CHALLENGES
from .curriculum_v4 import CurriculumFlappyEnvV4

def compare(scenarios,seeds,horizon=600):
    rows=[]
    for scenario in scenarios:
        for seed in seeds:
            rng=np.random.default_rng(int(seed)+987654)
            actions=rng.uniform(0,1,size=horizon).astype(np.float32)
            envs=[CurriculumFlappyEnv(scenario,max_steps=horizon),
                  CurriculumFlappyEnvV4(scenario,max_steps=horizon)]
            for env in envs:env.reset(seed=int(seed))
            outcomes=[]
            for name,env in zip(("v3","v4"),envs):
                result={"version":name,"steps_survived":0,"passed":0,"crashed":False,
                        "truncated":False,"total_reward":0.0}
                for i,action in enumerate(actions):
                    _,reward,terminated,truncated,info=env.step(np.array([action],dtype=np.float32))
                    result["steps_survived"]=i+1
                    result["total_reward"]+=float(reward)
                    result["passed"]=int(env.passed)
                    if terminated or truncated:
                        result["crashed"]=bool(terminated)
                        result["truncated"]=bool(truncated)
                        break
                outcomes.append(result)
                env.close()
            rows.append({"scenario":scenario,"seed":int(seed),"action_seed":int(seed)+987654,
                         "action_type":"fixed open-loop uniform random, shared across versions",
                         "v3":outcomes[0],"v4":outcomes[1]})
    return {"method":"paired open-loop random actions; independently seeded simulators",
            "horizon":horizon,"scenarios":list(scenarios),"seeds":[int(x) for x in seeds],
            "limitations":[
                "Random actions are not a learned policy or an optimal controller.",
                "A successful CI run does not imply PPO has learned to fly.",
                "Different collision times can change subsequent RNG consumption and obstacle generation.",
                "The v4 collision model assumes linear motion inside each discrete step.",
                "Results describe only the tested seeds, action streams and horizon."
            ],"episodes":rows}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,nargs="+",default=[7301,7302,7303,7304,7305])
    p.add_argument("--horizon",type=int,default=600)
    p.add_argument("--scenarios",nargs="+",choices=list(STAGES)+list(CHALLENGES),
                   default=list(STAGES)+list(CHALLENGES))
    p.add_argument("--output",default="runs/v3-v4-comparison.json")
    a=p.parse_args()
    report=compare(a.scenarios,a.seeds,a.horizon)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2))
    for s in a.scenarios:
        rows=[r for r in report["episodes"] if r["scenario"]==s]
        print(s,"v3 collisions",sum(r["v3"]["crashed"] for r in rows),
              "v4 collisions",sum(r["v4"]["crashed"] for r in rows),
              "mean survival steps",
              round(sum(r["v3"]["steps_survived"] for r in rows)/len(rows),1),
              round(sum(r["v4"]["steps_survived"] for r in rows)/len(rows),1))
if __name__=="__main__":main()
