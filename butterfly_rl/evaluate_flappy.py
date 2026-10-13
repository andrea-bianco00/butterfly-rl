"""Held-out checkpoint evaluation. Report failures as well as successes."""
import argparse
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from .flappy_env import FlappyButterflyEnv,LEVELS

def run(model,level,seed,record=False):
    env=FlappyButterflyEnv(level)
    obs,_=env.reset(seed=seed)
    total=0.0;frames=[];info={}
    for step in range(env.max_steps):
        action,_=model.predict(obs,deterministic=True)
        obs,r,terminated,truncated,info=env.step(action)
        total+=r
        if record:
            frames.append({"step":step,"t":round((step+1)*0.04,3),
                "x":info["x"],"y":info["y"],"vy":info["vy"],
                "wind":info["wind"],"passed":info["passed"],
                "gates":info["gates"],"action":np.asarray(action).tolist()})
        if terminated or truncated:break
    env.close()
    return {"return":total,"crashed":info["crashed"],
            "passed":info["passed"],"survival_seconds":(step+1)*0.04},frames

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="runs/flappy")
    p.add_argument("--output",default="runs/flappy-evaluation")
    p.add_argument("--episodes",type=int,default=100)
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    results={}
    for name in ["untrained",*LEVELS,"best"]:
        path=Path(a.input)/f"policy_{name}.zip"
        if not path.exists():continue
        model=PPO.load(str(path),device="cpu")
        results[name]={}
        for level in LEVELS:
            trials=[run(model,level,300000+i)[0] for i in range(a.episodes)]
            results[name][level]={
                "n":len(trials),
                "crash_rate":float(np.mean([r["crashed"] for r in trials])),
                "mean_seconds":float(np.mean([r["survival_seconds"] for r in trials])),
                "mean_gates":float(np.mean([r["passed"] for r in trials])),
                "mean_return":float(np.mean([r["return"] for r in trials])),
                "sd_return":float(np.std([r["return"] for r in trials],ddof=1))
                            if len(trials)>1 else None}
            if level in ("calm","storm"):
                _,frames=run(model,level,2027,True)
                (out/f"trajectory_{name}_{level}.json").write_text(json.dumps(frames))
        (out/"metrics.json").write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=="__main__":main()
