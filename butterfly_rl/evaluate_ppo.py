"""Compare saved PPO policies on held-out seeds and export real state trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from .robust_env import RobustButterflyEnv, LEVELS

def episode(model,level,seed,trajectory=False):
    env=RobustButterflyEnv(level=level)
    obs,_=env.reset(seed=seed)
    total=0.0
    records=[]
    info={}
    for t in range(env.max_steps):
        action,_=model.predict(obs,deterministic=True)
        obs,reward,terminated,truncated,info=env.step(action)
        total+=reward
        if trajectory:
            records.append({"t":round((t+1)*0.04,3),"x":info["x"],"z":info["z"],
                            "theta":info["theta"],"wind_x":info["wind_x"],
                            "wind_z":info["wind_z"],"action":np.asarray(action).tolist(),
                            "reward":reward})
        if terminated or truncated:
            break
    env.close()
    return {"return":total,"success":info["reached_target"],
            "crashed":info["crashed"],"seconds":(t+1)*0.04},records

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="runs/ppo")
    p.add_argument("--episodes",type=int,default=100)
    p.add_argument("--output",default="runs/evaluation")
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    models=Path(args.input)
    results={}
    for checkpoint in ["untrained",*LEVELS]:
        path=models/f"policy_{checkpoint}.zip"
        if not path.exists():
            print(f"Skipping missing checkpoint {path}")
            continue
        model=PPO.load(str(path),device="cpu")
        results[checkpoint]={}
        for level in LEVELS:
            samples=[episode(model,level,200_000+i)[0] for i in range(args.episodes)]
            results[checkpoint][level]={
                "n":len(samples),
                "success_rate":float(np.mean([s["success"] for s in samples])),
                "crash_rate":float(np.mean([s["crashed"] for s in samples])),
                "mean_return":float(np.mean([s["return"] for s in samples])),
                "return_sd":float(np.std([s["return"] for s in samples],ddof=1))
                             if len(samples)>1 else None,
                "mean_seconds":float(np.mean([s["seconds"] for s in samples]))}
            if level in ("calm","storm"):
                _,trajectory=episode(model,level,2027,True)
                (out/f"trajectory_{checkpoint}_{level}.json").write_text(
                    json.dumps(trajectory,indent=2))
        (out/"metrics.json").write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=="__main__":
    main()
