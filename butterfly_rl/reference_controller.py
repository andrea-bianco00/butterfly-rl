"""Independent deterministic PD baseline for Flappy Butterfly.

NOT a trained policy. This is an engineering baseline to test reachability
and to compare with PPO. It uses public environment state (oracle access),
so it must NEVER be presented as a learned policy or a fair observation-only
agent. Reports include failures without filtering.
"""
import argparse
import json
import numpy as np
from .flappy_env import FlappyButterflyEnv, LEVELS

def reference_action(env):
    """PD vertical tracking with actuator compensation; action in [0,1]."""
    gx,gc,gh=env._next_gate()
    distance=gx-env.x
    if env.with_gates and distance<2.6:
        target=float(np.clip(gc,0.6,3.4))
    else:
        target=2.0
    # Limit requested velocity to avoid violent steering near the gate.
    desired_v=np.clip(1.8*(target-env.y),-2.2,2.2)
    desired_acc=3.4*(desired_v-env.vy)
    # Invert the simulator's simplified vertical acceleration equation.
    target_lift=(9.81+(0.45+env.rain_drag)*(env.vy-env.wind)+desired_acc)/17.0
    # Compensate actuator lag, with finite bounded input.
    action=env.lift+0.7*(target_lift-env.lift)
    return np.array([np.clip(action,0,1)],dtype=np.float32)

def run_episode(level,seed,max_steps=1200,record=False):
    env=FlappyButterflyEnv(level,max_steps=max_steps)
    obs,info=env.reset(seed=seed)
    frames=[]
    for step in range(max_steps):
        action=reference_action(env)
        obs,reward,done,truncated,info=env.step(action)
        if record:
            frames.append({"step":step,"x":info["x"],"y":info["y"],
                           "vy":info["vy"],"wind":info["wind"],
                           "passed":info["passed"],"crashed":info["crashed"],
                           "gates":info["gates"],"action":float(action[0])})
        if done or truncated:break
    result={"seed":seed,"level":level,"steps":step+1,
            "survival_seconds":round((step+1)*0.04,3),
            "passed":info["passed"],"crashed":info["crashed"]}
    env.close()
    return result,frames

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--episodes",type=int,default=100)
    p.add_argument("--seed-start",type=int,default=700000)
    p.add_argument("--output",default="runs/reference-baseline.json")
    args=p.parse_args()
    if args.episodes<1:p.error("episodes must be positive")
    from pathlib import Path
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    results={}
    for level in LEVELS:
        episodes=[run_episode(level,args.seed_start+i)[0]
                  for i in range(args.episodes)]
        results[level]={"episodes":episodes,
                        "mean_passed":float(np.mean([e["passed"] for e in episodes])),
                        "crash_rate":float(np.mean([e["crashed"] for e in episodes])),
                        "mean_seconds":float(np.mean([e["survival_seconds"] for e in episodes]))}
    path.write_text(json.dumps({"controller":"untrained_oracle_PD",
                                "seed_start":args.seed_start,
                                "results":results},indent=2))
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!="episodes"}
                      for k,v in results.items()},indent=2))

if __name__=="__main__":main()
