"""Record true model actions and simulator states for website playback.

No invented flight, interpolation of success, or post-hoc policy correction.
"""
import argparse,json
from pathlib import Path
from stable_baselines3 import PPO
from .curriculum_v5 import CurriculumFlappyEnvV5
from .curriculum_v3 import STAGES,CHALLENGES

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[94001,94002,94003])
    p.add_argument("--horizon",type=int,default=1200)
    a=p.parse_args()
    model=PPO.load(a.model,device="cpu")
    out={"source":"PPO deterministic model.predict, real V5 environment steps",
         "checkpoint":Path(a.model).name,"horizon":a.horizon,"episodes":[]}
    for scenario in list(STAGES)+list(CHALLENGES):
        for seed in a.seeds:
            env=CurriculumFlappyEnvV5(scenario,max_steps=a.horizon)
            obs,info=env.reset(seed=seed)
            frames=[{"step":0,"action":None,"x":info["x"],"y":info["y"],"vy":info["vy"],"wind":info["wind"],"passed":info["passed"],"gates":info["gates"]}]
            for i in range(a.horizon):
                action,_=model.predict(obs,deterministic=True)
                obs,reward,terminated,truncated,info=env.step(action)
                frames.append({"step":i+1,"action":float(action[0]),"x":info["x"],
                    "y":info["y"],"vy":info["vy"],"wind":info["wind"],
                    "passed":info["passed"],"gates":info["gates"],
                    "reward":float(reward),"crashed":bool(terminated)})
                if terminated or truncated:break
            out["episodes"].append({"scenario":scenario,"seed":seed,
                "crashed":bool(terminated),"steps":len(frames)-1,
                "passed":info["passed"],"frames":frames})
            env.close()
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,separators=(",",":")))
    print("Recorded",len(out["episodes"]),"real model episodes to",path)
if __name__=="__main__":main()
