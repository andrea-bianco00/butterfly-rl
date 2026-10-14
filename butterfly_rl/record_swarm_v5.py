"""Record 100 authentic PPO flights on one shared course per checkpoint.

All agents share one environment seed, hence obstacle and wind RNG streams.
Only initial altitude/velocity vary, deterministically. Agents do not breed;
survival does not alter PPO weights. Any mismatch in live obstacle geometry
aborts the export rather than showing a misleading shared scene.
"""
import argparse,json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from .curriculum_v5 import CurriculumFlappyEnvV5

CHECKPOINTS=[("Untrained","policy_untrained.zip",0),("Wide","policy_wide.zip",100000),("Normal","policy_normal.zip",200000),("Dense","policy_dense.zip",300000),("Windy","policy_windy.zip",400000),("Storm","policy_storm.zip",500000),("Best","policy_best.zip",None)]

def round5(x):return round(float(x),5)
def gate_list(gates):return [[round5(v) for v in gate] for gate in gates]

def run_checkpoint(model,scenario,seed,count,horizon):
    rng=np.random.default_rng(77127)
    envs=[];obs=[];active=np.ones(count,dtype=bool)
    start=[];results=[None]*count
    for i in range(count):
        env=CurriculumFlappyEnvV5(scenario,max_steps=horizon)
        env.reset(seed=seed)
        # Different legitimate initial states, identical course and weather.
        env.y=float(np.clip(env.y+rng.uniform(-.32,.32),.7,3.3))
        env.vy=float(np.clip(env.vy+rng.uniform(-.22,.22),-.6,.6))
        start.append([round5(env.y),round5(env.vy)])
        obs.append(env._obs());envs.append(env)
    frames=[]
    try:
        for step in range(horizon+1):
            living=np.flatnonzero(active)
            if len(living)==0:break
            reference=envs[int(living[0])]._info(False)
            common=gate_list(reference["gates"])
            for i in living[1:]:
                info=envs[int(i)]._info(False)
                if gate_list(info["gates"])!=common or abs(info["x"]-reference["x"])>1e-8 or abs(info["wind"]-reference["wind"])>1e-8:
                    raise ValueError("Shared-course invariant failed at step "+str(step))
            frame={"step":step,"x":round5(reference["x"]),"gates":common,
                   "alive":[[int(i),round5(envs[int(i)].y),int(envs[int(i)].passed)] for i in living]}
            frames.append(frame)
            if step==horizon:break
            actions,_=model.predict(np.stack([obs[int(i)] for i in living]),deterministic=True)
            for j,i0 in enumerate(living):
                i=int(i0)
                nxt,_,terminated,truncated,info=envs[i].step(actions[j])
                obs[i]=nxt
                if terminated or truncated:
                    active[i]=False
                    results[i]={"id":i,"steps":step+1,"passed":int(info["passed"]),"crashed":bool(terminated)}
        for i in np.flatnonzero(active):
            results[int(i)]={"id":int(i),"steps":horizon,"passed":int(envs[int(i)].passed),"crashed":False}
        if any(r is None for r in results):raise ValueError("Missing outcome")
        return {"initial_states":start,"frames":frames,"outcomes":results}
    finally:
        for e in envs:e.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--models",required=True);p.add_argument("--out",required=True)
    p.add_argument("--scenario",default="normal");p.add_argument("--seed",type=int,default=94001)
    p.add_argument("--count",type=int,default=100);p.add_argument("--horizon",type=int,default=1200)
    a=p.parse_args()
    if a.count!=100:raise ValueError("This visualization is audited for exactly 100 agents")
    out={"source":"Actual PPO model.predict actions in 100 independent V5 simulators, synchronized identical course and wind",
         "training_seed":1729,"scenario":a.scenario,"course_seed":a.seed,"count":a.count,
         "horizon":a.horizon,"dt":0.04,"checkpoints":[]}
    for label,filename,steps in CHECKPOINTS:
        model=PPO.load(str(Path(a.models)/filename),device="cpu")
        result=run_checkpoint(model,a.scenario,a.seed,a.count,a.horizon)
        out["checkpoints"].append({"label":label,"model":filename,"training_steps_approx":steps,**result})
        print(label,"survivors",sum(not r["crashed"] for r in result["outcomes"]),
              "mean passed",round(sum(r["passed"] for r in result["outcomes"])/a.count,2),flush=True)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,separators=(",",":")))
    print("Published genuine synchronized trajectories:",path,path.stat().st_size,"bytes")
if __name__=="__main__":main()
