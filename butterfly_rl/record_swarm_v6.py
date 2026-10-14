"""Record 100 real PPO v6 impulse-flap flights per milestone on a shared course.

All agents run independent environments initialized with the same course seed.
Only initial altitude and velocity differ. We assert common obstacle geometry
and wind among survivors at every step; abort if the shared-scene claim fails.
"""
import argparse,json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from .flappy_impulse_v6 import FlappyImpulseEnvV6

def r(x):return round(float(x),5)
def gates(g):return [[r(z) for z in item] for item in g]

def simulate(model,seed,count,horizon):
    envs=[];obs=[];alive=np.ones(count,dtype=bool)
    rng=np.random.default_rng(77127)
    outcomes=[None]*count;frames=[]
    try:
        for i in range(count):
            e=FlappyImpulseEnvV6(max_steps=horizon)
            e.reset(seed=seed)
            e.y=float(np.clip(e.y+rng.uniform(-.3,.3),.7,3.3))
            e.vy=float(np.clip(e.vy+rng.uniform(-.2,.2),-.6,.6))
            envs.append(e);obs.append(e._obs())
        for step in range(horizon+1):
            ids=np.flatnonzero(alive)
            if len(ids)==0:break
            ref=envs[int(ids[0])]
            g=gates(ref.gates)
            for i in ids[1:]:
                e=envs[int(i)]
                if gates(e.gates)!=g or abs(e.x-ref.x)>1e-8 or abs(e.wind-ref.wind)>1e-8:
                    raise ValueError("Shared-course invariant violated at step "+str(step))
            frames.append({"step":step,"x":r(ref.x),"gates":g,
                           "alive":[[int(i),r(envs[int(i)].y),int(envs[int(i)].passed)] for i in ids]})
            if step==horizon:break
            actions,_=model.predict(np.stack([obs[int(i)] for i in ids]),deterministic=True)
            for j,i0 in enumerate(ids):
                i=int(i0)
                nxt,_,dead,limit,info=envs[i].step(int(actions[j]))
                obs[i]=nxt
                if dead or limit:
                    alive[i]=False
                    outcomes[i]={"id":i,"steps":step+1,"passed":int(info["passed"]),"crashed":bool(dead)}
        for i in np.flatnonzero(alive):
            outcomes[int(i)]={"id":int(i),"steps":horizon,"passed":int(envs[int(i)].passed),"crashed":False}
        if any(o is None for o in outcomes):raise ValueError("Missing agent outcome")
        return {"frames":frames,"outcomes":outcomes}
    finally:
        for e in envs:e.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--models",required=True);p.add_argument("--out",required=True)
    p.add_argument("--seed",type=int,default=94001)
    p.add_argument("--count",type=int,default=100)
    p.add_argument("--horizon",type=int,default=650)
    a=p.parse_args()
    if a.count!=100:raise ValueError("100 agents required")
    manifest=json.loads((Path(a.models)/"manifest.json").read_text())
    steps=manifest["checkpoints"]
    # Fixed chronological milestones; never choose visually flattering results.
    wanted=[0,50000,100000,200000,350000,500000,750000,1000000]
    selected=[]
    for target in wanted:
        candidates=[c for c in steps if c["step"]<=target]
        if candidates:
            c=max(candidates,key=lambda x:x["step"])
            if c not in selected:selected.append(c)
    final=steps[-1]
    if final not in selected:selected.append(final)
    out={"source":"PPO v6 deterministic actions in 100 synchronized impulse-flap environments",
         "training_seed":manifest["seed"],"scenario":"fixed normal",
         "course_seed":a.seed,"count":a.count,"horizon":a.horizon,"dt":0.04,"checkpoints":[]}
    for c in selected:
        model=PPO.load(str(Path(a.models)/c["file"]),device="cpu")
        record=simulate(model,a.seed,a.count,a.horizon)
        out["checkpoints"].append({"label":str(c["step"]),"model":c["file"],
                                   "training_steps_approx":c["step"],**record})
        print("steps",c["step"],"mean gates",round(sum(x["passed"] for x in record["outcomes"])/a.count,2),
              "finished",sum(x["passed"]>=12 for x in record["outcomes"]),flush=True)
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(out,separators=(",",":")))
    print("Verified:",len(out["checkpoints"]),"chronological policies, 100 agents each")
if __name__=="__main__":main()
