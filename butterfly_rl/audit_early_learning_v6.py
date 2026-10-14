"""Independent deterministic audits of impulse-flap PPO v6.

Tests collision boundaries, naive action baselines, and held-out course seeds.
The displayed course seed is excluded from this generalization evaluation.
"""
import argparse,json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from .flappy_impulse_v6 import FlappyImpulseEnvV6
from .curriculum_v4 import swept_collision

def run(seed,controller,horizon=650):
    e=FlappyImpulseEnvV6(max_steps=horizon)
    obs,_=e.reset(seed=seed)
    for step in range(horizon):
        a=controller(obs,step)
        obs,_,done,truncated,info=e.step(a)
        if done or truncated:break
    e.close()
    return {"passed":int(info["passed"]),"steps":step+1,"crashed":bool(done)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--models",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--seeds",type=int,default=40)
    a=p.parse_args()
    # Geometry sanity: outside gap collides; centered trajectory does not.
    g=[(3.0,2.0,.7)]
    assert swept_collision(2.5,2.0,3.5,2.0,g)==False
    assert swept_collision(2.5,3.0,3.5,3.0,g)==True
    assert swept_collision(2.5,1.0,3.5,1.0,g)==True
    assert swept_collision(2.5,.05,2.6,.05,[])==True
    seeds=list(range(20261015,20261015+a.seeds))
    baseline={
        "never_flap":lambda obs,t:0,
        "always_flap":lambda obs,t:1,
        "every_8_steps":lambda obs,t:int(t%8==0),
        "every_12_steps":lambda obs,t:int(t%12==0),
        "every_16_steps":lambda obs,t:int(t%16==0),
    }
    report={"evaluation_seeds":seeds,"display_seed_excluded":94001 not in seeds,
            "horizon":650,"collision_tests":"passed","baselines":{},"checkpoints":{}}
    for name,controller in baseline.items():
        outcomes=[run(s,controller) for s in seeds]
        report["baselines"][name]={"mean_passed":float(np.mean([x["passed"] for x in outcomes])),
                                   "completion_12_rate":float(np.mean([x["passed"]>=12 for x in outcomes])),
                                   "mean_steps":float(np.mean([x["steps"] for x in outcomes]))}
    manifest=json.loads((Path(a.models)/"manifest.json").read_text())
    wanted=list(range(0,50001,2000))+[100000]
    for target in wanted:
        choices=[c for c in manifest["checkpoints"] if c["step"]<=target]
        if not choices:continue
        cp=max(choices,key=lambda c:c["step"])
        model=PPO.load(str(Path(a.models)/cp["file"]),device="cpu")
        def controller(obs,t):return int(model.predict(obs,deterministic=True)[0])
        outcomes=[run(s,controller) for s in seeds]
        report["checkpoints"][str(cp["step"])]={
            "mean_passed":float(np.mean([x["passed"] for x in outcomes])),
            "completion_12_rate":float(np.mean([x["passed"]>=12 for x in outcomes])),
            "mean_steps":float(np.mean([x["steps"] for x in outcomes]))}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
