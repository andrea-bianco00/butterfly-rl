"""Actual PPO mixed-curriculum training with reproducible evaluation.

Training is separate from validation. We save EVERY stage checkpoint, and
select the best checkpoint by a held-out multi-scenario score, not final time.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from .mixed_env import MixedFlappyEnv
from .flappy_env import LEVELS
from .evaluate_flappy import run

def validate(model,episodes=20):
    scores={}
    for level in LEVELS:
        trials=[run(model,level,seed=500000+i)[0] for i in range(episodes)]
        scores[level]={
            "mean_seconds":float(np.mean([x["survival_seconds"] for x in trials])),
            "mean_gates":float(np.mean([x["passed"] for x in trials])),
            "crash_rate":float(np.mean([x["crashed"] for x in trials])),
        }
    # Reward robustness: emphasize obstacle/storm gates but require calm flight.
    # Not a training reward; used only to rank checkpoints.
    score=(scores["calm"]["mean_seconds"]/48
           +scores["wind"]["mean_seconds"]/48
           +scores["rain"]["mean_seconds"]/48
           +scores["obstacles"]["mean_gates"]/10
           +scores["storm"]["mean_gates"]/10)
    return float(score),scores

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--steps-per-level",type=int,default=200000)
    p.add_argument("--seed",type=int,default=1729)
    p.add_argument("--output",default="runs/flappy-v2")
    p.add_argument("--validation-episodes",type=int,default=20)
    a=p.parse_args()
    if a.steps_per_level<2048:p.error("At least 2048 steps required")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    manifest={"algorithm":"PPO","seed":a.seed,"steps_per_level":a.steps_per_level,
              "validation_seed_start":500000,"status":"running","stages":[]}
    model=None
    best=-float("inf")
    for i,stage in enumerate(LEVELS):
        env=MixedFlappyEnv(stage=stage,seed=a.seed+i)
        env.reset(seed=a.seed+i)
        if model is None:
            model=PPO("MlpPolicy",env,seed=a.seed,verbose=1,
                      n_steps=1024,batch_size=256,learning_rate=3e-4,
                      device="cpu")
            model.save(str(out/"policy_untrained"))
        else:
            model.set_env(env)
        callback=CheckpointCallback(save_freq=50000,save_path=str(out),
                                    name_prefix=f"checkpoint_{stage}")
        model.learn(total_timesteps=a.steps_per_level,callback=callback,
                    reset_num_timesteps=False)
        model.save(str(out/f"policy_{stage}"))
        score,metrics=validate(model,episodes=a.validation_episodes)
        if score>best:
            best=score
            model.save(str(out/"policy_best"))
            manifest["best_stage"]=stage
            manifest["best_validation_score"]=best
        manifest["stages"].append({"stage":stage,"timesteps":model.num_timesteps,
                                   "validation_score":score,"validation":metrics})
        (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
        env.close()
    manifest["status"]="completed"
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("Complete; best checkpoint:",manifest["best_stage"])

if __name__=="__main__":
    main()
