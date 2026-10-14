"""PPO v5: actual training, validation, held-out testing, no fabricated outcomes.

Run independently for each seed. Checkpoints are selected on validation
seeds only. The held-out test is run once after selection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from .curriculum_v3 import STAGES,SPECS,CHALLENGES
from .curriculum_v5 import CurriculumFlappyEnvV5 as CurriculumFlappyEnv, ReplayCurriculumEnvV5 as ReplayCurriculumEnv

def evaluate(model,scenario,seeds):
    episodes=[]
    for seed in seeds:
        env=CurriculumFlappyEnv(scenario)
        obs,_=env.reset(seed=int(seed))
        for t in range(env.max_steps):
            action,_=model.predict(obs,deterministic=True)
            obs,_,terminated,truncated,info=env.step(action)
            if terminated or truncated:break
        episodes.append({"seed":int(seed),"steps":t+1,
                         "seconds":round((t+1)*0.04,3),
                         "gates":int(info["passed"]),"crashed":bool(info["crashed"])})
        env.close()
    gates=np.array([e["gates"] for e in episodes])
    crashes=np.array([e["crashed"] for e in episodes])
    return {"episodes":episodes,"n":len(episodes),
            "mean_gates":float(np.mean(gates)),
            "median_gates":float(np.median(gates)),
            "min_gates":int(np.min(gates)),
            "crash_rate":float(np.mean(crashes)),
            "mean_seconds":float(np.mean([e["seconds"] for e in episodes]))}

def score(validation):
    # Selection criterion predeclared before held-out test.
    # Includes every stage to penalize catastrophic forgetting.
    return float(np.mean([np.log1p(validation[k]["mean_gates"])
                          +0.5*(1-validation[k]["crash_rate"]) for k in STAGES]))

def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=1729)
    p.add_argument("--steps-per-stage",type=int,default=100000)
    p.add_argument("--validation-episodes",type=int,default=20)
    p.add_argument("--test-episodes",type=int,default=100)
    p.add_argument("--output",default="runs/flappy-v5")
    a=p.parse_args()
    if a.steps_per_stage<2048 or a.validation_episodes<1 or a.test_episodes<1:
        p.error("Invalid episode/step budget")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    manifest={"experiment":"v5","algorithm":"PPO","seed":a.seed,
      "steps_per_stage":a.steps_per_stage,"stages":STAGES,
      "training_scenarios":SPECS,"held_out_scenarios":CHALLENGES,
      "validation_seeds":list(range(500000,500000+a.validation_episodes)),
      "test_seeds":list(range(900000,900000+a.test_episodes)),
      "status":"running","history":[]}
    model=None;best=-float("inf")
    for i,stage in enumerate(STAGES):
        env=ReplayCurriculumEnv(stage,seed=a.seed+i)
        if model is None:
            model=PPO("MlpPolicy",env,seed=a.seed,verbose=1,n_steps=1024,
                      batch_size=256,learning_rate=3e-4,device="cpu")
            model.save(str(out/"policy_untrained"))
        else:model.set_env(env)
        model.learn(total_timesteps=a.steps_per_stage,reset_num_timesteps=False,
                    callback=CheckpointCallback(save_freq=50000,save_path=str(out),
                                                name_prefix="intermediate_"+stage))
        model.save(str(out/("policy_"+stage)))
        val={k:evaluate(model,k,manifest["validation_seeds"]) for k in STAGES}
        metric=score(val)
        if metric>best:
            best=metric
            model.save(str(out/"policy_best"))
            manifest["best_stage"]=stage
            manifest["best_score"]=metric
        manifest["history"].append({"stage":stage,"timesteps":model.num_timesteps,
                                     "selection_score":metric,
                                     "validation":val})
        write_json(out/"manifest.json",manifest)
        env.close()
    # Locked test split: do not use this to choose checkpoints.
    best_model=PPO.load(str(out/"policy_best"),device="cpu")
    scenarios=list(STAGES)+list(CHALLENGES)
    manifest["held_out_test"]={k:evaluate(best_model,k,manifest["test_seeds"])
                               for k in scenarios}
    manifest["status"]="completed"
    manifest["environment"]="curriculum_v5: swept collision + corrected gate spacing"
    manifest["training_is_real"]=True
    write_json(out/"manifest.json",manifest)
    print("Selected stage:",manifest["best_stage"])
    print(json.dumps({k:{"mean_gates":v["mean_gates"],
                         "crash_rate":v["crash_rate"]}
                      for k,v in manifest["held_out_test"].items()},indent=2))

if __name__=="__main__":main()
