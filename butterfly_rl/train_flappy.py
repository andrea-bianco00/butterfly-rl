"""Train actual PPO checkpoint policies for Flappy Butterfly curriculum."""
import argparse
import json
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from .flappy_env import FlappyButterflyEnv,LEVELS

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--steps-per-level",type=int,default=100000)
    p.add_argument("--seed",type=int,default=1729)
    p.add_argument("--output",default="runs/flappy")
    a=p.parse_args()
    if a.steps_per_level<2048: p.error("at least 2048 steps required")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    manifest={"seed":a.seed,"steps_per_level":a.steps_per_level,
              "status":"running","levels":[]}
    model=None
    for i,level in enumerate(LEVELS):
        env=FlappyButterflyEnv(level=level)
        env.reset(seed=a.seed+i)
        if model is None:
            model=PPO("MlpPolicy",env,seed=a.seed,verbose=1,n_steps=1024,
                      batch_size=256,learning_rate=0.0003,device="cpu")
            model.save(str(out/"policy_untrained"))
        else:model.set_env(env)
        cb=CheckpointCallback(save_freq=20000,save_path=str(out),
                              name_prefix=f"checkpoint_{level}")
        model.learn(total_timesteps=a.steps_per_level,callback=cb,
                    reset_num_timesteps=False)
        model.save(str(out/f"policy_{level}"))
        manifest["levels"].append({"level":level,"timesteps":model.num_timesteps})
        (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
        env.close()
    manifest["status"]="completed"
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))

if __name__=="__main__":main()
