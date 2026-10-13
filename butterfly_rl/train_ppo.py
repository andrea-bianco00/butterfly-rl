"""Actual PPO training with checkpoint snapshots; never fabricate metrics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from .robust_env import RobustButterflyEnv, LEVELS

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--steps-per-level",type=int,default=100_000)
    p.add_argument("--seed",type=int,default=1729)
    p.add_argument("--output",default="runs/ppo")
    args=p.parse_args()
    if args.steps_per_level<2048:
        p.error("--steps-per-level must be >= 2048")
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    model=None
    metadata={"algorithm":"PPO","seed":args.seed,"steps_per_level":args.steps_per_level,
              "levels":[],"status":"running"}
    (out/"manifest.json").write_text(json.dumps(metadata,indent=2))
    for idx,level in enumerate(LEVELS):
        env=RobustButterflyEnv(level=level)
        env.reset(seed=args.seed+idx)
        if model is None:
            model=PPO("MlpPolicy",env,seed=args.seed,verbose=1,
                      n_steps=1024,batch_size=256,learning_rate=3e-4,
                      tensorboard_log=str(out/"tensorboard"))
            model.save(str(out/"policy_untrained"))
        else:
            model.set_env(env)
        callback=CheckpointCallback(save_freq=20_000,save_path=str(out),
                                    name_prefix=f"checkpoint_{level}")
        model.learn(total_timesteps=args.steps_per_level,callback=callback,
                    reset_num_timesteps=False,progress_bar=False)
        model.save(str(out/f"policy_{level}"))
        metadata["levels"].append({"level":level,"total_timesteps":model.num_timesteps,
                                   "checkpoint":f"policy_{level}.zip"})
        (out/"manifest.json").write_text(json.dumps(metadata,indent=2))
        env.close()
    metadata["status"]="completed"
    (out/"manifest.json").write_text(json.dumps(metadata,indent=2))
    print("Training finished; evaluate with python -m butterfly_rl.evaluate_ppo")

if __name__=="__main__":
    main()
