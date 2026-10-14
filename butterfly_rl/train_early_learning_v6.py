"""Train PPO on one unchanged Flappy impulse environment.

Checkpoints are saved at fixed timesteps, not curated to tell a story.
"""
import argparse,json
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from .flappy_impulse_v6 import FlappyImpulseEnvV6

class Milestones(BaseCallback):
    def __init__(self,out,interval):
        super().__init__();self.out=out;self.interval=interval;self.next_save=interval;self.saved=[]
    def _on_step(self):
        while self.num_timesteps>=self.next_save:
            name="policy_step_"+str(self.next_save)
            self.model.save(str(self.out/name))
            self.saved.append({"step":self.next_save,"file":name+".zip"})
            self.next_save+=self.interval
        return True

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=1729)
    p.add_argument("--steps",type=int,default=100000)
    p.add_argument("--interval",type=int,default=2000)
    p.add_argument("--out",default="runs/flappy-v6")
    a=p.parse_args()
    if a.steps<2048 or a.interval<2048:raise ValueError("Training budget too small")
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    env=FlappyImpulseEnvV6()
    model=PPO("MlpPolicy",env,seed=a.seed,n_steps=1000,batch_size=250,learning_rate=3e-4,verbose=1,device="cpu")
    model.save(str(out/"policy_step_0"))
    callback=Milestones(out,a.interval)
    model.learn(total_timesteps=a.steps,callback=callback)
    model.save(str(out/"policy_final"))
    manifest={"version":"v6","physics":"discrete impulse flap, gravity and swept collision","scenario":"normal throughout",
       "seed":a.seed,"training_steps":model.num_timesteps,"checkpoints":[{"step":0,"file":"policy_step_0.zip"},*callback.saved,{"step":model.num_timesteps,"file":"policy_final.zip"}],
       "note":"No checkpoint performance is assumed or fabricated."}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    env.close()
if __name__=="__main__":main()
