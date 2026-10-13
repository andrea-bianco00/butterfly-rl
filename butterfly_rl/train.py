"""Actual episodic policy search with Cross-Entropy Method (CEM), not PPO."""
import argparse
import json
from pathlib import Path
import numpy as np
from .env import ButterflyEnv
from .policy import act

def rollout(weights, seed, record=False):
    env = ButterflyEnv(seed)
    obs = env.observe()
    total = 0.0
    trajectory = []
    crashed = False
    for step in range(240):
        action = act(weights, obs)
        nxt, reward, done, info = env.step(action)
        if record:
            trajectory.append({"t": round(step*0.05,3), "z":float(env.z),
                "vz":float(env.vz), "theta":float(env.theta), "omega":float(env.omega),
                "thrust_command":float(action[0]), "torque_command":float(action[1]),
                "reward":reward})
        total += reward
        obs = nxt
        crashed = info["crashed"]
        if done:
            break
    return total, crashed, trajectory

def train(generations=35, population=32, seed=42):
    rng = np.random.default_rng(seed)
    mean = np.zeros(8)
    std = np.ones(8)*1.5
    history = []
    best = None
    best_score = -np.inf
    for gen in range(generations):
        candidates = rng.normal(mean, std, (population,8))
        seeds = [int(s) for s in rng.integers(0,2**31-1,5)]
        scores = np.array([np.mean([rollout(w,s)[0] for s in seeds]) for w in candidates])
        elites = candidates[np.argsort(scores)[-max(4,population//6):]]
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0),0.04)
        j = int(np.argmax(scores))
        if scores[j] > best_score:
            best_score = float(scores[j])
            best = candidates[j].copy()
        history.append({"generation":gen+1,"mean_population_return":float(scores.mean()),
            "best_generation_return":float(scores[j]),"best_seen_return":best_score})
        if (gen+1)%10==0:
            print(f"Generation {gen+1} | mean={scores.mean():.2f} | best={best_score:.2f}",flush=True)
    return best, history

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--generations",type=int,default=35)
    p.add_argument("--population",type=int,default=32)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output",default="results")
    a=p.parse_args()
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    weights,history=train(a.generations,a.population,a.seed)
    np.save(out/"policy.npy",weights)
    (out/"training_history.json").write_text(json.dumps(history,indent=2))
    print("Saved",out)

if __name__=="__main__":
    main()
