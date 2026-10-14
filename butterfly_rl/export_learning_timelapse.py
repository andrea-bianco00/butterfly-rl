"""Bundle deterministic replay trajectories of REAL saved PPO checkpoints.

All checkpoints are evaluated on identical scenario and seeds. Nothing is
interpolated between policies. Display trajectories are rounded to 5 decimals.
"""
import argparse,json
from pathlib import Path
from .export_replays_v5 import compact_frame

STAGES=[("Untrained","policy_untrained.zip",0),("Wide","policy_wide.zip",100000),("Normal","policy_normal.zip",200000),("Dense","policy_dense.zip",300000),("Windy","policy_windy.zip",400000),("Storm","policy_storm.zip",500000),("Best","policy_best.zip",None)]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--folder",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--scenario",default="normal")
    p.add_argument("--seeds",type=int,nargs="+",default=[94001,94002,94003])
    a=p.parse_args()
    out={"source":"Recorded PPO deterministic model.predict in CurriculumFlappyEnvV5","training_seed":1729,
         "scenario":a.scenario,"seeds":a.seeds,"frames_per_second":25,"checkpoints":[]}
    for label,filename,steps in STAGES:
        file=Path(a.folder)/(filename.replace(".zip",".json"))
        raw=json.loads(file.read_text())
        if raw["checkpoint"]!=filename:raise ValueError("Checkpoint mismatch: "+filename)
        episodes=[]
        for e in raw["episodes"]:
            if e["scenario"]!=a.scenario or e["seed"] not in a.seeds:continue
            frames=[compact_frame(f) for f in e["frames"]]
            if len(frames)!=e["steps"]+1 or bool(frames[-1]["crashed"])!=bool(e["crashed"]):raise ValueError("Replay inconsistent")
            episodes.append({"seed":e["seed"],"crashed":e["crashed"],"passed":e["passed"],"steps":e["steps"],"frames":frames})
        if len(episodes)!=len(a.seeds):raise ValueError("Missing episodes: "+label)
        out["checkpoints"].append({"label":label,"file":filename,"training_steps_approx":steps,"episodes":episodes})
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(out,separators=(",",":")))
    print("Verified",len(out["checkpoints"]),"checkpoints,",sum(len(x["episodes"]) for x in out["checkpoints"]),"real episodes")
if __name__=="__main__":main()
