"""Export *recorded* PPO V5 episodes to a compact, browser-ready replay.

This script does not synthesize trajectories or modify the actions, collision
outcomes, or episode endings. Numeric rounding affects display precision only.
"""
import argparse,json
from pathlib import Path

def compact_frame(frame):
    return {"step":int(frame["step"]),"action":None if frame["action"] is None else round(float(frame["action"]),5),
            "x":round(float(frame["x"]),5),"y":round(float(frame["y"]),5),
            "vy":round(float(frame["vy"]),5),"wind":round(float(frame["wind"]),5),
            "passed":int(frame["passed"]),
            "gates":[[round(float(v),5) for v in gate] for gate in frame["gates"]],
            "crashed":bool(frame.get("crashed",False))}

def export(source):
    if source.get("source")!="PPO deterministic model.predict, real V5 environment steps":
        raise ValueError("Unexpected replay provenance")
    output={"source":source["source"],"checkpoint":source["checkpoint"],
            "horizon":source["horizon"],"seed_of_training":1729,"episodes":[]}
    scenarios=list(dict.fromkeys(e["scenario"] for e in source["episodes"]))
    for scenario in scenarios:
        group=[e for e in source["episodes"] if e["scenario"]==scenario]
        # First episode plus first distinct crash, if any.
        selected=group[:1]
        crashed=next((e for e in group if e["crashed"] and e not in selected),None)
        if crashed is not None:selected.append(crashed)
        for e in selected:
            frames=[compact_frame(f) for f in e["frames"]]
            if len(frames)!=int(e["steps"])+1:raise ValueError("Missing frames")
            if frames[-1]["crashed"]!=bool(e["crashed"]):raise ValueError("Collision mismatch")
            if frames[-1]["passed"]!=int(e["passed"]):raise ValueError("Passed count mismatch")
            output["episodes"].append({"scenario":scenario,"seed":int(e["seed"]),
                "crashed":bool(e["crashed"]),"steps":int(e["steps"]),
                "passed":int(e["passed"]),"frames":frames})
    return output

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    result=export(json.loads(Path(a.input).read_text()))
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,separators=(",",":")))
    print("Real recorded episodes:",len(result["episodes"]))
    for e in result["episodes"]:
        print(e["scenario"],e["seed"],"steps",e["steps"],"passed",e["passed"],"crashed",e["crashed"])
if __name__=="__main__":main()
