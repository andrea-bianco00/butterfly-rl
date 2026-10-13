# Robust Butterfly RL: research protocol

## Status

**Implementation committed. PPO training NOT YET EXECUTED.**
No PPO success rates or learning curves are claimed until real checkpoints and evaluation outputs exist.

## Task

A stylized 2D agent must fly from x=0 to x=5 near altitude z=1.8.
The agent controls left and right wing activations, with delayed actuator response.
The environment includes gravity, drag, pitch dynamics and abstract lift/torque.

This is **not biologically validated butterfly aerodynamics**. "Sun" is only a
visual condition in the eventual portfolio, not a physical disturbance. Rain
is approximated by increased drag; wind by stochastic force. Obstacles are
stationary circles. The current observation includes local wind and nearest
obstacle position, so the policy has information for avoidance.

## Curriculum

1. `calm`: no wind, no rain, no obstacles.
2. `wind`: stochastic gusts.
3. `rain`: gusts plus increased drag.
4. `obstacles`: gusts plus fixed circular obstacles.
5. `storm`: stronger gusts, rain and obstacles.

The stages are **not monotonically nested** (the obstacle stage omits rain).
This is an initial curriculum to test, not a claim of guaranteed improvement.

## Training

```bash
python -m pip install -r requirements-ppo.txt
python -m pytest -q
python -m butterfly_rl.train_ppo --steps-per-level 100000 --seed 42
python -m butterfly_rl.evaluate_ppo --episodes 100
```

Training is expected to take substantial CPU time and may fail to learn.
Artifacts under `runs/` are created by executing the commands, not generated
from hand-authored values. `policy_untrained.zip` is saved before learning,
and subsequent policy snapshots are saved at the end of each curriculum stage.

## Evaluation standards

Compare **untrained** and every trained checkpoint across **all five levels**.
Use held-out evaluation seeds 100000–100099. Report success, crash, episode
duration, reward mean and standard deviation. Extend to multiple training
seeds and held-out weather regimes before claiming robust generalization.

A 100% success rate on finite trials is not "always" or "every situation".
Do not cherry-pick visually pleasing trajectories: keep evaluation tables and
identify any selected demonstration episode as illustrative.

## Animation contract

Replay only trajectories exported by `evaluate_ppo.py`. The visual wingbeat
and rendered butterfly are artistic; position and attitude must match logged
simulator states. No fake losses, episode numbers or PPO checkpoint claims.
