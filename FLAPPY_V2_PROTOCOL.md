# Flappy Butterfly RL v2 — experiment protocol

## Status

Source code and GitHub Actions workflow committed. **No v2 results are claimed until the workflow finishes and its artifacts are inspected.** v1 genuinely trained, but its final policy failed on obstacle courses and forgot calm-flight performance.

## Hypothesis

Sequential PPO training on one environment at a time causes performance regression. A mixed curriculum that replays previous conditions may preserve basic flight while acquiring obstacle skills.

## Changes

- `MixedFlappyEnv`: each episode samples either the current stage (50%) or one of the earlier stages (50%). The current stage is not forgotten.
- `FlappyButterflyEnv`: small local penalty for misalignment with the next gap (only when approaching an obstacle). The original gate-crossing bonus and collision penalty remain.
- Checkpoints saved after each stage and periodically during long training.
- `policy_best.zip` is chosen by a validation score across all five scenarios, **not** by last checkpoint or training reward.
- Held-out evaluation uses seeds 300000–300099, distinct from validation seeds 500000–500019 and training seeds.
- Seed 1729 is retained for the first training run.

## Run

GitHub Actions workflow: [Train Flappy PPO v2](.github/workflows/train-flappy-v2.yml)

On push it executes a **2048 steps/stage smoke test**, not a scientifically meaningful full training. For an actual experiment, manually select `100000`, `200000` or `400000` steps per stage. Each stage trains an additional PPO policy; evaluation exports genuine JSON trajectories.

## Metrics and limitations

Primary metrics: mean survival time, crash rate, number of gates passed, distribution of outcomes across levels. Evaluate *every* stage, including untrained and best checkpoint. Report failed runs.

The validation score is a heuristic, not a proof of optimality; it is calculated from calm/wind/rain survival and obstacle/storm gate count. Because the obstacle observation includes only the nearest gate and actions are simplified lift commands, the simulator is **Flappy-like**, not insect flight aerodynamics. No finite evaluation can demonstrate infinite safe flight.

## Next scientific checks

- Multiple training seeds, e.g. 1729, 2718, 3141, 5772, 8119.
- Confidence intervals for survival and collision rates.
- Stress tests on wind intensities and gap widths outside training distribution.
- Ablation: compare sequential v1 vs mixed replay v2 with matched training budgets.
- Inspect trajectories for reward hacking and degenerate behaviors.
