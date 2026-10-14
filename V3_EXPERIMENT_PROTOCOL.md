# Butterfly RL v3 — preregistered experiment

## Scientific status

Code committed, **training results are not available until GitHub Actions runs and the artifacts are inspected**. PPO is trained from scratch, not initialized from the oracle controller. No reward, success rate, or animation may be invented.

## Fixed design

- Stylized 2D Flappy-like flight, not realistic insect aerodynamics.
- Curriculum: wide → normal → dense → windy → storm. Each stage samples its own conditions 50% of episodes and previously visited stages 50%.
- All newly generated gates use the scenario's spacing and half-height ranges, including replacement gates.
- Training seeds: 1729, 2718, 3141, independently trained from scratch.
- Validation episodes: 20 per scenario, seeds 500000–500019, used for selecting the best stage checkpoint.
- Locked test: 100 episodes per scenario, seeds 900000–900099, used only after checkpoint selection.
- Four OOD tests: narrow gaps, dense gates, strong wind, combined stress. No OOD scenarios sampled during training.
- Baseline oracle controller uses hidden simulator state; PPO receives only the eight-dimensional observation. Their information budgets differ, so do not call their performance a controlled comparison.
- PPO architecture: MlpPolicy, 1024 rollout steps, batch size 256, learning rate 3e-4. One experiment per seed.
- Selection metric: mean over training scenarios of log(1 + mean gates) + 0.5 × (1 − collision rate). This is a heuristic, not the training reward.
- Report collision rate, gates mean/median/min, duration, and individual episode records, including failures.

## Important limitations and known confounds

1. Fixed evaluation seeds are reused across checkpoints during validation. This is legitimate model selection but may overfit validation. Only locked test scores estimate generalization.
2. Three seeds are a minimal replication, not enough for strong claims about reliability. Do not report statistical certainty from them alone.
3. Maximum 48 seconds per episode: 0 observed collisions does not prove infinite flight.
4. The reward function is inherited from v2; shaping and physics are not proven optimal.
5. Current obstacle collision checks use the vertical position at the end of a simulation step. This is an approximation, not a continuous swept-volume collision model. High-frequency collision and boundary tests remain a follow-up requirement.
6. The benchmark and training environment share the same simplified dynamics. OOD here means parameter shifts within this simulator, not transfer to the real world.
7. v2 results used a different obstacle/observation implementation; v2 and v3 reward values are **not directly comparable**. For a fair ablation, retrain both policies in the same frozen environment.

## Run procedure

Open GitHub Actions > Train PPO v3 - multi-seed progressive obstacles.
First run budget=2048 and seed_mode=smoke to test wiring. This is NOT a valid performance benchmark.
If tests pass, run budget=100000 and seed_mode=replicated for three independent real trainings.
Do not change the test split after seeing its results to improve the headline metrics.

## Definition of progress

Report whether the selected PPO policies exceed untrained baselines on held-out obstacle count and crash rate. A green GitHub workflow indicates only that execution succeeded. Preserve all outcomes.
