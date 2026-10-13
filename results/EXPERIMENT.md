# Verified baseline experiment

Actual CEM policy search run in Python, **not a neural network and not PPO**.

Training: seed 42, 35 generations, population 32, 5 shared environment seeds per candidate and generation.
Evaluation: 100 independent environment seeds (10000–10099), max 12 seconds per episode.

| Metric | Trained CEM | Zero-action baseline | One random fixed feedback policy |
|---|---:|---:|---:|
| Mean return | 238.3767 | -7.3696 | -33.5071 |
| Return SD | 1.2641 | 140.5814 | 7.8427 |
| Crash fraction | 0.00 | 0.52 | 1.00 |
| Mean episode seconds | 12.0000 | 8.9205 | 1.0225 |

These numbers were computed by executing the repository's training/evaluation logic. They are for a toy hovering task only, not validated insect aerodynamics. The random baseline is **one fixed random feedback policy**, not a new random action at every step. One training seed is not sufficient for robust statistical conclusions.

Reproduce:

```bash
pip install -r requirements.txt
python -m butterfly_rl.train --generations 35 --population 32 --seed 42
python -m butterfly_rl.evaluate
python -m pytest tests -q
```

Training generates `results/policy.npy` and `results/training_history.json`; evaluation exports real `results/trained_trajectory.json` and `results/untrained_trajectory.json`. The model files and trajectory exports are generated artifacts, not fabricated illustrations.
