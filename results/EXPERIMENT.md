# First verified CEM experiment

Executed with Python and NumPy. This is real policy search, **not PPO and not a neural network**.

- Seed: 42; 35 generations; population 32; 5 training episodes per candidate
- Evaluation: 100 held-out environment seeds 10000–10099
- Max episode: 240 steps × 0.05 seconds = 12 seconds

| Metric | Trained CEM | Zero action | One random fixed feedback policy |
|---|---:|---:|---:|
| Mean return | 238.2367 | -8.5052 | -34.9617 |
| Return SD | 1.3414 | 144.8384 | 5.8755 |
| Crash fraction | 0.00 | 0.50 | 1.00 |
| Mean episode length (seconds) | 12.0000 | 9.1165 | 0.9620 |

These are outputs from the executed baseline experiment. The environment uses abstract thrust and torque and does not simulate insect wing aerodynamics. One training seed and one random fixed baseline are insufficient to establish general robustness.

Reproduce with:

```bash
pip install -r requirements.txt
python -m butterfly_rl.train --generations 35 --population 32 --seed 42
python -m butterfly_rl.evaluate
python -m pytest tests -q
```

The trained weights are saved as `results/policy.npy`; the evaluation writes trajectories as JSON.
