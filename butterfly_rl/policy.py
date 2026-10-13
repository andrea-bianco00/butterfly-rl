"""Eight-parameter bounded linear feedback policy (NOT a neural network)."""
import numpy as np

def act(weights, observation):
    return np.tanh(np.asarray(weights).reshape(2, 4) @ observation)
