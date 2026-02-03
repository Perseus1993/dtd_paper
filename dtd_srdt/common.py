import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def bpr_time(free_flow_time: float, volume: float, capacity: float, alpha: float = 0.15, beta: float = 4.0) -> float:
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    x = volume / capacity
    return free_flow_time * (1.0 + alpha * (x ** beta))


def softmax_choice(costs: np.ndarray, theta: float, rng: np.random.Generator) -> int:
    utilities = -theta * costs
    utilities = utilities - np.max(utilities)
    probs = np.exp(utilities)
    s = float(np.sum(probs))
    if s <= 0 or not np.isfinite(s):
        probs = np.ones_like(probs) / len(costs)
    else:
        probs = probs / s
    return int(rng.choice(len(costs), p=probs))


@dataclass
class BayesianBelief:
    mu: float
    var: float


def bayes_update(mu_prior: float, var_prior: float, obs: float, var_obs: float) -> Tuple[float, float]:
    denom = var_prior + var_obs
    if denom <= 0:
        raise ValueError("invalid variance")
    mu_post = (var_obs * mu_prior + var_prior * obs) / denom
    var_post = (var_prior * var_obs) / denom
    return mu_post, var_post


def schedule_penalties(arrival_time: float, target_arrival: float) -> Tuple[float, float]:
    early = max(0.0, target_arrival - arrival_time)
    late = max(0.0, arrival_time - target_arrival)
    return early, late


def relative_gap(prev: np.ndarray, cur: np.ndarray) -> float:
    denom = float(np.sum(prev ** 2))
    if denom <= 0:
        return float("nan")
    return math.sqrt(float(np.sum((cur - prev) ** 2)) / denom)


def safe_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))
