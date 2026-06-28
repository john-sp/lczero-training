import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

_ASGO_CURVATURE_ABSOLUTE = 0
_ASGO_CURVATURE_PER_ITERATION = 1


def compute_gradient(
    deltas: Sequence[nnx.State],
    elo_diffs: Sequence[float],
    elo_clamp: float,
    rounds: int | None = None,
    curvature_weighting: bool = False,
    curvature_mode: int = _ASGO_CURVATURE_ABSOLUTE,
    noise_floor: float = 10.0,
) -> nnx.State:
    """Estimates a zeroth-order gradient from paired ELO differences."""
    if not deltas:
        raise ValueError("At least one perturbation delta is required.")
    rounds = len(deltas) if rounds is None else rounds
    if rounds != len(deltas) or rounds != len(elo_diffs):
        raise ValueError("rounds, deltas, and elo_diffs must have same length.")
    if rounds <= 0:
        raise ValueError("rounds must be positive.")
    if elo_clamp <= 0:
        raise ValueError("elo_clamp must be positive.")
    if noise_floor <= 0:
        raise ValueError("noise_floor must be positive.")

    elo_arr = jnp.asarray(elo_diffs, dtype=jnp.float32)
    clamped = elo_clamp * jnp.tanh(elo_arr / elo_clamp)
    if curvature_weighting:
        if curvature_mode == _ASGO_CURVATURE_ABSOLUTE:
            weights = jnp.abs(clamped) / noise_floor
            denom = jnp.asarray(rounds, dtype=jnp.float32)
        elif curvature_mode == _ASGO_CURVATURE_PER_ITERATION:
            mean_abs = jnp.mean(jnp.abs(clamped))
            weights = jnp.abs(clamped) / (mean_abs + 1e-8)
            denom = jnp.sum(weights) + 1e-8
        else:
            raise ValueError(f"Unknown ASGO curvature mode: {curvature_mode}")
    else:
        weights = jnp.ones((rounds,), dtype=jnp.float32)
        denom = jnp.asarray(rounds, dtype=jnp.float32)

    scale = _compute_scale(rounds)

    def leaf_gradient(*delta_leaves: jax.Array) -> jax.Array:
        result = jnp.zeros_like(delta_leaves[0])
        for idx, delta in enumerate(delta_leaves):
            variance = jnp.mean(jnp.square(delta)) + 1e-12
            result += (
                (weights[idx] / denom)
                * 0.5
                * clamped[idx]
                * delta
                / variance
            )
        return result * scale

    return jax.tree.map(leaf_gradient, *deltas)


def _compute_scale(rounds: int) -> float:
    """Returns multi-round normalization for symmetric perturbation signs."""
    if rounds <= 0:
        raise ValueError("rounds must be positive.")
    total = 0.0
    for k in range(rounds + 1):
        total += (
            math.comb(rounds, k)
            * 0.5**rounds
            * abs(2 * k - rounds)
            / rounds
        )
    if total == 0.0:
        return 1.0
    return round(1.0 / total, 3)
