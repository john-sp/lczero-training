import math

import jax
import jax.numpy as jnp
from flax import nnx
from google.protobuf.message import Message

_ASGO_OPTIMIZER_ADAM = 0
_ASGO_OPTIMIZER_MUON = 1
_ASGO_LR_CONSTANT = 0
_ASGO_LR_COSINE = 1


def asgo_optimizer_step(
    params: nnx.State,
    gradient: nnx.State,
    m: nnx.State,
    v: nnx.State,
    lr: float,
    beta1: float,
    beta2: float,
    beta1_product: float | jax.Array,
    beta2_product: float | jax.Array,
    epsilon: float,
    optimizer: int = _ASGO_OPTIMIZER_ADAM,
    muon_skip_smaller_than: int = 32,
) -> tuple[nnx.State, nnx.State, nnx.State, jax.Array, jax.Array]:
    """Applies one ASGO Adam/Muon update."""
    if lr <= 0:
        raise ValueError("lr must be positive.")
    if not 0.0 <= beta1 < 1.0:
        raise ValueError("beta1 must be in [0, 1).")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError("beta2 must be in [0, 1).")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if muon_skip_smaller_than <= 0:
        raise ValueError("muon_skip_smaller_than must be positive.")

    new_beta1_product = jnp.asarray(beta1_product) * beta1
    new_beta2_product = jnp.asarray(beta2_product) * beta2

    def update_m_leaf(
        param: jax.Array,
        old_m: jax.Array,
        grad: jax.Array,
    ) -> jax.Array:
        if _is_compact_zero_for_param(param, grad):
            return jnp.asarray(0, dtype=param.dtype)
        return beta1 * old_m + (1.0 - beta1) * grad

    def update_v_leaf(
        param: jax.Array,
        old_v: jax.Array,
        grad: jax.Array,
    ) -> jax.Array:
        if _is_compact_zero_for_param(param, grad):
            return jnp.asarray(0, dtype=param.dtype)
        return beta2 * old_v + (1.0 - beta2) * jnp.square(grad)

    new_m = jax.tree.map(update_m_leaf, params, m, gradient)
    new_v = jax.tree.map(update_v_leaf, params, v, gradient)

    def update_leaf(
        param: jax.Array,
        grad: jax.Array,
        m_leaf: jax.Array,
        v_leaf: jax.Array,
    ) -> jax.Array:
        if _is_compact_zero_for_param(param, grad):
            return param
        m_hat = m_leaf / (1.0 - new_beta1_product)
        v_hat = v_leaf / (1.0 - new_beta2_product)
        update = m_hat / (jnp.sqrt(v_hat) + epsilon)
        if _use_muon(update, optimizer, muon_skip_smaller_than):
            update = muon_orthogonalize(update)
        return param + lr * update.astype(param.dtype)

    new_params = jax.tree.map(update_leaf, params, gradient, new_m, new_v)
    return new_params, new_m, new_v, new_beta1_product, new_beta2_product


def anneal_value(schedule: Message, iteration: int) -> float:
    """Evaluates a three-phase ZO-AdaMU annealing schedule."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative.")
    if iteration < schedule.warmup_iters:
        return schedule.warmup_value
    if iteration < schedule.warmup_iters + schedule.steady_iters:
        if schedule.steady_iters <= 0:
            return schedule.steady_value
        progress = (
            iteration - schedule.warmup_iters
        ) / schedule.steady_iters
        t = 0.5 * (1.0 - math.cos(math.pi * progress))
        return schedule.warmup_value + (
            schedule.steady_value - schedule.warmup_value
        ) * t
    return schedule.steady_value


def learning_rate_for_iteration(config: Message, iteration: int) -> float:
    """Evaluates the ASGO learning-rate schedule."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative.")
    base_lr = config.learning_rate
    if (
        config.lr_warmup_iterations > 0
        and iteration < config.lr_warmup_iterations
    ):
        progress = iteration / config.lr_warmup_iterations
        if config.lr_warmup_quadratic:
            progress *= progress
        start_lr = base_lr * config.lr_warmup_start_factor
        return start_lr + (base_lr - start_lr) * progress

    if config.lr_schedule == _ASGO_LR_CONSTANT:
        return base_lr
    if config.lr_schedule != _ASGO_LR_COSINE:
        raise ValueError(f"Unknown ASGO LR schedule: {config.lr_schedule}")
    if iteration <= config.lr_decay_start:
        return base_lr
    if iteration >= config.lr_decay_end:
        return config.lr_end

    progress = (iteration - config.lr_decay_start) / (
        config.lr_decay_end - config.lr_decay_start
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.lr_end + (base_lr - config.lr_end) * cosine


def muon_orthogonalize(update: jax.Array, ns_steps: int = 5) -> jax.Array:
    """Applies Newton-Schulz orthogonalization to a matrix update."""
    if update.ndim != 2:
        return update
    a, b, c = 3.4445, -4.7750, 2.0315
    original_dtype = update.dtype
    original_norm = jnp.linalg.norm(update)
    x = update.astype(jnp.float32) / (original_norm + 1e-8)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(ns_steps):
        xx_t = x @ x.T
        x = a * x + b * xx_t @ x + c * xx_t @ xx_t @ x
    if transposed:
        x = x.T
    x *= original_norm / (jnp.linalg.norm(x) + 1e-8)
    return x.astype(original_dtype)


def _use_muon(
    update: jax.Array, optimizer: int, muon_skip_smaller_than: int
) -> bool:
    return (
        optimizer == _ASGO_OPTIMIZER_MUON
        and update.ndim == 2
        and min(update.shape) >= muon_skip_smaller_than
    )


def _is_compact_zero_for_param(param: jax.Array, grad: jax.Array) -> bool:
    return grad.shape == () and param.shape != ()
