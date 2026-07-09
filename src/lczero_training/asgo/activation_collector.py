from collections.abc import Iterable, Iterator, Mapping
import logging
from typing import Protocol, cast

import jax
import jax.numpy as jnp
from flax import nnx

from lczero_training.asgo.config import ASGO_DEFAULT_ACTIVATION_CACHE_BATCHES
from lczero_training.asgo.subspace import (
    ActivationSketch,
    activation_matrix,
)
from lczero_training.model.model import LczeroModel
from lczero_training.model.utils import get_dtype

logger = logging.getLogger(__name__)


class _DataLoaderLike(Protocol):
    def get_next(self) -> tuple[jax.Array, ...]: ...


class ActivationAccumulator:
    """Collects activations emitted by model tap points."""

    def __init__(self, tap_points: Iterable[str]) -> None:
        self.tap_points = {_canonical_tap_name(tap) for tap in tap_points}
        self.activations: dict[str, list[jax.Array]] = {
            tap: [] for tap in self.tap_points
        }

    def __call__(self, tap_name: str, activation: jax.Array) -> None:
        canonical_name = _canonical_tap_name(tap_name)
        if canonical_name in self.tap_points:
            self.activations[canonical_name].append(
                activation_matrix(activation).astype(jnp.float32)
            )

    def matrices(self) -> dict[str, jax.Array]:
        result = {}
        for tap_name, chunks in self.activations.items():
            if chunks:
                result[tap_name] = jnp.concatenate(chunks, axis=1)
        return result


class ActivationSketchSink:
    """Streams emitted activations into per-tap sketches."""

    def __init__(self, sketches: Mapping[str, ActivationSketch]) -> None:
        self.sketches = {
            _canonical_tap_name(tap): sketch for tap, sketch in sketches.items()
        }

    def __call__(self, tap_name: str, activation: jax.Array) -> None:
        canonical_name = _canonical_tap_name(tap_name)
        sketch = self.sketches.get(canonical_name)
        if sketch is not None:
            sketch.update(activation)


def collect_activations(
    model: LczeroModel,
    model_params: nnx.State,
    batches: list[jax.Array],
    tap_points: list[str],
) -> dict[str, jax.Array]:
    """Collects activation matrices for requested tap points.

    This direct collection path is intended for tests and small debug runs.
    Production AGZO refreshes should prefer collect_activation_bases(), which
    streams activations through compact sketches.
    """
    accumulator = ActivationAccumulator(tap_points)
    _run_model_over_cached_inputs(model, model_params, batches, accumulator)
    return accumulator.matrices()


def collect_activation_bases(
    model: LczeroModel,
    model_params: nnx.State,
    batches: list[jax.Array],
    ranks_by_tap: Mapping[str, int],
) -> dict[str, jax.Array]:
    """Streams activations and returns one basis per requested tap."""
    sketches: dict[str, ActivationSketch] = {}
    for tap_name, rank in ranks_by_tap.items():
        if rank <= 0:
            raise ValueError(f"Rank for {tap_name} must be positive.")
        sketches[_canonical_tap_name(tap_name)] = ActivationSketch(
            d_in=_expected_d_in(model, tap_name),
            rank=rank,
        )
    _update_activation_sketches_over_cached_inputs(
        model,
        model_params,
        batches,
        sketches,
    )
    return {tap_name: sketch.basis() for tap_name, sketch in sketches.items()}


def populate_activation_cache(
    dataloader: Iterable[tuple[jax.Array, ...]] | _DataLoaderLike,
    n_batches: int = ASGO_DEFAULT_ACTIVATION_CACHE_BATCHES,
) -> list[jax.Array]:
    """Extracts model input tensors from dataloader batches."""
    if n_batches <= 0:
        raise ValueError("n_batches must be positive.")
    batches = []
    for idx, batch in enumerate(_iter_dataloader_batches(dataloader)):
        logger.info(
            "Populating activation cache: batch %d/%d", idx + 1, n_batches
        )
        if idx >= n_batches:
            break
        if len(batch) not in (3, 5):
            raise ValueError(
                f"Expected tuple of 3 or 5 tensors, got {len(batch)}."
            )
        batches.append(jnp.asarray(batch[0]))
    return batches


def _iter_dataloader_batches(
    dataloader: Iterable[tuple[jax.Array, ...]] | _DataLoaderLike,
) -> Iterator[tuple[jax.Array, ...]]:
    logger.info("Iterating over dataloader batches.")
    if hasattr(dataloader, "get_next"):
        loader = cast(_DataLoaderLike, dataloader)
        while True:
            yield loader.get_next()
    else:
        yield from cast(Iterable[tuple[jax.Array, ...]], dataloader)


def should_refresh_agzo(iteration: int, config: object) -> bool:
    """Returns whether AGZO bases should be recomputed this iteration."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative.")
    if iteration == 0:
        return True
    if iteration < config.agzo_phase1_iterations:
        return False
    if config.agzo_refresh_interval <= 0:
        return False
    return iteration % config.agzo_refresh_interval == 0


def activation_cache_position_count(batches: Iterable[jax.Array]) -> int:
    """Returns the number of positions represented by cached input batches."""
    total = 0
    for batch in batches:
        total += int(batch.shape[0]) if batch.ndim >= 4 else 1
    return total


def _run_model_over_cached_inputs(
    model: LczeroModel,
    model_params: nnx.State,
    batches: list[jax.Array],
    activation_sink: object,
) -> None:
    graphdef, _ = nnx.split(model)
    restored_model = nnx.merge(graphdef, model_params)
    for batch in batches:
        inputs = jnp.asarray(batch)
        if inputs.ndim == 3:
            restored_model(inputs, activation_sink=activation_sink)
        elif inputs.ndim == 4:
            for sample_idx in range(inputs.shape[0]):
                restored_model(
                    inputs[sample_idx], activation_sink=activation_sink
                )
        else:
            raise ValueError(
                "Activation cache inputs must have shape "
                "[112, 8, 8] or [batch, 112, 8, 8]."
            )


def _update_activation_sketches_over_cached_inputs(
    model: LczeroModel,
    model_params: nnx.State,
    batches: list[jax.Array],
    sketches: Mapping[str, ActivationSketch],
) -> None:
    graphdef, _ = nnx.split(model)
    restored_model = nnx.merge(graphdef, model_params)
    tap_names = tuple(sketches.keys())

    @jax.jit
    def batch_covariances(
        inputs: jax.Array,
    ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        activations_by_tap = jax.vmap(
            lambda sample: _encoder_tap_activations(
                restored_model,
                sample,
                tap_names,
            )
        )(inputs)
        covariances = []
        sample_counts = []
        for activations in activations_by_tap:
            h_matrix = activation_matrix(activations).astype(jnp.float32)
            covariances.append(h_matrix @ h_matrix.T)
            sample_counts.append(
                jnp.asarray(h_matrix.shape[1], dtype=jnp.int32)
            )
        return tuple(covariances), tuple(sample_counts)

    for batch in batches:
        inputs = jnp.asarray(batch)
        if inputs.ndim == 3:
            inputs = inputs[None, ...]
        elif inputs.ndim != 4:
            raise ValueError(
                "Activation cache inputs must have shape "
                "[112, 8, 8] or [batch, 112, 8, 8]."
            )

        covariances, sample_counts = batch_covariances(inputs)
        for tap_name, covariance, sample_count in zip(
            tap_names,
            covariances,
            sample_counts,
            strict=True,
        ):
            sketches[tap_name].update_covariance(
                covariance,
                int(sample_count),
            )


def _encoder_tap_activations(
    model: LczeroModel,
    sample: jax.Array,
    tap_names: tuple[str, ...],
) -> tuple[jax.Array, ...]:
    tap_set = frozenset(tap_names)
    activations = {}

    def sink(tap_name: str, activation: jax.Array) -> None:
        canonical_name = _canonical_tap_name(tap_name)
        if canonical_name in tap_set:
            activations[canonical_name] = activation

    x = jnp.astype(sample, get_dtype(model.config.defaults.compute_dtype))
    x = jnp.transpose(x, (1, 2, 0))
    x = jnp.reshape(x, (64, model._input_channels))
    x = model.embedding(x)
    model.encoders(x, activation_sink=sink)
    return tuple(activations[tap_name] for tap_name in tap_names)


def _expected_d_in(model: LczeroModel, tap_name: str) -> int:
    tap_name = _canonical_tap_name(tap_name)
    if "/ffn/linear2/kernel" in tap_name:
        layer_idx = _layer_index(tap_name)
        layer_config = model.encoders.layer_configs[layer_idx]
        return layer_config.dff
    if "/mha/output_dense/kernel" in tap_name:
        return model.config.encoder.d_model
    if "/mha/smolgen/dense1/kernel" in tap_name:
        layer_idx = _layer_index(tap_name)
        smolgen = model.encoders.layer_configs[layer_idx].smolgen
        if smolgen is None:
            raise ValueError(f"Tap {tap_name} requires smolgen config.")
        return smolgen.hidden_channels * 64
    if "/mha/smolgen/dense2/kernel" in tap_name:
        layer_idx = _layer_index(tap_name)
        smolgen = model.encoders.layer_configs[layer_idx].smolgen
        if smolgen is None:
            raise ValueError(f"Tap {tap_name} requires smolgen config.")
        return smolgen.hidden_size
    return model.config.embedding.embedding_size


def _layer_index(tap_name: str) -> int:
    parts = _canonical_tap_name(tap_name).split("/")
    if len(parts) < 3 or parts[0] != "encoders" or parts[1] != "layers":
        raise ValueError(f"Unsupported ASGO tap name: {tap_name}")
    return int(parts[2])


def _canonical_tap_name(tap_name: str) -> str:
    parts = tap_name.split("/")
    if (
        len(parts) >= 4
        and parts[0] == "encoders"
        and parts[1] == "encoders"
        and parts[2] == "layers"
    ):
        return "/".join([parts[0], *parts[2:]])
    if len(parts) >= 3 and parts[0] == "encoders" and parts[1].isdigit():
        parts.insert(1, "layers")
        return "/".join(parts)
    return tap_name
