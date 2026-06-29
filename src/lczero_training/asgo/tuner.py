import dataclasses
import datetime
import glob
import gzip
import logging
import os
import signal
import time
from collections.abc import Sequence
from concurrent import futures
from typing import Callable, Protocol

import jax
import jax.numpy as jnp
from flax import nnx

from lczero_training.asgo.activation_collector import (
    activation_cache_position_count,
    collect_activation_bases,
    populate_activation_cache,
    should_refresh_agzo,
)
from lczero_training.asgo.checkpoint import (
    AsgoCheckpointManager,
    AsgoState,
)
from lczero_training.asgo.config import (
    ASGO_DEFAULT_ACTIVATION_CACHE_BATCHES,
    AsgoConfigError,
    hash_config,
    normalize_asgo_config,
    validate_asgo_config,
)
from lczero_training.asgo.gradient import compute_gradient
from lczero_training.asgo.optimizer import (
    anneal_value,
    asgo_optimizer_step,
    learning_rate_for_iteration,
)
from lczero_training.asgo.perturbation import PerturbationManager
from lczero_training.asgo.tournament import TournamentRunner
from lczero_training.convert.jax_to_leela import (
    LeelaExportOptions,
    jax_to_leela,
)
from lczero_training.convert.leela_to_jax import (
    LeelaImportOptions,
    leela_to_jax,
)
from lczero_training.dataloader import DataLoader, make_dataloader
from lczero_training.model.model import LczeroModel
from lczero_training.training.tensorboard import TensorboardLogger
from proto import hlo_pb2, net_pb2
from proto.root_config_pb2 import RootConfig

logger = logging.getLogger(__name__)


class TournamentMetricResult(Protocol):
    wins: int
    draws: int
    losses: int
    npm: float


@dataclasses.dataclass(frozen=True)
class RoundPreparation:
    """Files and delta prepared for one ASGO round."""

    round_idx: int
    subkey: jax.Array
    delta: nnx.State
    pos_path: str
    neg_path: str


class AsgoTuner:
    """Main single-machine ASGO tuning loop."""

    def __init__(
        self,
        config: RootConfig,
        *,
        data_loader_callback: Callable[[DataLoader | None], None] | None = None,
    ) -> None:
        self._data_loader_callback = data_loader_callback
        if not config.HasField("asgo"):
            raise AsgoConfigError("Config must contain an 'asgo' section.")
        self.config = RootConfig()
        self.config.CopyFrom(config)
        self.asgo = normalize_asgo_config(self.config.asgo)
        self.config.asgo.CopyFrom(self.asgo)
        validate_asgo_config(self.asgo)
        if self.asgo.activation_guided and not self.config.HasField(
            "data_loader"
        ):
            raise AsgoConfigError(
                "ASGO activation_guided mode requires root data_loader."
            )

        model = LczeroModel(config=self.config.model, rngs=nnx.Rngs(params=42))
        self.graphdef, empty_model_params = nnx.split(model)
        self._activation_model = LczeroModel(
            config=self.config.model,
            rngs=nnx.Rngs(params=42),
        )
        self._num_heads = self.config.model.encoder.heads

        self.checkpoint_mgr = AsgoCheckpointManager(
            self.asgo.checkpoint_path,
            create=False,
            max_to_keep=self.asgo.max_checkpoints,
        )
        empty_state = self._empty_state(empty_model_params)
        initial_state = self.checkpoint_mgr.restore_latest(empty_state)
        if initial_state is None:
            raise FileNotFoundError(
                f"No ASGO checkpoint found in {self.asgo.checkpoint_path}."
            )
        expected_hash = hash_config(self.asgo)
        if initial_state.config_hash != expected_hash:
            raise AsgoConfigError(
                "ASGO config hash does not match the latest checkpoint. "
                "Restore the original ASGO config or run an explicit "
                "checkpoint migration before resuming."
            )

        self.model_params = initial_state.model_params
        self.m = initial_state.m
        self.v = initial_state.v
        self.beta1_product = initial_state.beta1_product
        self.beta2_product = initial_state.beta2_product
        self.rng = initial_state.rng
        self.iteration = int(initial_state.iteration)

        self.perturbation_mgr = PerturbationManager(
            config=self.asgo,
            model_params=self.model_params,
            rng=self.rng,
            activation_bases=initial_state.activation_bases,
        )
        self.tournament = TournamentRunner(
            config=self.asgo.tournament,
            lc0_path=self.asgo.lc0_path,
            work_dir=self.asgo.checkpoint_path,
        )
        self.summary_writer = self._create_summary_writer()
        self._activation_cache = self._populate_activation_cache()
        self._agzo_cache_positions = activation_cache_position_count(
            self._activation_cache
        )
        self._agzo_cache_batch_size = _activation_cache_batch_size(
            self._activation_cache
        )
        self._last_agzo_refresh_time_s = 0.0
        self._last_agzo_basis_orthonormality_error = 0.0
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def run(self, max_iterations: int | None = None) -> None:
        """Runs the ASGO tuning loop from the current checkpoint."""
        if max_iterations is None:
            max_iterations = (
                self.asgo.max_iterations
                if self.asgo.max_iterations > 0
                else 1_000_000
            )

        try:
            while self.iteration < max_iterations:
                if self._shutdown_requested:
                    break
                self._run_iteration()
        finally:
            self.tournament.cleanup()
            if self.summary_writer is not None:
                self.summary_writer.close()

    def _run_iteration(self) -> None:
        logger.info("=== ASGO iteration %d ===", self.iteration)
        iter_start = time.time()
        lr = learning_rate_for_iteration(self.asgo, self.iteration)
        beta1 = anneal_value(self.asgo.beta1_schedule, self.iteration)
        beta2 = anneal_value(self.asgo.beta2_schedule, self.iteration)
        self._maybe_refresh_agzo()

        deltas = []
        elo_diffs = []
        tournament_results = []
        tournament_time = 0.0
        adaptive_masks = None
        if self.asgo.adaptive_mask:
            adaptive_masks = self.perturbation_mgr.make_adaptive_masks(
                self.v,
                self.asgo.adaptive_mask_exponent,
                self.asgo.adaptive_mask_floor,
            )

        round_keys = []
        for _round_idx in range(self.asgo.rounds_per_iteration):
            self.rng, subkey = jax.random.split(self.rng)
            round_keys.append(subkey)

        with futures.ThreadPoolExecutor(max_workers=1) as export_pool:
            prepare_future = export_pool.submit(
                self._prepare_round,
                0,
                round_keys[0],
                adaptive_masks,
            )
            for round_idx in range(self.asgo.rounds_per_iteration):
                logger.info(
                    "ASGO iteration %d round %d/%d",
                    self.iteration,
                    round_idx + 1,
                    self.asgo.rounds_per_iteration,
                )

                prepared = prepare_future.result()
                if prepared.round_idx != round_idx:
                    raise RuntimeError(
                        "ASGO round preparation completed out of order."
                    )
                deltas.append(prepared.delta)

                next_round_idx = round_idx + 1
                if next_round_idx < self.asgo.rounds_per_iteration:
                    prepare_future = export_pool.submit(
                        self._prepare_round,
                        next_round_idx,
                        round_keys[next_round_idx],
                        adaptive_masks,
                    )

                eval_start = time.time()
                if self.asgo.tournament.HasField("fixed_opponent"):
                    result = self.tournament.evaluate_against_opponents(
                        pos_weights_path=prepared.pos_path,
                        neg_weights_path=prepared.neg_path,
                        rng=prepared.subkey,
                    )
                else:
                    result = self.tournament.evaluate_pair(
                        pos_weights_path=prepared.pos_path,
                        neg_weights_path=prepared.neg_path,
                    )
                tournament_time += time.time() - eval_start
                elo_diffs.append(result.elo_diff)
                tournament_results.append(result)
                logger.info(
                    "ASGO round result: elo=%+.1f WDL=%d/%d/%d",
                    result.elo_diff,
                    result.wins,
                    result.draws,
                    result.losses,
                )

        gradient = compute_gradient(
            deltas=deltas,
            elo_diffs=elo_diffs,
            elo_clamp=self.asgo.elo_clamp,
            rounds=self.asgo.rounds_per_iteration,
            curvature_weighting=self.asgo.curvature_weighting,
            curvature_mode=self.asgo.curvature_mode,
            noise_floor=self.asgo.noise_floor,
        )

        prev_params = self.model_params
        (
            self.model_params,
            self.m,
            self.v,
            self.beta1_product,
            self.beta2_product,
        ) = asgo_optimizer_step(
            params=self.model_params,
            gradient=gradient,
            m=self.m,
            v=self.v,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            beta1_product=self.beta1_product,
            beta2_product=self.beta2_product,
            epsilon=self.asgo.adam_epsilon,
            optimizer=self.asgo.optimizer,
            muon_skip_smaller_than=self.asgo.muon_skip_smaller_than,
        )
        self.perturbation_mgr.model_params = self.model_params

        self._log_metrics(
            elo_diffs=elo_diffs,
            tournament_results=tournament_results,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            gradient=gradient,
            previous_params=prev_params,
            deltas=deltas,
            iteration_seconds=time.time() - iter_start,
            tournament_seconds=tournament_time,
        )
        completed_iteration = self.iteration + 1
        self.checkpoint_mgr.save(self._state_for_next_iteration())
        self._export_current(completed_iteration)
        self._cleanup_temp_files(self.iteration)
        self.iteration = completed_iteration

    def _empty_state(self, model_params: nnx.State) -> AsgoState:
        return AsgoState(
            iteration=0,
            model_params=model_params,
            m=jax.tree.map(jnp.zeros_like, model_params),
            v=jax.tree.map(jnp.zeros_like, model_params),
            beta1_product=jnp.asarray(1.0, dtype=jnp.float32),
            beta2_product=jnp.asarray(1.0, dtype=jnp.float32),
            rng=jax.random.PRNGKey(0),
            activation_bases={},
            config_hash=hash_config(self.asgo),
        )

    def _state_for_next_iteration(self) -> AsgoState:
        return AsgoState(
            iteration=self.iteration + 1,
            model_params=self.model_params,
            m=self.m,
            v=self.v,
            beta1_product=self.beta1_product,
            beta2_product=self.beta2_product,
            rng=self.rng,
            activation_bases=self.perturbation_mgr.activation_bases,
            config_hash=hash_config(self.asgo),
        )

    def _populate_activation_cache(self) -> list[jax.Array]:
        if not self.asgo.activation_guided:
            return []
        if self.asgo.coupling_spec:
            raise NotImplementedError(
                "Coupled AGZO basis refresh needs an explicit mapping from "
                "coupling specs to guided weight paths."
            )
        logger.info(
            "Populating ASGO activation cache with %d batches.",
            ASGO_DEFAULT_ACTIVATION_CACHE_BATCHES,
        )
        dataloader = make_dataloader(self.config.data_loader)
        logger.info(
            "Dataloader created.",
        )
        self._set_active_data_loader(dataloader)
        try:
            batches = populate_activation_cache(
                dataloader,
                n_batches=ASGO_DEFAULT_ACTIVATION_CACHE_BATCHES,
            )
        finally:
            self._set_active_data_loader(None)
            dataloader.stop()
        logger.info(
            "ASGO activation cache contains %d batches and %d positions.",
            len(batches),
            activation_cache_position_count(batches),
        )
        return batches

    def _set_active_data_loader(self, loader: DataLoader | None) -> None:
        if self._data_loader_callback is not None:
            self._data_loader_callback(loader)

    def _maybe_refresh_agzo(self) -> None:
        if not self.asgo.activation_guided:
            return
        if not should_refresh_agzo(self.iteration, self.asgo):
            return
        ranks_by_tap = {
            rank.tap_name: rank.rank for rank in self.asgo.agzo_rank
        }
        logger.info(
            "Refreshing ASGO AGZO bases for %d taps.", len(ranks_by_tap)
        )
        refresh_start = time.time()
        bases = collect_activation_bases(
            model=self._activation_model,
            model_params=self.model_params,
            batches=self._activation_cache,
            ranks_by_tap=ranks_by_tap,
        )
        self.perturbation_mgr.set_bases(bases)
        self._last_agzo_refresh_time_s = time.time() - refresh_start
        self._last_agzo_basis_orthonormality_error = _basis_error(bases)

    def _export_variant(
        self,
        params: nnx.State,
        iteration: int,
        round_idx: int,
        label: str,
    ) -> str:
        path = os.path.join(
            self.tournament.work_dir,
            f"iter{iteration}_r{round_idx}_{label}.pb.gz",
        )
        self._write_network(params, path, training_steps=iteration)
        return path

    def _prepare_round(
        self,
        round_idx: int,
        subkey: jax.Array,
        adaptive_masks: dict[str, jax.Array] | None,
    ) -> RoundPreparation:
        delta = self.perturbation_mgr.generate(
            iteration=self.iteration,
            rng=subkey,
            momentum=self.m,
            adaptive_masks=adaptive_masks,
        )

        params_plus = jax.tree.map(
            lambda param, diff: param + diff, self.model_params, delta
        )
        params_minus = jax.tree.map(
            lambda param, diff: param - diff, self.model_params, delta
        )
        pos_path = self._export_variant(
            params_plus, self.iteration, round_idx, "pos"
        )
        neg_path = self._export_variant(
            params_minus, self.iteration, round_idx, "neg"
        )
        return RoundPreparation(
            round_idx=round_idx,
            subkey=subkey,
            delta=delta,
            pos_path=pos_path,
            neg_path=neg_path,
        )

    def _export_current(self, iteration: int) -> None:
        if not self.config.export.destination_filename:
            return
        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        for template in self.config.export.destination_filename:
            path = template.format(datetime=date_str, step=iteration)
            self._write_network(
                self.model_params,
                path,
                training_steps=iteration,
            )
            logger.info("Exported ASGO model for iteration %d to %s", iteration, path)

    def _export_net(
        self,
        params: nnx.State,
        *,
        training_steps: int,
    ) -> net_pb2.Net:
        options = LeelaExportOptions(
            min_version="0.28",
            num_heads=self._num_heads,
            license=None,
            ffn_activation=self.config.model.defaults.ffn_activation,
            training_steps=training_steps,
        )
        return jax_to_leela(
            jax_weights=params,
            export_options=options,
            model_config=self.config.model,
        )

    def _write_network(
        self,
        params: nnx.State,
        path: str,
        *,
        training_steps: int,
    ) -> None:
        net = self._export_net(params, training_steps=training_steps)
        network_bytes = gzip.compress(net.SerializeToString())
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            f.write(network_bytes)

    def _log_metrics(
        self,
        *,
        elo_diffs: Sequence[float],
        tournament_results: Sequence[TournamentMetricResult],
        lr: float,
        beta1: float,
        beta2: float,
        gradient: nnx.State,
        previous_params: nnx.State,
        deltas: Sequence[nnx.State],
        iteration_seconds: float,
        tournament_seconds: float,
    ) -> None:
        elo_arr = jnp.asarray(elo_diffs, dtype=jnp.float32)
        total_games = sum(
            result.wins + result.draws + result.losses
            for result in tournament_results
        )
        logger.info(
            "ASGO iteration %d: mean elo=%+.1f lr=%g games=%d",
            self.iteration,
            float(jnp.mean(elo_arr)),
            lr,
            total_games,
        )
        if self.summary_writer is None:
            return

        wins = sum(result.wins for result in tournament_results)
        draws = sum(result.draws for result in tournament_results)
        losses = sum(result.losses for result in tournament_results)
        score = (wins + 0.5 * draws) / total_games if total_games else 0.0
        gradient_norm = _tree_norm(gradient)
        update_norm = _tree_delta_norm(self.model_params, previous_params)
        param_norm = _tree_norm(self.model_params)
        perturbation_norm = _mean([_tree_norm(delta) for delta in deltas])
        roundtrip_rms, roundtrip_norm = self._export_roundtrip_metrics(
            self.model_params
        )
        metrics = {
            "asgo/elo_diff_mean": float(jnp.mean(elo_arr)),
            "asgo/elo_diff_std": float(jnp.std(elo_arr)),
            "asgo/elo_diff_max": float(jnp.max(jnp.abs(elo_arr))),
            "asgo/score_mean": score,
            "asgo/wins": wins,
            "asgo/draws": draws,
            "asgo/losses": losses,
            "asgo/npm_mean": _mean(
                [result.npm for result in tournament_results]
            ),
            "asgo/export_roundtrip_rms": roundtrip_rms,
            "asgo/perturbation_to_quantization": _safe_ratio(
                perturbation_norm, roundtrip_norm
            ),
            "asgo/learning_rate": lr,
            "asgo/beta1": beta1,
            "asgo/beta2": beta2,
            "asgo/gradient_norm": gradient_norm,
            "asgo/param_update_norm": update_norm,
            "asgo/perturbation_norm": perturbation_norm,
            "asgo/delta_w_rel": _safe_ratio(update_norm, param_norm),
            "asgo/agzo_cache_positions": self._agzo_cache_positions,
            "asgo/agzo_cache_batch_size": self._agzo_cache_batch_size,
            "asgo/agzo_basis_orthonormality_error": (
                self._last_agzo_basis_orthonormality_error
            ),
            "asgo/agzo_basis_refresh_time_s": (
                self._last_agzo_refresh_time_s
            ),
            "asgo/iteration_time_s": iteration_seconds,
            "asgo/tournament_time_s": tournament_seconds,
        }
        for round_idx, elo_diff in enumerate(elo_diffs):
            metrics[f"asgo/elo_diff_round_{round_idx}"] = elo_diff
        self.summary_writer.log(self.iteration, metrics)
        self.summary_writer.log_text(
            self.iteration,
            "asgo/agzo_sketch_dtype",
            str(jnp.float16),
        )
        self.summary_writer.flush()

    def _export_roundtrip_metrics(
        self, params: nnx.State
    ) -> tuple[float, float]:
        net = self._export_net(params, training_steps=self.iteration)
        roundtrip = leela_to_jax(
            net,
            LeelaImportOptions(
                weights_dtype=hlo_pb2.XlaShapeProto.F32,
                compute_dtype=self.config.model.defaults.compute_dtype,
            ),
        )
        return (
            _tree_delta_rms(params, roundtrip),
            _tree_delta_norm(params, roundtrip),
        )

    def _cleanup_temp_files(self, iteration: int) -> None:
        pattern = os.path.join(
            self.tournament.work_dir, f"iter{iteration}_*.pb.gz"
        )
        for filename in glob.glob(pattern):
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass

    def _create_summary_writer(self) -> TensorboardLogger | None:
        if not (
            self.config.HasField("metrics")
            and self.config.metrics.tensorboard_path
        ):
            return None
        return TensorboardLogger(self.config.metrics.tensorboard_path)

    def _handle_shutdown(self, signum: int, frame: object) -> None:
        del signum, frame
        logger.info("Shutdown requested; finishing current ASGO iteration.")
        self._shutdown_requested = True


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _tree_norm(tree: object) -> float:
    leaves = [_as_array(leaf) for leaf in jax.tree.leaves(tree)]
    arrays = [leaf for leaf in leaves if leaf is not None]
    return _arrays_norm(arrays)


def _tree_delta_rms(new_tree: object, old_tree: object) -> float:
    return _arrays_rms(_tree_delta_arrays(new_tree, old_tree))


def _tree_delta_norm(new_tree: object, old_tree: object) -> float:
    return _arrays_norm(_tree_delta_arrays(new_tree, old_tree))


def _tree_delta_arrays(new_tree: object, old_tree: object) -> list[jax.Array]:
    arrays = []
    for new_leaf, old_leaf in zip(
        jax.tree.leaves(new_tree),
        jax.tree.leaves(old_tree),
        strict=True,
    ):
        new_array = _as_array(new_leaf)
        old_array = _as_array(old_leaf)
        if new_array is not None and old_array is not None:
            arrays.append(new_array - old_array)
    return arrays


def _arrays_norm(arrays: Sequence[jax.Array]) -> float:
    if not arrays:
        return 0.0
    square_sum = sum(float(jnp.sum(jnp.square(array))) for array in arrays)
    return float(square_sum**0.5)


def _arrays_rms(arrays: Sequence[jax.Array]) -> float:
    if not arrays:
        return 0.0
    square_sum = sum(float(jnp.sum(jnp.square(array))) for array in arrays)
    count = sum(array.size for array in arrays)
    return float((square_sum / max(1, count)) ** 0.5)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _basis_error(bases: dict[str, jax.Array]) -> float:
    if not bases:
        return 0.0
    errors = []
    for basis in bases.values():
        basis = basis.astype(jnp.float32)
        identity = jnp.eye(basis.shape[1], dtype=jnp.float32)
        gram = basis.T @ basis
        errors.append(float(jnp.max(jnp.abs(gram - identity))))
    return max(errors)


def _activation_cache_batch_size(batches: Sequence[jax.Array]) -> int:
    if not batches:
        return 0
    first = batches[0]
    return int(first.shape[0]) if first.ndim >= 4 else 1


def _as_array(value: object) -> jax.Array | None:
    if hasattr(value, "value"):
        value = value.value
    if not (hasattr(value, "shape") and hasattr(value, "dtype")):
        return None
    return jnp.asarray(value)
