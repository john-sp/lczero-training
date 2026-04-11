import dataclasses
import logging
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, Generator, Optional, Sequence, Tuple, cast

import jax
import jax.numpy as jnp
import jax.sharding as jshard
import numpy as np
import optax
from flax import nnx
from jax import tree_util
from jax.sharding import PartitionSpec as P

from lczero_training.dataloader import DataLoader
from lczero_training.model.loss_function import LczeroLoss
from lczero_training.model.model import LczeroModel
from lczero_training.training.utils import make_weights_mask
from lczero_training.training.state import (
    JitTrainingState,
    TrainingBatch,
    TrainingSample,
)
from proto import training_config_pb2 as training_config_pb2

MetricsDict = Dict[str, Any]
EPS = 1e-12


@dataclasses.dataclass(frozen=True)
class AdvancedMetricsOptions:
    expensive_metrics_period: int = 100
    enable_weight_decay_ratios: bool = True
    enable_update_to_weight_ratio: bool = True
    enable_auxiliary_loss_ratio: bool = True
    enable_gradient_conflict_ratio: bool = True
    enable_policy_entropy_logit_scale: bool = True
    enable_ln2_collapse_diagnostics: bool = True


def advanced_metrics_options_from_config(
    config: training_config_pb2.TrainingConfig,
) -> AdvancedMetricsOptions:
    if not config.HasField("advanced_metrics"):
        return AdvancedMetricsOptions()
    metrics = config.advanced_metrics
    period = (
        metrics.expensive_metrics_period
        if metrics.expensive_metrics_period > 0
        else 100
    )
    return AdvancedMetricsOptions(
        expensive_metrics_period=period,
        enable_weight_decay_ratios=(
            metrics.enable_weight_decay_ratios
            if metrics.HasField("enable_weight_decay_ratios")
            else True
        ),
        enable_update_to_weight_ratio=(
            metrics.enable_update_to_weight_ratio
            if metrics.HasField("enable_update_to_weight_ratio")
            else True
        ),
        enable_auxiliary_loss_ratio=(
            metrics.enable_auxiliary_loss_ratio
            if metrics.HasField("enable_auxiliary_loss_ratio")
            else True
        ),
        enable_gradient_conflict_ratio=(
            metrics.enable_gradient_conflict_ratio
            if metrics.HasField("enable_gradient_conflict_ratio")
            else True
        ),
        enable_policy_entropy_logit_scale=(
            metrics.enable_policy_entropy_logit_scale
            if metrics.HasField("enable_policy_entropy_logit_scale")
            else True
        ),
        enable_ln2_collapse_diagnostics=(
            metrics.enable_ln2_collapse_diagnostics
            if metrics.HasField("enable_ln2_collapse_diagnostics")
            else True
        ),
    )


def _get_array(value: Any) -> jax.Array:
    if isinstance(value, nnx.Variable):
        return jnp.asarray(value.value)
    return jnp.asarray(value)


def _leaf_l2_norm(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(0.0)
    total_sq = jnp.array(0.0)
    for leaf in leaves:
        array = _get_array(leaf)
        total_sq += jnp.sum(jnp.square(array))
    return jnp.sqrt(total_sq)


def _safe_divide(num: jax.Array, den: jax.Array) -> jax.Array:
    return num / jnp.maximum(den, jnp.array(EPS, dtype=num.dtype))


def _weighted_aux_ratio(
    weighted_losses: Dict[str, jax.Array], family: str, primary: str
) -> jax.Array:
    family_prefix = f"{family}/"
    total = jnp.array(0.0, dtype=jnp.float32)
    primary_value = jnp.array(0.0, dtype=jnp.float32)
    for key, value in weighted_losses.items():
        if not key.startswith(family_prefix):
            continue
        total += jnp.asarray(value, dtype=jnp.float32)
        if key == f"{family}/{primary}":
            primary_value = jnp.asarray(value, dtype=jnp.float32)
    aux = total - primary_value
    return _safe_divide(aux, total)


def _policy_entropy_and_logit_std(policy_logits: jax.Array) -> Tuple[jax.Array, jax.Array]:
    probs = jax.nn.softmax(policy_logits, axis=-1)
    entropy = -jnp.sum(probs * jnp.log(jnp.maximum(probs, EPS)), axis=-1)
    return jnp.mean(entropy), jnp.std(policy_logits)


def _flatten_for_cosine(tree: Any) -> jax.Array:
    leaves = [jnp.ravel(_get_array(leaf)) for leaf in jax.tree_util.tree_leaves(tree)]
    if not leaves:
        return jnp.zeros((0,), dtype=jnp.float32)
    return jnp.concatenate(leaves).astype(jnp.float32)


def _conflict_ratio_and_cosine(
    grad_a: Any,
    grad_b: Any,
) -> Tuple[jax.Array, jax.Array]:
    a = _flatten_for_cosine(grad_a)
    b = _flatten_for_cosine(grad_b)
    dots = a * b
    conflict_ratio = jnp.mean((dots < 0).astype(jnp.float32))
    cosine = _safe_divide(
        jnp.sum(dots),
        jnp.linalg.norm(a) * jnp.linalg.norm(b),
    )
    return conflict_ratio, cosine


def _channel_utilization_index(scales: jax.Array) -> jax.Array:
    magnitudes = jnp.abs(scales)
    probs = magnitudes / jnp.maximum(jnp.sum(magnitudes), EPS)
    entropy = -jnp.sum(probs * jnp.log(jnp.maximum(probs, EPS)))
    return jnp.exp(entropy) / jnp.array(scales.shape[0], dtype=jnp.float32)


@dataclasses.dataclass
class StepHookData:
    """Data passed to the step hook callback during training."""

    global_step: int
    local_step: int
    steps_per_epoch: int
    metrics: MetricsDict
    jit_state: JitTrainingState


StepHook = Callable[[StepHookData], None]

logger = logging.getLogger(__name__)


def from_dataloader(
    loader: DataLoader,
) -> Generator[tuple[np.ndarray, ...], None, None]:
    while True:
        yield loader.get_next()


class Training:
    optimizer_tx: optax.GradientTransformation
    train_step: Callable[
        [optax.GradientTransformation, JitTrainingState, TrainingBatch],
        Tuple[JitTrainingState, MetricsDict],
    ]
    _swa_config: Optional[training_config_pb2.SWAConfig]
    _dp_sharding: Optional[jshard.NamedSharding]
    _advanced_metrics: AdvancedMetricsOptions

    def __init__(
        self,
        optimizer_tx: optax.GradientTransformation,
        graphdef: nnx.GraphDef,
        loss_fn: LczeroLoss,
        optimizer_config: training_config_pb2.OptimizerConfig,
        lr_schedule: Optional[optax.Schedule] = None,
        swa_config: Optional[training_config_pb2.SWAConfig] = None,
        advanced_metrics: Optional[AdvancedMetricsOptions] = None,
    ):
        self.optimizer_tx = optimizer_tx
        self._swa_config = swa_config
        self._dp_sharding = None
        self._optimizer_config = optimizer_config
        self._lr_schedule = lr_schedule
        self._advanced_metrics = advanced_metrics or AdvancedMetricsOptions()
        self._expensive_metrics_period = max(
            self._advanced_metrics.expensive_metrics_period, 1
        )
        self._component_metric_keys: tuple[str, ...] = tuple(
            list(f"policy/{loss.metric_name}" for loss in loss_fn.policy_losses)
            + list(f"value/{loss.metric_name}" for loss in loss_fn.value_losses)
            + list(
                f"movesleft/{loss.metric_name}" for loss in loss_fn.movesleft_losses
            )
            + list(
                f"value_error/{loss.metric_name}"
                for loss in loss_fn.value_error_losses
            )
            + list(
                f"value_categorical/{loss.metric_name}"
                for loss in loss_fn.value_categorical_losses
            )
        )
        self._primary_component_keys: tuple[str, ...] = tuple(
            key
            for key in (
                "policy/vanilla",
                "value/winner",
                "movesleft/main",
            )
            if key in self._component_metric_keys
        )
        self._aux_component_keys: tuple[str, ...] = tuple(
            key
            for key in self._component_metric_keys
            if key not in self._primary_component_keys
        )

        jit_kwargs: Dict[str, Any] = {
            "static_argnames": ("optimizer_tx",),
            "donate_argnames": ("jit_state",),
        }
        if jax.device_count() > 1:
            num_devices = jax.device_count()
            logger.info(
                f"Multi-GPU training enabled: {num_devices} devices detected"
            )
            mesh = jshard.Mesh(jax.devices(), axis_names=("batch",))
            replicated = jshard.NamedSharding(mesh, P())
            dp_sharding = jshard.NamedSharding(mesh, P("batch"))
            self._dp_sharding = dp_sharding

            batch_sharding = TrainingBatch(
                inputs=dp_sharding,
                probabilities=dp_sharding,
                values=dp_sharding,
            )
            in_shardings = (replicated, batch_sharding)
            out_shardings = replicated

            jit_kwargs["in_shardings"] = in_shardings
            jit_kwargs["out_shardings"] = out_shardings

        @partial(jax.jit, **jit_kwargs)
        def _step(
            optimizer_tx: optax.GradientTransformation,
            jit_state: JitTrainingState,
            batch: TrainingBatch,
        ) -> Tuple[JitTrainingState, MetricsDict]:
            model = nnx.merge(graphdef, jit_state.model_state)

            def loss_for_grad(
                model_arg: LczeroModel, sample_arg: TrainingSample
            ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
                return loss_fn(model_arg, sample_arg)

            loss_vfn = jax.vmap(
                loss_for_grad,
                in_axes=(None, 0),  # (model_arg, sample_arg)
                out_axes=0,
            )

            def mean_loss_for_grad(
                model_arg: LczeroModel, batch_arg: TrainingBatch
            ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
                # vmap automatically distributes TrainingBatch over batch dimension,
                # calling loss_for_grad with TrainingSample (single samples).
                per_sample_data_loss, unweighted_losses = loss_vfn(
                    model_arg,
                    batch_arg,  # type: ignore[arg-type]
                )
                mean_loss = jnp.mean(per_sample_data_loss)
                return mean_loss, unweighted_losses

            grad_fn = nnx.value_and_grad(mean_loss_for_grad, has_aux=True)
            (mean_loss, unweighted_losses), mean_grads = grad_fn(model, batch)
            grad_norm = optax.global_norm(mean_grads)

            assert jit_state.opt_state is not None
            updates, new_opt_state = optimizer_tx.update(
                mean_grads, jit_state.opt_state, jit_state.model_state
            )
            new_model_state = optax.apply_updates(
                jit_state.model_state, updates
            )

            new_jit_state = jit_state.replace(
                step=jit_state.step + 1,
                model_state=new_model_state,
                opt_state=new_opt_state,
            )

            mean_unweighted = tree_util.tree_map(jnp.mean, unweighted_losses)
            weighted_losses = loss_fn.weighted_losses(mean_unweighted)
            lr_value = (
                jnp.asarray(self._lr_schedule(jit_state.step))
                if self._lr_schedule is not None
                else jnp.array(0.0, dtype=jnp.float32)
            )
            optimizer_metrics = self._compute_optimizer_metrics(
                jit_state.model_state,
                updates,
                lr_value,
            )
            loss_landscape_metrics = self._compute_loss_landscape_metrics(
                weighted_losses
            )
            policy_metrics = self._compute_policy_metrics(model, batch)
            ln2_metrics = self._compute_ln2_scale_metrics(model)
            metrics: MetricsDict = {
                "loss": mean_loss,
                "unweighted_losses": mean_unweighted,
                "weighted_losses": weighted_losses,
                "grad_norm": grad_norm,
                "optimizer_metrics": optimizer_metrics,
                "loss_landscape": loss_landscape_metrics,
                "policy_vanilla": policy_metrics,
                "ln2_collapse": ln2_metrics,
            }
            return new_jit_state, metrics

        self.train_step = cast(
            Callable[
                [optax.GradientTransformation, JitTrainingState, TrainingBatch],
                Tuple[JitTrainingState, MetricsDict],
            ],
            _step,
        )
        self._expensive_metrics_fn: Optional[
            Callable[[JitTrainingState, TrainingBatch], Dict[str, jax.Array]]
        ] = None
        if self._advanced_metrics.enable_gradient_conflict_ratio or (
            self._advanced_metrics.enable_ln2_collapse_diagnostics
        ):

            @jax.jit
            def _compute_expensive_metrics(
                jit_state: JitTrainingState,
                batch: TrainingBatch,
            ) -> Dict[str, jax.Array]:
                model = nnx.merge(graphdef, jit_state.model_state)
                metrics: Dict[str, jax.Array] = {}

                def loss_for_grad(
                    model_arg: LczeroModel, sample_arg: TrainingSample
                ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
                    return loss_fn(model_arg, sample_arg)

                loss_vfn = jax.vmap(
                    loss_for_grad,
                    in_axes=(None, 0),
                    out_axes=0,
                )

                @jax.checkpoint
                def remat_loss_vfn(
                    model_arg: LczeroModel,
                    batch_arg: TrainingBatch,
                ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
                    return loss_vfn(model_arg, batch_arg)

                def component_loss(
                    model_arg: LczeroModel,
                    batch_arg: TrainingBatch,
                    key: str,
                ) -> jax.Array:
                    _, unweighted = remat_loss_vfn(model_arg, batch_arg)
                    return jnp.mean(unweighted[key])

                def sum_component_loss(
                    model_arg: LczeroModel,
                    batch_arg: TrainingBatch,
                    keys: Sequence[str],
                ) -> jax.Array:
                    _, unweighted = remat_loss_vfn(model_arg, batch_arg)
                    total = jnp.array(0.0, dtype=jnp.float32)
                    for key in keys:
                        total += jnp.mean(unweighted[key])
                    return total

                if self._advanced_metrics.enable_gradient_conflict_ratio:
                    if (
                        "policy/vanilla" in self._component_metric_keys
                        and "value/winner" in self._component_metric_keys
                    ):
                        policy_grads = nnx.grad(
                            lambda m, b: component_loss(
                                m, b, "policy/vanilla"
                            )
                        )(model, batch)
                        value_grads = nnx.grad(
                            lambda m, b: component_loss(m, b, "value/winner")
                        )(model, batch)
                        conflict_ratio, mean_cosine = _conflict_ratio_and_cosine(
                            policy_grads["encoders"],
                            value_grads["encoders"],
                        )
                        metrics[
                            "grad_conflict/policy_vanilla_vs_value_winner/conflict_ratio"
                        ] = conflict_ratio
                        metrics[
                            "grad_conflict/policy_vanilla_vs_value_winner/mean_cosine"
                        ] = mean_cosine

                    if self._primary_component_keys and self._aux_component_keys:
                        primary_grads = nnx.grad(
                            lambda m, b: sum_component_loss(
                                m, b, self._primary_component_keys
                            )
                        )(model, batch)
                        aux_grads = nnx.grad(
                            lambda m, b: sum_component_loss(
                                m, b, self._aux_component_keys
                            )
                        )(model, batch)
                        conflict_ratio, mean_cosine = _conflict_ratio_and_cosine(
                            primary_grads["encoders"],
                            aux_grads["encoders"],
                        )
                        metrics[
                            "grad_conflict/primary_vs_aux/conflict_ratio"
                        ] = conflict_ratio
                        metrics[
                            "grad_conflict/primary_vs_aux/mean_cosine"
                        ] = mean_cosine

                if self._advanced_metrics.enable_ln2_collapse_diagnostics:

                    def encoder_out(sample_inputs: jax.Array) -> jax.Array:
                        x = jnp.transpose(sample_inputs, (1, 2, 0))
                        x = jnp.reshape(x, (64, model._input_channels))
                        x = model.embedding(x)
                        x = model.encoders(x)
                        return jnp.mean(x, axis=0)

                    outputs = jax.vmap(encoder_out)(batch.inputs)
                    centered = outputs - jnp.mean(outputs, axis=0, keepdims=True)
                    cov = jnp.matmul(centered.T, centered) / jnp.maximum(
                        outputs.shape[0] - 1, 1
                    )
                    eigvals = jnp.linalg.eigvalsh(cov)
                    eigvals = jnp.clip(eigvals, a_min=0.0)
                    eig_probs = eigvals / jnp.maximum(jnp.sum(eigvals), EPS)
                    eig_entropy = -jnp.sum(
                        eig_probs * jnp.log(jnp.maximum(eig_probs, EPS))
                    )
                    metrics["ln2_collapse/effective_rank_ratio"] = _safe_divide(
                        jnp.exp(eig_entropy),
                        jnp.array(eigvals.shape[0], dtype=jnp.float32),
                    )

                    final_encoder = model.encoders.encoders.layers[-1]
                    scales = jnp.asarray(final_encoder.ln2.scale.value)
                    active_mask = (jnp.abs(scales) >= 0.1).astype(jnp.float32)
                    channel_energy = jnp.var(outputs, axis=0)
                    active_energy = jnp.sum(channel_energy * active_mask)
                    total_energy = jnp.sum(channel_energy)
                    metrics["ln2_collapse/active_channel_energy_ratio"] = (
                        _safe_divide(active_energy, total_energy)
                    )

                return metrics

            self._expensive_metrics_fn = _compute_expensive_metrics

    def _compute_optimizer_metrics(
        self,
        model_state: nnx.State,
        updates: nnx.State,
        learning_rate: jax.Array,
    ) -> Dict[str, jax.Array]:
        metrics: Dict[str, jax.Array] = {}
        if not self._advanced_metrics.enable_update_to_weight_ratio and (
            not self._advanced_metrics.enable_weight_decay_ratios
        ):
            return metrics

        enc_state = model_state["encoders"]
        enc_update = updates["encoders"]
        weight_norm = _leaf_l2_norm(enc_state)
        update_norm = _leaf_l2_norm(enc_update)

        if self._advanced_metrics.enable_update_to_weight_ratio:
            metrics["encoder_body/update_to_weight_ratio"] = _safe_divide(
                update_norm, weight_norm
            )

        if (
            not self._advanced_metrics.enable_weight_decay_ratios
            or not self._optimizer_config.HasField("nadamw")
        ):
            return metrics

        nadamw = self._optimizer_config.nadamw
        decay_mask = make_weights_mask(nadamw.decay_selector, model_state)
        if self._optimizer_config.HasField("freeze_selector"):
            freeze_mask = make_weights_mask(
                self._optimizer_config.freeze_selector, model_state
            )
            decay_mask = jax.tree.map(
                lambda d, f: d and (not f), decay_mask, freeze_mask
            )
        decay_mask_enc = decay_mask["encoders"]

        wd_update = jax.tree.map(
            lambda p, m: (
                -learning_rate * nadamw.weight_decay * _get_array(p)
                if m
                else jnp.zeros_like(_get_array(p))
            ),
            enc_state,
            decay_mask_enc,
        )
        wd_norm = _leaf_l2_norm(wd_update)
        metrics["encoder_body/weight_decay_ratio_vs_update"] = _safe_divide(
            wd_norm, update_norm
        )
        metrics["encoder_body/weight_decay_ratio_vs_weight"] = _safe_divide(
            wd_norm, weight_norm
        )
        return metrics

    def _compute_loss_landscape_metrics(
        self, weighted_losses: Dict[str, jax.Array]
    ) -> Dict[str, jax.Array]:
        if not self._advanced_metrics.enable_auxiliary_loss_ratio:
            return {}
        return {
            "policy_aux_ratio": _weighted_aux_ratio(
                weighted_losses, "policy", "vanilla"
            ),
            "value_aux_ratio": _weighted_aux_ratio(
                weighted_losses, "value", "winner"
            ),
            "movesleft_aux_ratio": _weighted_aux_ratio(
                weighted_losses, "movesleft", "main"
            ),
        }

    def _compute_policy_metrics(
        self, model: LczeroModel, batch: TrainingBatch
    ) -> Dict[str, jax.Array]:
        if not self._advanced_metrics.enable_policy_entropy_logit_scale:
            return {}
        if "vanilla" not in model.policy_heads:
            return {}

        def vanilla_logits(sample_inputs: jax.Array) -> jax.Array:
            return model(sample_inputs).policy["vanilla"]

        logits = jax.vmap(vanilla_logits)(batch.inputs)
        entropy, logit_std = _policy_entropy_and_logit_std(logits)
        return {
            "entropy": entropy,
            "logit_std": logit_std,
        }

    def _compute_ln2_scale_metrics(
        self, model: LczeroModel
    ) -> Dict[str, jax.Array]:
        if not self._advanced_metrics.enable_ln2_collapse_diagnostics:
            return {}
        final_encoder = model.encoders.encoders.layers[-1]
        scales = jnp.asarray(final_encoder.ln2.scale.value)
        abs_scales = jnp.abs(scales)
        return {
            "active_channel_ratio": jnp.mean((abs_scales >= 0.1).astype(jnp.float32)),
            "near_zero_ratio": jnp.mean((abs_scales < 1e-3).astype(jnp.float32)),
            "channel_utilization_index": _channel_utilization_index(scales),
        }

    @staticmethod
    @jax.jit
    def _swa_tree_map(
        alpha: jax.Array,
        beta: jax.Array,
        swa_state: nnx.State,
        model_state: nnx.State,
    ) -> nnx.State:
        return tree_util.tree_map(
            lambda a, b: alpha * a + beta * b, swa_state, model_state
        )

    def update_swa(
        self, jit_state: JitTrainingState, weight: float
    ) -> JitTrainingState:
        """Update SWA using the provided weight for the current model.

        Assumes `jit_state.swa_state` is initialized and `_swa_config` present.
        """
        logger.info(
            "Updating SWA model, weight=%f, num_averages=%f",
            weight,
            jit_state.num_averages,
        )
        assert self._swa_config is not None
        assert jit_state.swa_state is not None
        assert weight > 0.0
        max_num_averages = self._swa_config.num_averages
        denom = jit_state.num_averages + weight
        alpha = jit_state.num_averages / denom
        beta = weight / denom
        new_swa_state = self._swa_tree_map(
            jnp.array(alpha),
            jnp.array(beta),
            jit_state.swa_state,
            jit_state.model_state,
        )
        new_num_averages = min(
            max_num_averages, jit_state.num_averages + weight
        )
        return jit_state.replace(
            swa_state=new_swa_state, num_averages=new_num_averages
        )

    def maybe_update_swa(
        self,
        jit_state: JitTrainingState,
        steps_completed: int,
        total_steps: int,
    ) -> JitTrainingState:
        """Optionally update SWA based on configured schedule and epoch progress.

        Returns the original jit_state when no update is scheduled.
        """
        if self._swa_config is None:
            return jit_state
        period_steps = self._swa_config.period_steps
        assert period_steps > 0
        if steps_completed % period_steps == 0:
            return self.update_swa(jit_state, 1.0)
        if steps_completed == total_steps:
            remainder = total_steps % period_steps
            return self.update_swa(jit_state, remainder / period_steps)
        return jit_state

    def _validate_and_prepare_batch(
        self, tensor_tuple: tuple[np.ndarray, ...]
    ) -> TrainingBatch:
        logger.info("Fetched batch from dataloader")

        # Convert tuple to TrainingBatch
        batch = TrainingBatch.from_tuple(tensor_tuple)

        # Ensure batch.inputs is jax.Array for shape access
        assert isinstance(batch.inputs, jax.Array)
        batch_size = batch.inputs.shape[0]
        if self._dp_sharding is not None:
            num_devices = jax.device_count()
            if batch_size % num_devices != 0:
                raise ValueError(
                    f"Batch size {batch_size} must be divisible by device "
                    f"count {num_devices} for multi-GPU training. "
                    f"Per-device batch size would be "
                    f"{batch_size / num_devices:.2f}"
                )
            per_device_batch_size = batch_size // num_devices
            logger.info(
                f"Multi-GPU batch: {batch_size} total "
                f"({per_device_batch_size} per device)"
            )

        if self._dp_sharding is not None:
            batch = jax.device_put(batch, self._dp_sharding)

        return batch

    def _log_step_metrics(
        self,
        step_value: int,
        local_step: int,
        num_steps: int,
        metrics: MetricsDict,
    ) -> None:
        loss = float(metrics["loss"])
        unweighted_losses = {
            k: float(v) for k, v in metrics["unweighted_losses"].items()
        }
        grad_norm = float(metrics["grad_norm"])
        logger.info(
            f"Step {step_value} ({local_step}/{num_steps}), Loss: {loss}, "
            f"Unweighted losses: {unweighted_losses}, Grad norm: {grad_norm}"
        )

    def _execute_step_hook(
        self,
        step_hook: Optional[StepHook],
        step_value: int,
        local_step: int,
        num_steps: int,
        metrics: MetricsDict,
        jit_state: JitTrainingState,
    ) -> None:
        if step_hook is None:
            return
        hook_data = StepHookData(
            global_step=step_value,
            local_step=local_step,
            steps_per_epoch=num_steps,
            metrics=metrics,
            jit_state=jit_state,
        )
        step_hook(hook_data)

    def run(
        self,
        jit_state: JitTrainingState,
        datagen: Generator[tuple[np.ndarray, ...], None, None],
        num_steps: int,
        step_hook: Optional[StepHook] = None,
        memory_profile_dir: Optional[str] = None,
    ) -> JitTrainingState:
        assert jit_state.opt_state is not None
        if self._dp_sharding is not None:
            replicated = jshard.NamedSharding(self._dp_sharding.mesh, P())
            jit_state = jax.device_put(jit_state, replicated)
        for local_step in range(num_steps):
            logger.info(f"Starting step {jit_state.step}")
            if memory_profile_dir is not None:
                jax.profiler.save_device_memory_profile(
                    f"{memory_profile_dir}/"
                    f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    f"_before_{int(jit_state.step)}.prof"
                )
            batch = self._validate_and_prepare_batch(next(datagen))
            jit_state, metrics = self.train_step(
                self.optimizer_tx, jit_state, batch
            )
            step_value = int(
                np.asarray(jax.device_get(jit_state.step)).reshape(())
            )
            should_run_expensive = (
                self._expensive_metrics_fn is not None
                and step_value % self._expensive_metrics_period == 0
            )
            if should_run_expensive:
                expensive_metrics = self._expensive_metrics_fn(jit_state, batch)
                metrics["expensive_metrics"] = expensive_metrics
            jit_state = self.maybe_update_swa(
                jit_state, local_step + 1, num_steps
            )
            self._execute_step_hook(
                step_hook, step_value, local_step, num_steps, metrics, jit_state
            )
            self._log_step_metrics(step_value, local_step, num_steps, metrics)
        return jit_state
