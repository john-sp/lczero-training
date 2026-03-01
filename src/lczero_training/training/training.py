import dataclasses
import logging
from functools import partial
from typing import Any, Callable, Dict, Generator, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import jax.sharding as jshard
import numpy as np
import optax
from flax import nnx
from jax import tree_util
from jax.ad_checkpoint import checkpoint as jax_checkpoint
from jax.sharding import PartitionSpec as P

from lczero_training.dataloader import DataLoader
from lczero_training.model.loss_function import LczeroLoss
from lczero_training.model.model import LczeroModel
from lczero_training.training.state import (
    JitTrainingState,
    TrainingBatch,
    TrainingSample,
)
from proto import training_config_pb2 as training_config_pb2

MetricsDict = Dict[str, Any]


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
        [
            optax.GradientTransformation,
            JitTrainingState,
            TrainingBatch,
            Optional[nnx.State],
        ],
        Tuple[JitTrainingState, MetricsDict],
    ]
    _swa_config: Optional[training_config_pb2.SWAConfig]
    _dp_sharding: Optional[jshard.NamedSharding]

    def __init__(
        self,
        optimizer_tx: optax.GradientTransformation,
        graphdef: nnx.GraphDef,
        loss_fn: LczeroLoss,
        swa_config: Optional[training_config_pb2.SWAConfig] = None,
        teacher_graphdef: Optional[nnx.GraphDef] = None,
        component_grad_norm_period: int = 0,
    ):
        self.optimizer_tx = optimizer_tx
        self._swa_config = swa_config
        self._dp_sharding = None
        self._component_grad_norm_period = component_grad_norm_period

        jit_kwargs: Dict[str, Any] = {"static_argnames": ("optimizer_tx",)}
        norms_jit_kwargs: Dict[str, Any] = {}
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
            # jit_state, batch, teacher_model_state
            # (optimizer_tx is static, so excluded from in_shardings.)
            in_shardings = (replicated, batch_sharding, replicated)
            out_shardings = replicated

            jit_kwargs["in_shardings"] = in_shardings
            jit_kwargs["out_shardings"] = out_shardings

            # Norms function: jit_state, batch, teacher_model_state.
            norms_jit_kwargs["in_shardings"] = (
                replicated,
                batch_sharding,
                replicated,
            )
            norms_jit_kwargs["out_shardings"] = replicated

        # Shared helper: per-sample loss used by both the train step and
        # the optional component-norm computation.
        def loss_for_grad(
            model_arg: LczeroModel,
            sample_arg: TrainingSample,
            teacher_model_arg: Optional[LczeroModel] = None,
        ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
            return loss_fn(model_arg, sample_arg, teacher_model_arg)

        loss_vfn = jax.vmap(
            loss_for_grad,
            in_axes=(None, 0, None),
            out_axes=0,
        )

        @partial(jax.jit, **jit_kwargs)
        def _step(
            optimizer_tx: optax.GradientTransformation,
            jit_state: JitTrainingState,
            batch: TrainingBatch,
            teacher_model_state: Optional[nnx.State] = None,
        ) -> Tuple[JitTrainingState, MetricsDict]:
            model = nnx.merge(graphdef, jit_state.model_state)
            teacher_model = (
                nnx.merge(teacher_graphdef, teacher_model_state)
                if teacher_graphdef is not None
                and teacher_model_state is not None
                else None
            )

            def mean_loss_for_grad(
                model_arg: LczeroModel,
                batch_arg: TrainingBatch,
                teacher_model_arg: Optional[LczeroModel] = None,
            ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
                # vmap distributes TrainingBatch over batch dimension,
                # calling loss_for_grad with TrainingSample (single
                # samples).
                per_sample_data_loss, unweighted_losses = loss_vfn(
                    model_arg,
                    batch_arg,
                    teacher_model_arg,
                )
                mean_loss = jnp.mean(per_sample_data_loss)
                return mean_loss, unweighted_losses

            grad_fn = nnx.value_and_grad(mean_loss_for_grad, has_aux=True)
            (mean_loss, unweighted_losses), mean_grads = grad_fn(
                model, batch, teacher_model
            )
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
            metrics: MetricsDict = {
                "loss": mean_loss,
                "unweighted_losses": mean_unweighted,
                "grad_norm": grad_norm,
            }
            return new_jit_state, metrics

        self.train_step = cast(
            Callable[
                [
                    optax.GradientTransformation,
                    JitTrainingState,
                    TrainingBatch,
                    Optional[nnx.State],
                ],
                Tuple[JitTrainingState, MetricsDict],
            ],
            _step,
        )

        # Build the optional component-norm JIT function.
        component_keys: Tuple[str, ...] = ()
        for pl in loss_fn.policy_losses:
            component_keys += (f"policy/{pl.metric_name}",)
        for vl in loss_fn.value_losses:
            component_keys += (f"value/{vl.metric_name}",)
        for ml in loss_fn.movesleft_losses:
            component_keys += (f"movesleft/{ml.metric_name}",)
        for vel in loss_fn.value_error_losses:
            component_keys += (f"value_error/{vel.metric_name}",)
        for vcl in loss_fn.value_categorical_losses:
            component_keys += (f"value_categorical/{vcl.metric_name}",)

        if component_grad_norm_period > 0 and component_keys:

            @partial(jax.jit, **norms_jit_kwargs)
            def _compute_component_norms(
                jit_state: JitTrainingState,
                batch: TrainingBatch,
                teacher_model_state: Optional[nnx.State] = None,
            ) -> Dict[str, jax.Array]:
                model = nnx.merge(graphdef, jit_state.model_state)
                teacher_model = (
                    nnx.merge(teacher_graphdef, teacher_model_state)
                    if teacher_graphdef is not None
                    and teacher_model_state is not None
                    else None
                )

                # Wrap forward pass in jax.checkpoint (remat) to
                # recompute activations during backward instead of
                # storing them.  This minimizes peak VRAM at the
                # cost of extra forward compute.
                @jax_checkpoint
                def remat_loss_vfn(
                    model_arg: LczeroModel,
                    batch_arg: TrainingBatch,
                    teacher_model_arg: Optional[LczeroModel],
                ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
                    return loss_vfn(model_arg, batch_arg, teacher_model_arg)

                norms: Dict[str, jax.Array] = {}
                for key in component_keys:

                    def single_loss_fn(
                        model_arg: LczeroModel,
                        batch_arg: TrainingBatch,
                        teacher_model_arg: Optional[LczeroModel] = None,
                        _key: str = key,
                    ) -> jax.Array:
                        _, unweighted = remat_loss_vfn(
                            model_arg,
                            batch_arg,
                            teacher_model_arg,
                        )
                        return jnp.mean(unweighted[_key])

                    grads = nnx.grad(single_loss_fn)(
                        model, batch, teacher_model
                    )
                    norms[key] = optax.global_norm(grads)
                return norms

            self._component_norms_fn: Optional[
                Callable[
                    [
                        JitTrainingState,
                        TrainingBatch,
                        Optional[nnx.State],
                    ],
                    Dict[str, jax.Array],
                ]
            ] = _compute_component_norms
        else:
            self._component_norms_fn = None

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
        new_swa_state = tree_util.tree_map(
            lambda a, b: alpha * a + beta * b,
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
        teacher_model_state: Optional[nnx.State] = None,
    ) -> JitTrainingState:
        assert jit_state.opt_state is not None
        if self._dp_sharding is not None:
            replicated = jshard.NamedSharding(self._dp_sharding.mesh, P())
            jit_state = jax.device_put(jit_state, replicated)
        for local_step in range(num_steps):
            logger.info(f"Starting step {jit_state.step}")
            batch = self._validate_and_prepare_batch(next(datagen))

            # Compute per-component gradient norms on periodic steps
            # using a separate JIT function to avoid extra VRAM on
            # normal steps.
            component_norms = None
            if (
                self._component_norms_fn is not None
                and self._component_grad_norm_period > 0
                and local_step % self._component_grad_norm_period == 0
            ):
                component_norms = self._component_norms_fn(
                    jit_state, batch, teacher_model_state
                )

            jit_state, metrics = self.train_step(
                self.optimizer_tx, jit_state, batch, teacher_model_state
            )

            if component_norms is not None:
                metrics["component_norms"] = component_norms

            step_value = int(
                np.asarray(jax.device_get(jit_state.step)).reshape(())
            )
            jit_state = self.maybe_update_swa(
                jit_state, local_step + 1, num_steps
            )
            self._execute_step_hook(
                step_hook, step_value, local_step, num_steps, metrics, jit_state
            )
            self._log_step_metrics(step_value, local_step, num_steps, metrics)
        return jit_state
