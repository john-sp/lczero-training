import logging
import os
import sys
from typing import Optional

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx, struct
from google.protobuf import text_format

from lczero_training.asgo.config import hash_config, normalize_asgo_config
from lczero_training.asgo.perturbation import selected_zero_state
from lczero_training.training.init import _load_lc0_model_state
from proto.root_config_pb2 import RootConfig

logger = logging.getLogger(__name__)


@struct.dataclass
class AsgoState:
    """Full ASGO tuner state checkpointed each iteration."""

    iteration: int
    model_params: nnx.State
    m: nnx.State
    v: nnx.State
    beta1_product: jax.Array
    beta2_product: jax.Array
    rng: jax.Array
    activation_bases: dict[str, jax.Array]
    config_hash: str


class AsgoCheckpointManager:
    """Small wrapper around Orbax for ASGO checkpoints."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        create: bool = True,
        max_to_keep: Optional[int] = None,
    ) -> None:
        options_kwargs: dict[str, object] = {"create": create}
        if max_to_keep is not None and max_to_keep > 0:
            options_kwargs["max_to_keep"] = max_to_keep
        self._manager = ocp.CheckpointManager(
            checkpoint_path,
            options=ocp.CheckpointManagerOptions(**options_kwargs),
        )

    def latest_step(self) -> Optional[int]:
        return self._manager.latest_step()

    def save(self, state: AsgoState) -> None:
        step = int(state.iteration)
        if step in self._manager.all_steps():
            logger.info(
                "Deleting existing ASGO checkpoint at iteration %d", step
            )
            self._manager.delete(step)
            self._manager.wait_until_finished()

        self._manager.save(step=step, args=ocp.args.PyTreeSave(state))
        self._manager.wait_until_finished()

    def restore_latest(self, empty_state: AsgoState) -> Optional[AsgoState]:
        step = self.latest_step()
        if step is None:
            return None
        restored = self._manager.restore(
            step, args=ocp.args.PyTreeRestore(empty_state)
        )
        assert restored is None or isinstance(restored, AsgoState)
        return restored


def asgo_init(
    config_filename: str,
    lczero_model: str,
    seed: int = 42,
    overwrite: bool = False,
    dry_run: bool = False,
    ignore_config_mismatch: bool = False,
) -> None:
    """Initialize an ASGO tuning checkpoint from an lc0 model."""
    config = RootConfig()
    logger.info("Reading configuration from proto file")
    with open(config_filename, "r") as f:
        text_format.Parse(f.read(), config)

    if not config.HasField("asgo"):
        logger.error("Config must contain an 'asgo' section.")
        sys.exit(1)

    asgo_config = normalize_asgo_config(config.asgo)
    checkpoint_path = asgo_config.checkpoint_path
    if not checkpoint_path:
        logger.error("asgo.checkpoint_path must be set.")
        sys.exit(1)

    checkpoint_exists = os.path.exists(checkpoint_path)
    if not dry_run and checkpoint_exists and not overwrite:
        logger.error("Checkpoint path %s already exists.", checkpoint_path)
        sys.exit(1)

    logger.info("Loading lczero model: %s", lczero_model)
    model_params, _training_steps = _load_lc0_model_state(
        lczero_model,
        config.model,
        config.model.defaults.compute_dtype,
        ignore_config_mismatch,
    )

    m = selected_zero_state(model_params, asgo_config.perturb_selector)
    v = selected_zero_state(model_params, asgo_config.perturb_selector)
    state = AsgoState(
        iteration=0,
        model_params=model_params,
        m=m,
        v=v,
        beta1_product=jnp.asarray(1.0, dtype=jnp.float32),
        beta2_product=jnp.asarray(1.0, dtype=jnp.float32),
        rng=jax.random.PRNGKey(seed),
        activation_bases={},
        config_hash=hash_config(asgo_config),
    )

    if dry_run:
        logger.info("Would save ASGO checkpoint to %s", checkpoint_path)
        logger.info(
            "Model params: %s", jax.tree.map(_shape_or_type_name, model_params)
        )
        logger.info("Optimizer m: %s", jax.tree.map(_shape_or_type_name, m))
        logger.info("Optimizer v: %s", jax.tree.map(_shape_or_type_name, v))
        return

    checkpoint_mgr = AsgoCheckpointManager(
        checkpoint_path, max_to_keep=asgo_config.max_checkpoints
    )
    checkpoint_mgr.save(state)
    logger.info("ASGO checkpoint initialized at %s", checkpoint_path)


def _shape_or_type_name(value: object) -> object:
    return getattr(value, "shape", type(value).__name__)
