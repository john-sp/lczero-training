import dataclasses
import logging
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import jax.sharding as jshard
import numpy as np
import optax
from flax import nnx
from flax.struct import dataclass

from lczero_training.model.model import LczeroModel
from lczero_training.training.lr_schedule import make_lr_schedule
from lczero_training.training.optimizer import (
    make_gradient_transformation,
    update_optimizer_step,
)
from proto.model_config_pb2 import ModelConfig
from proto.training_config_pb2 import TrainingConfig

logger = logging.getLogger(__name__)


# Number of value target rows in the values tensor.
NUM_VALUE_TYPES = 7

# Sentinel used in aux_indices for "no such move" (e.g. lookahead past the
# end of the game). Matches the uint16 sentinel in V7TrainingData.
INVALID_MOVE_INDEX = 65535


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TrainingSample:
    """Single training sample without batch dimension.

    Used for vmap over individual samples in loss computation.

    Fields:
        inputs: Input planes tensor [112, 8, 8]
        probabilities: Policy probabilities tensor [1858]
        values: Combined values tensor [7, 3] where:
            - Index 0: result [result_q, result_d, plies_left]
            - Index 1: best [best_q, best_d, best_m]
            - Index 2: played [played_q, played_d, played_m]
            - Index 3: orig [orig_q, orig_d, orig_m] (may contain NaN)
            - Index 4: root [root_q, root_d, root_m]
            - Index 5: st [q_st, d_st, NaN]
            - Index 6: st_censored [q_st_censored, d_st_censored, NaN]
        aux_indices: int32 move indices tensor [3]:
            [opp_played_idx, next_played_idx, played_idx];
            INVALID_MOVE_INDEX (65535) = no such move.
        aux_targets: float32 auxiliary targets tensor [2]:
            [provenance, played-move child-Q (NaN = none)].
    """

    inputs: jax.Array
    probabilities: jax.Array
    values: jax.Array
    aux_indices: Optional[jax.Array] = None
    aux_targets: Optional[jax.Array] = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TrainingBatch:
    """Batch of training data with inputs, probabilities, and values tensors.

    Fields:
        inputs: Input planes tensor [batch, 112, 8, 8]
        probabilities: Policy probabilities tensor [batch, 1858]
        values: Combined values tensor [batch, 7, 3] where:
            - Index 0: result [result_q, result_d, plies_left]
            - Index 1: best [best_q, best_d, best_m]
            - Index 2: played [played_q, played_d, played_m]
            - Index 3: orig [orig_q, orig_d, orig_m] (may contain NaN)
            - Index 4: root [root_q, root_d, root_m]
            - Index 5: st [q_st, d_st, NaN]
            - Index 6: st_censored [q_st_censored, d_st_censored, NaN]
        aux_indices: int32 move indices tensor [batch, 3]:
            [opp_played_idx, next_played_idx, played_idx];
            INVALID_MOVE_INDEX (65535) = no such move.
        aux_targets: float32 auxiliary targets tensor [batch, 2]:
            [provenance, played-move child-Q (NaN = none)].
    """

    inputs: Union[jax.Array, jshard.NamedSharding]
    probabilities: Union[jax.Array, jshard.NamedSharding]
    values: Union[jax.Array, jshard.NamedSharding]
    aux_indices: Optional[Union[jax.Array, jshard.NamedSharding]] = None
    aux_targets: Optional[Union[jax.Array, jshard.NamedSharding]] = None

    @classmethod
    def from_tuple(
        cls, tensor_tuple: tuple[np.ndarray, ...]
    ) -> "TrainingBatch":
        """Create TrainingBatch from tuple returned by DataLoader.

        The DataLoader emits 5 tensors: (inputs, probabilities, values,
        aux_indices, aux_targets). Legacy 3-tensor tuples (e.g. batches
        cached before the aux tensors existed) are still accepted; the aux
        tensors are then None and aux losses cannot be used.
        """
        if len(tensor_tuple) not in (3, 5):
            raise ValueError(
                f"Expected tuple of 5 (or legacy 3) tensors, "
                f"got {len(tensor_tuple)}"
            )
        return cls(
            inputs=jnp.asarray(tensor_tuple[0]),
            probabilities=jnp.asarray(tensor_tuple[1]),
            values=jnp.asarray(tensor_tuple[2]),
            aux_indices=(
                jnp.asarray(tensor_tuple[3]) if len(tensor_tuple) > 3 else None
            ),
            aux_targets=(
                jnp.asarray(tensor_tuple[4]) if len(tensor_tuple) > 4 else None
            ),
        )


@dataclass
class JitTrainingState:
    step: int
    model_state: nnx.State
    opt_state: Optional[optax.OptState]
    # SWA state mirrors model_state structure when enabled; None otherwise.
    # Marked non-pytree to exclude from JIT/pjit inputs and device transfers.
    swa_state: Optional[nnx.State]
    # Effective number of model snapshots accumulated into SWA (can be fractional).
    num_averages: float

    def replace(self, **changes: Any) -> "JitTrainingState":
        """Returns a new instance of the class with the specified changes."""
        return dataclasses.replace(self, **changes)


@dataclass
class TrainingState:
    jit_state: JitTrainingState
    # Last chunk source that was available when the last epoch started training.
    num_heads: int
    last_chunk_source: str = ""

    def replace(self, **changes: Any) -> "TrainingState":
        """Returns a new instance of the class with the specified changes."""
        return dataclasses.replace(self, **changes)

    def with_updated_step(self, step: int) -> "TrainingState":
        """Returns a copy with updated step in both jit_state and optimizer."""
        updated_opt_state = (
            update_optimizer_step(self.jit_state.opt_state, step)
            if self.jit_state.opt_state is not None
            else None
        )
        return self.replace(
            jit_state=self.jit_state.replace(
                step=step,
                opt_state=updated_opt_state,
            )
        )

    @staticmethod
    def new_from_config(
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ) -> "TrainingState":
        rngs = nnx.Rngs(params=42)
        model_state = nnx.state(LczeroModel(config=model_config, rngs=rngs))

        lr_sched = make_lr_schedule(training_config.lr_schedule)
        opt_state = make_gradient_transformation(
            training_config.optimizer,
            max_grad_norm=getattr(training_config, "max_grad_norm", 0.0),
            l2_regularization=getattr(
                training_config, "l2_regularization", 0.0
            ),
            lr_schedule=lr_sched,
        ).init(model_state)
        jit_state = JitTrainingState(
            step=0,
            model_state=model_state,
            opt_state=opt_state,
            swa_state=model_state,
            num_averages=0.0,
        )
        return TrainingState(
            jit_state=jit_state,
            num_heads=model_config.encoder.heads,
        )
