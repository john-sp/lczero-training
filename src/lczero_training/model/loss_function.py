from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.scipy.special import xlogy

from lczero_training.training.state import (
    INVALID_MOVE_INDEX,
    TrainingSample,
)
from lczero_training.training.utils import make_weights_mask
from proto.training_config_pb2 import (
    ChildQLossConfig,
    LossConfig,
    MovesLeftLossConfig,
    PolicyIndexLossConfig,
    PolicyLossConfig,
    RegularizationLossConfig,
    TeacherConfig,
    ValueCategoricalLossConfig,
    ValueErrorLossConfig,
    ValueLossConfig,
)

from .model import LczeroModel, ModelPrediction


MASKED_POLICY_LOGIT = -1.0e10

# Number of provenance classes in aux_targets column 0:
# 0=none, 1=tablebase, 2=noise-deblunder, 3=unintended-deblunder.
NUM_PROVENANCE_CLASSES = 4


def _masked_mean(losses: jax.Array, mask: jax.Array) -> jax.Array:
    """Sum of `losses` where `mask`, normalized by the number of unmasked
    entries (guarding against an all-masked batch, which yields 0.0).

    When called with per-sample scalars (inside the training vmap), this
    reduces to `loss` for unmasked samples and 0.0 for masked ones; the
    batch mean applied outside the vmap then normalizes by batch size
    rather than by unmasked count. When called on a full batch (as in unit
    tests or non-vmap evaluation), it normalizes by the unmasked count.
    """
    total = jnp.sum(jnp.where(mask, losses, 0.0))
    count = jnp.sum(mask.astype(losses.dtype))
    return total / jnp.maximum(count, 1.0)


def mask_illegal_policy_logits(
    policy_logits: jax.Array, policy_targets: jax.Array
) -> jax.Array:
    """Mask policy logits for illegal moves using training targets."""
    return jnp.where(policy_targets >= 0, policy_logits, MASKED_POLICY_LOGIT)


def _compute_q_from_wdl(wdl_logits: jax.Array) -> jax.Array:
    """Compute Q value from WDL logits."""
    wdl_probs = jax.nn.softmax(wdl_logits)
    q_weights = jnp.array([1.0, 0.0, -1.0])
    return jnp.dot(wdl_probs, q_weights)


def _positive_temperature_or_one(temperature: float) -> float:
    return temperature if temperature > 0 else 1.0


def _softmax_distillation_loss(
    student_logits: jax.Array,
    teacher_logits: jax.Array,
    temperature: float,
    mask: Optional[jax.Array] = None,
) -> jax.Array:
    teacher_probs = jax.nn.softmax(teacher_logits / temperature)

    if mask is not None:
        student_logits = jnp.where(mask, student_logits, MASKED_POLICY_LOGIT)
        teacher_probs = jnp.where(mask, teacher_probs, 0.0)
        teacher_probs_sum = jnp.sum(teacher_probs, axis=-1, keepdims=True)
        safe_sum = jnp.where(
            teacher_probs_sum > 0,
            teacher_probs_sum,
            jnp.ones_like(teacher_probs_sum),
        )
        teacher_probs = teacher_probs / safe_sum

    teacher_probs = jax.lax.stop_gradient(teacher_probs)
    kd_loss = optax.softmax_cross_entropy(
        logits=student_logits / temperature,
        labels=teacher_probs,
    )
    return kd_loss * (temperature * temperature)


def _teacher_kd_params(
    global_kd_alpha: float,
    global_temperature: float,
    head_config: Optional[Any],
) -> Tuple[float, float]:
    kd_alpha = global_kd_alpha
    temperature = global_temperature
    if head_config is not None:
        kd_alpha = head_config.kd_alpha
        if head_config.temperature > 0:
            temperature = head_config.temperature
    return kd_alpha, temperature


class LossBase:
    def __init__(
        self,
        config: Union[
            PolicyLossConfig,
            ValueLossConfig,
            MovesLeftLossConfig,
            ValueErrorLossConfig,
            ValueCategoricalLossConfig,
            PolicyIndexLossConfig,
            ChildQLossConfig,
        ],
    ) -> None:
        self.head_name = config.head_name
        self.metric_name = config.metric_name or config.head_name
        self.weight = config.weight

    def __call__(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
    ) -> jax.Array:
        raise NotImplementedError("Subclasses must implement __call__")


class RegularizationLoss:
    """Computes regularization loss on model parameters."""

    def __init__(self, config: RegularizationLossConfig) -> None:
        self.metric_name = config.metric_name or "l2"
        self.weight = config.weight
        self._selector = config.selector

    def __call__(self, model: LczeroModel) -> jax.Array:
        params = nnx.state(model, nnx.Param)
        mask = make_weights_mask(self._selector, params)
        masked_params = jax.tree.map(
            lambda p, m: p.value if m else jnp.zeros_like(p.value),
            params,
            mask,
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        )
        leaves = jax.tree.leaves(masked_params)
        return sum(
            (jnp.sum(jnp.square(leaf)) for leaf in leaves), jnp.array(0.0)
        )


class LczeroLoss:
    policy_losses: List["PolicyLoss"]
    value_losses: List["ValueLoss"]
    movesleft_losses: List["MovesLeftLoss"]
    value_error_losses: List["ValueErrorLoss"]
    value_categorical_losses: List["ValueCategoricalLoss"]
    regularization_losses: List["RegularizationLoss"]
    policy_index_losses: List["PolicyIndexLoss"]
    child_q_losses: List["ChildQLoss"]

    def __init__(
        self,
        config: LossConfig,
        teacher_config: Optional[TeacherConfig] = None,
    ) -> None:
        self.config = config
        self.teacher_config = teacher_config
        self.policy_losses = [
            PolicyLoss(loss_config) for loss_config in config.policy
        ]
        self.value_losses = [
            ValueLoss(loss_config) for loss_config in config.value
        ]
        self.movesleft_losses = [
            MovesLeftLoss(loss_config) for loss_config in config.movesleft
        ]
        self.value_error_losses = [
            ValueErrorLoss(loss_config) for loss_config in config.value_error
        ]
        self.value_categorical_losses = [
            ValueCategoricalLoss(loss_config)
            for loss_config in config.value_categorical
        ]
        self.regularization_losses = [
            RegularizationLoss(loss_config)
            for loss_config in config.regularization
        ]
        self.policy_index_losses = [
            PolicyIndexLoss(loss_config) for loss_config in config.policy_index
        ]
        self.child_q_losses = [
            ChildQLoss(loss_config) for loss_config in config.child_q
        ]

        def _validate_no_duplicate_metrics(
            loss_type_name: str,
            losses: Sequence[Union[LossBase, RegularizationLoss]],
        ) -> None:
            seen = set()
            for name in (loss.metric_name for loss in losses):
                if name in seen:
                    raise ValueError(
                        f"Duplicate metric name: {loss_type_name}/{name}"
                    )
                seen.add(name)

        _validate_no_duplicate_metrics("policy", self.policy_losses)
        _validate_no_duplicate_metrics("value", self.value_losses)
        _validate_no_duplicate_metrics("movesleft", self.movesleft_losses)
        _validate_no_duplicate_metrics("value_error", self.value_error_losses)
        _validate_no_duplicate_metrics(
            "value_categorical", self.value_categorical_losses
        )
        _validate_no_duplicate_metrics(
            "regularization", self.regularization_losses
        )
        _validate_no_duplicate_metrics("policy_index", self.policy_index_losses)
        _validate_no_duplicate_metrics("child_q", self.child_q_losses)
        self._loss_weights: Dict[str, float] = {}
        for policy_loss in self.policy_losses:
            self._loss_weights[f"policy/{policy_loss.metric_name}"] = (
                policy_loss.weight
            )
        for value_loss in self.value_losses:
            self._loss_weights[f"value/{value_loss.metric_name}"] = (
                value_loss.weight
            )
        for movesleft_loss in self.movesleft_losses:
            self._loss_weights[f"movesleft/{movesleft_loss.metric_name}"] = (
                movesleft_loss.weight
            )
        for value_error_loss in self.value_error_losses:
            self._loss_weights[
                f"value_error/{value_error_loss.metric_name}"
            ] = value_error_loss.weight
        for value_categorical_loss in self.value_categorical_losses:
            self._loss_weights[
                f"value_categorical/{value_categorical_loss.metric_name}"
            ] = value_categorical_loss.weight
        for reg_loss in self.regularization_losses:
            self._loss_weights[f"regularization/{reg_loss.metric_name}"] = (
                reg_loss.weight
            )
        for policy_index_loss in self.policy_index_losses:
            self._loss_weights[
                f"policy_index/{policy_index_loss.metric_name}"
            ] = policy_index_loss.weight
        for child_q_loss in self.child_q_losses:
            self._loss_weights[f"child_q/{child_q_loss.metric_name}"] = (
                child_q_loss.weight
            )
        if self.teacher_config is not None:
            self._register_teacher_loss_weights()

    @property
    def loss_weights(self) -> Dict[str, float]:
        return dict(self._loss_weights)

    def _register_teacher_loss_weights(self) -> None:
        assert self.teacher_config is not None
        self._register_teacher_head_loss_weights(
            "policy",
            list(self.teacher_config.model.policy_head)
            + list(self.teacher_config.model.simple_policy_head),
            self.teacher_config.policy,
        )
        self._register_teacher_head_loss_weights(
            "value",
            list(self.teacher_config.model.value_head)
            + list(self.teacher_config.model.simple_value_head),
            self.teacher_config.value,
        )

    def _register_teacher_head_loss_weights(
        self,
        family: str,
        head_configs: Sequence[Any],
        overrides: Sequence[Any],
    ) -> None:
        assert self.teacher_config is not None
        override_by_name = {config.head_name: config for config in overrides}
        for head_config in head_configs:
            if not head_config.name:
                continue
            kd_alpha = self.teacher_config.kd_alpha
            if head_config.name in override_by_name:
                kd_alpha = override_by_name[head_config.name].kd_alpha
            self._loss_weights[f"kd/{family}/{head_config.name}"] = kd_alpha

    def weighted_losses(
        self, unweighted_losses: Dict[str, jax.Array]
    ) -> Dict[str, jax.Array]:
        return {
            key: value * self._loss_weights.get(key, 1.0)
            for key, value in unweighted_losses.items()
        }

    def __call__(
        self,
        model: LczeroModel,
        sample: TrainingSample,
        teacher_model: Optional[LczeroModel] = None,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        # Run model forward pass.
        predictions = model(sample.inputs)

        unweighted_losses: Dict[str, jax.Array] = {}
        weighted_losses: List[jax.Array] = []

        for policy_loss in self.policy_losses:
            loss = policy_loss(predictions, sample)
            unweighted_losses[f"policy/{policy_loss.metric_name}"] = loss
            weighted_losses.append(loss * policy_loss.weight)

            # Compute policy accuracy as a metric (not contributing to loss).
            # Only compute for the "vanilla" policy head.
            if policy_loss.head_name == "vanilla":
                accuracy = policy_loss.compute_accuracy(predictions, sample)
                unweighted_losses[
                    f"policy/{policy_loss.metric_name}/accuracy_unmasked"
                ] = accuracy

                top_3_accuracy = policy_loss.compute_top_k_accuracy(
                    predictions, sample, k=3
                )
                unweighted_losses[
                    f"policy/{policy_loss.metric_name}/top_3_accuracy_unmasked"
                ] = top_3_accuracy

                top_5_accuracy = policy_loss.compute_top_k_accuracy(
                    predictions, sample, k=5
                )
                unweighted_losses[
                    f"policy/{policy_loss.metric_name}/top_5_accuracy_unmasked"
                ] = top_5_accuracy

                masked_accuracy = policy_loss.compute_masked_accuracy(
                    predictions, sample
                )
                unweighted_losses[
                    f"policy/{policy_loss.metric_name}/accuracy"
                ] = masked_accuracy

                masked_top_3_accuracy = policy_loss.compute_top_k_accuracy(
                    predictions, sample, k=3, masked=True
                )
                unweighted_losses[
                    f"policy/{policy_loss.metric_name}/top_3_accuracy"
                ] = masked_top_3_accuracy

                masked_top_5_accuracy = policy_loss.compute_top_k_accuracy(
                    predictions, sample, k=5, masked=True
                )
                unweighted_losses[
                    f"policy/{policy_loss.metric_name}/top_5_accuracy"
                ] = masked_top_5_accuracy

        for value_loss in self.value_losses:
            loss = value_loss(predictions, sample)
            unweighted_losses[f"value/{value_loss.metric_name}"] = loss
            weighted_losses.append(loss * value_loss.weight)

            accuracy = value_loss.compute_accuracy(predictions, sample)
            unweighted_losses[f"value/{value_loss.metric_name}/accuracy"] = (
                accuracy
            )

        for movesleft_loss in self.movesleft_losses:
            loss = movesleft_loss(predictions, sample)
            unweighted_losses[f"movesleft/{movesleft_loss.metric_name}"] = loss
            weighted_losses.append(loss * movesleft_loss.weight)

        for value_error_loss in self.value_error_losses:
            loss = value_error_loss(predictions, sample)
            unweighted_losses[f"value_error/{value_error_loss.metric_name}"] = (
                loss
            )
            weighted_losses.append(loss * value_error_loss.weight)

        for value_categorical_loss in self.value_categorical_losses:
            loss = value_categorical_loss(predictions, sample)
            unweighted_losses[
                f"value_categorical/{value_categorical_loss.metric_name}"
            ] = loss
            weighted_losses.append(loss * value_categorical_loss.weight)

        for policy_index_loss in self.policy_index_losses:
            loss = policy_index_loss(predictions, sample)
            unweighted_losses[
                f"policy_index/{policy_index_loss.metric_name}"
            ] = loss
            weighted_losses.append(loss * policy_index_loss.weight)

        for child_q_loss in self.child_q_losses:
            loss = child_q_loss(predictions, sample)
            unweighted_losses[f"child_q/{child_q_loss.metric_name}"] = loss
            weighted_losses.append(loss * child_q_loss.weight)

        for reg_loss in self.regularization_losses:
            loss = reg_loss(model)
            unweighted_losses[f"regularization/{reg_loss.metric_name}"] = loss
            weighted_losses.append(loss * reg_loss.weight)

        if teacher_model is not None and self.teacher_config is not None:
            teacher_predictions = teacher_model(sample.inputs)
            global_kd_alpha = self.teacher_config.kd_alpha
            global_temp = _positive_temperature_or_one(
                self.teacher_config.temperature
            )

            policy_map = {p.head_name: p for p in self.teacher_config.policy}
            value_map = {v.head_name: v for v in self.teacher_config.value}

            for head_name, student_logits in predictions.policy.items():
                if head_name in teacher_predictions.policy:
                    kd_alpha, temp = _teacher_kd_params(
                        global_kd_alpha,
                        global_temp,
                        policy_map.get(head_name),
                    )

                    if kd_alpha == 0:
                        continue

                    teacher_logits = teacher_predictions.policy[head_name]

                    policy_targets = jnp.asarray(
                        sample.probabilities, dtype=student_logits.dtype
                    )
                    kd_loss = _softmax_distillation_loss(
                        student_logits,
                        teacher_logits,
                        temp,
                        mask=policy_targets >= 0,
                    )

                    unweighted_losses[f"kd/policy/{head_name}"] = kd_loss
                    weighted_losses.append(kd_loss * kd_alpha)

            for head_name, student_value in predictions.value.items():
                if head_name in teacher_predictions.value:
                    kd_alpha, temp = _teacher_kd_params(
                        global_kd_alpha,
                        global_temp,
                        value_map.get(head_name),
                    )

                    if kd_alpha == 0:
                        continue

                    teacher_value = teacher_predictions.value[head_name]
                    kd_loss = _softmax_distillation_loss(
                        student_value[0],
                        teacher_value[0],
                        temp,
                    )

                    unweighted_losses[f"kd/value/{head_name}"] = kd_loss
                    weighted_losses.append(kd_loss * kd_alpha)

        data_loss = jnp.sum(jnp.array(weighted_losses))

        return data_loss, unweighted_losses


class ValueLoss(LossBase):
    def __init__(self, config: ValueLossConfig) -> None:
        super().__init__(config)
        self.value_type = config.value_type
        if len(config.provenance_weights) == 0:
            self.provenance_weights: Optional[jax.Array] = None
        else:
            if len(config.provenance_weights) != NUM_PROVENANCE_CLASSES:
                raise ValueError(
                    f"provenance_weights for value head "
                    f"'{config.head_name}' must have exactly "
                    f"{NUM_PROVENANCE_CLASSES} entries, got "
                    f"{len(config.provenance_weights)}"
                )
            self.provenance_weights = jnp.asarray(
                list(config.provenance_weights), dtype=jnp.float32
            )

    def __call__(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
    ) -> jax.Array:
        value_pred = predictions.value[self.head_name]
        value_logits = value_pred[0]
        # Extract raw q/d from sample and compute WDL.
        value_q = sample.values[self.value_type, 0]
        value_d = sample.values[self.value_type, 1]
        # Compute WDL: w = (1 + q - d) / 2, l = (1 - q - d) / 2
        value_w = (1.0 + value_q - value_d) / 2.0
        value_l = (1.0 - value_q - value_d) / 2.0
        value_wdl = jnp.stack([value_w, value_d, value_l], axis=-1)

        # The cross-entropy between the predicted value and the target value.
        value_cross_entropy = optax.softmax_cross_entropy(
            logits=value_logits, labels=jax.lax.stop_gradient(value_wdl)
        )
        assert isinstance(value_cross_entropy, jax.Array)

        # Optionally weight each sample by the provenance class of its
        # value target (aux_targets column 0).
        if self.provenance_weights is not None:
            assert sample.aux_targets is not None, (
                "provenance_weights requires the aux_targets tensor"
            )
            provenance = jnp.clip(
                sample.aux_targets[..., 0].astype(jnp.int32),
                0,
                NUM_PROVENANCE_CLASSES - 1,
            )
            value_cross_entropy = (
                value_cross_entropy * self.provenance_weights[provenance]
            )
        return value_cross_entropy

    def compute_accuracy(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
    ) -> jax.Array:
        """Compute value accuracy from the predicted and target WDL classes."""
        value_pred = predictions.value[self.head_name]
        value_logits = value_pred[0]
        value_q = sample.values[self.value_type, 0]
        value_d = sample.values[self.value_type, 1]
        value_w = (1.0 + value_q - value_d) / 2.0
        value_l = (1.0 - value_q - value_d) / 2.0
        value_wdl = jnp.stack([value_w, value_d, value_l], axis=-1)

        target_class = jnp.argmax(value_wdl, axis=-1)
        predicted_class = jnp.argmax(value_logits, axis=-1)
        return jnp.equal(target_class, predicted_class).astype(jnp.float32)


class PolicyLoss(LossBase):
    def __init__(self, config: PolicyLossConfig):
        super().__init__(config)
        self.config = config
        if config.type == PolicyLossConfig.LOSS_TYPE_UNSPECIFIED:
            raise ValueError(
                "Policy loss type must be specified for head "
                f"'{config.head_name}'."
            )
        self._loss_type = config.type
        temperature = config.temperature
        if temperature <= 0:
            temperature = 1.0
        self._temperature = temperature

        # Store optimistic config if present.
        if config.HasField("optimistic"):
            opt = config.optimistic
            self.opt_value_head: Optional[str] = opt.value_head_name
            self.opt_value_type = opt.value_type
            self.opt_strength = opt.strength
            self.opt_eps = opt.eps
            self.opt_alpha = opt.alpha
            self.opt_propagate_gradients = opt.propagate_value_gradients
        else:
            self.opt_value_head = None

    def _apply_temperature_and_normalize(
        self, policy_targets: jax.Array
    ) -> jax.Array:
        if self._temperature == 1.0:
            return policy_targets

        # Apply temperature scaling.
        policy_targets = jnp.power(policy_targets, 1.0 / self._temperature)

        # Renormalize after temperature scaling.
        target_sum = jnp.sum(policy_targets, axis=-1, keepdims=True)
        safe_sum = jnp.where(
            target_sum > 0, target_sum, jnp.ones_like(target_sum)
        )
        return policy_targets / safe_sum

    def _get_policy_predictions_and_targets(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
        *,
        masked: bool = False,
    ) -> Tuple[jax.Array, jax.Array]:
        policy_pred = predictions.policy[self.head_name]
        policy_targets = jnp.asarray(
            sample.probabilities, dtype=policy_pred.dtype
        )
        if masked:
            policy_pred = mask_illegal_policy_logits(
                policy_pred, policy_targets
            )
        policy_targets = jax.nn.relu(policy_targets)
        return policy_pred, policy_targets

    def _compute_optimistic_weight(
        self,
        value_pred: Tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]],
        target_q: jax.Array,
    ) -> jax.Array:
        """Compute optimistic policy weight from value head predictions."""
        wdl_logits = value_pred[0]
        error_pred = value_pred[1]
        assert error_pred is not None, (
            "Error prediction required for optimistic weighting"
        )

        # Optionally block gradients to value and error heads.
        if not self.opt_propagate_gradients:
            wdl_logits = jax.lax.stop_gradient(wdl_logits)
            error_pred = jax.lax.stop_gradient(error_pred)

        # Compute predicted Q from WDL.
        q_pred = _compute_q_from_wdl(wdl_logits)

        # Compute sigma and z-score.
        sigma = jnp.sqrt(error_pred.squeeze())
        z = (target_q - q_pred) / (sigma + self.opt_eps)

        # Compute weight.
        return jax.nn.sigmoid((z - self.opt_strength) * self.opt_alpha)

    def __call__(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
    ) -> jax.Array:
        policy_pred = predictions.policy[self.head_name]
        # Extract probabilities from sample.
        policy_targets = jnp.asarray(
            sample.probabilities, dtype=policy_pred.dtype
        )
        if self.config.illegal_moves == PolicyLossConfig.MASK:
            # Use a large negative value instead of -inf to avoid NaN in
            # gradients when computing 0 * log(softmax(-inf)).
            policy_pred = mask_illegal_policy_logits(
                policy_pred, policy_targets
            )

        # Zero out negative targets for illegal moves.
        policy_targets = jax.nn.relu(policy_targets)

        # Apply temperature scaling and renormalization if needed.
        policy_targets = self._apply_temperature_and_normalize(policy_targets)

        cross_entropy = cast(
            jax.Array,
            optax.safe_softmax_cross_entropy(
                logits=policy_pred, labels=policy_targets
            ),
        )
        if self._loss_type == PolicyLossConfig.CROSS_ENTROPY:
            loss = cross_entropy
        elif self._loss_type == PolicyLossConfig.KL:
            loss = cross_entropy + jnp.sum(
                xlogy(policy_targets, policy_targets), axis=-1
            )
        else:
            raise AssertionError(
                f"Unknown policy loss type: {self._loss_type}."
            )

        # Apply optimistic weighting if configured.
        if self.opt_value_head is not None:
            value_pred = predictions.value[self.opt_value_head]
            target_q = sample.values[self.opt_value_type, 0]
            loss = loss * self._compute_optimistic_weight(value_pred, target_q)

        return loss

    def compute_accuracy(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
    ) -> jax.Array:
        """Compute policy accuracy by comparing argmax targets and predictions.

        Returns:
            Scalar accuracy value (fraction of correct predictions per sample).
        """
        policy_pred, policy_targets = self._get_policy_predictions_and_targets(
            predictions, sample
        )

        target_move = jnp.argmax(policy_targets, axis=-1)
        predicted_move = jnp.argmax(policy_pred, axis=-1)
        correct = jnp.equal(target_move, predicted_move).astype(jnp.float32)

        return correct

    def compute_masked_accuracy(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
    ) -> jax.Array:
        """Compute policy accuracy with illegal moves masked out."""
        policy_pred, policy_targets = self._get_policy_predictions_and_targets(
            predictions, sample, masked=True
        )

        target_move = jnp.argmax(policy_targets, axis=-1)
        predicted_move = jnp.argmax(policy_pred, axis=-1)
        correct = jnp.equal(target_move, predicted_move).astype(jnp.float32)

        return correct

    def compute_top_k_accuracy(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
        k: int,
        *,
        masked: bool = False,
    ) -> jax.Array:
        """Compute whether the target move is within the top-k predictions."""
        policy_pred, policy_targets = self._get_policy_predictions_and_targets(
            predictions, sample, masked=masked
        )

        target_move = jnp.argmax(policy_targets, axis=-1)
        top_k = min(k, policy_pred.shape[-1])
        _, top_k_moves = jax.lax.top_k(policy_pred, top_k)
        return jnp.any(top_k_moves == target_move[..., None], axis=-1).astype(
            jnp.float32
        )


class PolicyIndexLoss(LossBase):
    """One-hot cross-entropy of a policy-shaped head against a move index.

    The target is a single move index from aux_indices (column 0 for
    OPP_PLAYED, column 1 for NEXT_PLAYED). Samples where the index is
    INVALID_MOVE_INDEX (65535, i.e. no such move near the end of a game)
    get zero weight; the result is normalized by the number of unmasked
    samples (see _masked_mean for the per-sample vmap caveat) and an
    all-masked batch yields 0.0 rather than NaN.
    """

    _TARGET_COLUMNS = {
        PolicyIndexLossConfig.OPP_PLAYED: 0,
        PolicyIndexLossConfig.NEXT_PLAYED: 1,
    }

    def __init__(self, config: PolicyIndexLossConfig) -> None:
        super().__init__(config)
        self.target_column = self._TARGET_COLUMNS[config.target]

    def __call__(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
    ) -> jax.Array:
        logits = predictions.policy[self.head_name]
        assert sample.aux_indices is not None, (
            "PolicyIndexLoss requires the aux_indices tensor"
        )
        index = sample.aux_indices[..., self.target_column].astype(jnp.int32)
        mask = index != INVALID_MOVE_INDEX
        safe_index = jnp.where(mask, index, 0)
        cross_entropy = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=safe_index
        )
        assert isinstance(cross_entropy, jax.Array)
        return _masked_mean(cross_entropy, mask)


class ChildQLoss(LossBase):
    """MSE between a policy-shaped head's output at the played-move index
    and the played-move child-Q target.

    The head output is gathered at aux_indices column 2 (played_idx) and
    regressed against aux_targets column 1 (child-Q). Samples with NaN
    child-Q (no value available, e.g. the last record of a game) get zero
    weight; the result is normalized by the number of unmasked samples
    (see _masked_mean for the per-sample vmap caveat) and an all-masked
    batch yields 0.0 rather than NaN.
    """

    PLAYED_INDEX_COLUMN = 2
    CHILD_Q_COLUMN = 1

    def __init__(self, config: ChildQLossConfig) -> None:
        super().__init__(config)

    def __call__(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
    ) -> jax.Array:
        outputs = predictions.policy[self.head_name]
        assert sample.aux_indices is not None, (
            "ChildQLoss requires the aux_indices tensor"
        )
        assert sample.aux_targets is not None, (
            "ChildQLoss requires the aux_targets tensor"
        )
        index = sample.aux_indices[..., self.PLAYED_INDEX_COLUMN].astype(
            jnp.int32
        )
        target = sample.aux_targets[..., self.CHILD_Q_COLUMN]
        mask = jnp.logical_and(
            jnp.logical_not(jnp.isnan(target)),
            index != INVALID_MOVE_INDEX,
        )
        safe_index = jnp.where(mask, index, 0)
        predicted = jnp.take_along_axis(
            outputs, jnp.expand_dims(safe_index, -1), axis=-1
        ).squeeze(-1)
        safe_target = jnp.where(mask, target, 0.0)
        squared_error = jnp.square(predicted - safe_target)
        return _masked_mean(squared_error, mask)


class MovesLeftLoss(LossBase):
    def __init__(self, config: MovesLeftLossConfig) -> None:
        super().__init__(config)
        self.value_type = config.value_type

    def __call__(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
    ) -> jax.Array:
        movesleft_pred = predictions.movesleft[self.head_name]
        # Extract movesleft from sample.
        # sample.values shape: [6, 3], component 2 is movesleft.
        movesleft_targets = sample.values[self.value_type, 2]

        # Scale the loss to similar range as other losses.
        scale = 20.0
        targets = movesleft_targets / scale
        scaled_predictions = movesleft_pred / scale

        # Huber loss
        huber_loss = optax.huber_loss(
            predictions=scaled_predictions, targets=targets, delta=10.0 / scale
        )
        assert isinstance(huber_loss, jax.Array)
        return huber_loss.squeeze()


class ValueErrorLoss(LossBase):
    def __init__(self, config: ValueErrorLossConfig) -> None:
        super().__init__(config)
        self.value_type = config.value_type
        self.propagate_value_gradients = config.propagate_value_gradients

    def __call__(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
    ) -> jax.Array:
        value_pred = predictions.value[self.head_name]
        wdl_logits = value_pred[0]
        error_pred = value_pred[1]
        assert error_pred is not None

        # Convert WDL to Q value.
        predicted_q = _compute_q_from_wdl(wdl_logits)

        # Get target Q value.
        target_q = sample.values[self.value_type, 0]

        # Compute actual squared error.
        actual_squared_error = jnp.square(predicted_q - target_q)

        # Optionally block gradients to WDL head.
        if not self.propagate_value_gradients:
            actual_squared_error = jax.lax.stop_gradient(actual_squared_error)

        # MSE between error prediction and actual error.
        loss = jnp.square(error_pred - actual_squared_error)

        return loss.squeeze()


class ValueCategoricalLoss(LossBase):
    def __init__(self, config: ValueCategoricalLossConfig) -> None:
        super().__init__(config)
        self.value_type = config.value_type

    def __call__(
        self,
        predictions: ModelPrediction,
        sample: TrainingSample,
    ) -> jax.Array:
        value_pred = predictions.value[self.head_name]
        categorical_logits = value_pred[2]
        assert categorical_logits is not None

        # Get target Q value from sample.
        target_q = sample.values[self.value_type, 0]

        # Convert Q to bucket index: map [-1, 1) to [0, num_buckets).
        num_buckets = categorical_logits.shape[-1]
        bucket_index = jnp.floor((target_q + 1.0) / 2.0 * num_buckets).astype(
            jnp.int32
        )
        bucket_index = jnp.clip(bucket_index, 0, num_buckets - 1)

        # Create one-hot target.
        target_one_hot = jax.nn.one_hot(bucket_index, num_buckets)

        # Compute softmax cross-entropy.
        loss = optax.softmax_cross_entropy(
            logits=categorical_logits,
            labels=jax.lax.stop_gradient(target_one_hot),
        )
        assert isinstance(loss, jax.Array)
        return loss
