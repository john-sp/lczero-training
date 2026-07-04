"""Unit tests for the aux-target losses and ST_CENSORED value selection."""

import jax.numpy as jnp
import numpy as np
import pytest

from lczero_training.model.loss_function import (
    ChildQLoss,
    PolicyIndexLoss,
    ValueLoss,
)
from lczero_training.model.model import ModelPrediction
from lczero_training.training.state import (
    INVALID_MOVE_INDEX,
    TrainingSample,
)
from proto.training_config_pb2 import (
    ChildQLossConfig,
    PolicyIndexLossConfig,
    ValueLossConfig,
    ValueType,
)

NUM_MOVES = 1858


def _prediction_with_policy(name: str, logits: jnp.ndarray) -> ModelPrediction:
    return ModelPrediction(value={}, policy={name: logits}, movesleft={})


def _prediction_with_value(
    name: str, wdl_logits: jnp.ndarray
) -> ModelPrediction:
    return ModelPrediction(
        value={name: (wdl_logits, None, None)}, policy={}, movesleft={}
    )


def _sample(
    values: np.ndarray | None = None,
    aux_indices: np.ndarray | None = None,
    aux_targets: np.ndarray | None = None,
) -> TrainingSample:
    batch = 1
    for arr in (values, aux_indices, aux_targets):
        if arr is not None:
            batch = arr.shape[0]
            break
    return TrainingSample(
        inputs=jnp.zeros((batch, 112, 8, 8), dtype=jnp.float32),
        probabilities=jnp.zeros((batch, NUM_MOVES), dtype=jnp.float32),
        values=(
            jnp.asarray(values)
            if values is not None
            else jnp.zeros((batch, 7, 3), dtype=jnp.float32)
        ),
        aux_indices=(
            jnp.asarray(aux_indices) if aux_indices is not None else None
        ),
        aux_targets=(
            jnp.asarray(aux_targets) if aux_targets is not None else None
        ),
    )


def _cross_entropy_at(logits: np.ndarray, index: int) -> float:
    """Reference softmax cross-entropy with a one-hot label at `index`."""
    logits = logits.astype(np.float64)
    log_norm = np.log(np.sum(np.exp(logits - logits.max()))) + logits.max()
    return float(log_norm - logits[index])


class TestPolicyIndexLoss:
    def _make_loss(
        self, target: int = PolicyIndexLossConfig.OPP_PLAYED
    ) -> PolicyIndexLoss:
        return PolicyIndexLoss(
            PolicyIndexLossConfig(
                head_name="opponent", weight=1.0, target=target
            )
        )

    def test_exact_value_unmasked(self) -> None:
        logits = np.zeros((2, NUM_MOVES), dtype=np.float32)
        logits[0, 5] = 2.0
        logits[1, 100] = -1.0
        aux_indices = np.array([[5, 0, 0], [100, 0, 0]], dtype=np.int32)
        loss = self._make_loss()(
            _prediction_with_policy("opponent", jnp.asarray(logits)),
            _sample(aux_indices=aux_indices),
        )
        expected = (
            _cross_entropy_at(logits[0], 5) + _cross_entropy_at(logits[1], 100)
        ) / 2.0
        np.testing.assert_allclose(float(loss), expected, rtol=1e-5)

    def test_masked_samples_excluded_and_count_normalized(self) -> None:
        logits = np.zeros((3, NUM_MOVES), dtype=np.float32)
        logits[0, 7] = 3.0
        aux_indices = np.array(
            [[7, 0, 0], [INVALID_MOVE_INDEX, 0, 0], [INVALID_MOVE_INDEX, 0, 0]],
            dtype=np.int32,
        )
        loss = self._make_loss()(
            _prediction_with_policy("opponent", jnp.asarray(logits)),
            _sample(aux_indices=aux_indices),
        )
        # Only sample 0 contributes; normalization is by unmasked count (1),
        # not batch size (3).
        expected = _cross_entropy_at(logits[0], 7)
        np.testing.assert_allclose(float(loss), expected, rtol=1e-5)

    def test_all_masked_batch_is_zero_not_nan(self) -> None:
        logits = jnp.zeros((2, NUM_MOVES), dtype=jnp.float32)
        aux_indices = np.full((2, 3), INVALID_MOVE_INDEX, dtype=np.int32)
        loss = self._make_loss()(
            _prediction_with_policy("opponent", logits),
            _sample(aux_indices=aux_indices),
        )
        assert np.isfinite(float(loss))
        np.testing.assert_allclose(float(loss), 0.0)

    def test_next_played_uses_column_1(self) -> None:
        logits = np.zeros((1, NUM_MOVES), dtype=np.float32)
        logits[0, 33] = 5.0
        aux_indices = np.array([[999, 33, 0]], dtype=np.int32)
        loss = self._make_loss(PolicyIndexLossConfig.NEXT_PLAYED)(
            _prediction_with_policy("opponent", jnp.asarray(logits)),
            _sample(aux_indices=aux_indices),
        )
        expected = _cross_entropy_at(logits[0], 33)
        np.testing.assert_allclose(float(loss), expected, rtol=1e-5)

    def test_per_sample_scalar_masked_is_zero(self) -> None:
        # Shape of a sample inside the training vmap: no batch dimension.
        logits = jnp.zeros((NUM_MOVES,), dtype=jnp.float32)
        sample = TrainingSample(
            inputs=jnp.zeros((112, 8, 8)),
            probabilities=jnp.zeros((NUM_MOVES,)),
            values=jnp.zeros((7, 3)),
            aux_indices=jnp.array([INVALID_MOVE_INDEX, 3, 5], dtype=jnp.int32),
            aux_targets=jnp.zeros((2,)),
        )
        loss = self._make_loss()(
            _prediction_with_policy("opponent", logits), sample
        )
        np.testing.assert_allclose(float(loss), 0.0)


class TestChildQLoss:
    def _make_loss(self) -> ChildQLoss:
        return ChildQLoss(ChildQLossConfig(head_name="child_q", weight=1.0))

    def test_exact_value_with_nan_masking(self) -> None:
        outputs = np.zeros((3, NUM_MOVES), dtype=np.float32)
        outputs[0, 3] = 0.9
        outputs[1, 10] = -0.4
        outputs[2, 20] = 0.7
        aux_indices = np.array(
            [[0, 0, 3], [0, 0, 10], [0, 0, 20]], dtype=np.int32
        )
        aux_targets = np.array(
            [[0.0, 0.5], [0.0, -0.5], [0.0, np.nan]], dtype=np.float32
        )
        loss = self._make_loss()(
            _prediction_with_policy("child_q", jnp.asarray(outputs)),
            _sample(aux_indices=aux_indices, aux_targets=aux_targets),
        )
        expected = ((0.9 - 0.5) ** 2 + (-0.4 - (-0.5)) ** 2) / 2.0
        np.testing.assert_allclose(float(loss), expected, rtol=1e-5)

    def test_all_nan_batch_is_zero_not_nan(self) -> None:
        outputs = jnp.zeros((2, NUM_MOVES), dtype=jnp.float32)
        aux_indices = np.array([[0, 0, 3], [0, 0, 4]], dtype=np.int32)
        aux_targets = np.array([[0.0, np.nan], [0.0, np.nan]], dtype=np.float32)
        loss = self._make_loss()(
            _prediction_with_policy("child_q", outputs),
            _sample(aux_indices=aux_indices, aux_targets=aux_targets),
        )
        assert np.isfinite(float(loss))
        np.testing.assert_allclose(float(loss), 0.0)

    def test_per_sample_scalar(self) -> None:
        outputs = np.zeros((NUM_MOVES,), dtype=np.float32)
        outputs[42] = 0.25
        sample = TrainingSample(
            inputs=jnp.zeros((112, 8, 8)),
            probabilities=jnp.zeros((NUM_MOVES,)),
            values=jnp.zeros((7, 3)),
            aux_indices=jnp.array([0, 0, 42], dtype=jnp.int32),
            aux_targets=jnp.array([0.0, -0.25], dtype=jnp.float32),
        )
        loss = self._make_loss()(
            _prediction_with_policy("child_q", jnp.asarray(outputs)), sample
        )
        np.testing.assert_allclose(float(loss), 0.25, rtol=1e-6)


def _reference_wdl_cross_entropy(
    wdl_logits: np.ndarray, q: float, d: float
) -> float:
    w = (1.0 + q - d) / 2.0
    ll = (1.0 - q - d) / 2.0
    labels = np.array([w, d, ll], dtype=np.float64)
    logits = wdl_logits.astype(np.float64)
    log_softmax = logits - (
        np.log(np.sum(np.exp(logits - logits.max()))) + logits.max()
    )
    return float(-np.sum(labels * log_softmax))


def _value_sample(
    values: np.ndarray, aux_targets: np.ndarray | None = None
) -> TrainingSample:
    """A single (unbatched) sample, as seen inside the training vmap.

    ValueLoss indexes `sample.values[value_type, 0]`, so it operates on
    per-sample tensors of shape [7, 3].
    """
    return TrainingSample(
        inputs=jnp.zeros((112, 8, 8), dtype=jnp.float32),
        probabilities=jnp.zeros((NUM_MOVES,), dtype=jnp.float32),
        values=jnp.asarray(values),
        aux_indices=jnp.zeros((3,), dtype=jnp.int32),
        aux_targets=(
            jnp.asarray(aux_targets) if aux_targets is not None else None
        ),
    )


class TestValueLossStCensored:
    def test_st_censored_selects_row_6(self) -> None:
        # Distinct q/d per row so a wrong row selection is detected.
        values = np.zeros((7, 3), dtype=np.float32)
        for row in range(7):
            values[row, 0] = 0.1 * row  # q
            values[row, 1] = 0.05 * row  # d
        wdl_logits = np.array([0.5, -0.25, 0.1], dtype=np.float32)
        loss_fn = ValueLoss(
            ValueLossConfig(
                head_name="winner",
                weight=1.0,
                value_type=ValueType.ST_CENSORED,
            )
        )
        loss = loss_fn(
            _prediction_with_value("winner", jnp.asarray(wdl_logits)),
            _value_sample(values),
        )
        # Row 6: q = 0.6, d = 0.3.
        expected = _reference_wdl_cross_entropy(wdl_logits, 0.6, 0.3)
        np.testing.assert_allclose(float(loss), expected, rtol=1e-5)

    def test_provenance_weighting(self) -> None:
        values = np.zeros((7, 3), dtype=np.float32)
        values[0, 0] = 0.4  # result_q
        values[0, 1] = 0.2  # result_d
        wdl_logits = np.array([0.3, -0.7, 0.2], dtype=np.float32)
        provenance_weights = [1.0, 2.0, 0.5, 0.0]
        loss_fn = ValueLoss(
            ValueLossConfig(
                head_name="winner",
                weight=1.0,
                value_type=ValueType.RESULT,
                provenance_weights=provenance_weights,
            )
        )
        base = _reference_wdl_cross_entropy(wdl_logits, 0.4, 0.2)
        for provenance in range(4):
            loss = loss_fn(
                _prediction_with_value("winner", jnp.asarray(wdl_logits)),
                _value_sample(
                    values,
                    aux_targets=np.array(
                        [float(provenance), 0.0], dtype=np.float32
                    ),
                ),
            )
            np.testing.assert_allclose(
                float(loss),
                base * provenance_weights[provenance],
                rtol=1e-5,
                err_msg=f"provenance={provenance}",
            )

    def test_no_provenance_weights_is_unweighted(self) -> None:
        values = np.zeros((7, 3), dtype=np.float32)
        values[0, 0] = -0.2
        values[0, 1] = 0.6
        wdl_logits = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        loss_fn = ValueLoss(
            ValueLossConfig(
                head_name="winner", weight=1.0, value_type=ValueType.RESULT
            )
        )
        loss = loss_fn(
            _prediction_with_value("winner", jnp.asarray(wdl_logits)),
            _value_sample(values),
        )
        expected = _reference_wdl_cross_entropy(wdl_logits, -0.2, 0.6)
        np.testing.assert_allclose(float(loss), expected, rtol=1e-5)

    def test_wrong_provenance_weights_length_raises(self) -> None:
        with pytest.raises(ValueError, match="provenance_weights"):
            ValueLoss(
                ValueLossConfig(
                    head_name="winner",
                    weight=1.0,
                    value_type=ValueType.RESULT,
                    provenance_weights=[1.0, 2.0],
                )
            )
