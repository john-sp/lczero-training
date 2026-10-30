"""Frozen ONNX teacher inference for JAX knowledge distillation."""

import logging

import jax
import numpy as np
from numpy.typing import NDArray

from proto.training_config_pb2 import TeacherConfig

from .model import ModelPrediction

logger = logging.getLogger(__name__)


class OnnxTeacher:
    """Run LC0 ONNX outputs through a batched, non-differentiable callback."""

    def __init__(self, config: TeacherConfig) -> None:
        if config.checkpoint_path or config.HasField("model"):
            raise ValueError(
                "onnx_path is mutually exclusive with checkpoint_path and model"
            )
        if not config.policy and not config.value:
            raise ValueError("ONNX teacher requires policy or value mappings")
        self._heads = [
            (
                family,
                head.head_name,
                head.output_name,
                head.output_is_probabilities,
                width,
            )
            for family, heads, width in (
                ("policy", config.policy, 1858),
                ("value", config.value, 3),
            )
            for head in heads
        ]
        seen = set()
        for family, name, output, _, _ in self._heads:
            if not name or not output:
                raise ValueError(
                    "ONNX mappings require head_name and output_name"
                )
            if (family, name) in seen:
                raise ValueError(f"Duplicate ONNX {family} head {name!r}")
            seen.add((family, name))

        import onnxruntime as ort

        providers = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in ort.get_available_providers()
        ]
        self._session = ort.InferenceSession(
            config.onnx_path, providers=providers
        )
        logger.info(
            "Loaded ONNX teacher from %s using %s",
            config.onnx_path,
            self._session.get_providers(),
        )
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("ONNX teacher must have exactly one input")
        self._input = inputs[0]
        if len(self._input.shape) != 4 or any(
            isinstance(actual, int) and actual != expected
            for actual, expected in zip(self._input.shape[1:], (112, 8, 8))
        ):
            raise ValueError(
                "ONNX teacher input must have shape [N, 112, 8, 8]"
            )
        batch_size = self._input.shape[0]
        if isinstance(batch_size, int) and batch_size <= 0:
            raise ValueError("ONNX teacher batch dimension must be positive")
        dtypes = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
        if self._input.type not in dtypes:
            raise ValueError("ONNX teacher input must be float32 or float16")
        self._dtype = dtypes[self._input.type]
        outputs = {
            output.name: output for output in self._session.get_outputs()
        }
        for _, _, name, _, width in self._heads:
            if name not in outputs:
                raise ValueError(f"ONNX teacher output {name!r} does not exist")
            shape = outputs[name].shape
            if len(shape) != 2 or (
                isinstance(shape[-1], int) and shape[-1] != width
            ):
                raise ValueError(
                    f"ONNX output {name!r} must have shape [N, {width}]"
                )

    def _infer(
        self, inputs: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], ...]:
        inputs = np.asarray(inputs)
        leading_shape = inputs.shape[:-3]
        planes = np.array(inputs, dtype=self._dtype, copy=True).reshape(
            -1, 112, 8, 8
        )
        # Training normalizes the rule-50 plane; LC0 ONNX expects raw counts.
        planes[:, 109, ...] *= 99
        names = list(dict.fromkeys(head[2] for head in self._heads))
        fixed_batch = self._input.shape[0]
        chunk_size = (
            fixed_batch if isinstance(fixed_batch, int) else len(planes)
        )
        if chunk_size <= 0:
            raise ValueError("ONNX teacher batch dimension must be positive")
        chunks: dict[str, list[NDArray[np.float32]]] = {
            name: [] for name in names
        }
        for start in range(0, len(planes), chunk_size):
            chunk = planes[start : start + chunk_size]
            count = len(chunk)
            if count < chunk_size:
                chunk = np.pad(
                    chunk, ((0, chunk_size - count), (0, 0), (0, 0), (0, 0))
                )
            raw = self._session.run(names, {self._input.name: chunk})
            for name, output in zip(names, raw):
                chunks[name].append(
                    np.asarray(output[:count], dtype=np.float32)
                )
        results = []
        for _, _, name, probabilities, width in self._heads:
            output = np.concatenate(chunks[name])
            if output.shape != (len(planes), width):
                raise ValueError(f"Unexpected ONNX output shape for {name!r}")
            if probabilities:
                output = np.log(np.maximum(output, np.finfo(np.float32).tiny))
            results.append(output.reshape(*leading_shape, width))
        return tuple(results)

    def __call__(self, inputs: jax.Array) -> ModelPrediction:
        shapes = tuple(
            jax.ShapeDtypeStruct((*inputs.shape[:-3], width), np.float32)
            for _, _, _, _, width in self._heads
        )
        outputs = jax.pure_callback(
            self._infer,
            shapes,
            jax.lax.stop_gradient(inputs),
            vmap_method="broadcast_all",
        )
        prediction = ModelPrediction(value={}, policy={}, movesleft={})
        for (family, name, _, _, _), output in zip(self._heads, outputs):
            if family == "policy":
                prediction.policy[name] = output
            else:
                prediction.value[name] = (output, None, None)
        return prediction
