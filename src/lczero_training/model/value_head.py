from collections.abc import Callable
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from proto import model_config_pb2

from .utils import get_activation, get_dtype

ActivationSink = Callable[[str, jax.Array], None]


class ValueHead(nnx.Module):
    def __init__(
        self,
        in_features: int,
        config: model_config_pb2.ValueHeadConfig,
        defaults: model_config_pb2.DefaultsConfig,
        rngs: nnx.Rngs,
    ):
        self.activation = defaults.activation
        self.has_error_output = config.has_error_output
        self.num_categorical_buckets = config.num_categorical_buckets
        cdt = get_dtype(defaults.compute_dtype)
        self.embed = nnx.Linear(
            in_features=in_features,
            out_features=config.num_channels,
            dtype=cdt,
            rngs=rngs,
        )
        self.dense1 = nnx.Linear(
            in_features=config.num_channels * 64,
            out_features=128,
            dtype=cdt,
            rngs=rngs,
        )
        self.wdl = nnx.Linear(
            in_features=128,
            out_features=3,
            dtype=cdt,
            rngs=rngs,
        )
        if self.has_error_output:
            self.error = nnx.Linear(
                in_features=128,
                out_features=1,
                dtype=cdt,
                rngs=rngs,
            )
        if self.num_categorical_buckets > 0:
            self.categorical = nnx.Linear(
                in_features=128,
                out_features=self.num_categorical_buckets,
                dtype=cdt,
                rngs=rngs,
            )

    def __call__(
        self,
        x: jax.Array,
        activation_sink: ActivationSink | None = None,
        tap_prefix: str = "value_head",
    ) -> Tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
        if activation_sink is not None:
            activation_sink(f"{tap_prefix}/embed/kernel", x)
        x = self.embed(x).flatten()
        x = get_activation(self.activation)(x)
        if activation_sink is not None:
            activation_sink(f"{tap_prefix}/dense1/kernel", x)
        x = self.dense1(x)
        x = get_activation(self.activation)(x)

        if activation_sink is not None:
            activation_sink(f"{tap_prefix}/wdl/kernel", x)
        # Cast head outputs to float32 so the loss/softmax run in full precision.
        wdl = self.wdl(x).astype(jnp.float32)
        if self.has_error_output:
            if activation_sink is not None:
                activation_sink(f"{tap_prefix}/error/kernel", x)
            error = nnx.sigmoid(self.error(x)).astype(jnp.float32)
        else:
            error = None
        if self.num_categorical_buckets > 0 and activation_sink is not None:
            activation_sink(f"{tap_prefix}/categorical/kernel", x)
        categorical = (
            self.categorical(x).astype(jnp.float32)
            if self.num_categorical_buckets > 0
            else None
        )

        return (wdl, error, categorical)

    def predict(self, x: jax.Array) -> jax.Array:
        return nnx.softmax(self(x)[0])
