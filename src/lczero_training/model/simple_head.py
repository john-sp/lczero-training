from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from proto import model_config_pb2, net_pb2

from .utils import get_activation, get_dtype


class SimpleHeadBase(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        output_size: int,
        activation: net_pb2.NetworkFormat.ActivationFunction,
        dtype: Any = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.activation = activation
        self.dense1 = nnx.Linear(
            in_features=in_features,
            out_features=hidden_size,
            dtype=dtype,
            rngs=rngs,
        )
        self.dense2 = nnx.Linear(
            in_features=hidden_size,
            out_features=output_size,
            dtype=dtype,
            rngs=rngs,
        )

    def hidden(self, x: jax.Array) -> jax.Array:
        x = self.dense1(x)
        return get_activation(self.activation)(x)


class SimpleValueHead(nnx.Module):
    def __init__(
        self,
        in_features: int,
        config: model_config_pb2.SimpleHeadConfig,
        defaults: model_config_pb2.DefaultsConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.has_error_output = config.has_error_output
        self.num_categorical_buckets = config.num_categorical_buckets
        cdt = get_dtype(defaults.compute_dtype)
        self.backbone = SimpleHeadBase(
            in_features=in_features,
            hidden_size=config.hidden_size,
            output_size=3,
            activation=defaults.activation,
            dtype=cdt,
            rngs=rngs,
        )
        if self.has_error_output:
            self.error = nnx.Linear(
                in_features=config.hidden_size,
                out_features=1,
                dtype=cdt,
                rngs=rngs,
            )
        if self.num_categorical_buckets > 0:
            self.categorical = nnx.Linear(
                in_features=config.hidden_size,
                out_features=self.num_categorical_buckets,
                dtype=cdt,
                rngs=rngs,
            )

    def __call__(
        self, x: jax.Array
    ) -> Tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
        hidden = self.backbone.hidden(x)
        # Cast head outputs to float32 so the loss/softmax run in full
        # precision.
        wdl = self.backbone.dense2(hidden).astype(jnp.float32)
        error = (
            nnx.sigmoid(self.error(hidden)).astype(jnp.float32)
            if self.has_error_output
            else None
        )
        categorical = (
            self.categorical(hidden).astype(jnp.float32)
            if self.num_categorical_buckets > 0
            else None
        )
        return (wdl, error, categorical)


class SimpleMovesLeftHead(nnx.Module):
    def __init__(
        self,
        in_features: int,
        config: model_config_pb2.SimpleHeadConfig,
        defaults: model_config_pb2.DefaultsConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.backbone = SimpleHeadBase(
            in_features=in_features,
            hidden_size=config.hidden_size,
            output_size=1,
            activation=defaults.activation,
            dtype=get_dtype(defaults.compute_dtype),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        hidden = self.backbone.hidden(x)
        return nnx.relu(self.backbone.dense2(hidden)).astype(jnp.float32)


class SimplePolicyHead(nnx.Module):
    def __init__(
        self,
        in_features: int,
        config: model_config_pb2.SimpleHeadConfig,
        defaults: model_config_pb2.DefaultsConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.backbone = SimpleHeadBase(
            in_features=in_features,
            hidden_size=config.hidden_size,
            output_size=1858,
            activation=defaults.activation,
            dtype=get_dtype(defaults.compute_dtype),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        hidden = self.backbone.hidden(x)
        return self.backbone.dense2(hidden).astype(jnp.float32)
