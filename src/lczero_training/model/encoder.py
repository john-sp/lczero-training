import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from flax.linen import initializers as flax_initializers

from proto import model_config_pb2

from .shared import Ffn
from .utils import get_activation, get_norm_layer


_SEQUENTIAL_BLOCK_STYLE = (
    model_config_pb2.EncoderConfig.ENCODER_BLOCK_STYLE_SEQUENTIAL
)
_PALM_PARALLEL_BLOCK_STYLE = (
    model_config_pb2.EncoderConfig.ENCODER_BLOCK_STYLE_PALM_PARALLEL
)


class _LayerConfig:
    def __init__(
        self,
        *,
        dff: int,
        heads: int,
        kv_heads: int,
        smolgen: Optional[model_config_pb2.SmolgenConfig],
    ):
        self.dff = dff
        self.heads = heads
        self.kv_heads = kv_heads
        self.smolgen = smolgen


def _build_layer_configs(
    config: model_config_pb2.EncoderConfig,
) -> list[_LayerConfig]:
    layer_configs = [
        _LayerConfig(
            dff=config.dff,
            heads=config.heads,
            kv_heads=config.kv_heads
            if config.HasField("kv_heads")
            else config.heads,
            smolgen=config.smolgen if config.HasField("smolgen") else None,
        )
        for _ in range(config.num_blocks)
    ]
    seen = [False] * config.num_blocks
    base_gen_size = (
        config.smolgen.gen_size if config.HasField("smolgen") else None
    )
    for override in config.layer_override:
        if not override.HasField("start_layer") or not override.HasField(
            "end_layer"
        ):
            raise ValueError(
                "layer_override requires start_layer and end_layer."
            )
        start = override.start_layer
        end = override.end_layer
        if start > end:
            raise ValueError("layer_override start_layer must be <= end_layer.")
        if end >= config.num_blocks:
            raise ValueError("layer_override range exceeds num_blocks.")
        for idx in range(start, end + 1):
            if seen[idx]:
                raise ValueError("Overlapping layer_override ranges are invalid.")
            seen[idx] = True
            layer_config = layer_configs[idx]
            if override.HasField("dff"):
                layer_config.dff = override.dff
            if override.HasField("heads"):
                layer_config.heads = override.heads
            if override.HasField("kv_heads"):
                layer_config.kv_heads = override.kv_heads
            if override.HasField("smolgen"):
                if base_gen_size is None:
                    raise ValueError(
                        "smolgen overrides require encoder.smolgen to be set."
                    )
                if (
                    override.smolgen.HasField("gen_size")
                    and override.smolgen.gen_size != base_gen_size
                ):
                    raise ValueError(
                        "smolgen.gen_size cannot vary by layer."
                    )
                layer_config.smolgen = override.smolgen
    for layer_config in layer_configs:
        if layer_config.kv_heads <= 0:
            raise ValueError("kv_heads must be greater than zero.")
        if layer_config.heads % layer_config.kv_heads != 0:
            raise ValueError("heads must be divisible by kv_heads.")
    return layer_configs


class EncoderTower(nnx.Module):
    def __init__(
        self,
        *,
        in_features: int,
        config: model_config_pb2.EncoderConfig,
        defaults: model_config_pb2.DefaultsConfig,
        deepnorm_beta: float,
        rngs: nnx.Rngs,
    ):
        layer_configs = _build_layer_configs(config)
        smolgen_shared_gen_dense = None
        if config.HasField("smolgen"):
            smolgen_shared_gen_dense = nnx.Linear(
                in_features=config.smolgen.gen_size,
                out_features=64 * 64,
                use_bias=False,
                rngs=rngs,
            )

        self.encoders = nnx.Sequential(
            *[
                EncoderBlock(
                    in_features=in_features,
                    config=config,
                    layer_config=layer_config,
                    defaults=defaults,
                    smol_gen_dense=smolgen_shared_gen_dense,
                    deepnorm_beta=deepnorm_beta,
                    rngs=rngs,
                )
                for layer_config in layer_configs
            ]
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.encoders(x)


class EncoderBlock(nnx.Module):
    """A single block of the transformer encoder."""

    def __init__(
        self,
        *,
        in_features: int,
        config: model_config_pb2.EncoderConfig,
        layer_config: _LayerConfig,
        defaults: model_config_pb2.DefaultsConfig,
        smol_gen_dense: Optional[nnx.Linear],
        deepnorm_beta: float,
        rngs: nnx.Rngs,
    ):
        assert (smol_gen_dense is not None) == (layer_config.smolgen is not None)
        self.mha = MultiHeadAttention(
            in_features=in_features,
            d_model=config.d_model,
            heads=layer_config.heads,
            kv_heads=layer_config.kv_heads,
            use_bias_q=config.use_bias_q,
            use_bias_k=config.use_bias_k,
            use_bias_v=config.use_bias_v,
            use_q_scale=config.use_q_scale,
            smolgen_config=layer_config.smolgen,
            defaults=defaults,
            smol_gen_dense=smol_gen_dense,
            deepnorm_beta=deepnorm_beta,
            rngs=rngs,
        )
        norm_layer = get_norm_layer(defaults.norm_type)

        self.alpha = math.pow(2.0 * config.num_blocks, -0.25)
        self.ln1 = norm_layer(in_features, epsilon=1e-3, rngs=rngs)
        self.ffn = Ffn(
            in_features=in_features,
            hidden_features=layer_config.dff,
            hidden_activation=defaults.ffn_activation,
            deepnorm_beta=deepnorm_beta,
            rngs=rngs,
        )
        self.ln2 = norm_layer(in_features, epsilon=1e-3, rngs=rngs)
        self.block_style = config.block_style
        if self.block_style not in (
            _SEQUENTIAL_BLOCK_STYLE,
            _PALM_PARALLEL_BLOCK_STYLE,
        ):
            raise ValueError(f"Unsupported block style: {self.block_style}")

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.block_style == _SEQUENTIAL_BLOCK_STYLE:
            x = x + self.mha(x) * self.alpha
            out1 = self.ln1(x)
            ffn_out = self.ffn(out1)
            return self.ln2(out1 + ffn_out * self.alpha)

        normed = self.ln1(x)
        attn_out = self.mha(normed)
        ffn_out = self.ffn(normed)
        return self.ln2(x + (attn_out + ffn_out) * self.alpha)


class MultiHeadAttention(nnx.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        in_features: int,
        d_model: int,
        heads: int,
        kv_heads: int,
        use_bias_q: bool,
        use_bias_k: bool,
        use_bias_v: bool,
        use_q_scale: bool,
        smolgen_config: Optional[model_config_pb2.SmolgenConfig],
        defaults: model_config_pb2.DefaultsConfig,
        smol_gen_dense: Optional[nnx.Linear],
        deepnorm_beta: float,
        *,
        rngs: nnx.Rngs,
    ):
        depth = d_model
        assert depth % heads == 0, (
            "Model depth must be divisible by the number of heads."
        )
        self.activation = defaults.activation
        self.depth = depth
        self.num_heads = heads
        self.kv_heads = kv_heads
        assert self.num_heads % self.kv_heads == 0
        head_depth = depth // self.num_heads

        self.q = nnx.Linear(
            in_features=in_features,
            out_features=depth,
            use_bias=use_bias_q,
            rngs=rngs,
        )
        self.q_scale = None
        if use_q_scale:
            self.q_scale = nnx.Param(
                jnp.ones((self.num_heads, 1, 1), dtype=jnp.float32)
            )
        self.k = nnx.Linear(
            in_features=in_features,
            out_features=self.kv_heads * head_depth,
            use_bias=use_bias_k,
            rngs=rngs,
        )
        deepnorm_init = flax_initializers.variance_scaling(
            scale=deepnorm_beta,
            mode="fan_avg",
            distribution="truncated_normal",
        )

        self.v = nnx.Linear(
            in_features=in_features,
            out_features=self.kv_heads * head_depth,
            use_bias=use_bias_v,
            kernel_init=deepnorm_init,
            rngs=rngs,
        )
        self.output_dense = nnx.Linear(
            in_features=depth,
            out_features=in_features,
            kernel_init=deepnorm_init,
            rngs=rngs,
        )

        assert (smol_gen_dense is not None) == (smolgen_config is not None)
        self.smolgen: Optional[Smolgen]
        if smol_gen_dense is not None:
            self.smolgen = Smolgen(
                in_features=in_features,
                config=smolgen_config,
                defaults=defaults,
                heads=heads,
                weight_gen_dense=smol_gen_dense,
                rngs=rngs,
            )
        else:
            self.smolgen = None

    def __call__(self, x: jax.Array) -> jax.Array:
        q, k, v = self.q(x), self.k(x), self.v(x)

        head_depth = self.depth // self.num_heads
        # Reshape for multi-head attention.
        q = q.reshape((-1, self.num_heads, head_depth)).transpose((1, 0, 2))
        k = k.reshape((-1, self.kv_heads, head_depth)).transpose((1, 0, 2))
        v = v.reshape((-1, self.kv_heads, head_depth)).transpose((1, 0, 2))

        if self.kv_heads != self.num_heads:
            group_size = self.num_heads // self.kv_heads
            k = jnp.repeat(k, group_size, axis=0)
            v = jnp.repeat(v, group_size, axis=0)

        if self.q_scale is not None:
            q = q * self.q_scale.value.astype(q.dtype)

        # Scaled dot-product attention.
        logits = jnp.einsum("...qd,...kd->...qk", q, k)
        logits /= jnp.sqrt(k.shape[-1]).astype(k.dtype)

        if self.smolgen is not None:
            logits += self.smolgen(x)

        attention_weights = nnx.softmax(logits, axis=-1)
        scaled_attention = jnp.matmul(attention_weights, v)

        # Reshape back to original dimensions.
        scaled_attention = scaled_attention.transpose((1, 0, 2)).reshape(
            (-1, self.depth)
        )
        return self.output_dense(scaled_attention)


class Smolgen(nnx.Module):
    """Smolgen module for generating attention biases."""

    def __init__(
        self,
        in_features: int,
        config: model_config_pb2.SmolgenConfig,
        defaults: model_config_pb2.DefaultsConfig,
        heads: int,
        weight_gen_dense: nnx.Linear,
        *,
        rngs: nnx.Rngs,
    ):
        self.heads = heads
        norm_layer = get_norm_layer(defaults.norm_type, allow_dynamic_erf=False)
        self.compress = nnx.Linear(
            in_features=in_features,
            out_features=config.hidden_channels,
            use_bias=False,
            rngs=rngs,
        )
        self.dense1 = nnx.Linear(
            in_features=config.hidden_channels * 64,
            out_features=config.hidden_size,
            use_bias=config.use_bias_dense1,
            rngs=rngs,
        )
        # Don't use RMSNorm in Smolgen.
        self.ln1 = nnx.LayerNorm(config.hidden_size, epsilon=1e-3, rngs=rngs)

        self.dense2 = nnx.Linear(
            in_features=config.hidden_size,
            out_features=config.gen_size * heads,
            use_bias=config.use_bias_dense2,
            rngs=rngs,
        )
        self.ln2 = norm_layer(config.gen_size * heads, epsilon=1e-3, rngs=rngs)
        self.weight_gen_dense = weight_gen_dense
        self.activation = config.activation or defaults.activation

    def __call__(self, x: jax.Array) -> jax.Array:
        compressed = self.compress(x).flatten()
        hidden = self.dense1(compressed)
        hidden = get_activation(self.activation)(hidden)
        hidden = self.ln1(hidden)

        gen_from = self.dense2(hidden)
        gen_from = get_activation(self.activation)(gen_from)
        gen_from = self.ln2(gen_from)
        gen_from = gen_from.reshape((self.heads, -1))

        out = self.weight_gen_dense(gen_from)
        return out.reshape((self.heads, 64, 64))
