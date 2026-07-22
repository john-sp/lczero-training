"""Tests for the encoder module."""

import jax
from flax import nnx

from lczero_training.model.encoder import MultiHeadAttention
from proto import model_config_pb2


def test_gqa_shapes() -> None:
    """Test GQA parameter shapes and forward pass."""
    # Configuration: 8 heads, 2 KV heads, d_model=64.
    # This means head_depth = 64 / 8 = 8.
    # KV output features should be 2 * 8 = 16.
    config = model_config_pb2.EncoderConfig()
    config.d_model = 64
    config.heads = 8
    config.kv_heads = 2
    config.use_bias_q = True
    config.use_bias_k = True
    config.use_bias_v = True

    defaults = model_config_pb2.DefaultsConfig()

    rngs = nnx.Rngs(params=42)

    mha = MultiHeadAttention(
        in_features=64,
        config=config,
        defaults=defaults,
        smol_gen_dense=None,
        deepnorm_beta=1.0,
        rngs=rngs,
    )

    # Check dimensions
    # Q: out_features = d_model = 64
    assert mha.q.kernel.value.shape == (64, 64)
    # K: out_features = kv_heads * head_depth = 2 * 8 = 16
    assert mha.k.kernel.value.shape == (64, 16)
    # V: out_features = kv_heads * head_depth = 2 * 8 = 16
    assert mha.v.kernel.value.shape == (64, 16)

    # Forward pass
    # Input shape: [seq_len, embedding_dim] = [10, 64]
    x = jax.random.normal(jax.random.key(0), (10, 64))

    output = mha(x)

    # Output should match input shape (residual connection handled outside MHA,
    # but MHA output dense projects back to d_model)
    # output_dense projects to in_features (which is 64 here).
    assert output.shape == (10, 64)


def test_mha_shapes() -> None:
    """Test standard MHA parameter shapes and forward pass."""
    # Configuration: 8 heads, 8 KV heads (implicit), d_model=64.
    config = model_config_pb2.EncoderConfig()
    config.d_model = 64
    config.heads = 8
    # kv_heads defaults to heads if not set? No, in __init__ logic:
    # kv_heads = config.kv_heads if config.HasField("kv_heads") else config.heads
    # So if we don't set it, it should be 8.
    config.use_bias_q = True
    config.use_bias_k = True
    config.use_bias_v = True

    defaults = model_config_pb2.DefaultsConfig()

    rngs = nnx.Rngs(params=42)

    mha = MultiHeadAttention(
        in_features=64,
        config=config,
        defaults=defaults,
        smol_gen_dense=None,
        deepnorm_beta=1.0,
        rngs=rngs,
    )

    # Check dimensions
    assert mha.q.kernel.value.shape == (64, 64)
    assert mha.k.kernel.value.shape == (64, 64)
    assert mha.v.kernel.value.shape == (64, 64)

    x = jax.random.normal(jax.random.key(0), (10, 64))
    output = mha(x)
    assert output.shape == (10, 64)
