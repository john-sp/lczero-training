#!/usr/bin/env python3
"""Calculate parameter counts for Lc0 model configurations.

Supports:
- regular heads (attention policy + token heads)
- headpremap + simple heads
- SwiGLU FFN accounting
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Optional


# =============================================================================
# CONFIG -- edit these values to match your model
# =============================================================================

INPUT_CHANNELS = 112

# Defaults
IS_RMSNORM = True  # True = RMSNorm (scale only), False = LayerNorm (scale+bias)
FFN_ACTIVATION: Literal["default", "swiglu"] = "swiglu"

# Embedding
EMB_DENSE_SIZE = 128
EMB_EMBEDDING_SIZE = 256
EMB_DFF = 512

# Encoder
ENC_NUM_BLOCKS = 10
ENC_DFF = 512
ENC_D_MODEL = 256
ENC_HEADS = 8
ENC_KV_HEADS: Optional[int] = 8  # None = same as ENC_HEADS (full MHA)
ENC_USE_BIAS_Q = False
ENC_USE_BIAS_K = False
ENC_USE_BIAS_V = False
ENC_USE_Q_SCALE = False


@dataclasses.dataclass
class SmolgenCfg:
    hidden_channels: int
    hidden_size: int
    gen_size: int
    use_bias_dense1: bool = True
    use_bias_dense2: bool = True


# Smolgen -- set ENC_SMOLGEN to None to disable.
ENC_SMOLGEN: Optional[SmolgenCfg] = None

# Example:
ENC_SMOLGEN = SmolgenCfg(hidden_channels=32, hidden_size=256, gen_size=256)

# Head architecture mode.
HEAD_ARCH: Literal["regular", "simple"] = "simple"

# ---------------------------
# Regular-head configuration
# ---------------------------

# Shared policy embedding -- set to None to give each head its own embedding.
SHARED_POLICY_EMBEDDING_SIZE: Optional[int] = 256


@dataclasses.dataclass
class PolicyHeadCfg:
    name: str
    embedding_size: int  # ignored when SHARED_POLICY_EMBEDDING_SIZE is set
    d_model: int


@dataclasses.dataclass
class ValueHeadCfg:
    name: str
    num_channels: int
    has_error_output: bool = False
    num_categorical_buckets: int = 0


@dataclasses.dataclass
class MovesLeftHeadCfg:
    name: str
    num_channels: int


POLICY_HEADS: list[PolicyHeadCfg] = [
    PolicyHeadCfg(name="vanilla", embedding_size=256, d_model=256),
    PolicyHeadCfg(name="optimistic_st", embedding_size=128, d_model=128),
    PolicyHeadCfg(name="soft", embedding_size=128, d_model=128),
]

VALUE_HEADS: list[ValueHeadCfg] = [
    ValueHeadCfg(name="winner", num_channels=32),
    ValueHeadCfg(
        name="q",
        num_channels=16,
        has_error_output=True,
        num_categorical_buckets=32,
    ),
    ValueHeadCfg(
        name="st",
        num_channels=32,
        has_error_output=True,
        num_categorical_buckets=32,
    ),
]

MOVESLEFT_HEADS: list[MovesLeftHeadCfg] = [
    MovesLeftHeadCfg(name="main", num_channels=8),
]

# --------------------------
# Simple-head configuration
# --------------------------


@dataclasses.dataclass
class HeadpremapCfg:
    intermediate_size: int
    output_size: int
    use_gating: bool = True


@dataclasses.dataclass
class SimpleHeadCfg:
    name: str
    hidden_size: int
    has_error_output: bool = False
    num_categorical_buckets: int = 0


HEADPREMAP = HeadpremapCfg(
    intermediate_size=64, output_size=284, use_gating=True
)

SIMPLE_POLICY_HEADS: list[SimpleHeadCfg] = [
    SimpleHeadCfg(name="vanilla", hidden_size=128),
    SimpleHeadCfg(name="optimistic_st", hidden_size=128),
    SimpleHeadCfg(name="soft", hidden_size=64),
]

SIMPLE_VALUE_HEADS: list[SimpleHeadCfg] = [
    SimpleHeadCfg(name="winner", hidden_size=256),
    SimpleHeadCfg(
        name="q",
        hidden_size=128,
        has_error_output=True,
        num_categorical_buckets=16,
    ),
    SimpleHeadCfg(
        name="st",
        hidden_size=128,
        has_error_output=True,
        num_categorical_buckets=16,
    ),
]

SIMPLE_MOVESLEFT_HEADS: list[SimpleHeadCfg] = [
    SimpleHeadCfg(name="main", hidden_size=32),
]


# =============================================================================
# CALCULATION -- nothing below here needs to change
# =============================================================================


def linear(in_f: int, out_f: int, bias: bool = True) -> int:
    return in_f * out_f + (out_f if bias else 0)


def norm(features: int) -> int:
    return features if IS_RMSNORM else 2 * features


def F(n: int) -> str:
    return f"{n:,}"


def _ffn_params(in_features: int, hidden_features: int) -> tuple[int, int]:
    dense1 = linear(in_features, hidden_features)
    dense_gate = (
        linear(in_features, hidden_features, bias=False)
        if FFN_ACTIVATION == "swiglu"
        else 0
    )
    dense2 = linear(hidden_features, in_features)
    return dense1 + dense_gate + dense2, dense_gate


def calc_embedding() -> tuple[int, int]:
    E = EMB_EMBEDDING_SIZE
    preprocess = linear(64 * 12, 64 * EMB_DENSE_SIZE)
    embedding = linear(INPUT_CHANNELS + EMB_DENSE_SIZE, E)
    norm1 = norm(E)
    gating = 2 * 64 * E  # mult_gate + add_gate
    ffn, ffn_gate = _ffn_params(E, EMB_DFF)
    out_norm = norm(E)
    return preprocess + embedding + norm1 + gating + ffn + out_norm, ffn_gate


def calc_encoder() -> tuple[int, int, int, int, int, int, int, int]:
    """Return per-block + component breakdown.

    Returns:
      (per_block, qkv, mha, smolgen_per_block, ffn, ln, smolgen_shared, ffn_gate)
    """
    E = EMB_EMBEDDING_SIZE
    heads = ENC_HEADS
    kv_heads = ENC_KV_HEADS if ENC_KV_HEADS is not None else heads
    head_depth = ENC_D_MODEL // heads
    kv_dim = kv_heads * head_depth

    q = linear(E, ENC_D_MODEL, bias=ENC_USE_BIAS_Q)
    k = linear(E, kv_dim, bias=ENC_USE_BIAS_K)
    v = linear(E, kv_dim, bias=ENC_USE_BIAS_V)
    qkv = q + k + v
    output_dense = linear(ENC_D_MODEL, E)
    q_scale = heads if ENC_USE_Q_SCALE else 0
    mha = qkv + output_dense + q_scale

    smolgen_per_block = 0
    smolgen_shared = 0
    if ENC_SMOLGEN is not None:
        sm = ENC_SMOLGEN
        smolgen_per_block += linear(E, sm.hidden_channels, bias=False)
        smolgen_per_block += linear(
            sm.hidden_channels * 64,
            sm.hidden_size,
            bias=sm.use_bias_dense1,
        )
        smolgen_per_block += norm(sm.hidden_size)
        smolgen_per_block += linear(
            sm.hidden_size,
            sm.gen_size * heads,
            bias=sm.use_bias_dense2,
        )
        smolgen_per_block += norm(sm.gen_size * heads)
        smolgen_shared = linear(sm.gen_size, 64 * 64, bias=False)

    ffn, ffn_gate = _ffn_params(E, ENC_DFF)
    ln = norm(E) * 2  # ln1 + ln2

    per_block = mha + smolgen_per_block + ffn + ln
    return (
        per_block,
        qkv,
        mha,
        smolgen_per_block,
        ffn,
        ln,
        smolgen_shared,
        ffn_gate,
    )


def calc_policy_regular() -> tuple[int, int, dict[str, int]]:
    """Return (total, shared_emb_params, per_head_params)."""
    E = EMB_EMBEDDING_SIZE
    shared_emb_params = 0
    if SHARED_POLICY_EMBEDDING_SIZE is not None:
        shared_emb_params = linear(E, SHARED_POLICY_EMBEDDING_SIZE)

    per_head: dict[str, int] = {}
    for head in POLICY_HEADS:
        if SHARED_POLICY_EMBEDDING_SIZE is not None:
            emb_size = SHARED_POLICY_EMBEDDING_SIZE
            tokens = 0
        else:
            emb_size = head.embedding_size
            tokens = linear(E, emb_size)
        q = linear(emb_size, head.d_model)
        k = linear(emb_size, head.d_model)
        promo = linear(head.d_model, 4, bias=False)
        per_head[head.name] = tokens + q + k + promo

    total = shared_emb_params + sum(per_head.values())
    return total, shared_emb_params, per_head


def calc_value_regular() -> tuple[int, dict[str, int]]:
    E = EMB_EMBEDDING_SIZE
    per_head: dict[str, int] = {}
    for head in VALUE_HEADS:
        total = linear(E, head.num_channels)
        total += linear(head.num_channels * 64, 128)
        total += linear(128, 3)
        if head.has_error_output:
            total += linear(128, 1)
        if head.num_categorical_buckets > 0:
            total += linear(128, head.num_categorical_buckets)
        per_head[head.name] = total
    return sum(per_head.values()), per_head


def calc_movesleft_regular() -> tuple[int, dict[str, int]]:
    E = EMB_EMBEDDING_SIZE
    per_head: dict[str, int] = {}
    for head in MOVESLEFT_HEADS:
        total = linear(E, head.num_channels)
        total += linear(head.num_channels * 64, 128)
        total += linear(128, 1)
        per_head[head.name] = total
    return sum(per_head.values()), per_head


def calc_headpremap() -> int:
    E = EMB_EMBEDDING_SIZE
    gate = E if HEADPREMAP.use_gating else 0
    reduce = linear(E, HEADPREMAP.intermediate_size)
    project = linear(64 * HEADPREMAP.intermediate_size, HEADPREMAP.output_size)
    return gate + reduce + project


def calc_policy_simple() -> tuple[int, dict[str, int]]:
    # Policy output is fixed at 1858.
    per_head: dict[str, int] = {}
    for head in SIMPLE_POLICY_HEADS:
        total = linear(HEADPREMAP.output_size, head.hidden_size)
        total += linear(head.hidden_size, 1858)
        per_head[head.name] = total
    return sum(per_head.values()), per_head


def calc_value_simple() -> tuple[int, dict[str, int]]:
    per_head: dict[str, int] = {}
    for head in SIMPLE_VALUE_HEADS:
        total = linear(HEADPREMAP.output_size, head.hidden_size)
        total += linear(head.hidden_size, 3)
        if head.has_error_output:
            total += linear(head.hidden_size, 1)
        if head.num_categorical_buckets > 0:
            total += linear(head.hidden_size, head.num_categorical_buckets)
        per_head[head.name] = total
    return sum(per_head.values()), per_head


def calc_movesleft_simple() -> tuple[int, dict[str, int]]:
    per_head: dict[str, int] = {}
    for head in SIMPLE_MOVESLEFT_HEADS:
        total = linear(HEADPREMAP.output_size, head.hidden_size)
        total += linear(head.hidden_size, 1)
        per_head[head.name] = total
    return sum(per_head.values()), per_head


def _validate_config() -> None:
    if HEAD_ARCH not in {"regular", "simple"}:
        raise ValueError("HEAD_ARCH must be 'regular' or 'simple'.")
    if FFN_ACTIVATION not in {"default", "swiglu"}:
        raise ValueError("FFN_ACTIVATION must be 'default' or 'swiglu'.")
    if ENC_D_MODEL % ENC_HEADS != 0:
        raise ValueError("ENC_D_MODEL must be divisible by ENC_HEADS.")
    if HEAD_ARCH == "simple":
        if HEADPREMAP.intermediate_size <= 0 or HEADPREMAP.output_size <= 0:
            raise ValueError(
                "HEADPREMAP intermediate_size/output_size must be positive."
            )
        if not (
            SIMPLE_POLICY_HEADS or SIMPLE_VALUE_HEADS or SIMPLE_MOVESLEFT_HEADS
        ):
            raise ValueError("Simple mode requires at least one simple head.")


def main() -> None:
    _validate_config()

    emb_total, emb_ffn_gate = calc_embedding()
    (
        per_block,
        qkv,
        mha,
        smolgen_pb,
        ffn,
        ln,
        smolgen_shared,
        enc_ffn_gate,
    ) = calc_encoder()
    enc_total = per_block * ENC_NUM_BLOCKS + smolgen_shared

    if HEAD_ARCH == "regular":
        policy_total, shared_emb_params, policy_per_head = calc_policy_regular()
        value_total, value_per_head = calc_value_regular()
        ml_total, ml_per_head = calc_movesleft_regular()
        headpremap_total = 0
    else:
        headpremap_total = calc_headpremap()
        policy_total, policy_per_head = calc_policy_simple()
        value_total, value_per_head = calc_value_simple()
        ml_total, ml_per_head = calc_movesleft_simple()
        shared_emb_params = 0

    grand_total = (
        emb_total
        + enc_total
        + headpremap_total
        + policy_total
        + value_total
        + ml_total
    )

    print(f"Head mode: {HEAD_ARCH}")
    print(f"FFN activation: {FFN_ACTIVATION}")
    print(f"Model parameters: {F(grand_total)}")
    print()
    print(f"Embedding:             {F(emb_total)}")
    if emb_ffn_gate:
        print(f"  embedding ffn gate: {F(emb_ffn_gate)}")
    print()
    print(f"Encoder ({ENC_NUM_BLOCKS} blocks): {F(enc_total)}")
    print(f"  Per block:  {F(per_block)}")
    print(f"    MHA:      {F(mha)}  (QKV: {F(qkv)})")
    if smolgen_pb:
        print(f"    Smolgen:  {F(smolgen_pb)}")
    print(f"    FFN:      {F(ffn)}")
    if enc_ffn_gate:
        print(f"      gate:   {F(enc_ffn_gate)}")
    print(f"    LN:       {F(ln)}")
    if smolgen_shared:
        print(f"  Shared (weight_gen_dense): {F(smolgen_shared)}")

    if HEAD_ARCH == "simple":
        print()
        print(f"Headpremap:            {F(headpremap_total)}")

    print()
    print(f"Policy heads:          {F(policy_total)}")
    if shared_emb_params:
        print(f"  Shared embedding:    {F(shared_emb_params)}")
    for name, count in policy_per_head.items():
        print(f"  {name}: {F(count)}")

    print()
    print(f"Value heads:           {F(value_total)}")
    for name, count in value_per_head.items():
        print(f"  {name}: {F(count)}")

    print()
    print(f"Moves-left heads:      {F(ml_total)}")
    for name, count in ml_per_head.items():
        print(f"  {name}: {F(count)}")


if __name__ == "__main__":
    main()
