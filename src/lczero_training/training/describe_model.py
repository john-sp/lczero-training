"""Parameter and FLOP accounting for transformer model configs."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from pathlib import Path

from google.protobuf import text_format
from google.protobuf.message import DecodeError

from proto import model_config_pb2, net_pb2
from proto.root_config_pb2 import RootConfig

_BOARD_SIZE = 64
_INPUT_CHANNELS = 112
_POSITIONAL_CHANNELS = 12
_VALUE_HIDDEN_SIZE = 128
_POLICY_OUTPUT_SIZE = 1858


@dataclasses.dataclass(frozen=True)
class Stats:
    params: int = 0
    flops: int = 0

    def __add__(self, other: "Stats") -> "Stats":
        return Stats(
            params=self.params + other.params,
            flops=self.flops + other.flops,
        )


@dataclasses.dataclass
class Breakdown:
    name: str
    stats: Stats = dataclasses.field(default_factory=Stats)
    children: list["Breakdown"] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class SmolgenConfig:
    hidden_channels: int
    hidden_size: int
    gen_size: int
    use_bias_dense1: bool
    use_bias_dense2: bool


@dataclasses.dataclass(frozen=True)
class LayerConfig:
    dff: int
    heads: int
    kv_heads: int
    smolgen: SmolgenConfig | None


def load_model_config(path: str | Path) -> model_config_pb2.ModelConfig:
    """Loads a RootConfig or standalone ModelConfig textproto."""
    contents = Path(path).read_text()
    root_config = RootConfig()
    try:
        text_format.Parse(contents, root_config)
    except (DecodeError, text_format.ParseError):
        model_config = model_config_pb2.ModelConfig()
        text_format.Parse(contents, model_config)
        return model_config

    if root_config.HasField("model"):
        return root_config.model

    model_config = model_config_pb2.ModelConfig()
    text_format.Parse(contents, model_config)
    return model_config


def describe_model_config(
    config: model_config_pb2.ModelConfig,
) -> Breakdown:
    """Builds a parameter and FLOP breakdown for one model invocation."""
    _validate_transformer_config(config)
    embedding_size = config.embedding.embedding_size
    root = Breakdown("Model")

    embedding = _embedding(config)
    root.children.append(embedding)

    encoder = _encoder(config)
    root.children.append(encoder)

    if config.HasField("headpremap"):
        headpremap = _headpremap(config)
        root.children.append(headpremap)
        root.children.append(_simple_policy_heads(config))
        root.children.append(_simple_value_heads(config))
        root.children.append(_simple_movesleft_heads(config))
    else:
        root.children.append(_policy_heads(config, embedding_size))
        root.children.append(_value_heads(config, embedding_size))
        root.children.append(_movesleft_heads(config, embedding_size))

    root.stats = _sum_children(root.children)
    return root


def format_breakdown(root: Breakdown) -> str:
    """Formats a model breakdown for CLI output."""
    lines = [
        f"Model parameters: {_fmt_int(root.stats.params)}",
        f"Model FLOPs:      {_fmt_int(root.stats.flops)}",
        "",
        (
            "FLOPs are for one forward pass of one position. Dense and "
            "attention matmuls count one multiply plus one add as two FLOPs; "
            "biases, norms, activations, softmax, and indexing are omitted."
        ),
        "",
    ]
    for child in root.children:
        _append_breakdown(lines, child, indent=0)
        lines.append("")
    return "\n".join(lines).rstrip()


def describe_model_text(config_filename: str) -> str:
    return format_breakdown(
        describe_model_config(load_model_config(config_filename))
    )


def _validate_transformer_config(config: model_config_pb2.ModelConfig) -> None:
    has_headpremap = config.HasField("headpremap")
    has_simple_heads = (
        len(config.simple_value_head) > 0
        or len(config.simple_movesleft_head) > 0
        or len(config.simple_policy_head) > 0
    )
    has_regular_heads = (
        len(config.value_head) > 0
        or len(config.movesleft_head) > 0
        or len(config.policy_head) > 0
    )
    if has_headpremap and has_regular_heads:
        raise ValueError(
            "model headpremap/simple heads cannot be mixed with regular heads."
        )
    if has_simple_heads and not has_headpremap:
        raise ValueError("model simple heads require headpremap.")
    if has_headpremap and not has_simple_heads:
        raise ValueError("model headpremap requires at least one simple head.")
    if has_headpremap and config.shared_policy_embedding_size:
        raise ValueError(
            "model shared_policy_embedding_size is unsupported with headpremap."
        )
    if config.encoder.num_blocks <= 0:
        raise ValueError("model.encoder.num_blocks must be greater than zero.")
    if config.embedding.embedding_size <= 0:
        raise ValueError("model.embedding.embedding_size must be set.")
    if config.embedding.dense_size <= 0:
        raise ValueError("model.embedding.dense_size must be set.")
    if config.embedding.dff <= 0:
        raise ValueError("model.embedding.dff must be set.")
    if config.encoder.d_model <= 0:
        raise ValueError("model.encoder.d_model must be set.")
    if config.encoder.heads <= 0:
        raise ValueError("model.encoder.heads must be set.")
    if config.encoder.d_model % config.encoder.heads != 0:
        raise ValueError("model.encoder.d_model must be divisible by heads.")


def _embedding(config: model_config_pb2.ModelConfig) -> Breakdown:
    dense_size = config.embedding.dense_size
    embedding_size = config.embedding.embedding_size
    dff = config.embedding.dff
    input_size = _INPUT_CHANNELS + dense_size
    preprocess_out = _BOARD_SIZE * dense_size

    children = [
        Breakdown(
            "Preprocess",
            _linear(
                _BOARD_SIZE * _POSITIONAL_CHANNELS,
                preprocess_out,
                tokens=1,
            ),
        ),
        Breakdown(
            "Token embedding",
            _linear(input_size, embedding_size, tokens=_BOARD_SIZE),
        ),
        Breakdown("Norm", _norm(embedding_size, config.defaults.norm_type)),
        Breakdown("MA gating", Stats(params=2 * _BOARD_SIZE * embedding_size)),
        Breakdown(
            "FFN",
            _ffn(
                embedding_size,
                dff,
                _ffn_activation(config),
                tokens=_BOARD_SIZE,
            ),
        ),
        Breakdown("Out norm", _norm(embedding_size, config.defaults.norm_type)),
    ]
    return Breakdown("Embedding", _sum_children(children), children)


def _encoder(config: model_config_pb2.ModelConfig) -> Breakdown:
    layer_configs = _build_layer_configs(config.encoder)
    embedding_size = config.embedding.embedding_size
    children = []
    per_layer = [
        _encoder_block(config, layer_config, embedding_size)
        for layer_config in layer_configs
    ]
    total = _sum_children(per_layer)

    if _has_smolgen(config.encoder):
        smolgen = _smolgen_from_proto(config.encoder.smolgen)
        shared = _linear(
            smolgen.gen_size,
            _BOARD_SIZE * _BOARD_SIZE,
            use_bias=False,
            tokens=config.encoder.heads,
        )
        total += shared
    else:
        shared = Stats()

    for label, block in _group_layer_breakdowns(per_layer):
        block.name = label
        children.append(block)

    if shared.params:
        children.append(Breakdown("Shared (weight_gen_dense)", shared))

    name = f"Encoder ({config.encoder.num_blocks} blocks)"
    return Breakdown(name, total, children)


def _encoder_block(
    config: model_config_pb2.ModelConfig,
    layer_config: LayerConfig,
    in_features: int,
) -> Breakdown:
    children = [
        _mha(config, layer_config, in_features),
        _smolgen(layer_config, config.defaults.norm_type, in_features),
        Breakdown(
            "FFN",
            _ffn(
                in_features,
                layer_config.dff,
                _ffn_activation(config),
                tokens=_BOARD_SIZE,
            ),
        ),
        Breakdown(
            "LN",
            _norm(in_features, config.defaults.norm_type)
            + _norm(in_features, config.defaults.norm_type),
        ),
    ]
    if children[1].stats.params == 0:
        children.pop(1)
    return Breakdown("Per block", _sum_children(children), children)


def _mha(
    config: model_config_pb2.ModelConfig,
    layer_config: LayerConfig,
    in_features: int,
) -> Breakdown:
    d_model = config.encoder.d_model
    heads = layer_config.heads
    kv_heads = layer_config.kv_heads
    head_depth = d_model // heads
    kv_depth = kv_heads * head_depth

    q = _linear(
        in_features,
        d_model,
        use_bias=config.encoder.use_bias_q,
        tokens=_BOARD_SIZE,
    )
    k = _linear(
        in_features,
        kv_depth,
        use_bias=config.encoder.use_bias_k,
        tokens=_BOARD_SIZE,
    )
    v = _linear(
        in_features,
        kv_depth,
        use_bias=config.encoder.use_bias_v,
        tokens=_BOARD_SIZE,
    )
    output = _linear(d_model, in_features, tokens=_BOARD_SIZE)
    q_scale = Stats(params=heads) if config.encoder.use_q_scale else Stats()
    attention = Stats(
        flops=(
            2 * heads * _BOARD_SIZE * _BOARD_SIZE * head_depth
            + 2 * heads * _BOARD_SIZE * _BOARD_SIZE * head_depth
        )
    )
    qkv = q + k + v + q_scale
    return Breakdown(
        "MHA",
        qkv + output + attention,
        [
            Breakdown("QKV", qkv),
            Breakdown("Attention", attention),
            Breakdown("Output dense", output),
        ],
    )


def _smolgen(
    layer_config: LayerConfig,
    norm_type: model_config_pb2.NormType,
    in_features: int,
) -> Breakdown:
    config = layer_config.smolgen
    if config is None:
        return Breakdown("Smolgen")
    generated_size = config.gen_size * layer_config.heads
    children = [
        Breakdown(
            "Compress",
            _linear(
                in_features,
                config.hidden_channels,
                use_bias=False,
                tokens=_BOARD_SIZE,
            ),
        ),
        Breakdown(
            "Dense 1",
            _linear(
                config.hidden_channels * _BOARD_SIZE,
                config.hidden_size,
                use_bias=config.use_bias_dense1,
            ),
        ),
        Breakdown("LN 1", _norm(config.hidden_size)),
        Breakdown(
            "Dense 2",
            _linear(
                config.hidden_size,
                generated_size,
                use_bias=config.use_bias_dense2,
            ),
        ),
        Breakdown("LN 2", _norm(generated_size, norm_type, dynamic_erf=False)),
    ]
    return Breakdown("Smolgen", _sum_children(children), children)


def _policy_heads(
    config: model_config_pb2.ModelConfig,
    in_features: int,
) -> Breakdown:
    children = []
    shared_embedding_size = config.shared_policy_embedding_size
    if shared_embedding_size:
        children.append(
            Breakdown(
                "Shared embedding",
                _linear(in_features, shared_embedding_size, tokens=_BOARD_SIZE),
            )
        )

    for head in config.policy_head:
        embedding_size = shared_embedding_size or head.embedding_size
        stats = Stats()
        if not shared_embedding_size:
            stats += _linear(in_features, embedding_size, tokens=_BOARD_SIZE)
        stats += _linear(embedding_size, head.d_model, tokens=_BOARD_SIZE)
        stats += _linear(embedding_size, head.d_model, tokens=_BOARD_SIZE)
        stats += Stats(flops=2 * _BOARD_SIZE * _BOARD_SIZE * head.d_model)
        stats += _linear(
            head.d_model,
            4,
            use_bias=False,
            tokens=8,
        )
        children.append(Breakdown(_head_name(head.name), stats))

    return Breakdown("Policy heads", _sum_children(children), children)


def _value_heads(
    config: model_config_pb2.ModelConfig,
    in_features: int,
) -> Breakdown:
    children = []
    for head in config.value_head:
        stats = _linear(in_features, head.num_channels, tokens=_BOARD_SIZE)
        stats += _linear(head.num_channels * _BOARD_SIZE, _VALUE_HIDDEN_SIZE)
        stats += _linear(_VALUE_HIDDEN_SIZE, 3)
        if head.has_error_output:
            stats += _linear(_VALUE_HIDDEN_SIZE, 1)
        if head.num_categorical_buckets:
            stats += _linear(_VALUE_HIDDEN_SIZE, head.num_categorical_buckets)
        children.append(Breakdown(_head_name(head.name), stats))
    return Breakdown("Value heads", _sum_children(children), children)


def _movesleft_heads(
    config: model_config_pb2.ModelConfig,
    in_features: int,
) -> Breakdown:
    children = []
    for head in config.movesleft_head:
        stats = _linear(in_features, head.num_channels, tokens=_BOARD_SIZE)
        stats += _linear(head.num_channels * _BOARD_SIZE, _VALUE_HIDDEN_SIZE)
        stats += _linear(_VALUE_HIDDEN_SIZE, 1)
        children.append(Breakdown(_head_name(head.name), stats))
    return Breakdown("Moves-left heads", _sum_children(children), children)


def _headpremap(config: model_config_pb2.ModelConfig) -> Breakdown:
    in_features = config.embedding.embedding_size
    intermediate_size = config.headpremap.intermediate_size
    output_size = config.headpremap.output_size
    stats = Stats()
    if config.headpremap.use_gating:
        stats += Stats(params=in_features)
    stats += _linear(in_features, intermediate_size, tokens=_BOARD_SIZE)
    stats += _linear(intermediate_size * _BOARD_SIZE, output_size)
    return Breakdown("Head premap", stats)


def _simple_policy_heads(config: model_config_pb2.ModelConfig) -> Breakdown:
    children = [
        Breakdown(
            _head_name(head.name),
            _simple_head(
                config.headpremap.output_size,
                head.hidden_size,
                _POLICY_OUTPUT_SIZE,
            ),
        )
        for head in config.simple_policy_head
    ]
    return Breakdown("Simple policy heads", _sum_children(children), children)


def _simple_value_heads(config: model_config_pb2.ModelConfig) -> Breakdown:
    children = []
    for head in config.simple_value_head:
        stats = _simple_head(config.headpremap.output_size, head.hidden_size, 3)
        if head.has_error_output:
            stats += _linear(head.hidden_size, 1)
        if head.num_categorical_buckets:
            stats += _linear(head.hidden_size, head.num_categorical_buckets)
        children.append(Breakdown(_head_name(head.name), stats))
    return Breakdown("Simple value heads", _sum_children(children), children)


def _simple_movesleft_heads(config: model_config_pb2.ModelConfig) -> Breakdown:
    children = [
        Breakdown(
            _head_name(head.name),
            _simple_head(config.headpremap.output_size, head.hidden_size, 1),
        )
        for head in config.simple_movesleft_head
    ]
    return Breakdown(
        "Simple moves-left heads",
        _sum_children(children),
        children,
    )


def _simple_head(in_features: int, hidden_size: int, output_size: int) -> Stats:
    return _linear(in_features, hidden_size) + _linear(hidden_size, output_size)


def _build_layer_configs(
    config: model_config_pb2.EncoderConfig,
) -> list[LayerConfig]:
    kv_heads = config.kv_heads or config.heads
    smolgen = (
        _smolgen_from_proto(config.smolgen) if _has_smolgen(config) else None
    )
    layers = [
        LayerConfig(
            dff=config.dff,
            heads=config.heads,
            kv_heads=kv_heads,
            smolgen=smolgen,
        )
        for _ in range(config.num_blocks)
    ]

    seen = [False] * config.num_blocks
    for override in config.layer_override:
        if override.end_layer >= config.num_blocks:
            raise ValueError("encoder.layer_override range exceeds num_blocks.")
        if override.start_layer > override.end_layer:
            raise ValueError(
                "encoder.layer_override start_layer must be <= end_layer."
            )
        for layer in range(override.start_layer, override.end_layer + 1):
            if seen[layer]:
                raise ValueError(
                    "Overlapping encoder.layer_override ranges are invalid."
                )
            seen[layer] = True
            current = layers[layer]
            override_smolgen = current.smolgen
            if override.HasField("smolgen"):
                if smolgen is None:
                    raise ValueError(
                        "encoder smolgen overrides require base smolgen."
                    )
                if (
                    override.smolgen.HasField("gen_size")
                    and override.smolgen.gen_size != smolgen.gen_size
                ):
                    raise ValueError(
                        "encoder smolgen.gen_size cannot vary by layer."
                    )
                override_smolgen = _smolgen_from_proto(override.smolgen)
            layers[layer] = LayerConfig(
                dff=override.dff or current.dff,
                heads=override.heads or current.heads,
                kv_heads=override.kv_heads or current.kv_heads,
                smolgen=override_smolgen,
            )

    for layer in layers:
        if layer.kv_heads <= 0:
            raise ValueError("encoder kv_heads must be greater than zero.")
        if layer.heads % layer.kv_heads != 0:
            raise ValueError("encoder heads must be divisible by kv_heads.")
    return layers


def _group_layer_breakdowns(
    blocks: list[Breakdown],
) -> Iterable[tuple[str, Breakdown]]:
    start = 0
    while start < len(blocks):
        end = start + 1
        while end < len(blocks) and _same_breakdown(blocks[start], blocks[end]):
            end += 1
        if start == 0 and end == len(blocks):
            label = "Per block"
        elif end == start + 1:
            label = f"Layer {start}"
        else:
            label = f"Layers {start}-{end - 1} per block"
        yield label, blocks[start]
        start = end


def _same_breakdown(left: Breakdown, right: Breakdown) -> bool:
    if left.stats != right.stats or len(left.children) != len(right.children):
        return False
    return all(
        l.name == r.name and _same_breakdown(l, r)
        for l, r in zip(left.children, right.children)
    )


def _smolgen_from_proto(
    config: model_config_pb2.SmolgenConfig,
) -> SmolgenConfig:
    return SmolgenConfig(
        hidden_channels=config.hidden_channels,
        hidden_size=config.hidden_size,
        gen_size=config.gen_size,
        use_bias_dense1=config.use_bias_dense1,
        use_bias_dense2=config.use_bias_dense2,
    )


def _has_smolgen(config: model_config_pb2.EncoderConfig) -> bool:
    return config.HasField("smolgen")


def _linear(
    in_features: int,
    out_features: int,
    *,
    use_bias: bool = True,
    tokens: int = 1,
) -> Stats:
    params = in_features * out_features
    if use_bias:
        params += out_features
    flops = tokens * 2 * in_features * out_features
    return Stats(params=params, flops=flops)


def _ffn(
    in_features: int,
    hidden_features: int,
    activation: int,
    *,
    tokens: int,
) -> Stats:
    stats = _linear(in_features, hidden_features, tokens=tokens)
    if activation == _swiglu_activation():
        stats += _linear(
            in_features,
            hidden_features,
            use_bias=False,
            tokens=tokens,
        )
    stats += _linear(hidden_features, in_features, tokens=tokens)
    return stats


def _ffn_activation(config: model_config_pb2.ModelConfig) -> int:
    if config.defaults.ffn_activation:
        return config.defaults.ffn_activation
    return config.defaults.activation


def _swiglu_activation() -> int | None:
    return getattr(net_pb2.NetworkFormat, "ACTIVATION_SWIGLU", None)


def _norm(
    size: int,
    norm_type: model_config_pb2.NormType = model_config_pb2.NORM_LAYER_NORM,
    *,
    dynamic_erf: bool = True,
) -> Stats:
    if norm_type == model_config_pb2.NORM_RMS_NORM:
        return Stats(params=size)
    if dynamic_erf and norm_type == model_config_pb2.NORM_DYNAMIC_ERF:
        return Stats(params=2 * size + 2)
    return Stats(params=2 * size)


def _sum_children(children: Iterable[Breakdown]) -> Stats:
    stats = Stats()
    for child in children:
        stats += child.stats
    return stats


def _append_breakdown(
    lines: list[str], node: Breakdown, *, indent: int
) -> None:
    prefix = " " * indent
    lines.append(
        f"{prefix}{node.name}: {_fmt_int(node.stats.params)} params, "
        f"{_fmt_int(node.stats.flops)} FLOPs"
    )
    for child in node.children:
        _append_breakdown(lines, child, indent=indent + 2)


def _head_name(name: str) -> str:
    return name or "<unnamed>"


def _fmt_int(value: int) -> str:
    return f"{value:,}"
