from typing import Optional

from proto import hlo_pb2, model_config_pb2, net_pb2


def _netformat_enum_or_none(name: str) -> Optional[int]:
    """Value of a NetworkFormat enum symbol, or None if this net.proto
    predates it (some symbols come from unpublished lc0 proto extensions).
    """
    return getattr(net_pb2.NetworkFormat, name, None)


def _safe_has_field(message: object, field_name: str) -> bool:
    """HasField that returns False when the field does not exist in this
    version of the proto (instead of raising ValueError)."""
    try:
        return message.HasField(field_name)  # type: ignore[attr-defined]
    except ValueError:
        return False


def _network_structure_to_block_style(
    network_structure: int,
) -> int:
    if (
        network_structure
        == net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
    ):
        return model_config_pb2.EncoderConfig.ENCODER_BLOCK_STYLE_SEQUENTIAL
    if network_structure == _netformat_enum_or_none(
        "NETWORK_ATTENTIONBODY_PALM_PARALLEL_WITH_MULTIHEADFORMAT"
    ):
        return model_config_pb2.EncoderConfig.ENCODER_BLOCK_STYLE_PALM_PARALLEL
    raise ValueError(
        "Unsupported network structure: {}".format(
            net_pb2.NetworkFormat.NetworkStructure.Name(network_structure)
        )
    )


def _defaultactivation_to_activation(
    activation: net_pb2.NetworkFormat.DefaultActivation,
) -> net_pb2.NetworkFormat.ActivationFunction:
    return {
        net_pb2.NetworkFormat.DEFAULT_ACTIVATION_RELU: net_pb2.NetworkFormat.ACTIVATION_RELU,
        net_pb2.NetworkFormat.DEFAULT_ACTIVATION_MISH: net_pb2.NetworkFormat.ACTIVATION_MISH,
    }[activation]


def leela_to_modelconfig(
    leela_net: net_pb2.Net,
    weights_dtype: hlo_pb2.XlaShapeProto.Type,
    compute_dtype: hlo_pb2.XlaShapeProto.Type,
) -> model_config_pb2.ModelConfig:
    assert weights_dtype == hlo_pb2.XlaShapeProto.F32, (
        "Only float32 weights are supported."
    )
    if leela_net.format.weights_encoding != net_pb2.Format.LINEAR16:
        # Nets exported with per-layer encodings (e.g. FLOAT16, requiring
        # lc0 >= 0.33) leave the global weights_encoding unset and declare
        # the encoding on each Layer instead.
        layer_encoding = leela_net.weights.ip_emb_b.encoding
        assert layer_encoding in (
            net_pb2.Weights.Layer.LINEAR16,
            net_pb2.Weights.Layer.FLOAT16,
        ), "Unsupported weights encoding: global={}, per-layer={}".format(
            leela_net.format.weights_encoding, layer_encoding
        )
    leela_net_format = leela_net.format.network_format
    model_config = model_config_pb2.ModelConfig()

    model_config.defaults.compute_dtype = compute_dtype
    model_config.defaults.activation = _defaultactivation_to_activation(
        leela_net_format.default_activation
    )
    model_config.defaults.ffn_activation = (
        leela_net_format.ffn_activation or model_config.defaults.activation
    )
    assert (
        leela_net_format.input_embedding
        == net_pb2.NetworkFormat.INPUT_EMBEDDING_PE_DENSE
    ), "Only dense positional embedding is supported, got {}".format(
        net_pb2.NetworkFormat.InputEmbeddingFormat.Name(
            leela_net_format.input_embedding
        )
    )
    policy_format = leela_net_format.policy
    value_format = leela_net_format.value
    moves_left_format = leela_net_format.moves_left
    # The *_SIMPLE formats only exist in an extended net.proto; when the
    # pinned proto lacks them, no net parsed with it can use them.
    is_simple = (
        policy_format == _netformat_enum_or_none("POLICY_SIMPLE")
        and value_format == _netformat_enum_or_none("VALUE_SIMPLE_WDL")
        and moves_left_format == _netformat_enum_or_none("MOVES_LEFT_SIMPLE")
    )
    is_regular = (
        policy_format == net_pb2.NetworkFormat.POLICY_ATTENTION
        and value_format == net_pb2.NetworkFormat.VALUE_WDL
        and moves_left_format == net_pb2.NetworkFormat.MOVES_LEFT_V1
    )
    assert is_simple or is_regular, (
        "Unsupported or mixed head formats: policy={}, value={}, moves_left={}"
    ).format(
        net_pb2.NetworkFormat.PolicyFormat.Name(policy_format),
        net_pb2.NetworkFormat.ValueFormat.Name(value_format),
        net_pb2.NetworkFormat.MovesLeftFormat.Name(moves_left_format),
    )

    def size(x: net_pb2.Weights.Layer) -> int:
        return len(x.params) // 2

    model_config.encoder.block_style = _network_structure_to_block_style(
        leela_net_format.network
    )
    weights = leela_net.weights
    model_config.input_format = leela_net_format.input
    model_config.embedding.dense_size = size(weights.ip_emb_preproc_b) // 64
    model_config.embedding.embedding_size = size(weights.ip_emb_b)
    assert size(weights.ip_mult_gate) > 0
    assert size(weights.ip_add_gate) > 0
    model_config.embedding.dff = size(weights.ip_emb_ffn.dense1_b)

    model_config.encoder.num_blocks = len(weights.encoder)
    assert model_config.encoder.num_blocks > 0
    encoder = weights.encoder[0]
    # The ln*_alphas/ln*_shifts (dynamic ERF) fields only exist in an
    # extended net.proto; treat them as absent when the proto lacks them.
    has_dynamic_erf = any(
        _safe_has_field(block, "ln1_alphas")
        or _safe_has_field(block, "ln1_shifts")
        or _safe_has_field(block, "ln2_alphas")
        or _safe_has_field(block, "ln2_shifts")
        for block in weights.encoder
    )
    if has_dynamic_erf:
        for block in weights.encoder:
            if not block.HasField("ln1_alphas") or not block.HasField(
                "ln1_shifts"
            ):
                raise ValueError(
                    "Dynamic ERF requires both ln1_alphas and ln1_shifts "
                    "for every encoder block."
                )
            if not block.HasField("ln2_alphas") or not block.HasField(
                "ln2_shifts"
            ):
                raise ValueError(
                    "Dynamic ERF requires both ln2_alphas and ln2_shifts "
                    "for every encoder block."
                )
        if not weights.HasField("ip_emb_ln_alphas") or not weights.HasField(
            "ip_emb_ln_shifts"
        ):
            raise ValueError(
                "Dynamic ERF requires ip_emb_ln_alphas and ip_emb_ln_shifts."
            )
        if not weights.HasField("ip_emb_ffn_ln_alphas") or not weights.HasField(
            "ip_emb_ffn_ln_shifts"
        ):
            raise ValueError(
                "Dynamic ERF requires ip_emb_ffn_ln_alphas and "
                "ip_emb_ffn_ln_shifts."
            )
        model_config.defaults.norm_type = model_config_pb2.NORM_DYNAMIC_ERF
    elif not encoder.HasField("ln1_betas"):
        model_config.defaults.norm_type = model_config_pb2.NORM_RMS_NORM
    if encoder.mha.HasField("q_b"):
        model_config.encoder.d_model = size(encoder.mha.q_b)
        model_config.encoder.use_bias_q = True
    else:
        model_config.encoder.d_model = (
            size(encoder.mha.q_w) // model_config.embedding.embedding_size
        )
        model_config.encoder.use_bias_q = False

    model_config.encoder.heads = weights.headcount
    head_depth = model_config.encoder.d_model // model_config.encoder.heads

    model_config.encoder.use_bias_k = encoder.mha.HasField("k_b")
    model_config.encoder.use_bias_v = encoder.mha.HasField("v_b")
    # q_scale only exists in an extended net.proto.
    model_config.encoder.use_q_scale = _safe_has_field(encoder.mha, "q_scale")

    if encoder.mha.HasField("k_b"):
        model_config.encoder.kv_heads = size(encoder.mha.k_b) // head_depth
    else:
        model_config.encoder.kv_heads = (
            size(encoder.mha.k_w) // model_config.embedding.embedding_size
        ) // head_depth

    model_config.encoder.dff = size(encoder.ffn.dense1_b)

    if weights.HasField("smolgen_w"):
        model_config.encoder.smolgen.activation = (
            leela_net_format.smolgen_activation
            or model_config.defaults.activation
        )
        model_config.encoder.smolgen.hidden_channels = (
            size(encoder.mha.smolgen.compress)
            // model_config.embedding.embedding_size
        )
        model_config.encoder.smolgen.gen_size = (
            size(encoder.mha.smolgen.dense2_b) // weights.headcount
        )
        model_config.encoder.smolgen.hidden_size = size(
            encoder.mha.smolgen.dense1_b
        )

    if is_simple:
        assert size(weights.ip_head_map_1_b) > 0
        assert size(weights.ip_head_map_2_b) > 0
        model_config.headpremap.intermediate_size = size(
            weights.ip_head_map_1_b
        )
        model_config.headpremap.output_size = size(weights.ip_head_map_2_b)
        model_config.headpremap.use_gating = weights.HasField(
            "ip_head_map_gate"
        )

        for head_name in ["vanilla", "optimistic_st", "soft", "opponent"]:
            if weights.policy_heads.HasField(head_name):
                head = getattr(weights.policy_heads, head_name)
                assert size(head.simple_ip1_pol_b) > 0
                assert size(head.simple_ip2_pol_b) == 1858
                policy_head = model_config.simple_policy_head.add()
                policy_head.name = head_name
                policy_head.hidden_size = size(head.simple_ip1_pol_b)

        for head_name in ["winner", "q", "st"]:
            if weights.value_heads.HasField(head_name):
                head = getattr(weights.value_heads, head_name)
                assert size(head.simple_ip1_val_b) > 0
                simple_head = model_config.simple_value_head.add()
                simple_head.name = head_name
                simple_head.hidden_size = size(head.simple_ip1_val_b)
                if head.HasField("ip_val_err_w"):
                    simple_head.has_error_output = True
                if head.HasField("ip_val_cat_b"):
                    simple_head.num_categorical_buckets = size(
                        head.ip_val_cat_b
                    )

        assert size(weights.ip1_mov_b) > 0
        simple_movesleft_head = model_config.simple_movesleft_head.add()
        simple_movesleft_head.name = "main"
        simple_movesleft_head.hidden_size = size(weights.ip1_mov_b)
    else:
        if weights.policy_heads.HasField("ip_pol_w"):
            model_config.shared_policy_embedding_size = size(
                weights.policy_heads.ip_pol_b
            )

        for head_name in ["vanilla", "optimistic_st", "soft", "opponent"]:
            if weights.policy_heads.HasField(head_name):
                head = getattr(weights.policy_heads, head_name)
                assert size(head.ip2_pol_b) > 0
                assert not head.HasField("ip_pol_w")
                policy_head = model_config.policy_head.add()
                policy_head.name = head_name
                if not model_config.HasField("shared_policy_embedding_size"):
                    policy_head.embedding_size = size(head.ip_pol_b)
                policy_head.d_model = size(head.ip2_pol_b)

        for head_name in ["winner", "q", "st"]:
            if weights.value_heads.HasField(head_name):
                head = getattr(weights.value_heads, head_name)
                assert size(head.ip_val_b) > 0
                value_head = model_config.value_head.add()
                value_head.name = head_name
                value_head.num_channels = size(head.ip_val_b)
                if head.HasField("ip_val_err_w"):
                    value_head.has_error_output = True
                if head.HasField("ip_val_cat_b"):
                    value_head.num_categorical_buckets = size(head.ip_val_cat_b)

        movesleft_head = model_config.movesleft_head.add()
        movesleft_head.name = "main"
        movesleft_head.num_channels = size(weights.ip_mov_b)

    return model_config
