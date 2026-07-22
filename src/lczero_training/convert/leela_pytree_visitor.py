import math
from typing import Any, Optional

from flax import nnx

from proto import net_pb2


def _optional_layer(
    message: Any, field_name: str
) -> Optional[net_pb2.Weights.Layer]:
    """Returns the given Layer field, or None when this version of
    net.proto does not define it (some fields come from unpublished lc0
    proto extensions)."""
    return getattr(message, field_name, None)


def _safe_has_field(message: Any, field_name: str) -> bool:
    """HasField that returns False when the field does not exist in this
    version of the proto (instead of raising ValueError)."""
    try:
        return bool(message.HasField(field_name))
    except ValueError:
        return False


class LeelaPytreeWeightsVisitor:
    def __init__(self, nnx_state: nnx.State, leela_net: net_pb2.Net) -> None:
        self.leela_net = leela_net
        self.nnx_state = nnx_state

    def run(self) -> None:
        state = self.nnx_state
        weights = self.leela_net.weights
        self.embedding_block(state["embedding"], weights)
        self.encoder_tower(state["encoders"], weights)
        if "headpremap" in state:
            self.headpremap(state["headpremap"], weights)
            self.simple_policy_heads(state["simple_policy_heads"], weights)
            self.simple_value_heads(state["simple_value_heads"], weights)
            assert "main" in state["simple_movesleft_heads"], (
                "movesleft head main missing in state"
            )
            self.simple_movesleft_head(
                state["simple_movesleft_heads"]["main"], weights
            )
        else:
            self.policy_heads(state, weights.policy_heads)
            for head_name in ["winner", "q", "st"]:
                if head_name in state["value_heads"]:
                    self.value_head(
                        state["value_heads"][head_name],
                        getattr(weights.value_heads, head_name),
                    )
            for head_name in ["main"]:
                assert head_name in state["movesleft_heads"], (
                    f"movesleft head {head_name} missing in state"
                )
                self.movesleft_head(
                    state["movesleft_heads"][head_name], weights
                )

    def headpremap(self, nnx_dict: nnx.State, weights: net_pb2.Weights) -> None:
        if "gate" in nnx_dict:
            self.tensor(nnx_dict["gate"], weights.ip_head_map_gate)
        self.matmul(
            nnx_dict["per_token_reduce"],
            weights.ip_head_map_1_w,
            weights.ip_head_map_1_b,
        )
        self.matmul(
            nnx_dict["global_project"],
            weights.ip_head_map_2_w,
            weights.ip_head_map_2_b,
        )

    def embedding_block(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights
    ) -> None:
        self.matmul(
            nnx_dict["preprocess"],
            weights.ip_emb_preproc_w,
            weights.ip_emb_preproc_b,
        )
        self.matmul(
            nnx_dict["embedding"],
            weights.ip_emb_w,
            weights.ip_emb_b,
        )
        self.layernorm(
            nnx_dict["norm"],
            weights.ip_emb_ln_gammas,
            weights.ip_emb_ln_betas,
            _optional_layer(weights, "ip_emb_ln_alphas"),
            _optional_layer(weights, "ip_emb_ln_shifts"),
        )
        self.tensor(
            nnx_dict["ma_gating"]["mult_gate"]["gate"], weights.ip_mult_gate
        )
        self.tensor(
            nnx_dict["ma_gating"]["add_gate"]["gate"], weights.ip_add_gate
        )
        self.ffn(nnx_dict["ffn"], weights.ip_emb_ffn)
        self.layernorm(
            nnx_dict["out_norm"],
            weights.ip_emb_ffn_ln_gammas,
            weights.ip_emb_ffn_ln_betas,
            _optional_layer(weights, "ip_emb_ffn_ln_alphas"),
            _optional_layer(weights, "ip_emb_ffn_ln_shifts"),
        )

    def encoder_tower(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights
    ) -> None:
        # Shared layer is stored at the point of the first usage.
        self.matmul(
            nnx_dict["encoders"]["layers"][0]["mha"]["smolgen"][
                "weight_gen_dense"
            ],
            weights.smolgen_w,
            None,
        )

        # assert len(nnx_dict["encoders"]["layers"]) == len(weights.encoder)
        for i in range(len(nnx_dict["encoders"]["layers"])):
            self.encoder_block(
                nnx_dict["encoders"]["layers"][i], weights.encoder[i]
            )

    def encoder_block(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.EncoderLayer
    ) -> None:
        self.mha(nnx_dict["mha"], weights.mha)
        self.layernorm(
            nnx_dict["ln1"],
            weights.ln1_gammas,
            weights.ln1_betas,
            _optional_layer(weights, "ln1_alphas"),
            _optional_layer(weights, "ln1_shifts"),
        )
        self.ffn(nnx_dict["ffn"], weights.ffn)
        self.layernorm(
            nnx_dict["ln2"],
            weights.ln2_gammas,
            weights.ln2_betas,
            _optional_layer(weights, "ln2_alphas"),
            _optional_layer(weights, "ln2_shifts"),
        )

    def mha(self, nnx_dict: nnx.State, weights: net_pb2.Weights.MHA) -> None:
        self.matmul(nnx_dict["q"], weights.q_w, weights.q_b)
        if "q_scale" in nnx_dict:
            q_scale = _optional_layer(weights, "q_scale")
            assert q_scale is not None, (
                "Model uses q_scale but this net.proto has no "
                "Weights.MHA.q_scale field."
            )
            self.tensor(nnx_dict["q_scale"], q_scale)
        self.matmul(nnx_dict["k"], weights.k_w, weights.k_b)
        self.matmul(nnx_dict["v"], weights.v_w, weights.v_b)
        self.smolgen(nnx_dict["smolgen"], weights.smolgen)
        self.matmul(nnx_dict["output_dense"], weights.dense_w, weights.dense_b)

    def smolgen(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.Smolgen
    ) -> None:
        self.matmul(nnx_dict["compress"], weights.compress, None)
        self.matmul(nnx_dict["dense1"], weights.dense1_w, weights.dense1_b)
        self.layernorm(nnx_dict["ln1"], weights.ln1_gammas, weights.ln1_betas)
        self.matmul(nnx_dict["dense2"], weights.dense2_w, weights.dense2_b)
        self.layernorm(nnx_dict["ln2"], weights.ln2_gammas, weights.ln2_betas)

    def layernorm(
        self,
        nnx_dict: nnx.State,
        scales: net_pb2.Weights.Layer,
        biases: net_pb2.Weights.Layer,
        alphas: Optional[net_pb2.Weights.Layer] = None,
        shifts: Optional[net_pb2.Weights.Layer] = None,
    ) -> None:
        if "weight" in nnx_dict:
            self.tensor(nnx_dict["weight"], scales)
        else:
            self.tensor(nnx_dict["scale"], scales)

        if "bias" in nnx_dict:
            self.tensor(nnx_dict["bias"], biases)

        if "alpha" in nnx_dict:
            assert alphas is not None and shifts is not None
            self.tensor(nnx_dict["alpha"], alphas)
            self.tensor(nnx_dict["shift"], shifts)

    def policy_heads(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.PolicyHeads
    ) -> None:
        if "policy_embedding_shared" in nnx_dict:
            self.matmul(
                nnx_dict["policy_embedding_shared"],
                weights.ip_pol_w,
                weights.ip_pol_b,
            )
        policy_heads_dict = nnx_dict["policy_heads"]
        for head_name in ["vanilla", "optimistic_st", "soft", "opponent"]:
            if head_name in policy_heads_dict:
                self.policy_head(
                    policy_heads_dict[head_name], getattr(weights, head_name)
                )

    def policy_head(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.PolicyHead
    ) -> None:
        if "tokens" in nnx_dict:
            self.matmul(nnx_dict["tokens"], weights.ip_pol_w, weights.ip_pol_b)
        self.matmul(nnx_dict["q"], weights.ip2_pol_w, weights.ip2_pol_b)
        self.matmul(nnx_dict["k"], weights.ip3_pol_w, weights.ip3_pol_b)
        self.matmul(nnx_dict["promotion_dense"], weights.ip4_pol_w, None)

    def simple_policy_heads(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights
    ) -> None:
        for head_name in ["vanilla", "optimistic_st", "soft", "opponent"]:
            if head_name in nnx_dict:
                self.simple_policy_head(
                    nnx_dict[head_name],
                    getattr(weights.policy_heads, head_name),
                )

    def simple_policy_head(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.PolicyHead
    ) -> None:
        self.matmul(
            nnx_dict["backbone"]["dense1"],
            weights.simple_ip1_pol_w,
            weights.simple_ip1_pol_b,
        )
        self.matmul(
            nnx_dict["backbone"]["dense2"],
            weights.simple_ip2_pol_w,
            weights.simple_ip2_pol_b,
        )

    def value_head(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.ValueHead
    ) -> None:
        self.matmul(nnx_dict["embed"], weights.ip_val_w, weights.ip_val_b)
        self.matmul(nnx_dict["dense1"], weights.ip1_val_w, weights.ip1_val_b)
        self.matmul(nnx_dict["wdl"], weights.ip2_val_w, weights.ip2_val_b)
        if "error" in nnx_dict:
            self.matmul(
                nnx_dict["error"], weights.ip_val_err_w, weights.ip_val_err_b
            )
        if "categorical" in nnx_dict:
            self.matmul(
                nnx_dict["categorical"],
                weights.ip_val_cat_w,
                weights.ip_val_cat_b,
            )

    def simple_value_heads(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights
    ) -> None:
        for head_name in ["winner", "q", "st"]:
            if head_name in nnx_dict:
                self.simple_value_head(
                    nnx_dict[head_name], getattr(weights.value_heads, head_name)
                )

    def simple_value_head(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.ValueHead
    ) -> None:
        self.matmul(
            nnx_dict["backbone"]["dense1"],
            weights.simple_ip1_val_w,
            weights.simple_ip1_val_b,
        )
        self.matmul(
            nnx_dict["backbone"]["dense2"],
            weights.simple_ip2_val_w,
            weights.simple_ip2_val_b,
        )
        if "error" in nnx_dict:
            self.matmul(
                nnx_dict["error"], weights.ip_val_err_w, weights.ip_val_err_b
            )
        if "categorical" in nnx_dict:
            self.matmul(
                nnx_dict["categorical"],
                weights.ip_val_cat_w,
                weights.ip_val_cat_b,
            )

    def movesleft_head(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights
    ) -> None:
        self.matmul(nnx_dict["embed"], weights.ip_mov_w, weights.ip_mov_b)
        self.matmul(nnx_dict["dense1"], weights.ip1_mov_w, weights.ip1_mov_b)
        self.matmul(nnx_dict["out"], weights.ip2_mov_w, weights.ip2_mov_b)

    def simple_movesleft_head(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights
    ) -> None:
        self.matmul(
            nnx_dict["backbone"]["dense1"],
            weights.ip1_mov_w,
            weights.ip1_mov_b,
        )
        self.matmul(
            nnx_dict["backbone"]["dense2"],
            weights.ip2_mov_w,
            weights.ip2_mov_b,
        )

    def ffn(self, nnx_dict: nnx.State, ffn: net_pb2.Weights.FFN) -> None:
        self.matmul(nnx_dict["linear1"], ffn.dense1_w, ffn.dense1_b)
        if "linear_gate" in nnx_dict:
            dense_gate_w = _optional_layer(ffn, "dense_gate_w")
            assert dense_gate_w is not None, (
                "Model uses a gated FFN but this net.proto has no "
                "Weights.FFN.dense_gate_w field."
            )
            self.matmul(nnx_dict["linear_gate"], dense_gate_w, None)
        else:
            assert not _safe_has_field(ffn, "dense_gate_w")
        self.matmul(nnx_dict["linear2"], ffn.dense2_w, ffn.dense2_b)

    def matmul(
        self,
        nnx_dict: nnx.State,
        weights: net_pb2.Weights.Layer,
        biases: Optional[net_pb2.Weights.Layer],
    ) -> None:
        self.tensor(nnx_dict["kernel"], weights)
        if biases and "bias" in nnx_dict:
            self.tensor(nnx_dict["bias"], biases)
        elif not biases:
            assert "bias" not in nnx_dict

    def tensor(
        self,
        param: Any,
        leela: net_pb2.Weights.Layer,
    ) -> None:
        print(
            param.shape,
            len(leela.params) // 2,
            math.prod(param.shape),
        )
        assert len(leela.params) // 2 == math.prod(param.shape)
        assert len(leela.params) != 0
