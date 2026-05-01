import dataclasses
import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from proto import model_config_pb2

from .embedding import Embedding
from .encoder import EncoderTower
from .headpremap import Headpremap
from .movesleft_head import MovesLeftHead
from .policy_head import PolicyHead
from .simple_head import SimpleMovesLeftHead, SimplePolicyHead, SimpleValueHead
from .utils import get_dtype
from .value_head import ValueHead


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ModelPrediction:
    """Output predictions from LczeroModel.

    Fields:
        value: Dictionary mapping head names to value prediction tuples.
        policy: Dictionary mapping head names to policy logits.
        movesleft: Dictionary mapping head names to moves-left predictions.
    """

    value: dict[str, Tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]]
    policy: dict[str, jax.Array]
    movesleft: dict[str, jax.Array]


class LczeroModel(nnx.Module):
    def __init__(self, config: model_config_pb2.ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self._input_channels = 112
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
                "headpremap/simple heads cannot be mixed with regular heads"
            )
        if has_simple_heads and not has_headpremap:
            raise ValueError("simple heads require headpremap")
        if has_headpremap and not has_simple_heads:
            raise ValueError("headpremap requires at least one simple head")
        if has_headpremap and config.HasField("shared_policy_embedding_size"):
            raise ValueError(
                "shared_policy_embedding_size is unsupported with headpremap"
            )

        self._use_headpremap = has_headpremap
        deepnorm_beta = math.pow(8.0 * config.encoder.num_blocks, -0.25)

        self.embedding = Embedding(
            input_channels=self._input_channels,
            config=config.embedding,
            defaults=config.defaults,
            deepnorm_alpha=math.pow(2.0 * config.encoder.num_blocks, -0.25),
            deepnorm_beta=deepnorm_beta,
            rngs=rngs,
        )

        assert self.config.encoder.num_blocks > 0

        self.encoders = EncoderTower(
            in_features=config.embedding.embedding_size,
            config=config.encoder,
            defaults=config.defaults,
            deepnorm_beta=deepnorm_beta,
            rngs=rngs,
        )

        self.policy_embedding_shared: Optional[nnx.Linear] = None
        if self._use_headpremap:
            if (
                config.headpremap.intermediate_size == 0
                or config.headpremap.output_size == 0
            ):
                raise ValueError(
                    "headpremap intermediate_size and output_size must be set"
                )
            self.headpremap = Headpremap(
                in_features=config.embedding.embedding_size,
                intermediate_size=config.headpremap.intermediate_size,
                output_size=config.headpremap.output_size,
                use_gating=config.headpremap.use_gating,
                rngs=rngs,
            )
            self.simple_value_heads = nnx.Dict(
                {
                    head_config.name: SimpleValueHead(
                        in_features=config.headpremap.output_size,
                        config=head_config,
                        defaults=config.defaults,
                        rngs=rngs,
                    )
                    for head_config in config.simple_value_head
                }
            )
            self.simple_policy_heads = nnx.Dict(
                {
                    head_config.name: SimplePolicyHead(
                        in_features=config.headpremap.output_size,
                        config=head_config,
                        defaults=config.defaults,
                        rngs=rngs,
                    )
                    for head_config in config.simple_policy_head
                }
            )
            self.simple_movesleft_heads = nnx.Dict(
                {
                    head_config.name: SimpleMovesLeftHead(
                        in_features=config.headpremap.output_size,
                        config=head_config,
                        defaults=config.defaults,
                        rngs=rngs,
                    )
                    for head_config in config.simple_movesleft_head
                }
            )
            self.value_heads = nnx.Dict({})
            self.policy_heads = nnx.Dict({})
            self.movesleft_heads = nnx.Dict({})
        else:
            self.headpremap = None
            self.simple_value_heads = nnx.Dict({})
            self.simple_policy_heads = nnx.Dict({})
            self.simple_movesleft_heads = nnx.Dict({})

            self.value_heads = nnx.Dict(
                {
                    head_config.name: ValueHead(
                        in_features=config.embedding.embedding_size,
                        config=head_config,
                        defaults=config.defaults,
                        rngs=rngs,
                    )
                    for head_config in config.value_head
                }
            )

            # Named to appear before 'policy_heads' alphabetically in pytree
            # state. This ensures shared embedding appears at parent level
            # during serialization.
            if config.HasField("shared_policy_embedding_size"):
                self.policy_embedding_shared = nnx.Linear(
                    in_features=config.embedding.embedding_size,
                    out_features=config.shared_policy_embedding_size,
                    rngs=rngs,
                )

            self.policy_heads = nnx.Dict(
                {
                    head_config.name: PolicyHead(
                        in_features=config.embedding.embedding_size,
                        config=head_config,
                        defaults=config.defaults,
                        shared_embedding=self.policy_embedding_shared,
                        rngs=rngs,
                    )
                    for head_config in config.policy_head
                }
            )
            self.movesleft_heads = nnx.Dict(
                {
                    head_config.name: MovesLeftHead(
                        in_features=config.embedding.embedding_size,
                        config=head_config,
                        defaults=config.defaults,
                        rngs=rngs,
                    )
                    for head_config in config.movesleft_head
                }
            )

    def __call__(self, x: jax.Array) -> ModelPrediction:
        x = jnp.astype(x, get_dtype(self.config.defaults.compute_dtype))
        x = jnp.transpose(x, (1, 2, 0))
        x = jnp.reshape(x, (64, self._input_channels))
        x = self.embedding(x)
        x = self.encoders(x)

        if self._use_headpremap:
            assert self.headpremap is not None
            x = self.headpremap(x)
            value = {
                name: head(x) for name, head in self.simple_value_heads.items()
            }
            policy = {
                name: head(x) for name, head in self.simple_policy_heads.items()
            }
            movesleft = {
                name: head(x)
                for name, head in self.simple_movesleft_heads.items()
            }
        else:
            value = {name: head(x) for name, head in self.value_heads.items()}
            policy = {name: head(x) for name, head in self.policy_heads.items()}
            movesleft = {
                name: head(x) for name, head in self.movesleft_heads.items()
            }

        return ModelPrediction(value=value, policy=policy, movesleft=movesleft)

    @property
    def policy_heads_for_metrics(self) -> nnx.Dict:
        if self._use_headpremap:
            return self.simple_policy_heads
        return self.policy_heads

    @property
    def value_heads_for_metrics(self) -> nnx.Dict:
        if self._use_headpremap:
            return self.simple_value_heads
        return self.value_heads

    @property
    def movesleft_heads_for_metrics(self) -> nnx.Dict:
        if self._use_headpremap:
            return self.simple_movesleft_heads
        return self.movesleft_heads
