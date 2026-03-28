import jax
from flax import nnx
from flax.linen import initializers as flax_initializers

from proto import net_pb2

from .utils import get_activation


class Ffn(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_activation: net_pb2.NetworkFormat.ActivationFunction,
        deepnorm_beta: float,
        *,
        rngs: nnx.Rngs,
    ):
        deepnorm_init = flax_initializers.variance_scaling(
            scale=deepnorm_beta,
            mode="fan_avg",
            distribution="truncated_normal",
        )
        out_features = in_features
        self.linear1 = nnx.Linear(
            in_features=in_features,
            out_features=hidden_features,
            kernel_init=deepnorm_init,
            rngs=rngs,
        )
        self.activation = hidden_activation
        self.linear_gate: nnx.Linear | None
        if self.activation == net_pb2.NetworkFormat.ACTIVATION_SWIGLU:
            self.linear_gate = nnx.Linear(
                in_features=in_features,
                out_features=hidden_features,
                use_bias=False,
                kernel_init=deepnorm_init,
                rngs=rngs,
            )
        else:
            self.linear_gate = None
        self.linear2 = nnx.Linear(
            in_features=hidden_features,
            out_features=out_features,
            kernel_init=deepnorm_init,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.activation == net_pb2.NetworkFormat.ACTIVATION_SWIGLU:
            assert self.linear_gate is not None
            gate = nnx.sigmoid(self.linear_gate(x))
            hidden = self.linear1(x)
            return self.linear2(gate * hidden)

        x = self.linear1(x)
        x = get_activation(self.activation)(x)
        return self.linear2(x)
