import jax
import jax.numpy as jnp
from flax import nnx


class Headpremap(nnx.Module):
    def __init__(
        self,
        in_features: int,
        intermediate_size: int,
        output_size: int,
        use_gating: bool,
        *,
        rngs: nnx.Rngs,
    ):
        self.use_gating = use_gating
        if use_gating:
            self.gate = nnx.Param(jnp.ones((in_features,), dtype=jnp.float32))

        self.per_token_reduce = nnx.Linear(
            in_features=in_features,
            out_features=intermediate_size,
            rngs=rngs,
        )
        self.global_project = nnx.Linear(
            in_features=64 * intermediate_size,
            out_features=output_size,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.use_gating:
            x = x * self.gate.value
        x = self.per_token_reduce(x)
        x = x.flatten()
        return self.global_project(x)
