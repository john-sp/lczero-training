from collections.abc import Mapping

import jax
import jax.numpy as jnp


def randomized_svd(
    h_matrix: jax.Array,
    rank: int,
    n_power_iters: int = 3,
    rng: jax.Array | None = None,
) -> jax.Array:
    """Computes an orthonormal basis for the top left singular subspace."""
    if h_matrix.ndim != 2:
        raise ValueError("h_matrix must have shape (d_in, samples).")
    if rank <= 0:
        raise ValueError("rank must be positive.")
    if n_power_iters < 0:
        raise ValueError("n_power_iters must be non-negative.")
    d_in, samples = h_matrix.shape
    if d_in <= 0 or samples <= 0:
        raise ValueError("h_matrix dimensions must be positive.")
    rank = min(rank, d_in, samples)
    if rng is None:
        rng = jax.random.PRNGKey(0)

    h_matrix = h_matrix.astype(jnp.float32)
    omega = jax.random.normal(rng, (samples, rank), dtype=jnp.float32)
    y = h_matrix @ omega
    for _ in range(n_power_iters):
        q, _ = jnp.linalg.qr(y, mode="reduced")
        y = h_matrix @ (h_matrix.T @ q)

    basis, _ = jnp.linalg.qr(y, mode="reduced")
    return basis[:, :rank].astype(jnp.float32)


class ActivationSketch:
    """Compact activation covariance sketch for AGZO basis extraction."""

    def __init__(
        self,
        d_in: int,
        rank: int,
        oversample: int = 16,
        rng: jax.Array | None = None,
        storage_dtype: jnp.dtype = jnp.float16,
        accumulator_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        if d_in <= 0:
            raise ValueError("d_in must be positive.")
        if rank <= 0:
            raise ValueError("rank must be positive.")
        if oversample < 0:
            raise ValueError("oversample must be non-negative.")
        del rng, storage_dtype
        self.d_in = d_in
        self.rank = min(rank, d_in)
        self.oversample = oversample
        self.accumulator_dtype = accumulator_dtype
        self._covariance = jnp.zeros((d_in, d_in), dtype=accumulator_dtype)
        self._samples = 0

    @property
    def samples(self) -> int:
        return self._samples

    def update(self, activations: jax.Array) -> None:
        """Consumes one activation block without storing all samples."""
        h_matrix = activation_matrix(activations).astype(
            self.accumulator_dtype
        )
        if h_matrix.shape[0] != self.d_in:
            raise ValueError(
                f"Expected d_in={self.d_in}, got {h_matrix.shape[0]}."
            )
        self._covariance = self._covariance + h_matrix @ h_matrix.T
        self._samples += h_matrix.shape[1]

    def update_covariance(
        self,
        covariance: jax.Array,
        samples: int,
    ) -> None:
        """Consumes a precomputed covariance contribution."""
        if covariance.shape != self._covariance.shape:
            raise ValueError(
                f"Expected covariance shape {self._covariance.shape}, "
                f"got {covariance.shape}."
            )
        if samples <= 0:
            raise ValueError("samples must be positive.")
        self._covariance = self._covariance + covariance.astype(
            self.accumulator_dtype
        )
        self._samples += samples

    def basis(self) -> jax.Array:
        """Returns a float32 orthonormal basis with shape (d_in, rank)."""
        if self._samples == 0:
            raise ValueError("Cannot extract a basis from an empty sketch.")
        eigenvalues, eigenvectors = jnp.linalg.eigh(
            self._covariance.astype(jnp.float32)
        )
        del eigenvalues
        return eigenvectors[:, -self.rank :].astype(jnp.float32)


def activation_matrix(activations: jax.Array) -> jax.Array:
    """Converts activations with trailing feature dim to (d_in, samples)."""
    if activations.ndim == 0:
        raise ValueError("Scalar activations cannot form an AGZO basis.")
    if activations.ndim == 1:
        return activations[:, None]
    return activations.reshape((-1, activations.shape[-1])).T


def extract_coupled_bases(
    activations_per_layer: Mapping[int, jax.Array],
    r_shared: int,
    r_uniq: int,
    rng: jax.Array,
) -> tuple[jax.Array, dict[int, jax.Array]]:
    """Extracts shared and per-layer unique bases for coupled AGZO."""
    if not activations_per_layer:
        raise ValueError("activations_per_layer must be non-empty.")
    if r_shared <= 0 or r_uniq <= 0:
        raise ValueError("r_shared and r_uniq must be positive.")
    shared_key, *unique_keys = jax.random.split(
        rng, len(activations_per_layer) + 1
    )
    h_joint = jnp.concatenate(
        [_h_matrix(h) for h in activations_per_layer.values()],
        axis=1,
    )
    a_shared = randomized_svd(h_joint, r_shared, rng=shared_key)
    projection = jnp.eye(a_shared.shape[0], dtype=jnp.float32)
    projection -= a_shared @ a_shared.T
    a_uniq = {}
    for key, (layer_idx, activations) in zip(
        unique_keys, activations_per_layer.items()
    ):
        h_proj = projection @ _h_matrix(activations)
        a_uniq[layer_idx] = randomized_svd(h_proj, r_uniq, rng=key)
    return a_shared, a_uniq


def orthonormality_error(basis: jax.Array) -> jax.Array:
    """Returns max absolute error in A.T @ A against identity."""
    if basis.ndim != 2:
        raise ValueError("basis must be a matrix.")
    identity = jnp.eye(basis.shape[1], dtype=jnp.float32)
    gram = basis.astype(jnp.float32).T @ basis.astype(jnp.float32)
    return jnp.max(jnp.abs(gram - identity))


def _h_matrix(value: jax.Array) -> jax.Array:
    if value.ndim != 2:
        raise ValueError(
            "Expected activation matrix with shape (d_in, samples)."
        )
    return value.astype(jnp.float32)
