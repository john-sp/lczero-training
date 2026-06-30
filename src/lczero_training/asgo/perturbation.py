import dataclasses
import fnmatch
from collections.abc import Mapping
from pathlib import PurePosixPath

import jax
import jax.numpy as jnp
from flax import nnx
from google.protobuf.message import Message

_ASGO_PERTURBATION_TENSOR_STD = 0
_ASGO_PERTURBATION_PER_WEIGHT = 1


@dataclasses.dataclass(frozen=True)
class CoupledBasis:
    """Shared and per-layer bases for coupled AGZO perturbations."""

    shared: jax.Array
    unique_by_layer: Mapping[int, jax.Array]
    r_shared: int
    r_uniq: int
    beta_shared: float


class PerturbationManager:
    """Generates ASGO perturbation trees matching an NNX model state."""

    def __init__(
        self,
        config: Message,
        model_params: nnx.State,
        rng: jax.Array,
        activation_bases: Mapping[str, jax.Array] | None = None,
    ) -> None:
        del rng
        self.config = config
        self.model_params = model_params
        self._activation_bases: dict[str, jax.Array] = {
            _canonical_weight_path(name): basis
            for name, basis in (activation_bases or {}).items()
        }
        self._coupled_bases: dict[str, CoupledBasis] = {}

    @property
    def activation_bases(self) -> dict[str, jax.Array]:
        """Returns the currently registered per-weight AGZO bases."""
        return dict(self._activation_bases)

    def set_basis(self, tap_name: str, basis: jax.Array) -> None:
        """Registers an AGZO basis for a guided weight path."""
        self._activation_bases[_canonical_weight_path(tap_name)] = basis

    def set_bases(self, bases: Mapping[str, jax.Array]) -> None:
        """Registers multiple AGZO bases."""
        for tap_name, basis in bases.items():
            self.set_basis(tap_name, basis)

    def set_coupled_basis(
        self,
        tap_name: str,
        *,
        shared: jax.Array,
        unique_by_layer: Mapping[int, jax.Array],
        r_shared: int,
        r_uniq: int,
        beta_shared: float,
    ) -> None:
        """Registers a coupled AGZO basis for one guided weight path."""
        self._coupled_bases[_canonical_weight_path(tap_name)] = CoupledBasis(
            shared=shared,
            unique_by_layer=dict(unique_by_layer),
            r_shared=r_shared,
            r_uniq=r_uniq,
            beta_shared=beta_shared,
        )

    def generate(
        self,
        iteration: int,
        rng: jax.Array,
        momentum: Mapping[str, jax.Array] | None = None,
        adaptive_masks: Mapping[str, jax.Array] | None = None,
    ) -> nnx.State:
        """Generates a perturbation delta for one ASGO round."""
        del iteration, momentum

        leaves: list[tuple[tuple[object, ...], object]] = []

        def collect(path: tuple[object, ...], variable: object) -> bool:
            leaves.append((path, variable))
            return False

        nnx.map_state(collect, self.model_params)
        keys = iter(jax.random.split(rng, max(1, len(leaves))))

        def make_delta(path: tuple[object, ...], variable: object) -> object:
            weight = _variable_value(variable)
            if not _is_array_like(weight):
                return variable
            path_name = _path_to_string(path)
            canonical_name = _canonical_weight_path(path_name)
            if not _selector_includes(self.config.perturb_selector, path_name):
                return _wrap_like_variable(
                    variable, jnp.asarray(0, dtype=weight.dtype)
                )

            key = next(keys)
            c = resolve_c(
                canonical_name,
                self.config.perturbation_size,
                self.config.perturbation_overrides,
            )
            delta = self._guided_delta(canonical_name, weight, c, key)

            if adaptive_masks is not None:
                mask = _lookup_by_path(adaptive_masks, canonical_name)
                if mask is not None:
                    delta = delta * mask
            return _wrap_like_variable(variable, delta.astype(weight.dtype))

        return nnx.map_state(make_delta, self.model_params)

    def make_adaptive_masks(
        self,
        v: nnx.State,
        exponent: float,
        floor: float,
    ) -> dict[str, jax.Array]:
        """Builds AdaMU-style sensitivity masks keyed by weight path."""
        v_by_path: dict[str, jax.Array] = {}

        def collect_v(path: tuple[object, ...], variable: object) -> bool:
            value = _variable_value(variable)
            if _is_array_like(value):
                path_name = _canonical_weight_path(_path_to_string(path))
                v_by_path[path_name] = value
            return False

        nnx.map_state(collect_v, v)
        masks = {}

        def collect_mask(path: tuple[object, ...], variable: object) -> bool:
            weight = _variable_value(variable)
            if not _is_array_like(weight):
                return False
            path_name = _path_to_string(path)
            canonical_name = _canonical_weight_path(path_name)
            if not _selector_includes(self.config.perturb_selector, path_name):
                return False
            v_leaf = v_by_path.get(canonical_name)
            if v_leaf is None:
                return False
            c = resolve_c(
                canonical_name,
                self.config.perturbation_size,
                self.config.perturbation_overrides,
            )
            masks[canonical_name] = adaptive_mask(
                weight,
                v_leaf,
                c,
                exponent,
                floor,
            )
            return False

        nnx.map_state(collect_mask, self.model_params)
        return masks

    def _guided_delta(
        self,
        path_name: str,
        weight: jax.Array,
        c: float,
        rng: jax.Array,
    ) -> jax.Array:
        layer_idx = _encoder_layer_index(path_name)
        coupled_basis = self._coupled_bases.get(path_name)
        if (
            coupled_basis is not None
            and layer_idx is not None
            and layer_idx in coupled_basis.unique_by_layer
            and weight.ndim == 2
        ):
            return coupled_delta(
                weight,
                coupled_basis.shared,
                coupled_basis.unique_by_layer[layer_idx],
                coupled_basis.r_shared,
                coupled_basis.r_uniq,
                c,
                coupled_basis.beta_shared,
                rng,
            )

        basis = self._activation_bases.get(path_name)
        if (
            self.config.activation_guided
            and basis is not None
            and weight.ndim == 2
        ):
            return agzo_delta(weight, basis, c, rng)

        return bernoulli_delta(
            weight,
            c,
            self.config.perturbation_mode,
            rng,
        )


def bernoulli_delta(
    weight: jax.Array,
    c: float,
    mode: int,
    rng: jax.Array,
) -> jax.Array:
    """Returns a Bernoulli +/- perturbation for one tensor."""
    sign = jax.random.bernoulli(rng, 0.5, shape=weight.shape)
    sign = jnp.where(sign, 1.0, -1.0).astype(weight.dtype)
    if mode == _ASGO_PERTURBATION_PER_WEIGHT:
        scale = c * jnp.abs(weight)
    else:
        scale = c * jnp.std(weight)
    return jnp.maximum(scale, 1e-8).astype(weight.dtype) * sign


def agzo_delta(
    weight: jax.Array,
    basis: jax.Array,
    c: float,
    rng: jax.Array,
) -> jax.Array:
    """Returns an activation-guided perturbation for a matrix weight."""
    d_in, d_out = weight.shape[0], weight.shape[1]
    rank = basis.shape[1]
    random_signs = jax.random.bernoulli(rng, 0.5, shape=(d_out, rank))
    random_signs = jnp.where(random_signs, 1.0, -1.0).astype(weight.dtype)
    delta = basis.astype(weight.dtype) @ random_signs.T
    target_norm = c * jnp.std(weight) * jnp.sqrt(d_in * d_out)
    return delta * (target_norm / (jnp.linalg.norm(delta) + 1e-8))


def coupled_delta(
    weight: jax.Array,
    a_shared: jax.Array,
    a_uniq: jax.Array,
    r_shared: int,
    r_uniq: int,
    c: float,
    beta_shared: float,
    rng: jax.Array,
) -> jax.Array:
    """Returns a coupled AGZO perturbation for a matrix weight."""
    sigma_w = jnp.std(weight)
    d_in, d_out = weight.shape
    r_shared_key, r_uniq_key = jax.random.split(rng)

    random_shared = jax.random.bernoulli(
        r_shared_key, 0.5, shape=(d_out, r_shared)
    )
    random_shared = jnp.where(random_shared, 1.0, -1.0).astype(weight.dtype)
    alpha_shared = c * sigma_w * jnp.sqrt(d_in / r_shared) * beta_shared
    delta_shared = alpha_shared * (
        a_shared.astype(weight.dtype) @ random_shared.T
    )

    random_uniq = jax.random.bernoulli(
        r_uniq_key, 0.5, shape=(d_out, r_uniq)
    )
    random_uniq = jnp.where(random_uniq, 1.0, -1.0).astype(weight.dtype)
    alpha_uniq = (
        c
        * sigma_w
        * jnp.sqrt(d_in / r_uniq)
        * jnp.sqrt(1.0 - beta_shared**2)
    )
    delta_uniq = alpha_uniq * (a_uniq.astype(weight.dtype) @ random_uniq.T)
    return delta_shared + delta_uniq


def guided_es_blend(
    delta_random: jax.Array,
    momentum: jax.Array,
    alpha: float,
    basis: jax.Array | None = None,
) -> jax.Array:
    """Blends a random perturbation with a momentum direction."""
    if basis is not None:
        basis = basis.astype(delta_random.dtype)
        m_proj = basis @ (basis.T @ momentum)
        m_dir = m_proj / (jnp.linalg.norm(m_proj) + 1e-8)
    else:
        m_dir = momentum / (jnp.linalg.norm(momentum) + 1e-8)
    projection = jnp.sum(delta_random * m_dir) * m_dir
    ortho = delta_random - projection
    ortho *= jnp.linalg.norm(delta_random) / (jnp.linalg.norm(ortho) + 1e-8)
    return alpha * jnp.linalg.norm(delta_random) * m_dir + (1.0 - alpha) * ortho


def apply_adaptive_mask(
    delta: jax.Array,
    weight: jax.Array,
    v: jax.Array,
    c: float,
    exponent: float,
    floor: float,
) -> jax.Array:
    """Applies an AdaMU-style sensitivity mask to one perturbation tensor."""
    mask = adaptive_mask(weight, v, c, exponent, floor)
    return delta * mask


def adaptive_mask(
    weight: jax.Array,
    v: jax.Array,
    c: float,
    exponent: float,
    floor: float,
) -> jax.Array:
    """Returns an AdaMU-style sensitivity mask for one tensor."""
    sensitivity = (jnp.sqrt(v) / (c * jnp.std(weight) + 1e-8)) ** exponent
    median_sens = jnp.median(sensitivity)
    mask = jnp.clip(median_sens / (sensitivity + 1e-8), floor, 1.0)
    return jnp.where(median_sens <= 1e-12, jnp.ones_like(mask), mask)


def resolve_c(
    weight_name: str,
    base_c: float,
    overrides: list[Message],
) -> float:
    """Resolves a per-tensor perturbation magnitude."""
    canonical_name = _canonical_weight_path(weight_name)
    for override in overrides:
        if fnmatch.fnmatch(canonical_name, override.pattern):
            return override.c
    return base_c


def _selector_includes(selector: Message, path_name: str) -> bool:
    names = {path_name, _canonical_weight_path(path_name)}
    for rule in selector.rule:
        for name in names:
            if PurePosixPath(name).full_match(rule.match):
                return rule.include
    return selector.otherwise_include


def _lookup_by_path(
    values: Mapping[str, jax.Array], canonical_name: str
) -> jax.Array | None:
    if canonical_name in values:
        return values[canonical_name]
    raw_name = _raw_encoder_path(canonical_name)
    return values.get(raw_name)


def _path_to_string(path: tuple[object, ...]) -> str:
    return str(PurePosixPath(*map(str, path)))


def _canonical_weight_path(path_name: str) -> str:
    parts = path_name.split("/")
    if len(parts) >= 3 and parts[0] == "encoders" and parts[1] == "encoders":
        parts[1] = "layers"
        return "/".join(parts)
    return path_name


def _raw_encoder_path(path_name: str) -> str:
    parts = path_name.split("/")
    if len(parts) >= 3 and parts[0] == "encoders" and parts[1] == "layers":
        parts[1] = "encoders"
        return "/".join(parts)
    return path_name


def _encoder_layer_index(path_name: str) -> int | None:
    parts = _canonical_weight_path(path_name).split("/")
    if len(parts) < 3 or parts[0] != "encoders" or parts[1] != "layers":
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def _variable_value(variable: object) -> object:
    return variable.value if hasattr(variable, "value") else variable


def _wrap_like_variable(variable: object, value: jax.Array) -> object:
    if not hasattr(variable, "value"):
        return value
    replace = getattr(variable, "replace", None)
    if callable(replace):
        try:
            return replace(value=value)
        except TypeError:
            pass
    return type(variable)(value)


def _is_array_like(value: object) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def selected_zero_state(
    model_params: nnx.State,
    selector: Message,
) -> nnx.State:
    """Returns zero optimizer state, compacting unselected leaves to scalars."""

    def make_zero(path: tuple[object, ...], variable: object) -> object:
        weight = _variable_value(variable)
        if not _is_array_like(weight):
            return variable
        path_name = _path_to_string(path)
        if _selector_includes(selector, path_name):
            return _wrap_like_variable(variable, jnp.zeros_like(weight))
        return _wrap_like_variable(
            variable, jnp.asarray(0, dtype=weight.dtype)
        )

    return nnx.map_state(make_zero, model_params)
