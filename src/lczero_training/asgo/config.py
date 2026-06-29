import hashlib
import os
from typing import Iterable

from google.protobuf.message import Message

_ASGO_LR_COSINE = 1
ASGO_DEFAULT_ACTIVATION_CACHE_BATCHES = 20
_RUNNER_OWNED_LC0_FLAGS = frozenset(
    {
        "--backend",
        "--backend-opts",
        "--black.visits",
        "--black.weights",
        "--games",
        "--mirror-openings",
        "--movetime",
        "--nodes",
        "--opening-seed",
        "--openings-pgn",
        "--parallelism",
        "--player1.weights",
        "--player2.weights",
        "--seed",
        "--visits",
        "--white.visits",
        "--white.weights",
        "--weights",
    }
)


class AsgoConfigError(ValueError):
    """Raised when an ASGO config is internally inconsistent."""


def hash_config(config: Message) -> str:
    """Returns a stable hash for an ASGO protobuf message."""
    config = _semantic_hash_config(config)
    return hashlib.sha256(
        config.SerializeToString(deterministic=True)
    ).hexdigest()


def normalize_asgo_config(config: Message) -> Message:
    """Returns a copy with derived ASGO defaults applied.

    Protobuf defaults cover scalar fields. This helper fills defaults that
    depend on other config values, such as annealing schedule steady values.
    """
    normalized = type(config)()
    normalized.CopyFrom(config)

    if (
        _has_field(normalized, "tournament")
        and normalized.tournament.WhichOneof("evaluation") is None
    ):
        normalized.tournament.direct_comparison.SetInParent()

    if not normalized.HasField("beta1_schedule"):
        normalized.beta1_schedule.steady_value = normalized.adam_beta1
    if not normalized.HasField("beta2_schedule"):
        normalized.beta2_schedule.steady_value = normalized.adam_beta2

    return normalized


def validate_asgo_config(
    config: Message, *, validate_paths: bool = True
) -> None:
    """Raises AsgoConfigError if an ASGO config is invalid.

    This validation intentionally does not execute lc0 or probe its supported
    flags. Callers that need binary feature detection should do that separately.
    """
    errors: list[str] = []

    if not getattr(config, "lc0_path", ""):
        errors.append("asgo.lc0_path must be set.")
    elif validate_paths and not os.path.isfile(config.lc0_path):
        errors.append(f"asgo.lc0_path does not exist: {config.lc0_path}")

    if not _has_field(config, "perturb_selector"):
        errors.append("asgo.perturb_selector must be set.")
    elif not any(rule.include for rule in config.perturb_selector.rule):
        errors.append("asgo.perturb_selector needs at least one include rule.")

    _check_positive(errors, "asgo.perturbation_size", config.perturbation_size)
    _check_positive(
        errors, "asgo.rounds_per_iteration", config.rounds_per_iteration
    )
    _check_positive(errors, "asgo.elo_clamp", config.elo_clamp)
    if config.curvature_weighting:
        _check_positive(errors, "asgo.noise_floor", config.noise_floor)

    if config.max_iterations < 0:
        errors.append("asgo.max_iterations must be non-negative.")

    _check_unit_interval(errors, "asgo.adam_beta1", config.adam_beta1)
    _check_unit_interval(errors, "asgo.adam_beta2", config.adam_beta2)
    _check_positive(errors, "asgo.adam_epsilon", config.adam_epsilon)
    _check_positive(
        errors,
        "asgo.muon_skip_smaller_than",
        config.muon_skip_smaller_than,
    )
    _check_positive(errors, "asgo.learning_rate", config.learning_rate)
    if config.lr_end < 0:
        errors.append("asgo.lr_end must be non-negative.")
    if config.lr_schedule == _ASGO_LR_COSINE:
        if config.lr_decay_start < 0 or config.lr_decay_end < 0:
            errors.append("ASGO LR decay bounds must be non-negative.")
        if config.lr_decay_end <= config.lr_decay_start:
            errors.append(
                "asgo.lr_decay_end must be greater than lr_decay_start."
            )
    if config.lr_warmup_iterations < 0:
        errors.append("asgo.lr_warmup_iterations must be non-negative.")
    if not 0.0 <= config.lr_warmup_start_factor <= 1.0:
        errors.append("asgo.lr_warmup_start_factor must be in [0, 1].")

    if config.adaptive_mask_exponent <= 0:
        errors.append("asgo.adaptive_mask_exponent must be positive.")
    if not 0.0 < config.adaptive_mask_floor <= 1.0:
        errors.append("asgo.adaptive_mask_floor must be in (0, 1].")

    for override in config.perturbation_overrides:
        if not override.pattern:
            errors.append("asgo.perturbation_overrides.pattern must be set.")
        _check_positive(
            errors,
            f"asgo.perturbation_overrides[{override.pattern}].c",
            override.c,
        )

    if not _has_field(config, "tournament"):
        errors.append("asgo.tournament must be set.")
    else:
        _validate_tournament(errors, config.tournament, validate_paths)

    if config.activation_guided:
        if config.agzo_refresh_interval < 0:
            errors.append("asgo.agzo_refresh_interval must be non-negative.")
        if not config.agzo_rank:
            errors.append(
                "asgo.agzo_rank must be set when activation_guided is true."
            )
    if config.agzo_phase1_iterations < 0:
        errors.append("asgo.agzo_phase1_iterations must be non-negative.")

    _validate_agzo_ranks(errors, config.agzo_rank)
    _validate_coupling_specs(errors, config.coupling_spec)

    if not getattr(config, "checkpoint_path", ""):
        errors.append("asgo.checkpoint_path must be set.")
    _check_positive(errors, "asgo.max_checkpoints", config.max_checkpoints)

    if errors:
        raise AsgoConfigError("\n".join(errors))


def _has_field(message: Message, field_name: str) -> bool:
    return field_name in message.DESCRIPTOR.fields_by_name and message.HasField(
        field_name
    )


def _semantic_hash_config(config: Message) -> Message:
    """Returns a copy with non-semantic ASGO runtime fields removed."""
    copied = type(config)()
    copied.CopyFrom(config)
    _clear_if_present(copied, "lc0_path")
    _clear_if_present(copied, "checkpoint_path")
    _clear_if_present(copied, "max_checkpoints")
    if _has_field(copied, "tournament"):
        tournament = copied.tournament
        _clear_if_present(tournament, "gpu")
        _clear_if_present(tournament, "timeout_seconds")
        _clear_if_present(tournament, "lc0_parallelism")
    return copied


def _clear_if_present(message: Message, field_name: str) -> None:
    if field_name in message.DESCRIPTOR.fields_by_name:
        message.ClearField(field_name)


def _check_positive(errors: list[str], name: str, value: float | int) -> None:
    if value <= 0:
        errors.append(f"{name} must be positive.")


def _check_unit_interval(errors: list[str], name: str, value: float) -> None:
    if not 0.0 <= value < 1.0:
        errors.append(f"{name} must be in [0, 1).")


def _validate_tournament(
    errors: list[str], tournament: Message, validate_paths: bool
) -> None:
    _check_positive(
        errors,
        "asgo.tournament.game_pairs_per_round",
        tournament.game_pairs_per_round,
    )
    if tournament.nodes <= 0 and tournament.movetime <= 0:
        errors.append(
            "Either asgo.tournament.nodes or movetime must be positive."
        )
    if tournament.nodes > 0 and tournament.movetime > 0:
        errors.append(
            "Only one of asgo.tournament.nodes or movetime may be set."
        )
    if tournament.opening_book and validate_paths and not os.path.isfile(
        tournament.opening_book
    ):
        errors.append(
            "asgo.tournament.opening_book does not exist: "
            f"{tournament.opening_book}"
        )
    if tournament.opening_seed < -1:
        errors.append("asgo.tournament.opening_seed must be >= -1.")
    if len(set(tournament.gpu)) != len(tournament.gpu) or any(
        gpu < 0 for gpu in tournament.gpu
    ):
        errors.append(
            "asgo.tournament.gpu entries must be unique and non-negative."
        )
    _check_positive(
        errors, "asgo.tournament.lc0_parallelism", tournament.lc0_parallelism
    )
    _check_positive(
        errors, "asgo.tournament.timeout_seconds", tournament.timeout_seconds
    )
    _validate_extra_args(
        errors, "asgo.tournament.extra_args", tournament.extra_args
    )

    evaluation = tournament.WhichOneof("evaluation")
    if evaluation is None:
        errors.append(
            "asgo.tournament evaluation mode must be set after normalization."
        )
    elif evaluation == "fixed_opponent":
        _validate_fixed_opponent(
            errors, tournament.fixed_opponent, validate_paths
        )


def _validate_fixed_opponent(
    errors: list[str], fixed_opponent: Message, validate_paths: bool
) -> None:
    if not fixed_opponent.opponent:
        errors.append(
            "asgo.tournament.fixed_opponent.opponent must be non-empty."
        )
    for opponent in fixed_opponent.opponent:
        label = opponent.name or "<unnamed>"
        if not opponent.name:
            errors.append(
                "asgo.tournament.fixed_opponent.opponent.name must be set."
            )
        if not opponent.weights:
            errors.append(f"Opponent {label} weights must be set.")
        elif validate_paths and not os.path.isfile(opponent.weights):
            errors.append(
                f"Opponent {label} weights do not exist: {opponent.weights}"
            )
        _check_positive(errors, f"Opponent {label} weight", opponent.weight)
        if opponent.nodes < 0:
            errors.append(f"Opponent {label} nodes must be non-negative.")
        _validate_extra_args(
            errors, f"Opponent {label} extra_args", opponent.extra_args
        )


def _validate_extra_args(
    errors: list[str], label: str, extra_args: Iterable[str]
) -> None:
    for arg in extra_args:
        flag = arg.split("=", 1)[0]
        if flag in _RUNNER_OWNED_LC0_FLAGS:
            errors.append(f"{label} must not contain runner-owned flag {flag}.")


def _validate_agzo_ranks(errors: list[str], ranks: Iterable[Message]) -> None:
    for rank in ranks:
        if not rank.tap_name:
            errors.append("asgo.agzo_rank.tap_name must be set.")
        _check_positive(
            errors, f"asgo.agzo_rank[{rank.tap_name}].rank", rank.rank
        )


def _validate_coupling_specs(
    errors: list[str], coupling_specs: Iterable[Message]
) -> None:
    for coupling in coupling_specs:
        label = coupling.name or "<unnamed>"
        if not coupling.name:
            errors.append("asgo.coupling_spec.name must be set.")
        if not coupling.HasField("spec"):
            errors.append(f"asgo.coupling_spec {label} must contain spec.")
            continue
        spec = coupling.spec
        if len(set(spec.shared_layers)) != len(spec.shared_layers):
            errors.append(
                f"asgo.coupling_spec {label} shared_layers must be unique."
            )
        if any(layer < 0 for layer in spec.shared_layers):
            errors.append(
                f"asgo.coupling_spec {label} shared_layers must be >= 0."
            )
        _check_positive(
            errors, f"asgo.coupling_spec {label} r_shared", spec.r_shared
        )
        _check_positive(
            errors, f"asgo.coupling_spec {label} r_uniq", spec.r_uniq
        )
        if not 0.0 <= spec.beta_shared <= 1.0:
            errors.append(
                f"asgo.coupling_spec {label} beta_shared must be in [0, 1]."
            )
