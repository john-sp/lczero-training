import dataclasses
import math
from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class DirectTournamentResult:
    """W/D/L result from one candidate's perspective."""

    wins: int
    draws: int
    losses: int
    npm: float = 0.0

    @property
    def total(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        if self.total <= 0:
            raise ValueError("Cannot compute score from an empty tournament.")
        return (self.wins + 0.5 * self.draws) / self.total

    @property
    def elo_diff(self) -> float:
        return score_to_elo(self.score, self.total)


def score_to_elo(score: float, total: int) -> float:
    """Converts a tournament score into an ELO difference."""
    if total <= 0:
        raise ValueError("Cannot compute ELO from an empty tournament.")
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be in [0, 1].")
    smoothed = (score * total + 0.5) / (total + 1.0)
    smoothed = min(max(smoothed, 1e-6), 1.0 - 1e-6)
    return 400.0 * math.log10(smoothed / (1.0 - smoothed))


@dataclasses.dataclass(frozen=True)
class OpponentComparisonResult:
    """Positive and negative perturbation results against one opponent."""

    pos_result: DirectTournamentResult
    neg_result: DirectTournamentResult
    opponent_name: str

    @property
    def elo_diff(self) -> float:
        return self.pos_result.elo_diff - self.neg_result.elo_diff

    @property
    def wins(self) -> int:
        return self.pos_result.wins

    @property
    def draws(self) -> int:
        return self.pos_result.draws

    @property
    def losses(self) -> int:
        return self.pos_result.losses

    @property
    def npm(self) -> float:
        return 0.5 * (self.pos_result.npm + self.neg_result.npm)


@dataclasses.dataclass(frozen=True)
class OpponentEvaluationResult:
    """Weighted aggregate over one or more fixed-opponent comparisons."""

    comparisons: tuple[tuple[float, OpponentComparisonResult], ...]

    @property
    def elo_diff(self) -> float:
        total_weight = sum(weight for weight, _ in self.comparisons)
        if total_weight <= 0.0:
            raise ValueError("Opponent result weights must sum positive.")
        weighted_sum = sum(
            weight * result.elo_diff for weight, result in self.comparisons
        )
        return weighted_sum / total_weight

    @property
    def wins(self) -> int:
        return sum(result.wins for _, result in self.comparisons)

    @property
    def draws(self) -> int:
        return sum(result.draws for _, result in self.comparisons)

    @property
    def losses(self) -> int:
        return sum(result.losses for _, result in self.comparisons)

    @property
    def npm(self) -> float:
        total_games = sum(
            result.pos_result.total + result.neg_result.total
            for _, result in self.comparisons
        )
        if total_games <= 0:
            return 0.0
        weighted_npm = sum(
            result.npm
            * (result.pos_result.total + result.neg_result.total)
            for _, result in self.comparisons
        )
        return weighted_npm / total_games


def combine_opponent_results(
    results: Sequence[tuple[float, OpponentComparisonResult]],
) -> OpponentEvaluationResult:
    if not results:
        raise ValueError("At least one opponent result is required.")
    return OpponentEvaluationResult(tuple(results))
