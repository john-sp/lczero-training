import pytest

from lczero_training.asgo.elo import DirectTournamentResult, score_to_elo


def test_score_to_elo_is_symmetric() -> None:
    positive = score_to_elo(0.75, 100)
    negative = score_to_elo(0.25, 100)

    assert positive == pytest.approx(-negative)


def test_score_to_elo_smooths_perfect_scores() -> None:
    elo = score_to_elo(1.0, 10)

    assert elo > 0
    assert elo < 2400


def test_direct_tournament_result_rejects_empty_score() -> None:
    result = DirectTournamentResult(wins=0, draws=0, losses=0)

    with pytest.raises(ValueError, match="empty tournament"):
        _ = result.score
