import subprocess
from types import SimpleNamespace

import pytest

from lczero_training.asgo.elo import DirectTournamentResult
from lczero_training.asgo.tournament import (
    ResultPerspective,
    TournamentError,
    TournamentRunner,
    _get_wld_and_npm,
    _split_games,
)


def _direct_config(**overrides: object) -> SimpleNamespace:
    values = {
        "game_pairs_per_round": 50,
        "nodes": 800,
        "movetime": 0,
        "opening_book": "book.pgn.gz",
        "opening_seed": -1,
        "gpu": [],
        "extra_args": [],
        "timeout_seconds": 3600,
        "lc0_parallelism": 8,
        "direct_comparison": SimpleNamespace(mirror_openings=True),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_direct_command_sets_player_order_and_games() -> None:
    runner = TournamentRunner(_direct_config(), "lc0", "/tmp/asgo")

    cmd = runner._build_direct_command("pos.pb.gz", "neg.pb.gz")

    assert cmd[:5] == [
        "lc0",
        "selfplay",
        "--player1.weights=pos.pb.gz",
        "--player2.weights=neg.pb.gz",
        "--games=100",
    ]
    assert "--mirror-openings" in cmd
    assert "--visits=800" in cmd
    assert "--openings-pgn=book.pgn.gz" in cmd


def test_split_games_keeps_mirrored_shards_even() -> None:
    assert _split_games(1004, shard_count=4, require_even=True) == [
        252,
        252,
        250,
        250,
    ]


def test_shard_command_adds_gpu_backend_flags() -> None:
    config = _direct_config(gpu=[0, 2], game_pairs_per_round=3)
    runner = TournamentRunner(config, "lc0", "/tmp/asgo")
    cmd = runner._build_direct_command("pos.pb.gz", "neg.pb.gz")

    shards = runner._shard_command(cmd, mirror_openings=True)

    assert [arg for arg in shards[0] if arg.startswith("--games=")] == [
        "--games=4"
    ]
    assert [arg for arg in shards[1] if arg.startswith("--games=")] == [
        "--games=2"
    ]
    assert "--backend=multiplexing" in shards[0]
    assert "--backend-opts=backend=cuda-auto,gpu=0" in shards[0]
    assert "--backend-opts=backend=cuda-auto,gpu=2" in shards[1]


def test_opening_seed_requires_explicit_binary_support() -> None:
    with pytest.raises(ValueError, match="--opening-seed"):
        TournamentRunner(
            _direct_config(opening_seed=123),
            "lc0",
            "/tmp/asgo",
        )


def test_parse_final_player1_result() -> None:
    line = (
        b"noise\n"
        b"tournamentstatus final games 100 npm 1234.5 P1: +42 -50 =8\n"
    )

    parsed = _get_wld_and_npm(line)

    assert parsed == (1234.5, 42, 50, 8)


def test_parse_latest_partial_status() -> None:
    output = (
        b"tournamentstatus games 10 npm 100.0 P1: +1 -2 =7\n"
        b"tournamentstatus games 20 npm 200.0 P1: +3 -4 =13\n"
    )

    parsed = _get_wld_and_npm(output, final_only=False)

    assert parsed == (200.0, 3, 4, 13)


def test_parse_white_perspective_from_color_breakdown() -> None:
    output = (
        b"tournamentstatus final npm 500.0 "
        b"P1: +4 -6 =10 P1-W: +3 -2 =5 P1-B: +1 -4 =5\n"
    )

    parsed = _get_wld_and_npm(
        output,
        perspective=ResultPerspective.WHITE,
    )

    assert parsed == (500.0, 7, 3, 10)


def test_parse_black_perspective_from_color_breakdown() -> None:
    output = (
        b"tournamentstatus final npm 500.0 "
        b"P1: +4 -6 =10 P1-W: +3 -2 =5 P1-B: +1 -4 =5\n"
    )

    parsed = _get_wld_and_npm(
        output,
        perspective=ResultPerspective.BLACK,
    )

    assert parsed == (500.0, 3, 7, 10)


def test_evaluate_pair_uses_subprocess_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=b"tournamentstatus final npm 10.0 P1: +6 -2 =2\n",
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = TournamentRunner(_direct_config(), "lc0", "/tmp/asgo")

    result = runner.evaluate_pair("pos.pb.gz", "neg.pb.gz")

    assert result == DirectTournamentResult(
        wins=6,
        draws=2,
        losses=2,
        npm=10.0,
    )
    assert result.elo_diff > 0


def test_parse_failure_raises_tail() -> None:
    runner = TournamentRunner(_direct_config(), "lc0", "/tmp/asgo")

    with pytest.raises(TournamentError, match="not a status"):
        runner._parse_output(b"not a status")
