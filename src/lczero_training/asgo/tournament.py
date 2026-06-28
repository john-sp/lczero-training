import enum
import logging
import re
import shlex
import subprocess
from collections.abc import Sequence
from concurrent import futures

from google.protobuf.message import Message

from lczero_training.asgo.elo import (
    DirectTournamentResult,
    OpponentComparisonResult,
    OpponentEvaluationResult,
    combine_opponent_results,
)

logger = logging.getLogger(__name__)

_ASGO_TUNED_BLACK = 1


class ResultPerspective(enum.Enum):
    PLAYER1 = enum.auto()
    WHITE = enum.auto()
    BLACK = enum.auto()


class TournamentError(RuntimeError):
    """Raised when lc0 selfplay cannot produce a complete result."""

    def __init__(
        self,
        message: str,
        *,
        partial_result: DirectTournamentResult | None = None,
    ) -> None:
        super().__init__(message)
        self.partial_result = partial_result


class TournamentTimeoutError(TournamentError):
    """Raised when lc0 selfplay times out."""


class TournamentRunner:
    """Runs ASGO perturbation tournaments through lc0 selfplay."""

    def __init__(
        self,
        config: Message,
        lc0_path: str,
        work_dir: str,
        *,
        supports_opening_seed: bool = False,
    ) -> None:
        self.config = config
        self.lc0_path = lc0_path
        self.work_dir = work_dir
        self.supports_opening_seed = supports_opening_seed
        if self.config.opening_seed >= 0 and not supports_opening_seed:
            raise ValueError(
                "asgo.tournament.opening_seed requires an lc0 binary with "
                "--opening-seed support."
            )

    def evaluate_pair(
        self,
        pos_weights_path: str,
        neg_weights_path: str,
    ) -> DirectTournamentResult:
        """Runs theta+ vs theta- and returns W/D/L for theta+."""
        cmd = self._build_direct_command(pos_weights_path, neg_weights_path)
        return self._run_command_set(
            cmd,
            perspective=ResultPerspective.PLAYER1,
            mirror_openings=self.config.direct_comparison.mirror_openings,
        )

    def evaluate_against_opponents(
        self,
        pos_weights_path: str,
        neg_weights_path: str,
        rng: object | None = None,
    ) -> OpponentEvaluationResult:
        """Evaluates theta+ and theta- against fixed opponent nets."""
        del rng
        results: list[tuple[float, OpponentComparisonResult]] = []
        for opponent in self.config.fixed_opponent.opponent:
            pos_cmd, perspective = self._build_fixed_opponent_command(
                pos_weights_path, opponent
            )
            neg_cmd, neg_perspective = self._build_fixed_opponent_command(
                neg_weights_path, opponent
            )
            if neg_perspective != perspective:
                raise TournamentError("Mismatched fixed-opponent perspective.")
            pos_result = self._run_command_set(
                pos_cmd,
                perspective=perspective,
                mirror_openings=self.config.fixed_opponent.mirror_openings,
            )
            neg_result = self._run_command_set(
                neg_cmd,
                perspective=perspective,
                mirror_openings=self.config.fixed_opponent.mirror_openings,
            )
            results.append(
                (
                    opponent.weight,
                    OpponentComparisonResult(
                        pos_result=pos_result,
                        neg_result=neg_result,
                        opponent_name=opponent.name,
                    ),
                )
            )
        return combine_opponent_results(results)

    def _build_direct_command(
        self,
        pos_weights_path: str,
        neg_weights_path: str,
    ) -> list[str]:
        games = 2 * self.config.game_pairs_per_round
        cmd = [
            self.lc0_path,
            "selfplay",
            f"--player1.weights={pos_weights_path}",
            f"--player2.weights={neg_weights_path}",
            f"--games={games}",
        ]
        if self.config.direct_comparison.mirror_openings:
            cmd.append("--mirror-openings")
        self._add_common_args(cmd)
        return cmd

    def _build_fixed_opponent_command(
        self,
        candidate_weights_path: str,
        opponent: Message,
    ) -> tuple[list[str], ResultPerspective]:
        games = 2 * self.config.game_pairs_per_round
        mode = self.config.fixed_opponent
        cmd = [self.lc0_path, "selfplay", f"--games={games}"]

        if mode.tuned_side == _ASGO_TUNED_BLACK:
            cmd.extend(
                [
                    f"--white.weights={opponent.weights}",
                    f"--black.weights={candidate_weights_path}",
                ]
            )
            opponent_side = "white"
            perspective = ResultPerspective.BLACK
        else:
            cmd.extend(
                [
                    f"--white.weights={candidate_weights_path}",
                    f"--black.weights={opponent.weights}",
                ]
            )
            opponent_side = "black"
            perspective = ResultPerspective.WHITE

        if mode.mirror_openings:
            cmd.append("--mirror-openings")
        self._add_common_args(cmd)
        if opponent.nodes > 0:
            cmd.append(f"--{opponent_side}.visits={opponent.nodes}")
        cmd.extend(opponent.extra_args)
        return cmd, perspective

    def _add_common_args(self, cmd: list[str]) -> None:
        if self.config.nodes > 0:
            cmd.append(f"--visits={self.config.nodes}")
        if self.config.movetime > 0:
            cmd.append(f"--movetime={self.config.movetime}")
        if self.config.opening_book:
            cmd.append(f"--openings-pgn={self.config.opening_book}")
        if self.config.opening_seed >= 0:
            cmd.append(f"--opening-seed={self.config.opening_seed}")
        cmd.extend(self.config.extra_args)

    def _run_command_set(
        self,
        cmd: Sequence[str],
        *,
        perspective: ResultPerspective,
        mirror_openings: bool,
    ) -> DirectTournamentResult:
        if not self.config.gpu:
            return self._run_lc0(list(cmd), perspective=perspective)

        shard_results = self._run_sharded_lc0(
            self._shard_command(cmd, mirror_openings),
            perspective=perspective,
        )
        return _combine_direct_results(shard_results)

    def _shard_command(
        self,
        cmd: Sequence[str],
        mirror_openings: bool,
    ) -> list[list[str]]:
        games = _get_int_arg(cmd, "--games")
        game_counts = _split_games(
            games,
            shard_count=len(self.config.gpu),
            require_even=mirror_openings,
        )
        shard_cmds = []
        for shard_idx, (gpu, shard_games) in enumerate(
            zip(self.config.gpu, game_counts, strict=True)
        ):
            if shard_games <= 0:
                continue
            shard_cmd = _replace_arg(cmd, "--games", str(shard_games))
            if self.config.opening_seed >= 0:
                seed = self.config.opening_seed + shard_idx
                shard_cmd = _replace_arg(shard_cmd, "--opening-seed", str(seed))
            shard_cmd.extend(
                [
                    f"--parallelism={self.config.lc0_parallelism}",
                    "--backend=multiplexing",
                    f"--backend-opts=backend=cuda-auto,gpu={gpu}",
                ]
            )
            shard_cmds.append(shard_cmd)
        return shard_cmds

    def _run_sharded_lc0(
        self,
        shard_cmds: Sequence[list[str]],
        *,
        perspective: ResultPerspective,
    ) -> list[DirectTournamentResult]:
        if not shard_cmds:
            raise ValueError("At least one lc0 shard command is required.")
        with futures.ThreadPoolExecutor(max_workers=len(shard_cmds)) as pool:
            running = [
                pool.submit(self._run_lc0, shard_cmd, perspective)
                for shard_cmd in shard_cmds
            ]
            return [task.result() for task in running]

    def _run_lc0(
        self,
        cmd: list[str],
        perspective: ResultPerspective = ResultPerspective.PLAYER1,
    ) -> DirectTournamentResult:
        logger.info("Running: %s", shlex.join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            partial = None
            try:
                partial = self._parse_output(
                    exc.stdout or b"",
                    allow_partial=True,
                    perspective=perspective,
                )
            except TournamentError:
                pass
            raise TournamentTimeoutError(
                "lc0 selfplay timed out.",
                partial_result=partial,
            ) from exc

        if result.returncode != 0:
            partial = None
            try:
                partial = self._parse_output(
                    result.stdout,
                    allow_partial=True,
                    perspective=perspective,
                )
            except TournamentError:
                pass
            stderr_tail = result.stderr.decode(
                "utf-8", errors="replace"
            )[-500:]
            raise TournamentError(
                f"lc0 exited with code {result.returncode}: {stderr_tail}",
                partial_result=partial,
            )

        return self._parse_output(
            result.stdout,
            allow_partial=False,
            perspective=perspective,
        )

    def _parse_output(
        self,
        output: bytes,
        allow_partial: bool = False,
        perspective: ResultPerspective = ResultPerspective.PLAYER1,
    ) -> DirectTournamentResult:
        parsed = _get_wld_and_npm(
            output,
            final_only=not allow_partial,
            perspective=perspective,
        )
        if parsed is None:
            tail = output.decode("utf-8", errors="replace")[-500:]
            raise TournamentError(
                "Could not parse tournamentstatus from lc0 output:\n"
                f"{tail}"
            )

        npm, wins, losses, draws = parsed
        return DirectTournamentResult(
            wins=wins,
            draws=draws,
            losses=losses,
            npm=npm,
        )

    def cleanup(self) -> None:
        """Hook for future runner-owned temporary resources."""


def _get_wld_and_npm(
    output: bytes,
    final_only: bool = True,
    perspective: ResultPerspective = ResultPerspective.PLAYER1,
) -> tuple[float, int, int, int] | None:
    """Returns (npm, wins, losses, draws) from lc0 status output."""
    selected_line = None
    for line in output.decode("utf-8", errors="replace").splitlines():
        if final_only:
            if line.startswith("tournamentstatus final"):
                selected_line = line
                break
        elif line.startswith("tournamentstatus"):
            selected_line = line

    if selected_line is None:
        return None

    npm_match = re.search(r"\bnpm\s+([\d.]+)", selected_line)
    if not npm_match:
        return None
    npm = float(npm_match.group(1))

    if perspective in (ResultPerspective.WHITE, ResultPerspective.BLACK):
        return _parse_color_breakdown(selected_line, perspective, npm)
    total_match = re.search(
        r"\bP1:\s*\+(\d+)\s+-(\d+)\s+=(\d+)", selected_line
    )
    if not total_match:
        return None
    wins, losses, draws = map(int, total_match.groups())
    return npm, wins, losses, draws


def _parse_color_breakdown(
    line: str,
    perspective: ResultPerspective,
    npm: float,
) -> tuple[float, int, int, int] | None:
    white_match = re.search(
        r"\bP1-W:\s*\+(\d+)\s+-(\d+)\s+=(\d+)", line
    )
    black_match = re.search(
        r"\bP1-B:\s*\+(\d+)\s+-(\d+)\s+=(\d+)", line
    )
    if not (white_match and black_match):
        return None

    white_win, white_loss, white_draw = map(int, white_match.groups())
    black_win, black_loss, black_draw = map(int, black_match.groups())
    if perspective == ResultPerspective.WHITE:
        wins = white_win + black_loss
        losses = white_loss + black_win
    else:
        wins = black_win + white_loss
        losses = black_loss + white_win
    draws = white_draw + black_draw
    return npm, wins, losses, draws


def _split_games(
    total_games: int,
    *,
    shard_count: int,
    require_even: bool,
) -> list[int]:
    if total_games <= 0:
        raise ValueError("total_games must be positive.")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive.")
    unit = 2 if require_even else 1
    if require_even and total_games % 2:
        raise ValueError("Mirrored tournaments require an even game count.")
    total_units = total_games // unit
    base_units, remainder = divmod(total_units, shard_count)
    return [
        unit * (base_units + (1 if idx < remainder else 0))
        for idx in range(shard_count)
    ]


def _combine_direct_results(
    results: Sequence[DirectTournamentResult],
) -> DirectTournamentResult:
    if not results:
        raise ValueError("At least one shard result is required.")
    total_games = sum(result.total for result in results)
    npm = (
        sum(result.npm * result.total for result in results) / total_games
        if total_games > 0
        else 0.0
    )
    return DirectTournamentResult(
        wins=sum(result.wins for result in results),
        draws=sum(result.draws for result in results),
        losses=sum(result.losses for result in results),
        npm=npm,
    )


def _get_int_arg(cmd: Sequence[str], flag: str) -> int:
    prefix = f"{flag}="
    for arg in cmd:
        if arg.startswith(prefix):
            return int(arg[len(prefix) :])
    raise ValueError(f"Command does not contain {flag}.")


def _replace_arg(cmd: Sequence[str], flag: str, value: str) -> list[str]:
    prefix = f"{flag}="
    replaced = False
    result = []
    for arg in cmd:
        if arg.startswith(prefix):
            result.append(f"{flag}={value}")
            replaced = True
        else:
            result.append(arg)
    if not replaced:
        result.append(f"{flag}={value}")
    return result
