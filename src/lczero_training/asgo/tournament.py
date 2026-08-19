import dataclasses
import enum
import json
import logging
import os
import re
import shutil
import shlex
import subprocess
import threading
import time
import urllib.parse
import uuid
from collections.abc import Callable, Sequence
from concurrent import futures
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TypeVar

from google.protobuf.message import Message

from lczero_training.asgo.elo import (
    DirectTournamentResult,
    FixedOpponentEvaluationResult,
    FixedOpponentResult,
    OpponentComparisonResult,
    OpponentEvaluationResult,
    combine_opponent_results,
)

logger = logging.getLogger(__name__)

_ASGO_TUNED_BLACK = 1
_ResultT = TypeVar("_ResultT")


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


@dataclasses.dataclass(frozen=True)
class RemoteTournamentTask:
    """One lc0 command set assigned to a remote worker."""

    cmd: list[str]
    perspective: ResultPerspective
    mirror_openings: bool
    timeout_seconds: float


@dataclasses.dataclass
class _RemoteJob:
    job_id: str
    payload: dict[str, object]
    timeout_seconds: float
    future: futures.Future[dict[str, object]]
    status: str = "pending"
    attempts: int = 0
    assigned_worker: str = ""
    started_monotonic: float = 0.0
    error: str = ""


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
        *,
        opening_seed: int | None = None,
    ) -> DirectTournamentResult:
        """Runs theta+ vs theta- and returns W/D/L for theta+."""
        cmd = self._build_direct_command(
            pos_weights_path,
            neg_weights_path,
            opening_seed=opening_seed,
        )
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
        *,
        opening_seed: int | None = None,
    ) -> OpponentEvaluationResult:
        """Evaluates theta+ and theta- against fixed opponent nets."""
        del rng
        results: list[tuple[float, OpponentComparisonResult]] = []
        for opponent in self.config.fixed_opponent.opponent:
            pos_cmd, perspective = self._build_fixed_opponent_command(
                pos_weights_path,
                opponent,
                opening_seed=opening_seed,
            )
            neg_cmd, neg_perspective = self._build_fixed_opponent_command(
                neg_weights_path,
                opponent,
                opening_seed=opening_seed,
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

    def evaluate_fixed_candidate(
        self,
        weights_path: str,
        *,
        opening_seed: int | None = None,
    ) -> FixedOpponentEvaluationResult:
        """Evaluates one network against all configured fixed opponents."""
        results = []
        for opponent in self.config.fixed_opponent.opponent:
            cmd, perspective = self._build_fixed_opponent_command(
                weights_path,
                opponent,
                opening_seed=opening_seed,
            )
            result = self._run_command_set(
                cmd,
                perspective=perspective,
                mirror_openings=self.config.fixed_opponent.mirror_openings,
            )
            results.append(
                (
                    opponent.weight,
                    FixedOpponentResult(
                        result=result,
                        opponent_name=opponent.name,
                    ),
                )
            )
        return FixedOpponentEvaluationResult(tuple(results))

    def _build_direct_command(
        self,
        pos_weights_path: str,
        neg_weights_path: str,
        *,
        opening_seed: int | None = None,
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
        self._add_common_args(cmd, opening_seed=opening_seed)
        return cmd

    def _build_fixed_opponent_command(
        self,
        candidate_weights_path: str,
        opponent: Message,
        *,
        opening_seed: int | None = None,
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
            candidate_side = "black"
            perspective = ResultPerspective.BLACK
        else:
            cmd.extend(
                [
                    f"--white.weights={candidate_weights_path}",
                    f"--black.weights={opponent.weights}",
                ]
            )
            opponent_side = "black"
            candidate_side = "white"
            perspective = ResultPerspective.WHITE

        if mode.mirror_openings:
            cmd.append("--mirror-openings")
        self._add_common_args(
            cmd,
            visits_side=candidate_side,
            opening_seed=opening_seed,
        )
        if opponent.nodes > 0:
            cmd.append(f"--{opponent_side}.visits={opponent.nodes}")
        cmd.extend(opponent.extra_args)
        return cmd, perspective

    def _add_common_args(
        self,
        cmd: list[str],
        *,
        visits_side: str | None = None,
        opening_seed: int | None = None,
    ) -> None:
        if self.config.nodes > 0:
            visits_arg = (
                f"--{visits_side}.visits"
                if visits_side is not None
                else "--visits"
            )
            cmd.append(f"{visits_arg}={self.config.nodes}")
        if self.config.movetime > 0:
            cmd.append(f"--movetime={self.config.movetime}")
        if self.config.opening_book:
            cmd.append(f"--openings-pgn={self.config.opening_book}")
        if opening_seed is None:
            opening_seed = self.config.opening_seed
        if opening_seed >= 0:
            cmd.append(f"--opening-seed={opening_seed}")
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
            opening_seed = _get_optional_int_arg(cmd, "--opening-seed")
            if opening_seed is not None:
                seed = opening_seed + shard_idx
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


class RemoteTournamentRunner(TournamentRunner):
    """Runs whole ASGO rounds on remote lc0 workers."""

    def __init__(
        self,
        config: Message,
        lc0_path: str,
        work_dir: str,
        *,
        remote_config: Message,
        supports_opening_seed: bool = True,
    ) -> None:
        super().__init__(
            config=config,
            lc0_path=lc0_path,
            work_dir=work_dir,
            supports_opening_seed=supports_opening_seed,
        )
        self.remote_config = remote_config
        self._server = RemoteTournamentServer(
            listen_host=remote_config.listen_host,
            port=remote_config.port,
            auth_token=remote_config.auth_token,
            long_poll_seconds=remote_config.long_poll_seconds,
            max_retries=remote_config.max_retries,
        )
        self._server.start()
        logger.info(
            "ASGO remote tournament server listening on %s",
            self.server_url,
        )

    @property
    def server_url(self) -> str:
        return self._server.url

    def evaluate_pair(
        self,
        pos_weights_path: str,
        neg_weights_path: str,
        *,
        opening_seed: int | None = None,
    ) -> DirectTournamentResult:
        return self.submit_pair(
            pos_weights_path,
            neg_weights_path,
            opening_seed=opening_seed,
        ).result()

    def submit_pair(
        self,
        pos_weights_path: str,
        neg_weights_path: str,
        *,
        opening_seed: int | None = None,
    ) -> futures.Future[DirectTournamentResult]:
        """Queues theta+ vs theta- as one remote ASGO round."""
        cmd = self._build_direct_command(
            pos_weights_path,
            neg_weights_path,
            opening_seed=opening_seed,
        )
        task = RemoteTournamentTask(
            cmd=cmd,
            perspective=ResultPerspective.PLAYER1,
            mirror_openings=self.config.direct_comparison.mirror_openings,
            timeout_seconds=float(self.config.timeout_seconds),
        )
        return self._submit_round(
            tasks=[task],
            file_paths=self._round_files(pos_weights_path, neg_weights_path),
            build_result=_direct_result_from_payload,
        )

    def evaluate_against_opponents(
        self,
        pos_weights_path: str,
        neg_weights_path: str,
        rng: object | None = None,
        *,
        opening_seed: int | None = None,
    ) -> OpponentEvaluationResult:
        return self.submit_against_opponents(
            pos_weights_path,
            neg_weights_path,
            rng=rng,
            opening_seed=opening_seed,
        ).result()

    def submit_against_opponents(
        self,
        pos_weights_path: str,
        neg_weights_path: str,
        rng: object | None = None,
        *,
        opening_seed: int | None = None,
    ) -> futures.Future[OpponentEvaluationResult]:
        """Queues all fixed-opponent commands for one ASGO round."""
        del rng
        tasks: list[RemoteTournamentTask] = []
        opponents = list(self.config.fixed_opponent.opponent)
        for opponent in opponents:
            pos_cmd, perspective = self._build_fixed_opponent_command(
                pos_weights_path,
                opponent,
                opening_seed=opening_seed,
            )
            neg_cmd, neg_perspective = self._build_fixed_opponent_command(
                neg_weights_path,
                opponent,
                opening_seed=opening_seed,
            )
            if neg_perspective != perspective:
                raise TournamentError("Mismatched fixed-opponent perspective.")
            tasks.append(
                RemoteTournamentTask(
                    cmd=pos_cmd,
                    perspective=perspective,
                    mirror_openings=self.config.fixed_opponent.mirror_openings,
                    timeout_seconds=float(self.config.timeout_seconds),
                )
            )
            tasks.append(
                RemoteTournamentTask(
                    cmd=neg_cmd,
                    perspective=perspective,
                    mirror_openings=self.config.fixed_opponent.mirror_openings,
                    timeout_seconds=float(self.config.timeout_seconds),
                )
            )

        return self._submit_round(
            tasks=tasks,
            file_paths=self._round_files(
                pos_weights_path,
                neg_weights_path,
                *(opponent.weights for opponent in opponents),
            ),
            build_result=lambda payload: _opponent_result_from_payload(
                payload,
                opponents,
            ),
        )

    def evaluate_fixed_candidate(
        self,
        weights_path: str,
        *,
        opening_seed: int | None = None,
    ) -> FixedOpponentEvaluationResult:
        return self.submit_fixed_candidate(
            weights_path,
            opening_seed=opening_seed,
        ).result()

    def submit_fixed_candidate(
        self,
        weights_path: str,
        *,
        opening_seed: int | None = None,
    ) -> futures.Future[FixedOpponentEvaluationResult]:
        """Queues one network against all configured fixed opponents."""
        tasks = []
        opponents = list(self.config.fixed_opponent.opponent)
        for opponent in opponents:
            cmd, perspective = self._build_fixed_opponent_command(
                weights_path,
                opponent,
                opening_seed=opening_seed,
            )
            tasks.append(
                RemoteTournamentTask(
                    cmd=cmd,
                    perspective=perspective,
                    mirror_openings=self.config.fixed_opponent.mirror_openings,
                    timeout_seconds=float(self.config.timeout_seconds),
                )
            )

        return self._submit_round(
            tasks=tasks,
            file_paths=self._round_files(
                weights_path,
                *(opponent.weights for opponent in opponents),
            ),
            build_result=lambda payload: _fixed_opponent_result_from_payload(
                payload,
                opponents,
            ),
        )

    def _round_files(self, *paths: str) -> list[str]:
        files = [path for path in paths if path]
        if self.config.opening_book:
            files.append(self.config.opening_book)
        return files

    def _submit_round(
        self,
        *,
        tasks: Sequence[RemoteTournamentTask],
        file_paths: Sequence[str],
        build_result: Callable[[dict[str, object]], _ResultT],
    ) -> futures.Future[_ResultT]:
        raw_future = self._server.submit(
            tasks=[_remote_task_payload(task) for task in tasks],
            file_paths=file_paths,
            timeout_seconds=self._remote_job_timeout(len(tasks)),
        )
        result_future: futures.Future[_ResultT] = futures.Future()

        def complete(done: futures.Future[dict[str, object]]) -> None:
            try:
                result_future.set_result(build_result(done.result()))
            except Exception as exc:
                result_future.set_exception(exc)

        raw_future.add_done_callback(complete)
        return result_future

    def _remote_job_timeout(self, task_count: int) -> float:
        if self.remote_config.job_timeout_seconds > 0:
            return float(self.remote_config.job_timeout_seconds)
        return float(self.config.timeout_seconds * max(1, task_count) + 60)

    def cleanup(self) -> None:
        self._server.stop()


class RemoteTournamentServer:
    """Small HTTP broker for remote ASGO tournament workers."""

    def __init__(
        self,
        *,
        listen_host: str,
        port: int,
        auth_token: str,
        long_poll_seconds: float,
        max_retries: int,
    ) -> None:
        self.listen_host = listen_host
        self.port = port
        self.auth_token = auth_token
        self.long_poll_seconds = long_poll_seconds
        self.max_retries = max_retries
        self._condition = threading.Condition()
        self._jobs: dict[str, _RemoteJob] = {}
        self._job_order: list[str] = []
        self._files: dict[str, str] = {}
        self._httpd: _RemoteHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._httpd is None:
            return f"http://{self.listen_host}:{self.port}"
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        self._stop_event.clear()
        handler = _make_remote_handler(self)
        self._httpd = _RemoteHTTPServer(
            (self.listen_host, self.port),
            handler,
        )
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="asgo-remote-tournament-server",
            daemon=True,
        )
        self._thread.start()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="asgo-remote-tournament-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=5.0)
            self._watchdog_thread = None
        with self._condition:
            for job in self._jobs.values():
                if not job.future.done():
                    job.future.cancel()
            self._condition.notify_all()

    def submit(
        self,
        *,
        tasks: Sequence[dict[str, object]],
        file_paths: Sequence[str],
        timeout_seconds: float,
    ) -> futures.Future[dict[str, object]]:
        job_id = uuid.uuid4().hex
        file_entries, path_rewrites = self._register_files(job_id, file_paths)
        future: futures.Future[dict[str, object]] = futures.Future()
        payload = {
            "job_id": job_id,
            "tasks": list(tasks),
            "files": file_entries,
            "path_rewrites": path_rewrites,
        }
        job = _RemoteJob(
            job_id=job_id,
            payload=payload,
            timeout_seconds=timeout_seconds,
            future=future,
        )
        with self._condition:
            self._jobs[job_id] = job
            self._job_order.append(job_id)
            self._condition.notify_all()
        return future

    def claim(self, worker_id: str) -> dict[str, object] | None:
        deadline = time.monotonic() + self.long_poll_seconds
        with self._condition:
            while True:
                self._expire_running_jobs_locked()
                for job_id in self._job_order:
                    job = self._jobs[job_id]
                    if job.status != "pending":
                        continue
                    job.status = "running"
                    job.attempts += 1
                    job.assigned_worker = worker_id
                    job.started_monotonic = time.monotonic()
                    logger.info(
                        "Assigned ASGO remote job %s to worker %s "
                        "(attempt %d).",
                        job.job_id,
                        worker_id,
                        job.attempts,
                    )
                    return dict(job.payload)

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def complete(
        self,
        *,
        job_id: str,
        worker_id: str,
        payload: dict[str, object],
    ) -> tuple[bool, str]:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                return False, f"Unknown job {job_id}."
            if job.status != "running":
                return False, f"Job {job_id} is not running."
            if job.assigned_worker and job.assigned_worker != worker_id:
                return False, f"Job {job_id} is assigned to another worker."

            if bool(payload.get("ok")):
                job.status = "done"
                if not job.future.done():
                    job.future.set_result(payload)
                self._forget_job_files_locked(job)
                logger.info("Completed ASGO remote job %s.", job_id)
                return True, ""

            error = str(payload.get("error") or "remote worker failed")
            logger.warning("ASGO remote job %s failed: %s", job_id, error)
            self._retry_or_fail_locked(job, TournamentError(error))
            self._condition.notify_all()
            return True, ""

    def file_path(self, file_id: str) -> str | None:
        with self._condition:
            return self._files.get(file_id)

    def _register_files(
        self,
        job_id: str,
        file_paths: Sequence[str],
    ) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
        unique_paths = _unique_paths(file_paths)
        file_entries: list[dict[str, object]] = []
        path_rewrites: list[dict[str, str]] = []
        for idx, path in enumerate(unique_paths):
            if not os.path.isfile(path):
                raise TournamentError(f"Remote tournament file missing: {path}")
            file_id = f"{job_id}-{idx}"
            filename = f"{idx}-{os.path.basename(path) or 'file'}"
            file_entries.append(
                {
                    "id": file_id,
                    "name": filename,
                    "size": os.path.getsize(path),
                }
            )
            path_rewrites.append({"from": path, "file_id": file_id})
            with self._condition:
                self._files[file_id] = path
        return file_entries, path_rewrites

    def _expire_running_jobs_locked(self) -> None:
        now = time.monotonic()
        for job in self._jobs.values():
            if job.status != "running":
                continue
            if now - job.started_monotonic < job.timeout_seconds:
                continue
            error = TournamentTimeoutError(
                f"Remote tournament job {job.job_id} timed out."
            )
            logger.warning("%s", error)
            self._retry_or_fail_locked(job, error)

    def _retry_or_fail_locked(
        self,
        job: _RemoteJob,
        error: TournamentError,
    ) -> None:
        job.error = str(error)
        job.assigned_worker = ""
        job.started_monotonic = 0.0
        if job.attempts <= self.max_retries:
            job.status = "pending"
            return
        job.status = "failed"
        if not job.future.done():
            job.future.set_exception(error)
        self._forget_job_files_locked(job)

    def _watchdog_loop(self) -> None:
        while not self._stop_event.wait(1.0):
            with self._condition:
                self._expire_running_jobs_locked()
                self._condition.notify_all()

    def _forget_job_files_locked(self, job: _RemoteJob) -> None:
        files = job.payload.get("files", [])
        if not isinstance(files, list):
            return
        for file_info in files:
            if isinstance(file_info, dict):
                file_id = file_info.get("id")
                if isinstance(file_id, str):
                    self._files.pop(file_id, None)


class _RemoteHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def _make_remote_handler(
    server: RemoteTournamentServer,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if not self._authorized():
                return
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/health":
                self._send_json({"ok": True})
                return
            prefix = "/v1/files/"
            if parsed.path.startswith(prefix):
                self._send_file(parsed.path[len(prefix) :])
                return
            self._send_json(
                {"ok": False, "error": "not found"},
                status=HTTPStatus.NOT_FOUND,
            )

        def do_POST(self) -> None:
            if not self._authorized():
                return
            parsed = urllib.parse.urlparse(self.path)
            body = self._read_json()
            if parsed.path == "/v1/jobs/claim":
                worker_id = str(body.get("worker_id") or "")
                if not worker_id:
                    self._send_json(
                        {"ok": False, "error": "worker_id is required"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                job = server.claim(worker_id)
                if job is None:
                    self._send_json({"ok": True, "status": "no_job"})
                else:
                    self._send_json(
                        {"ok": True, "status": "job", "job": job}
                    )
                return

            suffix = "/complete"
            if parsed.path.startswith("/v1/jobs/") and parsed.path.endswith(
                suffix
            ):
                job_id = parsed.path[len("/v1/jobs/") : -len(suffix)]
                worker_id = str(body.get("worker_id") or "")
                ok, error = server.complete(
                    job_id=job_id,
                    worker_id=worker_id,
                    payload=body,
                )
                status = HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST
                self._send_json({"ok": ok, "error": error}, status=status)
                return

            self._send_json(
                {"ok": False, "error": "not found"},
                status=HTTPStatus.NOT_FOUND,
            )

        def log_message(self, format: str, *args: object) -> None:
            logger.debug("ASGO remote HTTP: " + format, *args)

        def _authorized(self) -> bool:
            if not server.auth_token:
                return True
            expected = f"Bearer {server.auth_token}"
            if self.headers.get("Authorization") == expected:
                return True
            self._send_json(
                {"ok": False, "error": "unauthorized"},
                status=HTTPStatus.UNAUTHORIZED,
            )
            return False

        def _read_json(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            data = self.rfile.read(length)
            return json.loads(data.decode("utf-8"))

        def _send_json(
            self,
            payload: dict[str, object],
            *,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, file_id: str) -> None:
            file_id = urllib.parse.unquote(file_id)
            path = server.file_path(file_id)
            if path is None:
                self._send_json(
                    {"ok": False, "error": "unknown file"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(os.path.getsize(path)))
            self.end_headers()
            with open(path, "rb") as f:
                shutil.copyfileobj(f, self.wfile)

    return Handler


def _remote_task_payload(
    task: RemoteTournamentTask,
) -> dict[str, object]:
    return {
        "cmd": list(task.cmd),
        "perspective": task.perspective.name,
        "mirror_openings": task.mirror_openings,
        "timeout_seconds": task.timeout_seconds,
    }


def _direct_result_from_payload(
    payload: dict[str, object],
) -> DirectTournamentResult:
    results = _payload_results(payload, expected_count=1)
    return _direct_result_from_dict(results[0])


def _fixed_opponent_result_from_payload(
    payload: dict[str, object],
    opponents: Sequence[Message],
) -> FixedOpponentEvaluationResult:
    """Builds a single-network result from a remote worker payload."""
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise TournamentError("Remote result payload has no results list.")
    duplicated_payload = dict(payload)
    duplicated_payload["results"] = [
        result for result in raw_results for _ in range(2)
    ]
    comparison = _opponent_result_from_payload(
        duplicated_payload,
        opponents,
    )
    return FixedOpponentEvaluationResult(
        tuple(
            (
                weight,
                FixedOpponentResult(
                    result=result.pos_result,
                    opponent_name=result.opponent_name,
                ),
            )
            for weight, result in comparison.comparisons
        )
    )


def _opponent_result_from_payload(
    payload: dict[str, object],
    opponents: Sequence[Message],
) -> OpponentEvaluationResult:
    results = _payload_results(payload, expected_count=2 * len(opponents))
    comparisons: list[tuple[float, OpponentComparisonResult]] = []
    for idx, opponent in enumerate(opponents):
        pos_result = _direct_result_from_dict(results[2 * idx])
        neg_result = _direct_result_from_dict(results[2 * idx + 1])
        comparisons.append(
            (
                opponent.weight,
                OpponentComparisonResult(
                    pos_result=pos_result,
                    neg_result=neg_result,
                    opponent_name=opponent.name,
                ),
            )
        )
    return combine_opponent_results(comparisons)


def _payload_results(
    payload: dict[str, object],
    *,
    expected_count: int,
) -> list[dict[str, object]]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise TournamentError("Remote worker response did not contain results.")
    if len(results) != expected_count:
        raise TournamentError(
            "Remote worker returned "
            f"{len(results)} result(s), expected {expected_count}."
        )
    checked = []
    for result in results:
        if not isinstance(result, dict):
            raise TournamentError("Remote worker result must be an object.")
        checked.append(result)
    return checked


def _direct_result_from_dict(data: dict[str, object]) -> DirectTournamentResult:
    try:
        return DirectTournamentResult(
            wins=int(data["wins"]),
            draws=int(data["draws"]),
            losses=int(data["losses"]),
            npm=float(data.get("npm", 0.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TournamentError("Invalid remote tournament result.") from exc


def _unique_paths(paths: Sequence[str]) -> list[str]:
    seen = set()
    result = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


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


def _get_optional_int_arg(cmd: Sequence[str], flag: str) -> int | None:
    prefix = f"{flag}="
    for arg in cmd:
        if arg.startswith(prefix):
            return int(arg[len(prefix) :])
    return None


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
