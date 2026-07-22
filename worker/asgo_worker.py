#!/usr/bin/env python3
"""Remote lc0 worker for ASGO tournament evaluation.

The worker intentionally depends only on the Python standard library. It polls
the ASGO host HTTP server, downloads files needed for one whole ASGO round,
runs lc0 locally, and posts W/D/L results back to the host.
"""

import argparse
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from concurrent import futures

logger = logging.getLogger(__name__)


class WorkerError(RuntimeError):
    """Raised when a remote ASGO worker cannot complete a job."""


class AsgoWorker:
    def __init__(
        self,
        *,
        host: str,
        lc0_path: str,
        work_dir: str,
        worker_id: str,
        auth_token: str,
        gpus: Sequence[int],
        lc0_parallelism: int,
        request_timeout_seconds: float,
        keep_workdirs: bool,
    ) -> None:
        self.host = _normalize_host_url(host)
        self.lc0_path = lc0_path
        self.work_dir = work_dir
        self.worker_id = worker_id
        self.auth_token = auth_token
        self.gpus = list(gpus)
        self.lc0_parallelism = lc0_parallelism
        self.request_timeout_seconds = request_timeout_seconds
        self.keep_workdirs = keep_workdirs

    def run(self, *, once: bool) -> int:
        os.makedirs(self.work_dir, exist_ok=True)
        while True:
            response = self._request_json(
                "POST",
                "/v1/jobs/claim",
                {"worker_id": self.worker_id},
            )
            if response.get("status") == "no_job":
                if once:
                    return 0
                continue
            if response.get("status") != "job":
                raise WorkerError(f"Unexpected claim response: {response}")
            self._run_job(_expect_dict(response.get("job")))
            if once:
                return 0

    def _run_job(self, job: dict[str, object]) -> None:
        job_id = str(job.get("job_id") or "")
        if not job_id:
            raise WorkerError("Claimed job did not include job_id.")
        job_dir = os.path.join(self.work_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        logger.info("Claimed ASGO remote job %s.", job_id)
        try:
            rewrites = self._download_job_files(job, job_dir)
            results = [
                self._run_task(_expect_dict(task), rewrites)
                for task in _expect_list(job.get("tasks"))
            ]
            payload: dict[str, object] = {
                "ok": True,
                "worker_id": self.worker_id,
                "results": results,
            }
        except Exception as exc:
            logger.exception("ASGO remote job %s failed.", job_id)
            payload = {
                "ok": False,
                "worker_id": self.worker_id,
                "error": str(exc),
            }
        finally:
            if not self.keep_workdirs:
                shutil.rmtree(job_dir, ignore_errors=True)
        self._request_json("POST", f"/v1/jobs/{job_id}/complete", payload)

    def _download_job_files(
        self,
        job: dict[str, object],
        job_dir: str,
    ) -> dict[str, str]:
        by_file_id: dict[str, str] = {}
        for file_info_obj in _expect_list(job.get("files")):
            file_info = _expect_dict(file_info_obj)
            file_id = str(file_info.get("id") or "")
            name = os.path.basename(str(file_info.get("name") or file_id))
            if not file_id or not name:
                raise WorkerError("Job file entry is missing id or name.")
            local_path = os.path.join(job_dir, name)
            self._download_file(file_id, local_path)
            expected_size = int(file_info.get("size") or -1)
            downloaded_size = os.path.getsize(local_path)
            if expected_size >= 0 and downloaded_size != expected_size:
                raise WorkerError(
                    f"Downloaded file {file_id} has the wrong size."
                )
            by_file_id[file_id] = local_path

        rewrites: dict[str, str] = {}
        for rewrite_obj in _expect_list(job.get("path_rewrites")):
            rewrite = _expect_dict(rewrite_obj)
            original = str(rewrite.get("from") or "")
            file_id = str(rewrite.get("file_id") or "")
            if original and file_id in by_file_id:
                rewrites[original] = by_file_id[file_id]
        return rewrites

    def _run_task(
        self,
        task: dict[str, object],
        rewrites: dict[str, str],
    ) -> dict[str, object]:
        cmd = _rewrite_command(
            _expect_str_list(task.get("cmd")),
            rewrites,
            self.lc0_path,
        )
        perspective = str(task.get("perspective") or "PLAYER1")
        mirror_openings = bool(task.get("mirror_openings"))
        timeout_seconds = float(task.get("timeout_seconds") or 0.0)
        if self.gpus:
            shard_cmds = _shard_command(
                cmd,
                gpus=self.gpus,
                lc0_parallelism=self.lc0_parallelism,
                mirror_openings=mirror_openings,
            )
            results = self._run_sharded_lc0(
                shard_cmds,
                perspective=perspective,
                timeout_seconds=timeout_seconds,
            )
            return _result_to_dict(_combine_results(results))
        result = _run_lc0(
            cmd,
            perspective=perspective,
            timeout_seconds=timeout_seconds,
        )
        return _result_to_dict(result)

    def _run_sharded_lc0(
        self,
        shard_cmds: Sequence[list[str]],
        *,
        perspective: str,
        timeout_seconds: float,
    ) -> list[dict[str, float | int]]:
        if not shard_cmds:
            raise WorkerError("At least one lc0 shard command is required.")
        with futures.ThreadPoolExecutor(max_workers=len(shard_cmds)) as pool:
            running = [
                pool.submit(
                    _run_lc0,
                    shard_cmd,
                    perspective=perspective,
                    timeout_seconds=timeout_seconds,
                )
                for shard_cmd in shard_cmds
            ]
            return [task.result() for task in running]

    def _download_file(self, file_id: str, local_path: str) -> None:
        quoted = urllib.parse.quote(file_id, safe="")
        url = f"{self.host}/v1/files/{quoted}"
        request = urllib.request.Request(url)
        self._add_auth(request)
        with urllib.request.urlopen(
            request,
            timeout=self.request_timeout_seconds,
        ) as response:
            with open(local_path, "wb") as out:
                shutil.copyfileobj(response, out)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        self._add_auth(request)
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.request_timeout_seconds,
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise WorkerError(
                f"Host request failed with HTTP {exc.code}: {body}"
            ) from exc
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise WorkerError("Host response must be a JSON object.")
        if not parsed.get("ok"):
            raise WorkerError(f"Host returned an error: {parsed}")
        return parsed

    def _add_auth(self, request: urllib.request.Request) -> None:
        if self.auth_token:
            request.add_header("Authorization", f"Bearer {self.auth_token}")


def _run_lc0(
    cmd: list[str],
    *,
    perspective: str,
    timeout_seconds: float,
) -> dict[str, float | int]:
    logger.info("Running: %s", " ".join(_shell_quote(arg) for arg in cmd))
    timeout = timeout_seconds if timeout_seconds > 0 else None
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        partial = None
        try:
            partial = _parse_output(
                exc.stdout or b"",
                allow_partial=True,
                perspective=perspective,
            )
        except WorkerError:
            pass
        raise WorkerError(f"lc0 timed out with partial result {partial}.")

    if result.returncode != 0:
        partial = None
        try:
            partial = _parse_output(
                result.stdout,
                allow_partial=True,
                perspective=perspective,
            )
        except WorkerError:
            pass
        stderr_tail = result.stderr.decode("utf-8", errors="replace")[-500:]
        raise WorkerError(
            "lc0 exited with code "
            f"{result.returncode}: {stderr_tail}; partial={partial}"
        )
    return _parse_output(
        result.stdout,
        allow_partial=False,
        perspective=perspective,
    )


def _parse_output(
    output: bytes,
    *,
    allow_partial: bool,
    perspective: str,
) -> dict[str, float | int]:
    parsed = _get_wld_and_npm(
        output,
        final_only=not allow_partial,
        perspective=perspective,
    )
    if parsed is None:
        tail = output.decode("utf-8", errors="replace")[-500:]
        raise WorkerError(f"Could not parse tournamentstatus:\n{tail}")
    npm, wins, losses, draws = parsed
    return {"wins": wins, "draws": draws, "losses": losses, "npm": npm}


def _get_wld_and_npm(
    output: bytes,
    *,
    final_only: bool,
    perspective: str,
) -> tuple[float, int, int, int] | None:
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

    if perspective in ("WHITE", "BLACK"):
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
    perspective: str,
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
    if perspective == "WHITE":
        wins = white_win + black_loss
        losses = white_loss + black_win
    else:
        wins = black_win + white_loss
        losses = black_loss + white_win
    draws = white_draw + black_draw
    return npm, wins, losses, draws


def _shard_command(
    cmd: Sequence[str],
    *,
    gpus: Sequence[int],
    lc0_parallelism: int,
    mirror_openings: bool,
) -> list[list[str]]:
    games = _get_int_arg(cmd, "--games")
    game_counts = _split_games(
        games,
        shard_count=len(gpus),
        require_even=mirror_openings,
    )
    shard_cmds = []
    for shard_idx, (gpu, shard_games) in enumerate(
        zip(gpus, game_counts, strict=True)
    ):
        if shard_games <= 0:
            continue
        shard_cmd = _replace_arg(cmd, "--games", str(shard_games))
        try:
            seed = _get_int_arg(shard_cmd, "--opening-seed")
            shard_cmd = _replace_arg(
                shard_cmd,
                "--opening-seed",
                str(seed + shard_idx),
            )
        except ValueError:
            pass
        shard_cmd.extend(
            [
                f"--parallelism={lc0_parallelism}",
                "--backend=multiplexing",
                f"--backend-opts=backend=cuda-auto,gpu={gpu}",
            ]
        )
        shard_cmds.append(shard_cmd)
    return shard_cmds


def _split_games(
    total_games: int,
    *,
    shard_count: int,
    require_even: bool,
) -> list[int]:
    if total_games <= 0:
        raise WorkerError("total_games must be positive.")
    if shard_count <= 0:
        raise WorkerError("shard_count must be positive.")
    unit = 2 if require_even else 1
    if require_even and total_games % 2:
        raise WorkerError("Mirrored tournaments require an even game count.")
    total_units = total_games // unit
    base_units, remainder = divmod(total_units, shard_count)
    return [
        unit * (base_units + (1 if idx < remainder else 0))
        for idx in range(shard_count)
    ]


def _combine_results(
    results: Sequence[dict[str, float | int]],
) -> dict[str, float | int]:
    if not results:
        raise WorkerError("At least one shard result is required.")
    total_games = sum(_result_total(result) for result in results)
    npm = (
        sum(float(result["npm"]) * _result_total(result) for result in results)
        / total_games
        if total_games > 0
        else 0.0
    )
    return {
        "wins": sum(int(result["wins"]) for result in results),
        "draws": sum(int(result["draws"]) for result in results),
        "losses": sum(int(result["losses"]) for result in results),
        "npm": npm,
    }


def _result_total(result: dict[str, float | int]) -> int:
    return int(result["wins"]) + int(result["draws"]) + int(result["losses"])


def _result_to_dict(result: dict[str, float | int]) -> dict[str, object]:
    return {
        "wins": int(result["wins"]),
        "draws": int(result["draws"]),
        "losses": int(result["losses"]),
        "npm": float(result["npm"]),
    }


def _rewrite_command(
    cmd: Sequence[str],
    rewrites: dict[str, str],
    lc0_path: str,
) -> list[str]:
    if not cmd:
        raise WorkerError("Remote task command is empty.")
    rewritten = [lc0_path]
    for arg in cmd[1:]:
        rewritten.append(_rewrite_arg(arg, rewrites))
    return rewritten


def _rewrite_arg(arg: str, rewrites: dict[str, str]) -> str:
    if arg in rewrites:
        return rewrites[arg]
    if "=" not in arg:
        return arg
    flag, value = arg.split("=", 1)
    return f"{flag}={rewrites.get(value, value)}"


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


def _expect_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise WorkerError("Expected a JSON object.")
    return value


def _expect_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise WorkerError("Expected a JSON array.")
    return value


def _expect_str_list(value: object) -> list[str]:
    result = []
    for item in _expect_list(value):
        if not isinstance(item, str):
            raise WorkerError("Expected a string array.")
        result.append(item)
    return result


def _parse_gpus(value: str) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.split(",") if part]


def _normalize_host_url(host: str) -> str:
    host = host.rstrip("/")
    if "://" not in host:
        host = f"http://{host}"
    return host


def _shell_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=+-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASGO remote lc0 worker.")
    parser.add_argument(
        "--host",
        required=True,
        help="ASGO host URL or hostname:port, for example 127.0.0.1:8765.",
    )
    parser.add_argument(
        "--lc0-path",
        required=True,
        help="Path to the lc0 binary on this worker machine.",
    )
    parser.add_argument(
        "--work-dir",
        default="/tmp/lczero-asgo-worker",
        help="Directory used for downloaded job files.",
    )
    parser.add_argument(
        "--worker-id",
        default=_default_worker_id(),
        help="Stable worker id shown in host logs.",
    )
    parser.add_argument(
        "--auth-token",
        default="",
        help="Bearer token matching asgo.tournament.remote.auth_token.",
    )
    parser.add_argument(
        "--gpus",
        default="",
        help="Comma-separated GPU ids for local lc0 sharding, e.g. 0,1.",
    )
    parser.add_argument(
        "--lc0-parallelism",
        type=int,
        default=8,
        help="lc0 --parallelism value when --gpus is set.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP request timeout for host communication.",
    )
    parser.add_argument(
        "--keep-workdirs",
        action="store_true",
        help="Keep downloaded job directories for debugging.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit after one no_job response or one completed job.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    worker = AsgoWorker(
        host=args.host,
        lc0_path=args.lc0_path,
        work_dir=args.work_dir,
        worker_id=args.worker_id,
        auth_token=args.auth_token,
        gpus=_parse_gpus(args.gpus),
        lc0_parallelism=args.lc0_parallelism,
        request_timeout_seconds=args.request_timeout_seconds,
        keep_workdirs=args.keep_workdirs,
    )
    while True:
        try:
            return worker.run(once=args.once)
        except KeyboardInterrupt:
            return 130
        except Exception:
            logger.exception("Worker loop failed; retrying in 5 seconds.")
            if args.once:
                return 1
            time.sleep(5.0)


if __name__ == "__main__":
    sys.exit(main())
