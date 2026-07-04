import json
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from lczero_training.asgo.elo import DirectTournamentResult
from lczero_training.asgo.tournament import RemoteTournamentRunner


def _remote_config() -> SimpleNamespace:
    return SimpleNamespace(
        listen_host="127.0.0.1",
        port=0,
        auth_token="",
        long_poll_seconds=0.1,
        job_timeout_seconds=30,
        max_retries=0,
    )


def _direct_config(**overrides: object) -> SimpleNamespace:
    values = {
        "game_pairs_per_round": 5,
        "nodes": 800,
        "movetime": 0,
        "opening_book": "",
        "opening_seed": -1,
        "gpu": [],
        "extra_args": [],
        "timeout_seconds": 30,
        "lc0_parallelism": 8,
        "direct_comparison": SimpleNamespace(mirror_openings=True),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _post_json(url: str, path: str, payload: object) -> dict[str, object]:
    request = urllib.request.Request(
        f"{url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5.0) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def _as_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _as_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def test_remote_direct_round_claim_and_complete(tmp_path: Path) -> None:
    pos = tmp_path / "pos.pb.gz"
    neg = tmp_path / "neg.pb.gz"
    pos.write_bytes(b"pos")
    neg.write_bytes(b"neg")
    runner = RemoteTournamentRunner(
        _direct_config(),
        "host-lc0",
        str(tmp_path),
        remote_config=_remote_config(),
    )
    try:
        future = runner.submit_pair(str(pos), str(neg))

        claimed = _post_json(
            runner.server_url,
            "/v1/jobs/claim",
            {"worker_id": "worker-a"},
        )
        assert claimed["status"] == "job"
        job = _as_dict(claimed["job"])
        assert len(_as_list(job["tasks"])) == 1
        assert len(_as_list(job["files"])) == 2

        completed = _post_json(
            runner.server_url,
            f"/v1/jobs/{str(job['job_id'])}/complete",
            {
                "ok": True,
                "worker_id": "worker-a",
                "results": [
                    {"wins": 6, "draws": 2, "losses": 2, "npm": 10.0}
                ],
            },
        )
        assert completed["ok"]
        assert future.result(timeout=1.0) == DirectTournamentResult(
            wins=6,
            draws=2,
            losses=2,
            npm=10.0,
        )
    finally:
        runner.cleanup()


def test_remote_fixed_opponent_queues_whole_round(tmp_path: Path) -> None:
    pos = tmp_path / "pos.pb.gz"
    neg = tmp_path / "neg.pb.gz"
    opponent = tmp_path / "opponent.pb.gz"
    pos.write_bytes(b"pos")
    neg.write_bytes(b"neg")
    opponent.write_bytes(b"opponent")
    config = _direct_config(
        fixed_opponent=SimpleNamespace(
            tuned_side=0,
            mirror_openings=False,
            opponent=[
                SimpleNamespace(
                    name="baseline",
                    weights=str(opponent),
                    weight=1.0,
                    nodes=0,
                    extra_args=[],
                )
            ],
        )
    )
    runner = RemoteTournamentRunner(
        config,
        "host-lc0",
        str(tmp_path),
        remote_config=_remote_config(),
    )
    try:
        future = runner.submit_against_opponents(str(pos), str(neg))

        claimed = _post_json(
            runner.server_url,
            "/v1/jobs/claim",
            {"worker_id": "worker-a"},
        )
        job = _as_dict(claimed["job"])
        assert len(_as_list(job["tasks"])) == 2
        assert len(_as_list(job["files"])) == 3

        _post_json(
            runner.server_url,
            f"/v1/jobs/{str(job['job_id'])}/complete",
            {
                "ok": True,
                "worker_id": "worker-a",
                "results": [
                    {"wins": 6, "draws": 2, "losses": 2, "npm": 10.0},
                    {"wins": 2, "draws": 2, "losses": 6, "npm": 10.0},
                ],
            },
        )
        result = future.result(timeout=1.0)
        assert result.elo_diff > 0
        assert result.wins == 6
    finally:
        runner.cleanup()


def test_remote_rejects_completion_from_wrong_worker(tmp_path: Path) -> None:
    pos = tmp_path / "pos.pb.gz"
    neg = tmp_path / "neg.pb.gz"
    pos.write_bytes(b"pos")
    neg.write_bytes(b"neg")
    runner = RemoteTournamentRunner(
        _direct_config(),
        "host-lc0",
        str(tmp_path),
        remote_config=_remote_config(),
    )
    try:
        runner.submit_pair(str(pos), str(neg))
        claimed = _post_json(
            runner.server_url,
            "/v1/jobs/claim",
            {"worker_id": "worker-a"},
        )
        job = _as_dict(claimed["job"])
        request = urllib.request.Request(
            f"{runner.server_url}/v1/jobs/{str(job['job_id'])}/complete",
            data=json.dumps(
                {
                    "ok": True,
                    "worker_id": "worker-b",
                    "results": [
                        {"wins": 1, "draws": 0, "losses": 0, "npm": 1.0}
                    ],
                }
            ).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            urllib.request.urlopen(request, timeout=5.0)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
        else:
            raise AssertionError("wrong-worker completion was accepted")
    finally:
        runner.cleanup()


def test_remote_runner_allows_opening_seed_without_host_lc0_probe(
    tmp_path: Path,
) -> None:
    runner = RemoteTournamentRunner(
        _direct_config(opening_seed=123),
        "host-lc0",
        str(tmp_path),
        remote_config=_remote_config(),
    )
    runner.cleanup()
