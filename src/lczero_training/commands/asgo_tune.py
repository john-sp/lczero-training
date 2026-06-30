import argparse
import logging
import sys

from google.protobuf import text_format

from lczero_training.commands import configure_root_logging
from proto.root_config_pb2 import RootConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ASGO zeroth-order tuning for ELO optimization."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the ASGO tuning config textproto.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override config max_iterations.",
    )
    parser.add_argument(
        "--override-lc0-config-conflict",
        action="store_true",
        help=(
            "Allow runner-owned lc0 flags and resume from a checkpoint whose "
            "ASGO config hash differs."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Lazy import keeps --help responsive and avoids importing JAX eagerly.
    from lczero_training.asgo.tuner import AsgoTuner

    config = RootConfig()
    logging.info("Reading configuration from %s", args.config)
    with open(args.config, "r") as f:
        text_format.Parse(f.read(), config)

    if not config.HasField("asgo"):
        logging.error("Config must contain an 'asgo' section.")
        return 1

    tuner = AsgoTuner(
        config,
        override_lc0_config_conflict=args.override_lc0_config_conflict,
    )
    tuner.run(max_iterations=args.max_iterations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
