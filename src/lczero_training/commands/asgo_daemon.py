import argparse
import logging
import sys

from lczero_training.asgo.daemon import AsgoDaemon
from lczero_training.commands import configure_root_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ASGO TUI daemon.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override ASGO config max_iterations.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)
    parser = _build_parser()
    args = parser.parse_args(argv)

    daemon = AsgoDaemon(max_iterations=args.max_iterations)
    daemon.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
