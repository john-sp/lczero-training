import argparse
import logging
import sys

from lczero_training.commands import configure_root_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize an ASGO tuning run from an lc0 model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the ASGO tuning config textproto.",
    )
    parser.add_argument(
        "--lczero_model",
        type=str,
        required=True,
        help="Path to an existing lc0 model (.pb.gz) to start from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for initial state.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip checkpoint creation.",
    )
    parser.add_argument(
        "--ignore-config-mismatch",
        action="store_true",
        help="Load weights even if the inferred model config differs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Lazy import to keep --help responsive and avoid heavy deps unless needed.
    from lczero_training.asgo.checkpoint import asgo_init

    asgo_init(
        config_filename=args.config,
        lczero_model=args.lczero_model,
        seed=args.seed,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        ignore_config_mismatch=args.ignore_config_mismatch,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
