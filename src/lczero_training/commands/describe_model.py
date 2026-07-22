import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Describe model parameters and FLOPs from a textproto."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a RootConfig or ModelConfig textproto file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    from lczero_training.training.describe_model import describe_model_text

    print(describe_model_text(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
