import argparse

from lczero_training.tui.app import TrainingTuiApp


class AsgoTuiApp(TrainingTuiApp):
    """TUI runner for ASGO with the existing DataLoader visualization."""

    DAEMON_MODULE = "lczero_training.commands.asgo_daemon"
    DATA_PIPELINE_TITLE = "ASGO activation cache data pipeline"
    TRAINING_SCHEDULE_TITLE = "ASGO Status"

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        TrainingTuiApp.add_arguments(parser)
        parser.add_argument(
            "--max-iterations",
            type=int,
            default=None,
            help="Override ASGO config max_iterations.",
        )

    def __init__(self, args: argparse.Namespace | None = None) -> None:
        super().__init__(args)
        if args is not None and args.max_iterations is not None:
            self._daemon_flags.extend(
                ["--max-iterations", str(args.max_iterations)]
            )
