import logging
import sys
import threading
import time

import anyio
from google.protobuf import text_format

import proto.training_metrics_pb2 as training_metrics_pb2

from lczero_training._lczero_training import DataLoader
from lczero_training.daemon.protocol.communicator import Communicator
from lczero_training.daemon.protocol.messages import (
    StartTrainingImmediatelyPayload,
    StartTrainingPayload,
    TrainingStatusPayload,
)
from proto.root_config_pb2 import RootConfig

logger = logging.getLogger(__name__)


def _configure_file_logging(config: RootConfig) -> None:
    if config.HasField("log_filename"):
        file_handler = logging.FileHandler(config.log_filename)
        file_handler.setFormatter(
            logging.Formatter(
                "%(levelname).1s%(asctime)s.%(msecs)03d %(name)s "
                "%(filename)s:%(lineno)d] %(message)s",
                datefmt="%m%d %H:%M:%S",
            )
        )
        logging.getLogger().addHandler(file_handler)
        logger.info("Added file logging to %s", config.log_filename)


def _read_config_file(config_filepath: str) -> RootConfig:
    config = RootConfig()
    with open(config_filepath, "r") as f:
        text_format.Parse(f.read(), config)
    return config


class AsgoDaemon:
    """Daemon wrapper that streams ASGO DataLoader metrics to the TUI."""

    def __init__(self, *, max_iterations: int | None = None) -> None:
        self._max_iterations = max_iterations
        self._config_filepath: str | None = None
        self._data_loader: DataLoader | None = None
        self._data_loader_lock = threading.Lock()
        self._communicator = Communicator(self, sys.stdin, sys.stdout)
        self._communicator_thread = threading.Thread(
            target=lambda: self._communicator.run(),
            daemon=True,
        )
        self._communicator_thread.start()
        self._metrics_thread = threading.Thread(
            target=lambda: anyio.run(self._metrics_main),
            daemon=True,
        )
        self._metrics_thread.start()

    def run(self) -> None:
        while self._config_filepath is None:
            logger.info("Waiting for ASGO config.")
            time.sleep(1)

        logger.info("ASGO config received. Starting tuner.")
        config = _read_config_file(self._config_filepath)
        _configure_file_logging(config)

        from lczero_training.asgo.tuner import AsgoTuner

        tuner = AsgoTuner(
            config,
            data_loader_callback=self._set_data_loader,
        )
        tuner.run(max_iterations=self._max_iterations)

    def on_start_training(self, payload: StartTrainingPayload) -> None:
        self._config_filepath = payload.config_filepath

    def on_start_training_immediately(
        self, payload: StartTrainingImmediatelyPayload
    ) -> None:
        del payload
        logger.info("Ignoring immediate-start request in ASGO TUI mode.")

    def _set_data_loader(self, data_loader: DataLoader | None) -> None:
        with self._data_loader_lock:
            self._data_loader = data_loader

    def _get_data_loader(self) -> DataLoader | None:
        with self._data_loader_lock:
            return self._data_loader

    async def _metrics_main(self) -> None:
        while True:
            await anyio.sleep(1.1)
            self._communicator.send(self._status_payload())

    def _status_payload(self) -> TrainingStatusPayload:
        data_loader = self._get_data_loader()
        if data_loader is None:
            return TrainingStatusPayload()

        try:
            stats_1_second_bytes, _ = data_loader.get_bucket_metrics(0, False)
            stats_total_bytes, update_secs = (
                data_loader.get_aggregate_ending_now(float("inf"), False)
            )
        except Exception:
            logger.exception("Failed to read ASGO DataLoader metrics.")
            return TrainingStatusPayload()

        dataloader_1_second = training_metrics_pb2.DataLoaderMetricsProto()
        dataloader_1_second.ParseFromString(stats_1_second_bytes)
        dataloader_total = training_metrics_pb2.DataLoaderMetricsProto()
        dataloader_total.ParseFromString(stats_total_bytes)
        return TrainingStatusPayload(
            dataloader_update_secs=update_secs,
            dataloader_1_second=dataloader_1_second,
            dataloader_total=dataloader_total,
        )
