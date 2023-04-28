import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from pathlib import Path
from config import LOG_DIR, LOG_LEVEL


class Logger:
    def __init__(self, logger_name, log_file_name):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(LOG_LEVEL)
        self.log_file_name = log_file_name
        self.log_file_path = Path(LOG_DIR) / self.log_file_name
        self._configure_logger()

    def _configure_logger(self):
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(LOG_LEVEL)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        file_handler = TimedRotatingFileHandler(
            filename=self.log_file_path,
            when="midnight",
            backupCount=7,
            utc=True,
        )
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

    def critical(self, message):
        self.logger.critical(message)

    def exception(self, message):
        self.logger.exception(message)


