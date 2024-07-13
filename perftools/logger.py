
import logging
import time
from logging.handlers import RotatingFileHandler
from typing import Union

from . import __version__


class Logger(logging.Logger):
    _name = "perftools"

    # default formatter
    _formatter = logging.Formatter("[{asctime}]({process}) - {levelname} - {filename}:{lineno} - {message}", "%Y-%m-%d %H:%M:%S", "{")
    _formatter.converter = time.localtime

    # base stream formatter
    _stream_handler = logging.StreamHandler()
    _stream_handler.setLevel(logging.DEBUG)
    _stream_handler.setFormatter(_formatter)

    def __init__(self) -> None:
        super().__init__(self._name)

        self.addHandler(self._stream_handler)
        self.warning("Logger starts logging. (Version %s)", __version__)

    def add_file_handler(self, path: str, level: Union[str, int] = 0):
        file_handler = RotatingFileHandler(path, encoding="utf8", maxBytes=50 << 20, backupCount=5)
        file_handler.setLevel(level)
        file_handler.setFormatter(self._formatter)
        self.addHandler(file_handler)
        self.warning("Logger starts logging to file %s. (Version %s)", path, __version__)
        return file_handler


# global instance
logger = Logger()
