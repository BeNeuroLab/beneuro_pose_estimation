"""
Initialize macro variables and functions
"""
import logging
from pathlib import Path

from rich.logging import RichHandler


def set_logging(file_path: Path | None = None, overwrite: bool = True) -> None:
    frmt = "%(asctime)s | %(name)s | %(levelname)s | > %(message)s"

    if file_path is not None:
        logging.basicConfig(
            filename=file_path,
            level=logging.INFO,
            format=frmt
        )
    else:
        if overwrite is True and file_path.exists() is True:
            file_path.unlink()

        logging.basicConfig(
            handlers=[RichHandler(level="NOTSET")],
            level=logging.INFO,
            format=frmt
        )
