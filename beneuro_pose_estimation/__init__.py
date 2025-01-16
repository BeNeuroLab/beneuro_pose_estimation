"""
Initialize macro variables and functions
"""

import logging
from pathlib import Path

from rich.logging import RichHandler


def set_logging(file_path=None, overwrite=True):
    frmt = "%(asctime)s - %(levelname)s - %(message)s"

    if file_path is not None:
        file_path = Path(file_path)
        if overwrite is True and file_path.exists() is True:
            file_path.unlink()
        logging.basicConfig(
            filename=file_path,
            level=logging.INFO,
            format=frmt,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logging.basicConfig(
            handlers=[RichHandler(level="NOTSET")],
            level=logging.INFO,
            format=frmt,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
