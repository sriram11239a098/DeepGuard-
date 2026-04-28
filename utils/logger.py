"""
logger.py — Logging setup for DeepGuard / Sach-AI.

Usage:
    from utils.logger import setup_logger
    logger = setup_logger("training", config.LOGS_DIR)
    logger.info("Epoch 1 started")
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str, log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Create (or retrieve) a named logger that writes to both stdout and a
    rotating log file inside `log_dir`.

    Args:
        name    : logger name (used as the log file stem, e.g. "training")
        log_dir : directory where the .log file will be written
        level   : logging level (default INFO)

    Returns:
        Configured logging.Logger instance.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid duplicate handlers if setup_logger is called more than once
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
