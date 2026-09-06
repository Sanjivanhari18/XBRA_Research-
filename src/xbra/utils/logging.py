"""Centralised logging setup for XBRA.

All modules import logger from here so format/level is consistent.
Call `setup_logging()` once at application startup (Streamlit app.py or CLI).
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

_LOG_DIR = Path(__file__).resolve().parents[4] / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _logger.remove()
    _logger.add(sys.stderr, level=level, colorize=True,
                format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | "
                       "<cyan>{name}:{line}</cyan> — {message}")
    _logger.add(_LOG_DIR / "xbra_{time:YYYY-MM-DD}.log",
                rotation="00:00", retention="7 days", level="DEBUG")
    _CONFIGURED = True


# Module-level convenience: just `from src.xbra.utils.logging import logger`
logger = _logger
