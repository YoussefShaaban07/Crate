"""
Logging setup.

Everything the system does — retrieval hits, LLM calls, guardrail rejections,
fallbacks — goes through here. The console stays readable (INFO and above,
message only) while the log file keeps the full timestamped trace for grading
and debugging.
"""

from __future__ import annotations

import logging
import sys

from src.config import LOG_DIR, LOG_FILE

_CONFIGURED = False


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure root logging once and return the application logger."""
    global _CONFIGURED

    logger = logging.getLogger("vibescope")

    if _CONFIGURED:
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s | %(message)s")
    )
    logger.addHandler(file_handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console)

    _CONFIGURED = True
    logger.debug("logging initialised (file=%s, verbose=%s)", LOG_FILE, verbose)
    return logger


def get_logger(name: str = "") -> logging.Logger:
    """Child logger under the application logger."""
    base = setup_logging()
    return base.getChild(name) if name else base
