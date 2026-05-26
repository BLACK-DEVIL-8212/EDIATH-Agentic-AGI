"""
Centralized Logging - Production Safe + Structlog Compatible
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

# ------------------------------------------------------------
# OPTIONAL STRUCTLOG
# ------------------------------------------------------------
try:
    import structlog

    STRUCTLOG_AVAILABLE = True

except Exception:
    structlog = None
    STRUCTLOG_AVAILABLE = False

# ------------------------------------------------------------
# LOG DIRECTORY
# ------------------------------------------------------------
LOG_DIR = Path("logs")

try:
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
except Exception:
    pass

# ------------------------------------------------------------
# UTF-8 SAFE STDOUT/STDERR
# ------------------------------------------------------------
try:

    if hasattr(sys.stdout, "reconfigure"):

        try:
            sys.stdout.reconfigure(
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            pass

    if hasattr(sys.stderr, "reconfigure"):

        try:
            sys.stderr.reconfigure(
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            pass

except Exception:
    pass

# ------------------------------------------------------------
# LOG FORMAT
# ------------------------------------------------------------
LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ------------------------------------------------------------
# ROOT LOGGER
# ------------------------------------------------------------
_root_logger = logging.getLogger("EDIATH")

_root_logger.setLevel(logging.INFO)

_root_logger.propagate = False

# ------------------------------------------------------------
# PREVENT DUPLICATE HANDLERS
# ------------------------------------------------------------
if not _root_logger.handlers:

    # --------------------------------------------------------
    # CONSOLE HANDLER
    # --------------------------------------------------------
    console_handler = logging.StreamHandler(
        stream=sys.stdout
    )

    try:

        if hasattr(
            console_handler.stream,
            "reconfigure",
        ):

            console_handler.stream.reconfigure(
                encoding="utf-8",
                errors="replace",
            )

    except Exception:
        pass

    console_handler.setLevel(logging.INFO)

    console_handler.setFormatter(

        logging.Formatter(
            LOG_FORMAT,
            datefmt=DATE_FORMAT,
        )
    )

    _root_logger.addHandler(
        console_handler
    )

    # --------------------------------------------------------
    # FILE HANDLER
    # --------------------------------------------------------
    try:

        file_handler = (
            logging.handlers.RotatingFileHandler(
                LOG_DIR / "EDIATH.log",
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
        )

        file_handler.setLevel(
            logging.INFO
        )

        file_handler.setFormatter(

            logging.Formatter(
                LOG_FORMAT,
                datefmt=DATE_FORMAT,
            )
        )

        _root_logger.addHandler(
            file_handler
        )

    except Exception:
        pass

# ------------------------------------------------------------
# STRUCTLOG SETUP
# ------------------------------------------------------------
def setup_structured_logging() -> None:
    """
    Configure structlog safely.
    """

    if not STRUCTLOG_AVAILABLE:
        return

    try:

        structlog.configure(

            processors=[

                structlog.contextvars.merge_contextvars,

                structlog.processors.TimeStamper(
                    fmt="iso"
                ),

                structlog.stdlib.add_log_level,

                structlog.stdlib.filter_by_level,

                structlog.processors.StackInfoRenderer(),

                structlog.processors.format_exc_info,

                structlog.processors.UnicodeDecoder(),

                structlog.processors.JSONRenderer(),
            ],

            wrapper_class=(
                structlog.stdlib.BoundLogger
            ),

            logger_factory=(
                structlog.stdlib.LoggerFactory()
            ),

            cache_logger_on_first_use=True,
        )

    except Exception as e:

        try:

            _root_logger.warning(
                f"Structlog setup failed: {e}"
            )

        except Exception:
            pass

# ------------------------------------------------------------
# LOGGER FACTORY
# ------------------------------------------------------------
def get_logger(
    name: Optional[str] = None,
):
    """
    Get production-safe logger.

    Supports:
    - structlog
    - stdlib logging
    - fallback safety
    - duplicate handler prevention
    """

    logger_name = (
        str(name)
        if name
        else "EDIATH"
    )

    # --------------------------------------------------------
    # STRUCTLOG LOGGER
    # --------------------------------------------------------
    if STRUCTLOG_AVAILABLE:

        try:

            return structlog.get_logger(
                logger_name
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # STANDARD LOGGER
    # --------------------------------------------------------
    try:

        logger = logging.getLogger(
            logger_name
        )

        logger.setLevel(logging.INFO)

        logger.propagate = False

        # avoid duplicate handlers
        if not logger.handlers:

            for handler in _root_logger.handlers:

                try:
                    logger.addHandler(
                        handler
                    )
                except Exception:
                    continue

        return logger

    except Exception:

        return _root_logger

# ------------------------------------------------------------
# GLOBAL LOGGER EXPORT
# ------------------------------------------------------------
logger = get_logger("EDIATH")

# ------------------------------------------------------------
# SAFE STARTUP MESSAGE
# ------------------------------------------------------------
try:

    logger.debug(
        "✓ Logging system initialized"
    )

except Exception:
    pass

# ------------------------------------------------------------
# LEGACY COMPATIBILITY
# ------------------------------------------------------------
get_logger_old = get_logger

# ------------------------------------------------------------
# EXPORTS
# ------------------------------------------------------------
__all__ = [

    "logger",

    "get_logger",

    "get_logger_old",

    "setup_structured_logging",

    "STRUCTLOG_AVAILABLE",
]