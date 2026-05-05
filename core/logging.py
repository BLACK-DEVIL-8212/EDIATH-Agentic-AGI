"""Centralized Logging - Production structlog"""

import logging
import sys

try:
    import structlog

    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None

# Stdlib fallback logger
_fallback_logger = logging.getLogger("ediath.fallback")
_fallback_logger.setLevel(logging.INFO)
if not _fallback_logger.handlers:
    # Ensure stdout/stderr use UTF-8 to avoid UnicodeEncodeError on Windows consoles
    try:
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        if hasattr(sys.stderr, "reconfigure"):
            try:
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    except Exception:
        pass

    handler = logging.StreamHandler(stream=sys.stdout)
    try:
        handler.stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _fallback_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get logger with structlog if available, fallback otherwise"""
    if STRUCTLOG_AVAILABLE:
        try:
            return structlog.get_logger(name)
        except Exception:
            pass

    logger = logging.getLogger(name)
    if name not in [h.name for h in _fallback_logger.handlers]:
        logger.addHandler(_fallback_logger.handlers[0])
    return logger


# Structured logging setup
def setup_structured_logging():
    if not STRUCTLOG_AVAILABLE:
        return

    try:
        structlog.configure(
            processors=[
                structlog.thread.ThreadLocalDictProcessor(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.stdlib.filter_by_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
    except Exception as e:
        print(f"Structured logging setup failed: {e}")


# Legacy compat
get_logger_old = get_logger  # For files still using old import

__all__ = ["get_logger", "setup_structured_logging"]
