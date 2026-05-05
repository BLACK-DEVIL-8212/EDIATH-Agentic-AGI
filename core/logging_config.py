import logging
import logging.handlers
import sys
from pathlib import Path


class SafeStreamHandler(logging.StreamHandler):
    """Safe stream handler that handles Unicode errors gracefully"""

    def __init__(self, stream=None):
        super().__init__(stream)
        self._stream = stream

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                # Fallback: encode with 'replace' error handling
                try:
                    encoded_msg = msg.encode("ascii", "replace").decode("ascii")
                    stream.write(encoded_msg + self.terminator)
                except:
                    pass
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)


def setup_structured_logging(base_path: str = "logs", level: str = "INFO"):
    """
    Setup structured logging for EDIATH with rotation and JSON formatting
    """

    # Create logs directory
    log_dir = Path(base_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Main logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    # File handler with rotation (10MB max, 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "EDIATH.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s | PID:%(process)d | TID:%(thread)d",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - using SafeStreamHandler to avoid reconfigure issues
    console_handler = SafeStreamHandler(sys.stdout)
    console_formatter = logging.Formatter("%(levelname)-8s | %(name)s | %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=10,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)

    # Task handler
    task_handler = logging.handlers.RotatingFileHandler(
        log_dir / "tasks.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    task_handler.setFormatter(file_formatter)
    logger.addHandler(task_handler)

    # Voice handler
    voice_handler = logging.handlers.RotatingFileHandler(
        log_dir / "voice.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    voice_handler.setFormatter(file_formatter)
    logger.addHandler(voice_handler)

    # Vision handler
    vision_handler = logging.handlers.RotatingFileHandler(
        log_dir / "vision.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    vision_handler.setFormatter(file_formatter)
    logger.addHandler(vision_handler)

    # Performance handler
    perf_handler = logging.handlers.RotatingFileHandler(
        log_dir / "performance.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    perf_handler.setFormatter(file_formatter)
    logger.addHandler(perf_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get named logger"""
    logger = logging.getLogger(name)
    if not logger.handlers:  # Avoid duplicate handlers
        logger.addHandler(logging.NullHandler())
    return logger


# Initialize default logger on module load
default_logger = setup_structured_logging()


__all__ = [
    "setup_structured_logging",
    "get_logger",
    "default_logger",
    "SafeStreamHandler",
]
