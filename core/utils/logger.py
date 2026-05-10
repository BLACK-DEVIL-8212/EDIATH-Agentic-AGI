"""
🔥 EDIATH LOGGER - COMPLETE KIVY ISOLATION
"""

import os

# Disable Kivy logging BEFORE importing anything Kivy-related
os.environ["KIVY_NO_FILELOG"] = "1"
os.environ["KIVY_LOG_MODE"] = "PYTHON"

try:
    from loguru import logger as loguru_logger

    _HAS_LOGURU = True
except ImportError:
    loguru_logger = None
    _HAS_LOGURU = False


def _safe_import_helpers():
    try:
        from core.utils.helpers import ensure_dir, get_project_root

        return ensure_dir, get_project_root
    except ImportError:

        def ensure_dir(path):
            os.makedirs(path, exist_ok=True)
            return path

        def get_project_root():
            return os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )

        return ensure_dir, get_project_root


ensure_dir, get_project_root = _safe_import_helpers()


class EDIATHLogger:
    _instance = None
    _log_active = True

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.log_dir = ensure_dir(os.path.join(get_project_root(), "logs"))
        self._recursion_depth = 0

        # COMPLETELY DISABLE KIVY LOGGING
        self._disable_kivy_logging()

        # Setup loguru or fallback
        if _HAS_LOGURU:
            self._setup_loguru()
        else:
            self._setup_fallback()

    def _disable_kivy_logging(self):
        """Completely disable Kivy's logging system to prevent recursion"""
        try:
            # Method 1: Redirect Kivy Logger to nowhere
            import kivy.logger

            kivy.logger.Logger.handlers = []
            kivy.logger.Logger.propagate = False

            # Method 2: Override Kivy's write method
            class NullWriter:
                def write(self, *args, **kwargs):
                    pass

                def flush(self, *args, **kwargs):
                    pass

            null_writer = NullWriter()
            if hasattr(kivy.logger, "Logger"):
                kivy.logger.Logger._out = null_writer
                kivy.logger.Logger._err = null_writer
        except Exception:
            pass

    def _setup_loguru(self):
        """Setup loguru with safe handlers"""
        try:
            self.logger = loguru_logger
            self.logger.remove()  # Remove default handlers

            # Add console handler ONLY to stdout (not stderr)
            import sys

            self.logger.add(
                sys.stdout,
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
                level="INFO",
                colorize=True,
                catch=True,
            )

            # Add file handler
            log_path = os.path.join(self.log_dir, "EDIATH.log")
            self.logger.add(
                log_path,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                level="DEBUG",
                rotation="10 MB",
                catch=True,
            )
        except Exception as e:
            print(f"Loguru setup failed: {e}")
            self._setup_fallback()

    def _setup_fallback(self):
        """Fallback to simple print logging"""
        self.logger = None

    def _log(self, level: str, message: str, *args):
        """Simple log method without recursion"""
        if not self.__class__._log_active:
            return

        if self._recursion_depth > 3:
            return

        self._recursion_depth += 1

        try:
            # Format message
            if args:
                try:
                    msg = (
                        message % args
                        if "%" in message
                        else f"{message} {' '.join(str(a) for a in args)}"
                    )
                except Exception:
                    msg = f"{message} {args}"
            else:
                msg = str(message)

            # Log using loguru or print
            if _HAS_LOGURU and hasattr(self, "logger") and self.logger:
                getattr(self.logger, level, self.logger.info)(msg)
            else:
                # Simple print fallback
                print(f"[{level.upper()}] {msg}")
        except Exception:
            # Ultimate fallback
            try:
                print(f"[{level.upper()}] {message}")
            except Exception:
                pass
        finally:
            self._recursion_depth -= 1

    def debug(self, msg: str, *args):
        self._log("debug", msg, *args)

    def info(self, msg: str, *args):
        self._log("info", msg, *args)

    def warning(self, msg: str, *args):
        self._log("warning", msg, *args)

    def error(self, msg: str, *args):
        self._log("error", msg, *args)

    def critical(self, msg: str, *args):
        self._log("critical", msg, *args)

    def exception(self, msg: str, *args):
        self._log("error", msg, *args)


# Global instance
logger = EDIATHLogger()

__all__ = ["logger", "EDIATHLogger"]
