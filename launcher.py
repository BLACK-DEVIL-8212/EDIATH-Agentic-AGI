import asyncio
import threading
import logging
import sys
import signal
import time
from pathlib import Path

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

shutdown_event = threading.Event()


# =========================
# SIGNAL HANDLING
# =========================
def signal_handler(signum, frame):
    logger.info(f"📡 Signal {signum} received → shutting down")
    shutdown_event.set()


def setup_signal_handlers():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# =========================
# SAFE ASYNC RUNNER
# =========================
def run_async_safe(coro_func, *args, **kwargs):
    def target():
        while not shutdown_event.is_set():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                logger.info("🧠 Backend loop starting...")
                loop.run_until_complete(coro_func(*args, **kwargs))

            except Exception as e:
                logger.error(f"💥 Backend crashed: {e}", exc_info=True)
                logger.info("🔁 Restarting backend in 3s...")
                time.sleep(3)

            finally:
                try:
                    loop.close()
                except Exception:
                    pass

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


# =========================
# BACKEND START
# =========================
def start_backend():
    try:
        from main import run_backend_only

        logger.info("🔧 Starting backend...")
        return run_async_safe(run_backend_only)

    except Exception as e:
        logger.error(f"❌ Backend import failed: {e}")
        return None


# =========================
# UI START
# =========================
def start_full_ui():
    try:
        from main import main_interactive

        logger.info("🎨 Starting UI + backend...")
        main_interactive()

    except Exception as e:
        logger.error(f"💥 UI crashed: {e}", exc_info=True)


# =========================
# MONITOR LOOP (FIXES FREEZE)
# =========================
def monitor_backend(thread):
    logger.info("🖥️ Monitoring backend health...")

    while not shutdown_event.is_set():
        if not thread.is_alive():
            logger.error("❌ Backend died → restarting...")
            thread = start_backend()

        # CPU cooldown (prevents 100% lock)
        time.sleep(1.5)

    logger.info("🛑 Monitor stopping...")


# =========================
# VALIDATION
# =========================
def validate_main():
    path = Path(__file__).parent / "main.py"
    if not path.exists():
        logger.error("❌ main.py missing")
        return False
    return True


# =========================
# MAIN
# =========================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ui", "backend"], default="ui")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════╗
║                 EDIATH LAUNCHER FIXED                ║
╚══════════════════════════════════════════════════════╝
    """)

    setup_signal_handlers()

    if not validate_main():
        sys.exit(1)

    try:
        if args.mode == "backend":
            thread = start_backend()
            monitor_backend(thread)

        else:
            start_full_ui()

    except KeyboardInterrupt:
        logger.info("🛑 User stopped system")

    finally:
        shutdown_event.set()
        time.sleep(1)
        logger.info("👋 Shutdown complete")


if __name__ == "__main__":
    main()
