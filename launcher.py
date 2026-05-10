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
<<<<<<< HEAD
# MODEL CONFIGURATION
# =========================
def get_model_path():
    """Get the correct model path with fallback options"""
    base_path = Path("E:/EDIATH/models/EDIATH-q4_k_m.gguf")
    alt_path = Path("C:/models/EDIATH-q4_k_m.gguf")
    local_path = Path("./models/EDIATH-q4_k_m.gguf")
    
    # Check which path exists
    if base_path.exists():
        return str(base_path)
    elif alt_path.exists():
        return str(alt_path)
    elif local_path.exists():
        return str(local_path)
    else:
        logger.error("❌ Model file not found in any location!")
        logger.info(f"   Checked: {base_path}")
        logger.info(f"   Checked: {alt_path}")
        logger.info(f"   Checked: {local_path}")
        return str(base_path)  # Return default anyway


# =========================
=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
<<<<<<< HEAD
        # Set environment variable for model path before importing
        import os
        os.environ['EDIATH_MODEL_PATH'] = get_model_path()
        
        from main import run_backend_only

        logger.info("🔧 Starting backend...")
        logger.info(f"📁 Using model: {os.environ['EDIATH_MODEL_PATH']}")
=======
        from main import run_backend_only

        logger.info("🔧 Starting backend...")
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        return run_async_safe(run_backend_only)

    except Exception as e:
        logger.error(f"❌ Backend import failed: {e}")
        return None


# =========================
# UI START
# =========================
def start_full_ui():
    try:
<<<<<<< HEAD
        # Set environment variable for model path before importing
        import os
        os.environ['EDIATH_MODEL_PATH'] = get_model_path()
        
        from main import main_interactive

        logger.info("🎨 Starting UI + backend...")
        logger.info(f"📁 Using model: {os.environ['EDIATH_MODEL_PATH']}")
=======
        from main import main_interactive

        logger.info("🎨 Starting UI + backend...")
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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


<<<<<<< HEAD
def check_llama_version():
    """Check if llama-cpp-python is installed and version"""
    try:
        import llama_cpp
        version = getattr(llama_cpp, '__version__', 'unknown')
        logger.info(f"✅ llama-cpp-python version: {version}")
        if version != 'unknown' and version < '0.3.16':
            logger.warning("⚠️ Version is older than 0.3.16 - update recommended!")
            logger.warning("   Run: pip install --upgrade llama-cpp-python")
        return True
    except ImportError:
        logger.error("❌ llama-cpp-python not installed!")
        logger.info("   Install with: pip install llama-cpp-python")
        return False


=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
║                   EDIATH  LAUNCHER                   ║
╚══════════════════════════════════════════════════════╝
    """)

    setup_signal_handlers()

    if not validate_main():
        sys.exit(1)

<<<<<<< HEAD
    # Check llama-cpp-python before starting
    if not check_llama_version():
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Display model info
    model_path = get_model_path()
    print(f"\n📁 Model path: {model_path}")
    if Path(model_path).exists():
        size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        print(f"📊 Model size: {size_mb:.1f} MB")
    else:
        print(f"⚠️  WARNING: Model file not found at {model_path}")
        print("   Please ensure the model file exists or update get_model_path()")
        
    print()

=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
