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
        # Set environment variable for model path before importing
        import os
        os.environ['EDIATH_MODEL_PATH'] = get_model_path()
        
        from main import run_backend_only
        
        logger.info("🔧 Starting backend...")
        logger.info(f"📁 Using model: {os.environ['EDIATH_MODEL_PATH']}")
        return run_async_safe(run_backend_only)
        
    except Exception as e:
        logger.error(f"❌ Backend import failed: {e}")
        return None


# =========================
# UI START
# =========================
def start_full_ui():
    try:
        # Set environment variable for model path before importing
        import os
        os.environ['EDIATH_MODEL_PATH'] = get_model_path()
        
        from main import main_interactive
        
        logger.info("🎨 Starting UI + backend...")
        logger.info(f"📁 Using model: {os.environ['EDIATH_MODEL_PATH']}")
        main_interactive()
        
    except Exception as e:
        logger.error(f"💥 UI crashed: {e}", exc_info=True)


# =========================
# MONITOR LOOP (FIXES FREEZE)
# =========================
def monitor_backend(thread):
    logger.info("🖥️ Monitoring backend health...")
    
    while not shutdown_event.is_set():
        if thread is None or not thread.is_alive():
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


def check_dependencies():
    """Check all required dependencies"""
    missing = []
    
    # Core dependencies
    try:
        import llama_cpp
        logger.info("✅ llama-cpp-python: installed")
    except ImportError:
        missing.append("llama-cpp-python")
    
    try:
        import numpy
        logger.info(f"✅ numpy: {numpy.__version__}")
    except ImportError:
        missing.append("numpy")
    
    try:
        import aiohttp
        logger.info(f"✅ aiohttp: {aiohttp.__version__}")
    except ImportError:
        missing.append("aiohttp")
    
    try:
        import tenacity
        logger.info(f"✅ tenacity: {tenacity.__version__}")
    except ImportError:
        missing.append("tenacity")
    
    # Optional dependencies
    try:
        import faiss
        logger.info("✅ faiss: installed")
    except ImportError:
        logger.warning("⚠️ faiss-cpu not installed (optional for vector search)")
    
    try:
        import sklearn
        logger.info(f"✅ scikit-learn: {sklearn.__version__}")
    except ImportError:
        logger.warning("⚠️ scikit-learn not installed (optional for clustering)")
    
    if missing:
        logger.error(f"❌ Missing dependencies: {missing}")
        logger.info(f"   Install with: pip install {' '.join(missing)}")
        return False
    
    return True


def print_banner():
    """Print EDIATH banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   ███████╗██████╗ ██╗ █████╗ ████████╗██╗  ██╗                     ║
║   ██╔════╝██╔══██╗██║██╔══██╗╚══██╔══╝██║  ██║                     ║
║   █████╗  ██║  ██║██║███████║   ██║   ███████║                     ║
║   ██╔══╝  ██║  ██║██║██╔══██║   ██║   ██╔══██║                     ║
║   ███████╗██████╔╝██║██║  ██║   ██║   ██║  ██║                     ║
║   ╚══════╝╚═════╝ ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝                     ║
║                                                                   ║
║         Autonomous AI Framework with Unified Brain                ║
║                     Ultimate Edition v2.0                         ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


# =========================
# MAIN
# =========================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="EDIATH - Autonomous AI Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["ui", "backend"], default="ui",
                       help="Run mode: ui (full interface) or backend (only backend)")
    parser.add_argument("--check-deps", action="store_true",
                       help="Check dependencies and exit")
    parser.add_argument("--skip-model-check", action="store_true",
                       help="Skip model file existence check")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Validate main.py exists
    if not validate_main():
        sys.exit(1)
    
    # Check dependencies if requested
    if args.check_deps:
        logger.info("🔍 Checking dependencies...")
        if check_dependencies():
            print("\n✅ All dependencies satisfied!")
        else:
            print("\n❌ Some dependencies are missing.")
        sys.exit(0)
    
    # Check llama-cpp-python
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
    elif not args.skip_model_check:
        print(f"⚠️  WARNING: Model file not found at {model_path}")
        print("   Please ensure the model file exists or update get_model_path()")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print()
    
    try:
        if args.mode == "backend":
            thread = start_backend()
            if thread:
                monitor_backend(thread)
            else:
                logger.error("Failed to start backend")
                sys.exit(1)
        else:
            start_full_ui()
            
    except KeyboardInterrupt:
        logger.info("\n🛑 User stopped system")
        
    finally:
        shutdown_event.set()
        time.sleep(1)
        logger.info("👋 Shutdown complete")


if __name__ == "__main__":
    main()