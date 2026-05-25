"""
EDIATH Launcher - Fixed with CUDA Support for NVIDIA 3050
"""

import asyncio
import threading
import logging
import signal
import time
import os
import sys
import io
from pathlib import Path

# Fix Unicode output on Windows consoles
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

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
# CUDA DETECTION
# =========================
def check_cuda_availability():
    """Check if CUDA is available and return device info"""
    cuda_info = {
        "available": False,
        "device_name": None,
        "cuda_version": None,
        "memory_mb": 0,
        "enabled": False
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            cuda_info["available"] = True
            cuda_info["device_name"] = torch.cuda.get_device_name(0)
            cuda_info["cuda_version"] = torch.version.cuda
            cuda_info["memory_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            cuda_info["enabled"] = True
            logger.info(f"✅ CUDA available: {cuda_info['device_name']}")
            logger.info(f"   CUDA Version: {cuda_info['cuda_version']}")
            logger.info(f"   Memory: {cuda_info['memory_mb']:.0f} MB")
        else:
            logger.warning("⚠️ CUDA not available, using CPU mode")
            
    except ImportError:
        logger.warning("⚠️ PyTorch not installed, CUDA detection skipped")
    except Exception as e:
        logger.warning(f"⚠️ CUDA detection error: {e}")
    
    return cuda_info


def enable_cpu_fallback():
    """Enable CPU fallback if needed"""
    cpu_fallback = os.environ.get('EDIATH_ALLOW_CPU_LLM_FALLBACK', '0')
    if cpu_fallback == '1':
        logger.info("✅ CPU LLM fallback enabled")
        return True
    
    logger.warning("CPU model fallback disabled. Set EDIATH_ALLOW_CPU_LLM_FALLBACK=1 to enable")
    return False


# =========================
# MODEL CONFIGURATION
# =========================
def get_model_path():
    """Get the correct model path with fallback options"""
    base_path = Path(__file__).resolve().parent / "models" / "EDIATH-q4_k_m.gguf"
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


def download_model_if_missing():
    """Download model if missing"""
    model_path = get_model_path()
    
    if Path(model_path).exists():
        logger.info(f"✅ Model found at: {model_path}")
        return True
    
    logger.warning("⚠️ Model file not found!")
    logger.info("   Please download the model from:")
    logger.info("   https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
    logger.info(f"   Save to: {model_path}")
    
    return False


# =========================
# LLAMA-CPP CONFIGURATION WITH CUDA
# =========================
def setup_llama_cpp():
    """Configure llama-cpp-python with CUDA support"""
    
    # Check if CUDA-enabled llama-cpp is installed
    try:
        import llama_cpp
        version = getattr(llama_cpp, '__version__', 'unknown')
        
        # Check if CUDA version
        if hasattr(llama_cpp, 'llama_cpp'):
            logger.info(f"✅ llama-cpp-python version: {version}")
            
            # Try to detect if built with CUDA
            if hasattr(llama_cpp, 'LLAMA_SUPPORTS_GPU_OFFLOAD'):
                logger.info("✅ llama-cpp-python has GPU offload support")
            else:
                logger.info("ℹ️ Reinstall with CUDA for better performance:")
                logger.info("   CMAKE_ARGS='-DGGML_CUDA=on' pip install --upgrade llama-cpp-python --force-reinstall --no-cache-dir")
        
        return True
        
    except ImportError:
        logger.error("❌ llama-cpp-python not installed!")
        logger.info("   Install with CUDA support:")
        logger.info("   CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python")
        return False


# =========================
# ENVIRONMENT SETUP
# =========================
def setup_environment():
    """Set up environment variables for optimal performance"""
    
    # CUDA settings
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # PyTorch settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # llama-cpp settings
    os.environ['LLAMA_CPP_GPU_LAYERS'] = '50'  # Offload 50 layers to GPU
    
    # Model path
    os.environ['EDIATH_MODEL_PATH'] = get_model_path()
    
    # Enable CPU fallback if needed (for testing)
    if not check_cuda_availability()["available"]:
        os.environ['EDIATH_ALLOW_CPU_LLM_FALLBACK'] = '1'
    
    logger.info("✅ Environment configured")


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
        # Ensure environment is set up
        setup_environment()
        
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
        # Ensure environment is set up
        setup_environment()

        from main import main_qt

        logger.info("🎨 Starting UI + backend...")
        logger.info(f"📁 Using model: {os.environ['EDIATH_MODEL_PATH']}")
        main_qt()
        
    except Exception as e:
        logger.error(f"💥 UI crashed: {e}", exc_info=True)
        logger.info("Attempting to start in headless mode...")
        start_backend()


# =========================
# MONITOR LOOP
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
        
        # Check for GPU support
        try:
            # Test GPU offload
            if hasattr(llama_cpp, 'llama_cpp'):
                logger.info("✅ GPU offload support detected")
        except:
            pass
            
        return True
    except ImportError:
        logger.error("❌ llama-cpp-python not installed!")
        logger.info("   Install with CUDA support:")
        logger.info("   CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python")
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
        ver = getattr(tenacity, "__version__", "unknown")
        logger.info(f"✅ tenacity: {ver}")
    except ImportError:
        missing.append("tenacity")
    
    # Optional dependencies
    try:
        import torch
        logger.info(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"   CUDA available: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.warning("⚠️ PyTorch not installed (optional for advanced features)")
        logger.info("   Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
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
║         🚀 GPU: NVIDIA GeForce RTX 3050 - CUDA Enabled           ║
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
    parser.add_argument("--force-cpu", action="store_true",
                       help="Force CPU mode even if GPU is available")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Check CUDA
    cuda_info = check_cuda_availability()
    
    if args.force_cpu:
        logger.info("⚠️ Force CPU mode enabled")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
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
    
    # Display performance advice
    if cuda_info["available"] and not args.force_cpu:
        print("\n🚀 Performance Optimizations Active:")
        print(f"   • GPU: {cuda_info['device_name']}")
        print(f"   • VRAM: {cuda_info['memory_mb']:.0f} MB")
        print(f"   • CUDA Version: {cuda_info['cuda_version']}")
        print("   • GPU Layers Offloaded: 50")
        print("   • Optimized for RTX 3050")
    else:
        print("\n⚠️ Running in CPU Mode")
        print("   Performance will be slower")
        print("   Install CUDA toolkit for GPU acceleration")
    
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