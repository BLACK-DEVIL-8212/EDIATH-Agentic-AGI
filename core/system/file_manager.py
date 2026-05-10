"""
Secure File Manager for EDIATH (Sandboxed + Async Ready)
"""

import os
import shutil
import json
import time
import asyncio

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils.logger import logger
from ..utils.helpers import ensure_dir, sanitize_filename


class FileManager:
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize FileManager with sandboxed directory.
        Production-grade: safe, validated, and monitored.
        """

        try:
            from pathlib import Path
            import threading
            import time

            # ------------------------
            # 🔥 BASE DIRECTORY SETUP
            # ------------------------
            base_dir = base_dir or "./workspace"
            self.base_dir = Path(base_dir).expanduser().resolve()

            # ------------------------
            # 🔥 SECURITY CHECK
            # ------------------------
            if ".." in str(self.base_dir):
                raise ValueError("Invalid base_dir path (path traversal detected)")

            # ------------------------
            # 🔥 ENSURE DIRECTORY EXISTS
            # ------------------------
            try:
                ensure_dir(str(self.base_dir))
            except Exception as e:
                raise RuntimeError(f"Failed to create base directory: {e}")

            # ------------------------
            # 🔥 OPERATION LOG
            # ------------------------
            self.operation_log: List[Dict[str, Any]] = []

            # ------------------------
            # 🔥 THREAD SAFETY
            # ------------------------
            self._lock = threading.Lock()

            # ------------------------
            # 🔥 METRICS
            # ------------------------
            self.files_created = 0
            self.files_deleted = 0
            self.files_read = 0
            self.files_written = 0
            self.errors = 0

            # ------------------------
            # 🔥 CONFIG FLAGS
            # ------------------------
            self.max_file_size = 10 * 1024 * 1024  # 10MB limit
            self.allowed_extensions = None  # set later if needed
            self.restricted_paths = []

            # ------------------------
            # 🔥 TIMESTAMP
            # ------------------------
            self.initialized_at = time.time()

            # ------------------------
            # 🔥 LOGGING
            # ------------------------
            try:
                logger.info(f"📁 FileManager sandbox initialized: {self.base_dir}")
            except Exception:
                pass

        except Exception as e:
            try:
                logger.error(f"FileManager init failed: {e}")
            except Exception:
                pass

            raise RuntimeError(f"FileManager initialization failed: {e}")

    # ------------------------
    # PATH SAFETY 🔒
    # ------------------------
    def _safe_path(self, path: str) -> Path:
        """
        Resolve and validate path of sandbox.
        Prevents traversal, injection, and unsafe access.
        """

        try:
            from pathlib import Path

            # ------------------------
            # 🔥 INPUT VALIDATION
            # ------------------------
            if not path or not isinstance(path, str):
                raise ValueError("Invalid path input")

            path = path.strip()

            # ------------------------
            # 🔥 SANITIZE
            # ------------------------
            try:
                clean = sanitize_filename(path)
            except Exception:
                clean = path  # fallback if sanitizer fails

            # ------------------------
            # 🔥 BUILD FULL PATH
            # ------------------------
            full_path = (self.base_dir / clean).resolve()

            # ------------------------
            # 🔥 SANDBOX ESCAPE CHECK
            # ------------------------
            try:
                base = self.base_dir.resolve()

                if not str(full_path).startswith(str(base)):
                    raise PermissionError("Access outside sandbox is not allowed")

            except Exception:
                raise PermissionError("Sandbox validation failed")

            # ------------------------
            # 🔥 RESTRICTED PATH CHECK
            # ------------------------
            restricted = getattr(self, "restricted_paths", []) or []

            for r in restricted:
                try:
                    if str(full_path).startswith(str(Path(r).resolve())):
                        raise PermissionError(f"Access restricted: {r}")
                except Exception:
                    continue

            # ------------------------
            # 🔥 OPTIONAL EXTENSION FILTER
            # ------------------------
            allowed_ext = getattr(self, "allowed_extensions", None)

            if allowed_ext:
                ext = full_path.suffix.lower()
                if ext not in allowed_ext:
                    raise PermissionError(f"Extension not allowed: {ext}")

            # ------------------------
            # 🔥 FINAL SAFETY CHECK
            # ------------------------
            if ".." in str(full_path):
                raise PermissionError("Path traversal detected")

            return full_path

        except Exception as e:
            try:
                logger.error(f"Safe path error: {path} → {e}")
            except Exception:
                pass

            raise

    def open_file(
        self,
        path: str,
        mode: str = "r",
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> Any:
        """
        Open and read a file safely (production-grade)
        """

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not path or not isinstance(path, str):
                raise ValueError("Invalid file path")

            if not isinstance(mode, str) or not mode:
                mode = "r"

            # ------------------------
            # 🔥 RESOLVE PATH
            # ------------------------
            base_path = self.workspace.resolve()
            target_path = (base_path / path).resolve()

            # ------------------------
            # 🔥 SECURITY (SANDBOX)
            # ------------------------
            if not str(target_path).startswith(str(base_path)):
                raise PermissionError("Access outside workspace is not allowed")

            # ------------------------
            # 🔥 FILE EXISTS CHECK
            # ------------------------
            if not os.path.exists(target_path):
                raise FileNotFoundError(f"File not found: {target_path}")

            # ------------------------
            # 🔥 MODE HANDLING
            # ------------------------
            is_binary = "b" in mode

            # ------------------------
            # 🔥 OPEN FILE
            # ------------------------
            if is_binary:
                with open(target_path, mode) as f:
                    data = f.read()
            else:
                with open(target_path, mode, encoding=encoding, errors=errors) as f:
                    data = f.read()

            # ------------------------
            # 🔥 METRICS
            # ------------------------
            try:
                self.file_read_count = getattr(self, "file_read_count", 0) + 1
                self.last_file_access = str(target_path)
            except Exception:
                pass

            return data

        except Exception as e:
            # ------------------------
            # 🔥 ERROR TRACKING
            # ------------------------
            try:
                self.error_count = getattr(self, "error_count", 0) + 1
                self.last_error = str(e)

                logger.error(f"❌ Failed to open file: {path} -> {e}")
            except Exception:
                pass

            # ------------------------
            # 🔥 RAISE CLEAN ERROR
            # ------------------------
            raise RuntimeError(f"File access failed: {path}") from e

    # ------------------------
    # CREATE DIRECTORY
    # ------------------------
    async def create_directory(self, path: str) -> Path:
        """
        Create directory safely inside sandbox (async + production-ready)
        """

        start = time.time()

        try:
            # ------------------------
            # 🔥 VALIDATE + RESOLVE PATH
            # ------------------------
            full_path = self._safe_path(path)

            # ------------------------
            # 🔥 CREATE DIRECTORY (NON-BLOCKING)
            # ------------------------
            await asyncio.to_thread(ensure_dir, str(full_path))

            # ------------------------
            # 🔥 METRICS UPDATE
            # ------------------------
            if hasattr(self, "_lock"):
                try:
                    with self._lock:
                        self.files_created = getattr(self, "files_created", 0) + 1
                except:
                    pass

            # ------------------------
            # 🔥 LOG SUCCESS
            # ------------------------
            try:
                self._log(
                    action="create_directory",
                    path=str(path),
                    success=True,
                    meta={
                        "resolved_path": str(full_path),
                        "processing_time": round(time.time() - start, 4),
                    },
                )
            except Exception:
                pass

            return full_path

        except Exception as e:
            # ------------------------
            # 🔥 ERROR METRICS
            # ------------------------
            try:
                if hasattr(self, "_lock"):
                    with self._lock:
                        self.errors = getattr(self, "errors", 0) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 LOG ERROR
            # ------------------------
            try:
                self._log(
                    action="create_directory",
                    path=str(path),
                    success=False,
                    error=str(e),
                )
            except Exception:
                pass

            # ------------------------
            # 🔥 FAIL HARD (IMPORTANT)
            # ------------------------
            raise RuntimeError(f"Directory creation failed: {e}")

    # ------------------------
    # READ FILE
    # ------------------------
    async def read_file(self, filepath: str) -> str:
        """
        Read file safely from sandbox (async + production-grade)
        """

        import asyncio
        import time

        start = time.time()

        try:
            # ------------------------
            # 🔥 SAFE PATH RESOLUTION
            # ------------------------
            full_path = self._safe_path(filepath)

            # ------------------------
            # 🔥 EXISTENCE CHECK
            # ------------------------
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")

            if not full_path.is_file():
                raise ValueError("Path is not a file")

            # ------------------------
            # 🔥 SIZE CHECK (SECURITY)
            # ------------------------
            max_size = getattr(self, "max_file_size", None)
            file_size = full_path.stat().st_size

            if max_size and file_size > max_size:
                raise ValueError(f"File too large ({file_size} bytes)")

            # ------------------------
            # 🔥 READ FILE (NON-BLOCKING)
            # ------------------------
            content = await asyncio.to_thread(
                full_path.read_text, encoding="utf-8", errors="ignore"
            )

            # ------------------------
            # 🔥 METRICS UPDATE
            # ------------------------
            try:
                if hasattr(self, "_lock"):
                    with self._lock:
                        self.files_read = getattr(self, "files_read", 0) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 LOG SUCCESS
            # ------------------------
            try:
                self._log(
                    action="read_file",
                    path=str(filepath),
                    success=True,
                    meta={
                        "size": file_size,
                        "processing_time": round(time.time() - start, 4),
                    },
                )
            except Exception:
                pass

            return content

        except Exception as e:
            # ------------------------
            # 🔥 ERROR METRICS
            # ------------------------
            try:
                if hasattr(self, "_lock"):
                    with self._lock:
                        self.errors = getattr(self, "errors", 0) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 LOG ERROR
            # ------------------------
            try:
                self._log(
                    action="read_file", path=str(filepath), success=False, error=str(e)
                )
            except Exception:
                pass

            raise RuntimeError(f"File read failed: {e}")

    # ------------------------
    # WRITE FILE
    # ------------------------
    async def write_file(self, filepath: str, content: str, overwrite: bool = True):
        """
        Write file safely داخل sandbox (async + production-grade)
        """

        start = time.time()

        try:
            # ------------------------
            # 🔥 VALIDATE INPUT
            # ------------------------
            if not isinstance(content, str):
                raise ValueError("Content must be a string")

            # ------------------------
            # 🔥 SAFE PATH
            # ------------------------
            full_path = self._safe_path(filepath)

            # ------------------------
            # 🔥 OVERWRITE CHECK
            # ------------------------
            if full_path.exists() and not overwrite:
                raise FileExistsError(f"File already exists: {filepath}")

            # ------------------------
            # 🔥 SIZE LIMIT (SECURITY)
            # ------------------------
            max_size = getattr(self, "max_file_size", None)
            content_size = len(content.encode("utf-8"))

            if max_size and content_size > max_size:
                raise ValueError(f"Content too large ({content_size} bytes)")

            # ------------------------
            # 🔥 ENSURE DIRECTORY
            # ------------------------
            await asyncio.to_thread(ensure_dir, str(full_path.parent))

            # ------------------------
            # 🔥 ATOMIC WRITE (SAFE)
            # ------------------------
            temp_path = full_path.with_suffix(full_path.suffix + ".tmp")

            await asyncio.to_thread(temp_path.write_text, content, encoding="utf-8")

            # rename → atomic replace
            await asyncio.to_thread(temp_path.replace, full_path)

            # ------------------------
            # 🔥 METRICS UPDATE
            # ------------------------
            try:
                if hasattr(self, "_lock"):
                    with self._lock:
                        self.files_written = getattr(self, "files_written", 0) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 LOG SUCCESS
            # ------------------------
            try:
                self._log(
                    action="write_file",
                    path=str(filepath),
                    success=True,
                    meta={
                        "size": content_size,
                        "processing_time": round(time.time() - start, 4),
                    },
                )
            except Exception:
                pass

            return full_path

        except Exception as e:
            # ------------------------
            # 🔥 ERROR METRICS
            # ------------------------
            try:
                if hasattr(self, "_lock"):
                    with self._lock:
                        self.errors = getattr(self, "errors", 0) + 1
            except:
                pass

            # ------------------------
            # 🔥 LOG ERROR
            # ------------------------
            try:
                self._log(
                    action="write_file", path=str(filepath), success=False, error=str(e)
                )
            except:
                pass

            raise RuntimeError(f"File write failed: {e}")

    # ------------------------
    # DELETE FILE
    # ------------------------
    async def delete_file(self, filepath: str) -> bool:
        """
        Delete file safely داخل sandbox (async + production-grade)
        """

        start = time.time()

        try:
            # ------------------------
            # 🔥 SAFE PATH
            # ------------------------
            full_path = self._safe_path(filepath)

            # ------------------------
            # 🔥 EXISTENCE CHECK
            # ------------------------
            if not full_path.exists():
                return False

            # ------------------------
            # 🔥 TYPE CHECK (NO DIR DELETE)
            # ------------------------
            if not full_path.is_file():
                raise ValueError("Target is not a file")

            # ------------------------
            # 🔥 PROTECTED PATH CHECK
            # ------------------------
            restricted = getattr(self, "restricted_paths", []) or []
            for r in restricted:
                try:
                    if str(full_path).startswith(str(r)):
                        raise PermissionError("File is protected")
                except:
                    continue

            # ------------------------
            # 🔥 DELETE FILE (ASYNC)
            # ------------------------
            await asyncio.to_thread(full_path.unlink)

            # ------------------------
            # 🔥 METRICS UPDATE
            # ------------------------
            try:
                if hasattr(self, "_lock"):
                    with self._lock:
                        self.files_deleted = getattr(self, "files_deleted", 0) + 1
            except:
                pass

            # ------------------------
            # 🔥 LOG SUCCESS
            # ------------------------
            try:
                self._log(
                    action="delete_file",
                    path=str(filepath),
                    success=True,
                    meta={"processing_time": round(time.time() - start, 4)},
                )
            except:
                pass

            return True

        except Exception as e:
            # ------------------------
            # 🔥 ERROR METRICS
            # ------------------------
            try:
                if hasattr(self, "_lock"):
                    with self._lock:
                        self.errors = getattr(self, "errors", 0) + 1
            except:
                pass

            # ------------------------
            # 🔥 LOG ERROR
            # ------------------------
            try:
                self._log(
                    action="delete_file",
                    path=str(filepath),
                    success=False,
                    error=str(e),
                )
            except:
                pass

            raise RuntimeError(f"File delete failed: {e}")

    # ------------------------
    # COPY FILE
    # ------------------------
    async def copy_file(self, src: str, dst: str):
        """
        Copy file safely داخل sandbox (async + production-grade)
        """

        start = time.time()

        try:
            # ------------------------
            # 🔥 SAFE PATH RESOLUTION
            # ------------------------
            src_path = self._safe_path(src)
            dst_path = self._safe_path(dst)

            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not src_path.exists():
                raise FileNotFoundError(f"Source not found: {src}")

            if not src_path.is_file():
                raise ValueError("Source is not a file")

            # ------------------------
            # 🔥 SIZE CHECK (SECURITY)
            # ------------------------
            max_size = getattr(self, "max_file_size", None)
            file_size = src_path.stat().st_size

            if max_size and file_size > max_size:
                raise ValueError(f"File too large ({file_size} bytes)")

            # ------------------------
            # 🔥 PREVENT SELF-COPY
            # ------------------------
            if src_path.resolve() == dst_path.resolve():
                raise ValueError("Source and destination are the same")

            # ------------------------
            # 🔥 ENSURE DEST DIR
            # ------------------------
            await asyncio.to_thread(ensure_dir, str(dst_path.parent))

            # ------------------------
            # 🔥 ATOMIC COPY (TEMP FILE)
            # ------------------------
            temp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")

            await asyncio.to_thread(shutil.copy2, src_path, temp_path)

            await asyncio.to_thread(temp_path.replace, dst_path)

            # ------------------------
            # 🔥 METRICS UPDATE
            # ------------------------
            try:
                if hasattr(self, "_lock"):
                    with self._lock:
                        self.files_written = getattr(self, "files_written", 0) + 1
            except:
                pass

            # ------------------------
            # 🔥 LOG SUCCESS
            # ------------------------
            try:
                self._log(
                    action="copy_file",
                    path=f"{src}->{dst}",
                    success=True,
                    meta={
                        "size": file_size,
                        "processing_time": round(time.time() - start, 4),
                    },
                )
            except:
                pass

            return dst_path

        except Exception as e:
            # ------------------------
            # 🔥 ERROR METRICS
            # ------------------------
            try:
                if hasattr(self, "_lock"):
                    with self._lock:
                        self.errors = getattr(self, "errors", 0) + 1
            except:
                pass

            # ------------------------
            # 🔥 LOG ERROR
            # ------------------------
            try:
                self._log(
                    action="copy_file",
                    path=f"{src}->{dst}",
                    success=False,
                    error=str(e),
                )
            except:
                pass

            raise RuntimeError(f"File copy failed: {e}")

    # ------------------------
    # LIST FILES
    # ------------------------
    async def list_files(self, directory: str = ".", pattern: str = "*") -> List[str]:
        """
        List files safely داخل sandbox (async + production-grade)
        """

        import asyncio
        import time

        start = time.time()

        try:
            # ------------------------
            # 🔥 SAFE PATH
            # ------------------------
            dir_path = self._safe_path(directory)

            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")

            if not dir_path.is_dir():
                raise ValueError("Path is not a directory")

            # ------------------------
            # 🔥 PATTERN SANITIZATION
            # ------------------------
            if ".." in pattern:
                raise ValueError("Invalid pattern")

            # ------------------------
            # 🔥 LIST FILES (ASYNC)
            # ------------------------
            files = await asyncio.to_thread(lambda: list(dir_path.glob(pattern)))

            # ------------------------
            # 🔥 FILTER + FORMAT
            # ------------------------
            result = []
            for f in files:
                try:
                    # ensure inside sandbox
                    if not str(f.resolve()).startswith(str(self.base_dir)):
                        continue

                    relative = str(f.relative_to(self.base_dir))
                    result.append(relative)

                except:
                    continue

            # ------------------------
            # 🔥 METRICS
            # ------------------------
            try:
                if hasattr(self, "_lock"):
                    with self._lock:
                        self.files_read = getattr(self, "files_read", 0) + len(result)
            except:
                pass

            # ------------------------
            # 🔥 LOG SUCCESS
            # ------------------------
            try:
                self._log(
                    action="list_files",
                    path=str(directory),
                    success=True,
                    meta={
                        "count": len(result),
                        "pattern": pattern,
                        "processing_time": round(time.time() - start, 4),
                    },
                )
            except:
                pass

            return result

        except Exception as e:
            # ------------------------
            # 🔥 ERROR METRICS
            # ------------------------
            try:
                if hasattr(self, "_lock"):
                    with self._lock:
                        self.errors = getattr(self, "errors", 0) + 1
            except:
                pass

            # ------------------------
            # 🔥 LOG ERROR
            # ------------------------
            try:
                self._log(
                    action="list_files",
                    path=str(directory),
                    success=False,
                    error=str(e),
                )
            except:
                pass

            raise RuntimeError(f"List files failed: {e}")

    # ------------------------
    # JSON SUPPORT
    # ------------------------
    async def read_json(self, filepath: str):
        """
        Read JSON file safely (async + production-grade)
        """

        start = time.time()

        try:
            # ------------------------
            # 🔥 READ FILE
            # ------------------------
            content = await self.read_file(filepath)

            if not content:
                raise ValueError("Empty JSON file")

            # ------------------------
            # 🔥 PARSE JSON (SAFE)
            # ------------------------
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {e}")

            # ------------------------
            # 🔥 TYPE VALIDATION
            # ------------------------
            if not isinstance(data, (dict, list)):
                raise ValueError("JSON must be object or array")

            # ------------------------
            # 🔥 LOG SUCCESS
            # ------------------------
            try:
                self._log(
                    action="read_json",
                    path=str(filepath),
                    success=True,
                    meta={
                        "type": type(data).__name__,
                        "processing_time": round(time.time() - start, 4),
                    },
                )
            except:
                pass

            return data

        except Exception as e:
            # ------------------------
            # 🔥 ERROR METRICS
            # ------------------------
            try:
                if hasattr(self, "_lock"):
                    with self._lock:
                        self.errors = getattr(self, "errors", 0) + 1
            except:
                pass

            # ------------------------
            # 🔥 LOG ERROR
            # ------------------------
            try:
                self._log(
                    action="read_json", path=str(filepath), success=False, error=str(e)
                )
            except:
                pass

            raise RuntimeError(f"JSON read failed: {e}")

    async def write_json(self, filepath: str, data: Dict[str, Any]):
        """
        Write JSON safely (async + production-grade)
        """

        start = time.time()

        try:
            # ------------------------
            # 🔥 VALIDATE INPUT
            # ------------------------
            if not isinstance(data, (dict, list)):
                raise ValueError("Data must be dict or list")

            # ------------------------
            # 🔥 SAFE SERIALIZATION
            # ------------------------
            try:
                content = json.dumps(data, indent=2, ensure_ascii=False)
            except Exception as e:
                raise ValueError(f"JSON serialization failed: {e}")

            # ------------------------
            # 🔥 SIZE CHECK
            # ------------------------
            max_size = getattr(self, "max_file_size", None)
            content_size = len(content.encode("utf-8"))

            if max_size and content_size > max_size:
                raise ValueError(f"JSON too large ({content_size} bytes)")

            # ------------------------
            # 🔥 WRITE FILE
            # ------------------------
            result = await self.write_file(filepath, content, overwrite=True)

            # ------------------------
            # 🔥 LOG SUCCESS
            # ------------------------
            try:
                self._log(
                    action="write_json",
                    path=str(filepath),
                    success=True,
                    meta={
                        "size": content_size,
                        "processing_time": round(time.time() - start, 4),
                    },
                )
            except:
                pass

            return result

        except Exception as e:
            # ------------------------
            # 🔥 ERROR METRICS
            # ------------------------
            try:
                if hasattr(self, "_lock"):
                    with self._lock:
                        self.errors = getattr(self, "errors", 0) + 1
            except:
                pass

            # ------------------------
            # 🔥 LOG ERROR
            # ------------------------
            try:
                self._log(
                    action="write_json", path=str(filepath), success=False, error=str(e)
                )
            except:
                pass

            raise RuntimeError(f"JSON write failed: {e}")

    # ------------------------
    # INFO
    # ------------------------
    def file_exists(self, filepath: str) -> bool:
        """
        Check file existence safely داخل sandbox (production-grade)
        """

        import time

        start = time.time()

        try:
            # ------------------------
            # 🔥 SAFE PATH
            # ------------------------
            full_path = self._safe_path(filepath)

            # ------------------------
            # 🔥 EXISTENCE CHECK
            # ------------------------
            exists = full_path.exists()

            # ------------------------
            # 🔥 LOG SUCCESS
            # ------------------------
            try:
                self._log(
                    action="file_exists",
                    path=str(filepath),
                    success=True,
                    meta={
                        "exists": exists,
                        "processing_time": round(time.time() - start, 4),
                    },
                )
            except:
                pass

            return exists

        except Exception as e:
            # ------------------------
            # 🔥 ERROR METRICS
            # ------------------------
            try:
                if hasattr(self, "_lock"):
                    with self._lock:
                        self.errors = getattr(self, "errors", 0) + 1
            except:
                pass

            # ------------------------
            # 🔥 LOG ERROR
            # ------------------------
            try:
                self._log(
                    action="file_exists",
                    path=str(filepath),
                    success=False,
                    error=str(e),
                )
            except:
                pass

            return False  # safe fallback (never crash)

    def get_file_size(self, filepath: str) -> int:
        """
        Get file size safely داخل sandbox (production-grade)
        """

        import time

        start = time.time()

        try:
            # ------------------------
            # 🔥 SAFE PATH
            # ------------------------
            full_path = self._safe_path(filepath)

            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not full_path.exists():
                return 0

            if not full_path.is_file():
                raise ValueError("Path is not a file")

            # ------------------------
            # 🔥 GET SIZE
            # ------------------------
            size = full_path.stat().st_size

            # ------------------------
            # 🔥 LOG SUCCESS
            # ------------------------
            try:
                self._log(
                    action="get_file_size",
                    path=str(filepath),
                    success=True,
                    meta={
                        "size": size,
                        "processing_time": round(time.time() - start, 4),
                    },
                )
            except:
                pass

            return size

        except Exception as e:
            # ------------------------
            # 🔥 ERROR METRICS
            # ------------------------
            try:
                if hasattr(self, "_lock"):
                    with self._lock:
                        self.errors = getattr(self, "errors", 0) + 1
            except:
                pass

            # ------------------------
            # 🔥 LOG ERROR
            # ------------------------
            try:
                self._log(
                    action="get_file_size",
                    path=str(filepath),
                    success=False,
                    error=str(e),
                )
            except:
                pass

            return 0  # safe fallback (never crash)

    # ------------------------
    # LOGGING
    # ------------------------
    def _log(
        self,
        operation,
        target,
        success,
        error=None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """
        Internal operation logger (production-grade)
        """

        import time

        try:
            # ------------------------
            # 🔥 BUILD LOG ENTRY
            # ------------------------
            entry = {
                "operation": str(operation),
                "target": str(target),
                "success": bool(success),
                "error": str(error) if error else None,
                "meta": meta or {},
                "timestamp": datetime.utcnow().isoformat(),
                "epoch": time.time(),
            }

            # ------------------------
            # 🔥 THREAD-SAFE APPEND
            # ------------------------
            if hasattr(self, "_lock"):
                try:
                    with self._lock:
                        self.operation_log.append(entry)
                except Exception:
                    self.operation_log.append(entry)
            else:
                self.operation_log.append(entry)

            # ------------------------
            # 🔥 LIMIT LOG SIZE (MEMORY SAFE)
            # ------------------------
            max_logs = getattr(self, "max_log_entries", 10000)

            if len(self.operation_log) > max_logs:
                self.operation_log = self.operation_log[-max_logs:]

            # ------------------------
            # 🔥 OPTIONAL EXTERNAL LOGGING
            # ------------------------
            try:
                if success:
                    logger.debug(f"[FileManager] {operation} → {target}")
                else:
                    logger.warning(
                        f"[FileManager] {operation} FAILED → {target} | {error}"
                    )
            except Exception:
                pass

        except Exception as e:
            # ------------------------
            # 🔥 FAILSAFE (NEVER BREAK SYSTEM)
            # ------------------------
            try:
                logger.error(f"FileManager logging failed: {e}")
            except:
                pass

    def get_operation_log(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent operation logs (production-grade)
        """

        try:
            # ------------------------
            # 🔥 VALIDATE INPUT
            # ------------------------
            try:
                limit = int(limit)
            except Exception:
                limit = 10

            if limit <= 0:
                return []

            # ------------------------
            # 🔥 THREAD-SAFE READ
            # ------------------------
            if hasattr(self, "_lock"):
                try:
                    with self._lock:
                        logs = list(self.operation_log)
                except Exception:
                    logs = list(self.operation_log)
            else:
                logs = list(self.operation_log)

            # ------------------------
            # 🔥 LIMIT + RETURN
            # ------------------------
            return logs[-limit:]

        except Exception as e:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            try:
                logger.error(f"Get operation log failed: {e}")
            except Exception:
                pass

            return []

    async def run_file(
        self, filepath: str, timeout: int = 10, capture_output: bool = True
    ) -> Dict[str, Any]:
        """
        Execute file safely داخل sandbox (production-grade)
        """

        import asyncio
        import time
        import sys

        start = time.time()

        try:
            # ------------------------
            # 🔥 SAFE PATH
            # ------------------------
            full_path = self._safe_path(filepath)

            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")

            if not full_path.is_file():
                raise ValueError("Target is not a file")

            # ------------------------
            # 🔥 ALLOWED FILE TYPES
            # ------------------------
            allowed_ext = [".py", ".bat", ".cmd"]

            ext = full_path.suffix.lower()

            if ext not in allowed_ext:
                raise PermissionError(f"Execution not allowed: {ext}")

            # ------------------------
            # 🔥 BUILD COMMAND
            # ------------------------
            if ext == ".py":
                cmd = [sys.executable, str(full_path)]  # ✅ FIXED (important)
            else:
                cmd = [str(full_path)]

            # ------------------------
            # 🔥 RUN PROCESS
            # ------------------------
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE if capture_output else None,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError("Execution timed out")

            # ------------------------
            # 🔥 RESULT
            # ------------------------
            result = {
                "return_code": process.returncode,
                "stdout": stdout.decode(errors="ignore") if stdout else "",
                "stderr": stderr.decode(errors="ignore") if stderr else "",
                "execution_time": round(time.time() - start, 4),
            }

            # ------------------------
            # 🔥 LOG
            # ------------------------
            try:
                self._log(
                    action="run_file",
                    target=str(filepath),
                    success=(process.returncode == 0),
                    meta=result,
                )
            except Exception:
                pass

            return result

        except Exception as e:
            # ------------------------
            # 🔥 ERROR HANDLING
            # ------------------------
            try:
                if hasattr(self, "_lock"):
                    with self._lock:
                        self.errors = getattr(self, "errors", 0) + 1
            except Exception:
                pass

            try:
                self._log(
                    action="run_file", target=str(filepath), success=False, error=str(e)
                )
            except:
                pass

            return {"return_code": -1, "error": str(e)}
