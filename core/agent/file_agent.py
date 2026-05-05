"""
File Agent for EDIATH
Handles file operations: read, write, edit, search, organize, and manage files
"""

import os
import shutil
import json
import yaml
import csv
import re
import hashlib
import zipfile
import tarfile
import tempfile
import platform
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
import logging
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed


class FileOperation(Enum):
    """File operation types"""

    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    DELETE = "delete"
    COPY = "copy"
    MOVE = "move"
    RENAME = "rename"
    SEARCH = "search"
    COMPRESS = "compress"
    EXTRACT = "extract"


class FileType(Enum):
    """Supported file types"""

    TEXT = "text"
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"
    PYTHON = "python"
    MARKDOWN = "markdown"
    HTML = "html"
    XML = "xml"
    BINARY = "binary"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    LOG = "log"
    CONFIG = "config"
    SCRIPT = "script"


@dataclass
class FileInfo:
    """File information container"""

    name: str
    path: str
    size: int
    created: datetime
    modified: datetime
    accessed: datetime
    is_directory: bool
    extension: str
    file_type: FileType
    hash: Optional[str] = None


class FileAgent:
    """
    Advanced file management agent capable of:
    - Reading files in various formats (text, JSON, YAML, CSV, binary)
    - Writing and creating files
    - Editing file content (line-based, pattern-based, append/prepend)
    - Searching within files (regex, plain text)
    - File organization and batch operations
    - File compression and extraction
    - Safe operations with backup and rollback
    - File monitoring and watching
    - Permission management
    - Desktop operations (create files on desktop, etc.)
    """

    def __init__(
        self, workspace_root: Optional[str] = None, config: Optional[Dict] = None
    ):
        """
        Initialize File Agent

        Args:
            workspace_root: Root directory for file operations (security boundary)
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Get desktop path for cross-platform support
        self.desktop_path = self._get_desktop_path()

        # Set workspace root (security boundary)
        if workspace_root:
            self.workspace_root = Path(workspace_root).resolve()
        else:
            self.workspace_root = Path.cwd() / "workspace"

        # Create workspace if it doesn't exist
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.max_file_size = self.config.get(
            "max_file_size", 100 * 1024 * 1024
        )  # 100MB
        self.create_backups = self.config.get("create_backups", True)
        self.backup_dir = self.workspace_root / ".backups"
        if self.create_backups:
            self.backup_dir.mkdir(exist_ok=True)

        self.encoding = self.config.get("encoding", "utf-8")
        self.thread_pool_size = self.config.get("thread_pool_size", 4)
        self.allow_desktop_access = self.config.get("allow_desktop_access", True)
        self.allow_system_access = self.config.get("allow_system_access", False)

        # Supported extensions mapping
        self.extension_map = {
            ".txt": FileType.TEXT,
            ".json": FileType.JSON,
            ".yaml": FileType.YAML,
            ".yml": FileType.YAML,
            ".csv": FileType.CSV,
            ".py": FileType.PYTHON,
            ".md": FileType.MARKDOWN,
            ".html": FileType.HTML,
            ".htm": FileType.HTML,
            ".xml": FileType.XML,
            ".log": FileType.LOG,
            ".ini": FileType.CONFIG,
            ".cfg": FileType.CONFIG,
            ".conf": FileType.CONFIG,
            ".sh": FileType.SCRIPT,
            ".bat": FileType.SCRIPT,
            ".ps1": FileType.SCRIPT,
            ".jpg": FileType.IMAGE,
            ".jpeg": FileType.IMAGE,
            ".png": FileType.IMAGE,
            ".gif": FileType.IMAGE,
            ".bmp": FileType.IMAGE,
            ".mp3": FileType.AUDIO,
            ".wav": FileType.AUDIO,
            ".mp4": FileType.VIDEO,
            ".avi": FileType.VIDEO,
        }

        # Operation statistics
        self.stats = {
            "operations": 0,
            "bytes_read": 0,
            "bytes_written": 0,
            "files_created": 0,
            "files_deleted": 0,
            "errors": 0,
        }

        self.logger.info(
            f"File Agent initialized with workspace: {self.workspace_root}"
        )
        self.logger.info(f"Desktop path: {self.desktop_path}")

    def _get_desktop_path(self) -> Path:
        """Get desktop path for current platform"""
        system = platform.system()

        if system == "Windows":
            desktop = Path.home() / "Desktop"
        elif system == "Darwin":  # macOS
            desktop = Path.home() / "Desktop"
        else:  # Linux
            desktop = Path.home() / "Desktop"

        # If Desktop doesn't exist, try alternative paths
        if not desktop.exists():
            if system == "Windows":
                # Try OneDrive Desktop
                desktop = Path.home() / "OneDrive" / "Desktop"
            elif system == "Linux":
                # Try common Linux desktop paths
                for path in [
                    Path.home() / "Desktop",
                    Path.home() / "Escritorio",
                    Path.home() / "桌面",
                    Path.home() / "桌面",
                ]:
                    if path.exists():
                        desktop = path
                        break

        return desktop

    def _resolve_path(self, path: str, create_dirs: bool = False) -> Path:
        """
        Resolve path and ensure it's within allowed boundaries (security)

        Args:
            path: File or directory path
            create_dirs: Whether to create parent directories

        Returns:
            Resolved Path object
        """
        try:
            # Convert to absolute path
            target_path = Path(path)

            # Handle desktop paths
            if self.allow_desktop_access and (
                "desktop" in str(path).lower() or "Desktop" in str(path)
            ):
                # Allow desktop access
                if not target_path.is_absolute():
                    # Check if it's a relative path from desktop
                    if str(target_path).startswith("Desktop") or str(
                        target_path
                    ).startswith("desktop"):
                        target_path = self.desktop_path / str(target_path).replace(
                            "Desktop", ""
                        ).replace("desktop", "").lstrip("/\\")
                    else:
                        target_path = self.desktop_path / target_path
            elif not target_path.is_absolute():
                target_path = self.workspace_root / target_path

            # Resolve to absolute path
            target_path = target_path.resolve()

            # Check if within allowed boundaries
            allowed = False

            # Check workspace
            if (
                self.workspace_root in target_path.parents
                or target_path == self.workspace_root
            ):
                allowed = True

            # Check desktop if allowed
            if self.allow_desktop_access and self.desktop_path in target_path.parents:
                allowed = True

            # Check system if allowed
            if self.allow_system_access:
                allowed = True

            if not allowed:
                raise PermissionError(f"Path {path} is outside allowed boundaries")

            # Create parent directories if requested
            if create_dirs:
                target_path.parent.mkdir(parents=True, exist_ok=True)

            return target_path

        except Exception as e:
            self.logger.error(f"Path resolution error: {str(e)}")
            raise

    def _create_backup(self, file_path: Path) -> Optional[Path]:
        """Create backup of a file before modification"""
        if not self.create_backups:
            return None

        if not file_path.exists():
            return None

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            backup_name = f"{file_path.stem}.{timestamp}{file_path.suffix}.bak"
            backup_path = self.backup_dir / backup_name
            shutil.copy2(file_path, backup_path)
            self.logger.debug(f"Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.warning(f"Failed to create backup for {file_path}: {e}")
            return None

    def _get_file_type(self, file_path: Path) -> FileType:
        """Determine file type based on extension"""
        extension = file_path.suffix.lower()
        return self.extension_map.get(extension, FileType.BINARY)

    def create_file_on_desktop(
        self, filename: str, content: str = "", extension: str = ".txt"
    ) -> Dict[str, Any]:
        """
        Create a file directly on the desktop

        Args:
            filename: Name of the file (without extension)
            content: File content
            extension: File extension (default: .txt)

        Returns:
            Dictionary with creation result
        """
        try:
            if not self.allow_desktop_access:
                return {
                    "success": False,
                    "error": "Desktop access is disabled in configuration",
                }

            # Ensure filename has proper extension
            if not filename.endswith(extension):
                filename = filename + extension

            desktop_file = self.desktop_path / filename

            # Write the file
            with open(desktop_file, "w", encoding=self.encoding) as f:
                f.write(content)

            self.stats["operations"] += 1
            self.stats["files_created"] += 1
            self.stats["bytes_written"] += len(content.encode(self.encoding))

            self.logger.info(f"File created on desktop: {desktop_file}")

            return {
                "success": True,
                "path": str(desktop_file),
                "filename": filename,
                "size": len(content),
                "message": f"Successfully created {filename} on your desktop",
            }

        except Exception as e:
            self.logger.error(f"Desktop file creation error: {str(e)}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e), "filename": filename}

    def read_file(
        self, file_path: str, encoding: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Read file content

        Args:
            file_path: Path to the file
            encoding: File encoding (default: configured encoding)

        Returns:
            Dictionary with file content and metadata
        """
        try:
            path = self._resolve_path(file_path)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if path.is_dir():
                raise IsADirectoryError(f"Path is a directory: {file_path}")

            # Check file size
            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                raise ValueError(
                    f"File too large: {file_size} bytes > {self.max_file_size} bytes"
                )

            file_type = self._get_file_type(path)
            encoding = encoding or self.encoding

            # Read based on file type
            if file_type in [
                FileType.TEXT,
                FileType.PYTHON,
                FileType.MARKDOWN,
                FileType.HTML,
                FileType.XML,
                FileType.LOG,
                FileType.CONFIG,
                FileType.SCRIPT,
            ]:
                with open(path, "r", encoding=encoding) as f:
                    content = f.read()
                return self._create_read_response(content, path, file_type)

            elif file_type == FileType.JSON:
                with open(path, "r", encoding=encoding) as f:
                    content = json.load(f)
                return self._create_read_response(content, path, file_type)

            elif file_type == FileType.YAML:
                with open(path, "r", encoding=encoding) as f:
                    content = yaml.safe_load(f)
                return self._create_read_response(content, path, file_type)

            elif file_type == FileType.CSV:
                with open(path, "r", encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    content = list(reader)
                return self._create_read_response(content, path, file_type)

            else:
                # Binary file
                with open(path, "rb") as f:
                    content = f.read()
                return self._create_read_response(content, path, file_type)

        except Exception as e:
            self.logger.error(f"Read error: {str(e)}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e), "file_path": file_path}

    def _create_read_response(
        self, content: Any, path: Path, file_type: FileType
    ) -> Dict:
        """Create standardized read response"""
        self.stats["operations"] += 1
        if isinstance(content, str):
            self.stats["bytes_read"] += len(content.encode(self.encoding))
        elif isinstance(content, bytes):
            self.stats["bytes_read"] += len(content)

        return {
            "success": True,
            "content": content,
            "path": str(path),
            "file_type": file_type.value,
            "size": path.stat().st_size,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }

    def write_file(
        self,
        file_path: str,
        content: Any,
        encoding: Optional[str] = None,
        append: bool = False,
        create_backup: bool = True,
    ) -> Dict[str, Any]:
        """
        Write content to file

        Args:
            file_path: Path to the file
            content: Content to write (string, bytes, dict, list)
            encoding: File encoding
            append: Append to existing file instead of overwriting
            create_backup: Create backup before overwriting

        Returns:
            Dictionary with operation result
        """
        try:
            path = self._resolve_path(file_path, create_dirs=True)
            file_type = self._get_file_type(path)
            encoding = encoding or self.encoding

            # Create backup if needed
            if create_backup and path.exists() and not append:
                self._create_backup(path)

            # Convert content based on type
            if isinstance(content, (dict, list)):
                if file_type == FileType.JSON:
                    content_str = json.dumps(content, indent=2, default=str)
                elif file_type == FileType.YAML:
                    content_str = yaml.dump(content, default_flow_style=False)
                else:
                    content_str = str(content)
                content_bytes = content_str.encode(encoding)
                mode = "ab" if append else "wb"

                with open(path, mode) as f:
                    f.write(content_bytes)

            elif isinstance(content, str):
                mode = "a" if append else "w"
                with open(path, mode, encoding=encoding) as f:
                    f.write(content)

            elif isinstance(content, bytes):
                mode = "ab" if append else "wb"
                with open(path, mode) as f:
                    f.write(content)

            else:
                # Convert to string
                mode = "a" if append else "w"
                with open(path, mode, encoding=encoding) as f:
                    f.write(str(content))

            self.stats["operations"] += 1
            self.stats["bytes_written"] += (
                len(content) if isinstance(content, (str, bytes)) else 0
            )
            if not path.exists() or (append and not path.exists()):
                self.stats["files_created"] += 1

            return {
                "success": True,
                "path": str(path),
                "operation": "append" if append else "write",
                "size": path.stat().st_size,
                "message": f"Successfully wrote to {file_path}",
            }

        except Exception as e:
            self.logger.error(f"Write error: {str(e)}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e), "file_path": file_path}

    def edit_file(self, file_path: str, edits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Edit file with multiple edit operations

        Edit operations:
        - {'type': 'replace', 'old': 'text_to_replace', 'new': 'new_text'}
        - {'type': 'insert_line', 'line': 5, 'content': 'text to insert'}
        - {'type': 'delete_line', 'line': 5}
        - {'type': 'replace_line', 'line': 5, 'content': 'new line content'}
        - {'type': 'regex_replace', 'pattern': r'regex', 'replacement': 'new'}
        - {'type': 'append', 'content': 'text to append'}
        - {'type': 'prepend', 'content': 'text to prepend'}

        Args:
            file_path: Path to the file
            edits: List of edit operations to apply

        Returns:
            Dictionary with edit results
        """
        try:
            # Read current content
            read_result = self.read_file(file_path)
            if not read_result["success"]:
                return read_result

            content = read_result["content"]
            if not isinstance(content, str):
                raise ValueError("Edit operations only supported on text files")

            # Create backup
            path = self._resolve_path(file_path)
            self._create_backup(path)

            # Apply each edit
            lines = content.splitlines(keepends=True)
            modifications = []

            for edit in edits:
                edit_type = edit.get("type")

                if edit_type == "replace":
                    old = edit.get("old", "")
                    new = edit.get("new", "")
                    count = content.count(old)
                    content = content.replace(old, new)
                    modifications.append(
                        f"Replaced '{old}' with '{new}' ({count} occurrences)"
                    )

                elif edit_type == "regex_replace":
                    pattern = edit.get("pattern", "")
                    replacement = edit.get("replacement", "")
                    new_content, count = re.subn(pattern, replacement, content)
                    content = new_content
                    modifications.append(
                        f"Applied regex pattern: {pattern} ({count} replacements)"
                    )

                elif edit_type == "insert_line":
                    line_num = edit.get("line", 0)
                    insert_content = edit.get("content", "")
                    if 0 <= line_num <= len(lines):
                        lines.insert(line_num, insert_content + "\n")
                        modifications.append(f"Inserted line at position {line_num}")
                    else:
                        raise ValueError(f"Invalid line number: {line_num}")

                elif edit_type == "delete_line":
                    line_num = edit.get("line", 0)
                    if 0 <= line_num < len(lines):
                        deleted = lines.pop(line_num)
                        modifications.append(
                            f"Deleted line {line_num}: {deleted.strip()}"
                        )
                    else:
                        raise ValueError(f"Invalid line number: {line_num}")

                elif edit_type == "replace_line":
                    line_num = edit.get("line", 0)
                    new_content = edit.get("content", "")
                    if 0 <= line_num < len(lines):
                        old = lines[line_num]
                        lines[line_num] = new_content + (
                            "\n" if not new_content.endswith("\n") else ""
                        )
                        modifications.append(
                            f"Replaced line {line_num}: {old.strip()} -> {new_content}"
                        )
                    else:
                        raise ValueError(f"Invalid line number: {line_num}")

                elif edit_type == "append":
                    append_content = edit.get("content", "")
                    content += append_content
                    modifications.append("Appended content")

                elif edit_type == "prepend":
                    prepend_content = edit.get("content", "")
                    content = prepend_content + content
                    modifications.append("Prepended content")

                elif edit_type == "find_and_replace_all":
                    search = edit.get("search", "")
                    replace = edit.get("replace", "")
                    count = content.count(search)
                    content = content.replace(search, replace)
                    modifications.append(f"Replaced {count} occurrences of '{search}'")

                else:
                    modifications.append(f"Unknown edit type: {edit_type}")

            # Rebuild content if line operations were performed
            if any(
                e.get("type") in ["insert_line", "delete_line", "replace_line"]
                for e in edits
            ):
                content = "".join(lines)

            # Write modified content
            write_result = self.write_file(file_path, content, create_backup=False)

            if write_result["success"]:
                write_result["edits_applied"] = modifications
                write_result["edit_count"] = len(edits)

            return write_result

        except Exception as e:
            self.logger.error(f"Edit error: {str(e)}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e), "file_path": file_path}

    def delete_file(self, file_path: str, safe_mode: bool = True) -> Dict[str, Any]:
        """
        Delete a file or directory

        Args:
            file_path: Path to delete
            safe_mode: If True, moves to trash instead of permanent deletion

        Returns:
            Dictionary with deletion result
        """
        try:
            path = self._resolve_path(file_path)

            if not path.exists():
                raise FileNotFoundError(f"Path not found: {file_path}")

            info = self.get_file_info(file_path)

            if safe_mode:
                # Move to trash/backup instead of deleting
                trash_dir = self.workspace_root / ".trash"
                trash_dir.mkdir(exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                dest_path = trash_dir / f"{path.name}.{timestamp}"
                shutil.move(str(path), str(dest_path))

                self.stats["operations"] += 1

                return {
                    "success": True,
                    "operation": "move_to_trash",
                    "original_path": str(path),
                    "trash_path": str(dest_path),
                    "info": info,
                }
            else:
                # Permanent deletion
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

                self.stats["operations"] += 1
                self.stats["files_deleted"] += 1

                return {
                    "success": True,
                    "operation": "permanent_delete",
                    "deleted_path": str(path),
                    "info": info,
                }

        except Exception as e:
            self.logger.error(f"Delete error: {str(e)}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e), "file_path": file_path}

    def copy_file(
        self, source_path: str, dest_path: str, overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Copy file or directory

        Args:
            source_path: Source path
            dest_path: Destination path
            overwrite: Overwrite if destination exists

        Returns:
            Dictionary with copy result
        """
        try:
            src = self._resolve_path(source_path)
            dst = self._resolve_path(dest_path, create_dirs=True)

            if not src.exists():
                raise FileNotFoundError(f"Source not found: {source_path}")

            if dst.exists() and not overwrite:
                raise FileExistsError(f"Destination exists: {dest_path}")

            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                operation = "copy_directory"
            else:
                shutil.copy2(src, dst)
                operation = "copy_file"

            self.stats["operations"] += 1

            return {
                "success": True,
                "operation": operation,
                "source": str(src),
                "destination": str(dst),
                "size": dst.stat().st_size if dst.exists() else None,
            }

        except Exception as e:
            self.logger.error(f"Copy error: {str(e)}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": str(e),
                "source": source_path,
                "destination": dest_path,
            }

    def move_file(
        self, source_path: str, dest_path: str, overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Move file or directory

        Args:
            source_path: Source path
            dest_path: Destination path
            overwrite: Overwrite if destination exists

        Returns:
            Dictionary with move result
        """
        try:
            src = self._resolve_path(source_path)
            dst = self._resolve_path(dest_path, create_dirs=True)

            if not src.exists():
                raise FileNotFoundError(f"Source not found: {source_path}")

            if dst.exists() and not overwrite:
                raise FileExistsError(f"Destination exists: {dest_path}")

            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()

            shutil.move(str(src), str(dst))

            self.stats["operations"] += 1

            return {
                "success": True,
                "operation": "move",
                "source": str(src),
                "destination": str(dst),
                "size": dst.stat().st_size if dst.exists() else None,
            }

        except Exception as e:
            self.logger.error(f"Move error: {str(e)}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": str(e),
                "source": source_path,
                "destination": dest_path,
            }

    def search_files(
        self,
        directory: str,
        pattern: str,
        file_pattern: str = "*",
        recursive: bool = True,
        content_search: bool = False,
    ) -> Dict[str, Any]:
        """
        Search for files matching pattern

        Args:
            directory: Directory to search in
            pattern: Search pattern (regex or plain text)
            file_pattern: File name pattern (glob)
            recursive: Search subdirectories
            content_search: Search within file contents

        Returns:
            Dictionary with search results
        """
        try:
            path = self._resolve_path(directory)
            if not path.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")

            results = []
            search_regex = re.compile(
                pattern, re.IGNORECASE if not content_search else 0
            )

            # Walk through directory
            iterator = (
                path.rglob(file_pattern) if recursive else path.glob(file_pattern)
            )

            for file_path in iterator:
                if file_path.is_file():
                    match = False
                    match_info = {}

                    if content_search:
                        # Search in file content
                        try:
                            read_result = self.read_file(str(file_path))
                            if read_result["success"] and isinstance(
                                read_result["content"], str
                            ):
                                content = read_result["content"]
                                matches = list(search_regex.finditer(content))
                                if matches:
                                    match = True
                                    match_info = {
                                        "matches": len(matches),
                                        "preview": [
                                            content[
                                                max(0, m.start() - 50) : m.end() + 50
                                            ]
                                            for m in matches[:3]
                                        ],
                                    }
                        except:
                            continue
                    else:
                        # Search in filename
                        if search_regex.search(file_path.name):
                            match = True
                            match_info = {"matched_name": file_path.name}

                    if match:
                        results.append(
                            {
                                "path": str(file_path),
                                "name": file_path.name,
                                "size": file_path.stat().st_size,
                                "modified": datetime.fromtimestamp(
                                    file_path.stat().st_mtime
                                ).isoformat(),
                                **match_info,
                            }
                        )

            return {
                "success": True,
                "search_pattern": pattern,
                "content_search": content_search,
                "directory": str(path),
                "total_matches": len(results),
                "results": results,
            }

        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e), "directory": directory}

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get detailed file information

        Args:
            file_path: Path to file or directory

        Returns:
            Dictionary with file information
        """
        try:
            path = self._resolve_path(file_path)

            if not path.exists():
                raise FileNotFoundError(f"Path not found: {file_path}")

            stat_info = path.stat()

            info = {
                "success": True,
                "name": path.name,
                "path": str(path),
                "absolute_path": str(path.absolute()),
                "size": stat_info.st_size,
                "size_human": self._human_readable_size(stat_info.st_size),
                "created": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat_info.st_atime).isoformat(),
                "is_directory": path.is_dir(),
                "is_file": path.is_file(),
                "extension": path.suffix if not path.is_dir() else "",
                "file_type": (
                    self._get_file_type(path).value
                    if not path.is_dir()
                    else "directory"
                ),
                "permissions": oct(stat_info.st_mode)[-3:],
                "owner": stat_info.st_uid,
            }

            # Add hash for files
            if path.is_file() and path.stat().st_size < 10 * 1024 * 1024:  # < 10MB
                info["md5"] = self._calculate_hash(path, "md5")
                info["sha256"] = self._calculate_hash(path, "sha256")

            return info

        except Exception as e:
            self.logger.error(f"Info error: {str(e)}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e), "file_path": file_path}

    def list_directory(
        self, directory: str, pattern: str = "*", show_hidden: bool = False
    ) -> Dict[str, Any]:
        """
        List contents of a directory

        Args:
            directory: Directory path
            pattern: Glob pattern to filter files
            show_hidden: Include hidden files

        Returns:
            Dictionary with directory listing
        """
        try:
            path = self._resolve_path(directory)

            if not path.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")

            if not path.is_dir():
                raise NotADirectoryError(f"Not a directory: {directory}")

            items = []
            for item_path in path.glob(pattern):
                if not show_hidden and item_path.name.startswith("."):
                    continue

                try:
                    stat_info = item_path.stat()
                    items.append(
                        {
                            "name": item_path.name,
                            "path": str(item_path),
                            "is_directory": item_path.is_dir(),
                            "size": stat_info.st_size if item_path.is_file() else 0,
                            "size_human": (
                                self._human_readable_size(stat_info.st_size)
                                if item_path.is_file()
                                else ""
                            ),
                            "modified": datetime.fromtimestamp(
                                stat_info.st_mtime
                            ).isoformat(),
                            "extension": (
                                item_path.suffix if item_path.is_file() else ""
                            ),
                        }
                    )
                except:
                    continue

            # Sort directories first, then files
            items.sort(key=lambda x: (not x["is_directory"], x["name"].lower()))

            return {
                "success": True,
                "directory": str(path),
                "total_items": len(items),
                "items": items,
            }

        except Exception as e:
            self.logger.error(f"List error: {str(e)}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e), "directory": directory}

    def compress_files(
        self,
        source_paths: List[str],
        output_path: str,
        format: str = "zip",
        compression_level: int = 6,
    ) -> Dict[str, Any]:
        """
        Compress files or directories into archive

        Args:
            source_paths: List of paths to compress
            output_path: Output archive path
            format: Archive format ('zip', 'tar', 'tar.gz', 'tar.bz2')
            compression_level: Compression level (1-9)

        Returns:
            Dictionary with compression result
        """
        try:
            # Resolve paths
            sources = [self._resolve_path(p) for p in source_paths]
            output = self._resolve_path(output_path, create_dirs=True)

            # Verify sources exist
            for src in sources:
                if not src.exists():
                    raise FileNotFoundError(f"Source not found: {src}")

            # Compress based on format
            if format == "zip":
                with zipfile.ZipFile(
                    output, "w", zipfile.ZIP_DEFLATED, compresslevel=compression_level
                ) as zipf:
                    for src in sources:
                        if src.is_dir():
                            for file in src.rglob("*"):
                                if file.is_file():
                                    zipf.write(file, file.relative_to(src.parent))
                        else:
                            zipf.write(src, src.name)

            elif format in ["tar", "tar.gz", "tar.bz2"]:
                mode = "w"
                if format == "tar.gz":
                    mode = "w:gz"
                elif format == "tar.bz2":
                    mode = "w:bz2"

                with tarfile.open(output, mode) as tarf:
                    for src in sources:
                        tarf.add(src, arcname=src.name)

            else:
                raise ValueError(f"Unsupported format: {format}")

            self.stats["operations"] += 1

            return {
                "success": True,
                "archive_path": str(output),
                "format": format,
                "source_count": len(sources),
                "archive_size": output.stat().st_size,
                "archive_size_human": self._human_readable_size(output.stat().st_size),
            }

        except Exception as e:
            self.logger.error(f"Compress error: {str(e)}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e), "output_path": output_path}

    def extract_archive(
        self, archive_path: str, output_dir: str, delete_after: bool = False
    ) -> Dict[str, Any]:
        """
        Extract archive file

        Args:
            archive_path: Path to archive file
            output_dir: Output directory for extracted files
            delete_after: Delete archive after extraction

        Returns:
            Dictionary with extraction result
        """
        try:
            archive = self._resolve_path(archive_path)
            output = self._resolve_path(output_dir, create_dirs=True)

            if not archive.exists():
                raise FileNotFoundError(f"Archive not found: {archive_path}")

            # Extract based on extension
            extracted_files = []

            if archive.suffix == ".zip":
                with zipfile.ZipFile(archive, "r") as zipf:
                    zipf.extractall(output)
                    extracted_files = zipf.namelist()

            elif archive.suffix in [".tar", ".gz", ".bz2"]:
                mode = "r"
                if archive.suffix == ".gz":
                    mode = "r:gz"
                elif archive.suffix == ".bz2":
                    mode = "r:bz2"

                with tarfile.open(archive, mode) as tarf:
                    tarf.extractall(output)
                    extracted_files = tarf.getnames()

            else:
                raise ValueError(f"Unsupported archive format: {archive.suffix}")

            # Delete archive if requested
            if delete_after:
                archive.unlink()

            self.stats["operations"] += 1

            return {
                "success": True,
                "archive_path": str(archive),
                "output_dir": str(output),
                "extracted_count": len(extracted_files),
                "extracted_files": extracted_files[:50],  # Limit output
            }

        except Exception as e:
            self.logger.error(f"Extract error: {str(e)}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e), "archive_path": archive_path}

    def batch_operation(
        self, operations: List[Dict[str, Any]], parallel: bool = False
    ) -> Dict[str, Any]:
        """
        Execute multiple file operations in batch

        Args:
            operations: List of operation dictionaries
            parallel: Execute operations in parallel

        Returns:
            Dictionary with batch results
        """
        results = []
        errors = []

        def execute_op(op):
            op_type = op.get("type")
            if op_type == "read":
                return self.read_file(op.get("path"))
            elif op_type == "write":
                return self.write_file(
                    op.get("path"), op.get("content"), append=op.get("append", False)
                )
            elif op_type == "delete":
                return self.delete_file(op.get("path"), op.get("safe_mode", True))
            elif op_type == "copy":
                return self.copy_file(
                    op.get("source"),
                    op.get("dest"),
                    overwrite=op.get("overwrite", False),
                )
            elif op_type == "move":
                return self.move_file(
                    op.get("source"),
                    op.get("dest"),
                    overwrite=op.get("overwrite", False),
                )
            elif op_type == "info":
                return self.get_file_info(op.get("path"))
            else:
                return {"success": False, "error": f"Unknown operation: {op_type}"}

        if parallel:
            with ThreadPoolExecutor(max_workers=self.thread_pool_size) as executor:
                future_to_op = {
                    executor.submit(execute_op, op): op for op in operations
                }
                for future in as_completed(future_to_op):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        errors.append(str(e))
        else:
            for op in operations:
                try:
                    results.append(execute_op(op))
                except Exception as e:
                    errors.append(str(e))

        successful = sum(1 for r in results if r.get("success", False))

        return {
            "success": successful > 0,
            "total_operations": len(operations),
            "successful_operations": successful,
            "failed_operations": len(operations) - successful,
            "results": results,
            "errors": errors if errors else None,
        }

    def _human_readable_size(self, size: int) -> str:
        """Convert size to human readable format"""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"

    def _calculate_hash(self, file_path: Path, algorithm: str) -> str:
        """Calculate file hash"""
        hash_func = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            **self.stats,
            "workspace": str(self.workspace_root),
            "desktop_path": (
                str(self.desktop_path) if self.allow_desktop_access else "disabled"
            ),
            "backup_enabled": self.create_backups,
        }

    def create_temp_file(
        self, content: Any = None, suffix: str = ".txt"
    ) -> Dict[str, Any]:
        """
        Create a temporary file

        Args:
            content: Optional content to write
            suffix: File suffix

        Returns:
            Dictionary with temp file info
        """
        try:
            fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=self.workspace_root)
            os.close(fd)

            temp_path = Path(temp_path)

            if content:
                self.write_file(str(temp_path), content)

            return {
                "success": True,
                "path": str(temp_path),
                "name": temp_path.name,
                "temporary": True,
            }

        except Exception as e:
            self.logger.error(f"Temp file creation error: {str(e)}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e)}

    def restore_backup(self, backup_name: str, target_path: str) -> Dict[str, Any]:
        """
        Restore a file from backup

        Args:
            backup_name: Name of the backup file
            target_path: Path to restore to

        Returns:
            Dictionary with restore result
        """
        try:
            backup_path = self.backup_dir / backup_name

            if not backup_path.exists():
                raise FileNotFoundError(f"Backup not found: {backup_name}")

            target = self._resolve_path(target_path)

            # Create backup of current file before restoring
            if target.exists():
                self._create_backup(target)

            shutil.copy2(backup_path, target)

            return {
                "success": True,
                "restored_from": str(backup_path),
                "restored_to": str(target),
                "message": f"Successfully restored from {backup_name}",
            }

        except Exception as e:
            self.logger.error(f"Restore error: {str(e)}")
            return {"success": False, "error": str(e)}

    def list_backups(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        List available backups

        Args:
            file_path: Optional specific file to list backups for

        Returns:
            Dictionary with backups list
        """
        try:
            backups = []

            for backup_file in self.backup_dir.glob("*.bak"):
                if file_path:
                    # Filter backups for specific file
                    original_name = Path(file_path).stem
                    if backup_file.stem.startswith(original_name):
                        backups.append(
                            {
                                "name": backup_file.name,
                                "size": backup_file.stat().st_size,
                                "created": datetime.fromtimestamp(
                                    backup_file.stat().st_ctime
                                ).isoformat(),
                            }
                        )
                else:
                    backups.append(
                        {
                            "name": backup_file.name,
                            "size": backup_file.stat().st_size,
                            "created": datetime.fromtimestamp(
                                backup_file.stat().st_ctime
                            ).isoformat(),
                        }
                    )

            backups.sort(key=lambda x: x["created"], reverse=True)

            return {"success": True, "backups": backups, "total_backups": len(backups)}

        except Exception as e:
            self.logger.error(f"List backups error: {str(e)}")
            return {"success": False, "error": str(e)}


# Integration wrapper for EDIATH
class FileAgentWrapper:
    """
    Wrapper class to integrate FileAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        workspace = config.get("workspace") if config else None
        self.file_agent = FileAgent(workspace_root=workspace, config=config)
        self.agent_type = "file_manager"
        self.capabilities = [
            "read_file",
            "write_file",
            "edit_file",
            "delete_file",
            "copy_file",
            "move_file",
            "search_files",
            "list_directory",
            "get_file_info",
            "compress_files",
            "extract_archive",
            "batch_operations",
            "create_file_on_desktop",
            "restore_backup",
            "list_backups",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a file operation request

        Request format:
        {
            'operation': 'read|write|edit|delete|copy|move|search|list|info|compress|extract|batch|create_desktop',
            ... operation specific parameters ...
        }
        """
        operation = request.get("operation")

        if operation == "read":
            return self.file_agent.read_file(
                request.get("path"), encoding=request.get("encoding")
            )
        elif operation == "write":
            return self.file_agent.write_file(
                request.get("path"),
                request.get("content"),
                encoding=request.get("encoding"),
                append=request.get("append", False),
                create_backup=request.get("create_backup", True),
            )
        elif operation == "edit":
            return self.file_agent.edit_file(
                request.get("path"), request.get("edits", [])
            )
        elif operation == "delete":
            return self.file_agent.delete_file(
                request.get("path"), safe_mode=request.get("safe_mode", True)
            )
        elif operation == "copy":
            return self.file_agent.copy_file(
                request.get("source"),
                request.get("destination"),
                overwrite=request.get("overwrite", False),
            )
        elif operation == "move":
            return self.file_agent.move_file(
                request.get("source"),
                request.get("destination"),
                overwrite=request.get("overwrite", False),
            )
        elif operation == "search":
            return self.file_agent.search_files(
                request.get("directory"),
                request.get("pattern"),
                file_pattern=request.get("file_pattern", "*"),
                recursive=request.get("recursive", True),
                content_search=request.get("content_search", False),
            )
        elif operation == "list":
            return self.file_agent.list_directory(
                request.get("directory"),
                pattern=request.get("pattern", "*"),
                show_hidden=request.get("show_hidden", False),
            )
        elif operation == "info":
            return self.file_agent.get_file_info(request.get("path"))
        elif operation == "compress":
            return self.file_agent.compress_files(
                request.get("sources", []),
                request.get("output"),
                format=request.get("format", "zip"),
                compression_level=request.get("compression_level", 6),
            )
        elif operation == "extract":
            return self.file_agent.extract_archive(
                request.get("archive"),
                request.get("output_dir"),
                delete_after=request.get("delete_after", False),
            )
        elif operation == "batch":
            return self.file_agent.batch_operation(
                request.get("operations", []), parallel=request.get("parallel", False)
            )
        elif operation == "create_desktop":
            return self.file_agent.create_file_on_desktop(
                request.get("filename"),
                request.get("content", ""),
                request.get("extension", ".txt"),
            )
        elif operation == "restore_backup":
            return self.file_agent.restore_backup(
                request.get("backup_name"), request.get("target_path")
            )
        elif operation == "list_backups":
            return self.file_agent.list_backups(request.get("file_path"))
        elif operation == "stats":
            return self.file_agent.get_stats()
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "FileAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "workspace": str(self.file_agent.workspace_root),
            "desktop_access": self.file_agent.allow_desktop_access,
            "stats": self.file_agent.get_stats(),
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the file agent
    agent = FileAgent(
        workspace_root="./test_workspace", config={"allow_desktop_access": True}
    )

    print("=== File Agent Test ===\n")

    # Test desktop file creation
    print("1. Creating file on desktop...")
    result = agent.create_file_on_desktop(
        "EDIATH_Test_File",
        "Hello! This file was created by EDIATH.\nTimestamp: "
        + datetime.now().isoformat(),
        ".txt",
    )
    print(f"   Result: {result.get('message', result.get('error', 'Unknown'))}")

    # Test write and read
    print("\n2. Write and Read test...")
    result = agent.write_file("test.txt", "Hello, World!\nThis is a test file.")
    print(f"   Write: {result.get('message')}")

    result = agent.read_file("test.txt")
    print(f"   Read content: {result.get('content', '')[:50]}")

    # Test editing
    print("\n3. Edit File test...")
    edits = [
        {"type": "replace", "old": "World", "new": "EDIATH"},
        {"type": "append", "content": "\nAppended line."},
        {"type": "insert_line", "line": 1, "content": "Inserted line"},
    ]
    result = agent.edit_file("test.txt", edits)
    if result.get("success"):
        print(f"   Edits applied: {len(result.get('edits_applied', []))}")
        for edit in result.get("edits_applied", [])[:3]:
            print(f"     - {edit}")

    # Test search
    print("\n4. Search test...")
    result = agent.search_files(".", "test", content_search=False)
    print(f"   Found {result.get('total_matches', 0)} files")

    # Test directory listing
    print("\n5. Directory listing...")
    result = agent.list_directory(".")
    for item in result.get("items", [])[:5]:
        print(f"   {item['name']} ({'DIR' if item['is_directory'] else 'FILE'})")

    # Test file info
    print("\n6. File info...")
    result = agent.get_file_info("test.txt")
    if result.get("success"):
        print(f"   Name: {result.get('name')}")
        print(f"   Size: {result.get('size_human')}")
        print(f"   Modified: {result.get('modified')}")

    # Test backup listing
    print("\n7. Backups...")
    result = agent.list_backups()
    if result.get("success"):
        print(f"   Total backups: {result.get('total_backups')}")
        for backup in result.get("backups", [])[:3]:
            print(f"     - {backup['name']} ({backup['size']} bytes)")

    # Get statistics
    print("\n8. Agent Statistics...")
    stats = agent.get_stats()
    print(f"   Operations: {stats['operations']}")
    print(f"   Files created: {stats['files_created']}")
    print(f"   Bytes written: {stats['bytes_written']}")
    print(f"   Workspace: {stats['workspace']}")
    print(f"   Desktop path: {stats['desktop_path']}")

    # Clean up
    print("\n9. Cleanup...")
    agent.delete_file("test.txt", safe_mode=False)
    print("   Test files cleaned up")

    print("\n=== Test Complete ===")
