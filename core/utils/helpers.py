"""Core utility functions and helpers for EDIATH."""

import os
import json
import hashlib
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
import asyncio


def get_project_root() -> str:
    """Get the root directory of the EDIATH project."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(path: str) -> str:
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)
    return path


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file with error handling."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"response": ""}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def calculate_hash(data: str, algorithm: str = "sha256") -> str:
    """Calculate hash of a string."""
    if algorithm == "sha256":
        return hashlib.sha256(data.encode()).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(data.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def retry_on_exception(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying functions on exception."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            current_delay = delay

            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise
                    asyncio.sleep(current_delay)
                    current_delay *= backoff

            return None

        return wrapper

    return decorator


def time_execution(func: Callable) -> Callable:
    """Decorator to measure function execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        return result, duration

    return wrapper


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to max length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename


def format_timestamp(
    dt: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """Format datetime object to string."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime(format_str)


def parse_duration(duration_str: str) -> timedelta:
    """Parse duration string like '1h30m' to timedelta."""
    parts = {"h": 3600, "m": 60, "s": 1}
    seconds = 0

    import re

    for match in re.finditer(r"(\d+)([hms])", duration_str):
        value, unit = match.groups()
        seconds += int(value) * parts[unit]

    return timedelta(seconds=seconds)


def merge_dicts(base: Dict, updates: Dict, deep: bool = True) -> Dict:
    """Merge two dictionaries."""
    result = base.copy()

    for key, value in updates.items():
        if (
            deep
            and key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value, deep=True)
        else:
            result[key] = value

    return result


def chunk_list(items: List, chunk_size: int) -> List[List]:
    """Split list into chunks."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def flatten_list(nested_list: List[List]) -> List:
    """Flatten nested list."""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


__all__ = [
    "get_project_root",
    "ensure_dir",
    "load_json",
    "save_json",
    "calculate_hash",
    "retry_on_exception",
    "time_execution",
    "truncate_string",
    "sanitize_filename",
    "format_timestamp",
    "parse_duration",
    "merge_dicts",
    "chunk_list",
    "flatten_list",
]
