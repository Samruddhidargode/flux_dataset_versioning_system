"""File utility helpers for FLUX versioning system."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """Create a directory (and parents) if it doesn't exist.

    Args:
        path: Directory path to create.

    Returns:
        Path object of the created directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Read and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Union[str, Path], data: Dict[str, Any], indent: int = 2) -> None:
    """Write a dictionary as a JSON file.

    Args:
        path: Output file path.
        data: Dictionary to serialize.
        indent: JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """Copy a file from src to dst.

    Args:
        src: Source file path.
        dst: Destination file path.

    Returns:
        Path to the copied file.

    Raises:
        FileNotFoundError: If source file doesn't exist.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    return dst


def is_flux_repo(path: Union[str, Path]) -> bool:
    """Check if a path contains a valid FLUX repository.

    A valid repo has 'versions/' directory and 'refs.json' file.

    Args:
        path: Path to check.

    Returns:
        True if it's a valid FLUX repo.
    """
    path = Path(path)
    return (
        path.is_dir()
        and (path / "versions").is_dir()
        and (path / "refs.json").is_file()
    )


def safe_filename(name: str) -> str:
    """Sanitize a string for use as a filename.

    Args:
        name: Input string.

    Returns:
        Sanitized string safe for filenames.
    """
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
