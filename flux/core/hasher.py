"""Hash computation utilities for FLUX versioning system.

Provides functions to compute SHA-256 hashes of raw data files,
canonicalize JSON configs, and combine hashes into version identifiers.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Union[str, Path], chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file's contents.

    Args:
        file_path: Path to the file to hash.
        chunk_size: Size of chunks to read at a time (default 8192 bytes).

    Returns:
        Hex-encoded SHA-256 hash string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be read.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)

    file_hash = sha256.hexdigest()
    logger.debug("File hash for '%s': %s", file_path, file_hash)
    return file_hash


def canonicalize_config(config: Dict[str, Any]) -> str:
    """Canonicalize a config dict to a deterministic JSON string.

    Sorts keys recursively and uses consistent formatting to ensure
    the same config always produces the same string representation.

    Args:
        config: Configuration dictionary.

    Returns:
        Canonical JSON string representation.
    """
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of a canonicalized config.

    Args:
        config: Configuration dictionary.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    canonical = canonicalize_config(config)
    config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    logger.debug("Config hash: %s", config_hash)
    return config_hash


def compute_version_hash(raw_hash: str, config_hash: str) -> str:
    """Compute version hash by combining raw data hash and config hash.

    Version ID = SHA-256(raw_hash + config_hash).

    Args:
        raw_hash: SHA-256 hex hash of the raw data file.
        config_hash: SHA-256 hex hash of the canonicalized config.

    Returns:
        Hex-encoded SHA-256 version hash string.
    """
    combined = raw_hash + config_hash
    version_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    logger.debug("Version hash: %s (raw=%s, config=%s)", version_hash, raw_hash, config_hash)
    return version_hash


def compute_string_hash(data: str) -> str:
    """Compute SHA-256 hash of a string.

    Args:
        data: Input string.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
