"""Repository operations for FLUX versioning system.

Provides functions for initializing repos, creating versions,
listing/loading versions, tagging, and export/import.
"""

import json
import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from flux.core.hasher import compute_config_hash, compute_file_hash, compute_version_hash
from flux.core.locker import FileLock
from flux.core.metrics import compute_metrics
from flux.core.preprocessor import apply_pipeline
from flux.exceptions import (
    DataFormatError,
    ImportError_,
    InvalidConfigError,
    RepositoryExistsError,
    RepositoryNotFoundError,
    TagNotFoundError,
    VersionNotFoundError,
)
from flux.models.version import VersionInfo
from flux.utils.file_utils import (
    copy_file,
    ensure_directory,
    is_flux_repo,
    read_json,
    write_json,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Repository Initialization
# ---------------------------------------------------------------------------

def init_repo(path: Union[str, Path]) -> Path:
    """Initialize a new FLUX repository at the given path.

    Creates the directory structure: versions/, locks/, refs.json.
    If the path already contains a valid repo, it is detected and reused.

    Args:
        path: Path where the repository should be created.

    Returns:
        Path to the initialized repository.

    Raises:
        RepositoryExistsError: If trying to re-init (optional; currently reuses).
    """
    repo_path = Path(path).resolve()

    if is_flux_repo(repo_path):
        logger.info("Repository already exists at %s, reusing.", repo_path)
        return repo_path

    ensure_directory(repo_path / "versions")
    ensure_directory(repo_path / "locks")

    refs_path = repo_path / "refs.json"
    if not refs_path.exists():
        write_json(refs_path, {})

    logger.info("Initialized FLUX repository at %s", repo_path)
    return repo_path


def _validate_repo(repo_path: Path) -> None:
    """Validate that a path is a FLUX repository.

    Args:
        repo_path: Path to validate.

    Raises:
        RepositoryNotFoundError: If not a valid repository.
    """
    if not is_flux_repo(repo_path):
        raise RepositoryNotFoundError(
            f"Not a valid FLUX repository: {repo_path}. "
            f"Run 'flux init' first."
        )


# ---------------------------------------------------------------------------
# Version Creation
# ---------------------------------------------------------------------------

def create_version(
    raw_path: Union[str, Path],
    config: Union[str, Path, Dict[str, Any]],
    repo_path: Union[str, Path],
    user: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> str:
    """Create a new dataset version.

    Args:
        raw_path: Path to the raw CSV data file.
        config: Preprocessing config as a dict, or path to a JSON config file.
        repo_path: Path to the FLUX repository.
        user: Optional user identifier.
        tags: Optional list of tags to assign.

    Returns:
        The version hash string.

    Raises:
        RepositoryNotFoundError: If repo_path is not a valid repository.
        FileNotFoundError: If raw_path doesn't exist.
        DataFormatError: If the CSV doesn't have a 'text' column.
        InvalidConfigError: If the config is invalid.
    """
    repo_path = Path(repo_path).resolve()
    raw_path = Path(raw_path).resolve()
    _validate_repo(repo_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    # Load config
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config_dict = read_json(config_path)
    else:
        config_dict = config

    # Validate config has pipeline
    if "pipeline" not in config_dict:
        raise InvalidConfigError("Config must contain a 'pipeline' key.")

    # Compute hashes
    raw_hash = compute_file_hash(raw_path)
    config_hash = compute_config_hash(config_dict)
    version_hash = compute_version_hash(raw_hash, config_hash)

    version_dir = repo_path / "versions" / version_hash
    lock_dir = repo_path / "locks"

    # Acquire lock
    with FileLock(lock_dir, "create_version"):
        # Idempotent: if version already exists, return existing hash
        if version_dir.exists():
            logger.info("Version %s already exists, returning existing.", version_hash)
            if tags:
                for t in tags:
                    tag_version(version_hash, t, repo_path)
            return version_hash

        # Create version directory
        ensure_directory(version_dir)

        try:
            # Copy raw file
            copy_file(raw_path, version_dir / "raw.csv")

            # Write config
            write_json(version_dir / "config.json", config_dict)

            # Load and validate raw data
            df = pd.read_csv(raw_path, encoding="utf-8")
            if "text" not in df.columns:
                raise DataFormatError(
                    f"CSV file must have a 'text' column. "
                    f"Found columns: {list(df.columns)}"
                )

            # Apply preprocessing pipeline
            pipeline = config_dict["pipeline"]
            processed_df = apply_pipeline(df, pipeline)

            # Save processed data
            processed_df.to_csv(version_dir / "processed.csv", index=False, encoding="utf-8")

            # Determine if tokenization was applied
            is_tokenized = any(s.get("step") == "tokenize" for s in pipeline)

            # Compute and save metrics
            metadata = compute_metrics(processed_df, user=user, is_tokenized=is_tokenized)
            metadata["raw_hash"] = raw_hash
            metadata["config_hash"] = config_hash
            metadata["version_hash"] = version_hash
            write_json(version_dir / "metadata.json", metadata)

            logger.info("Created version %s", version_hash)

        except Exception:
            # Clean up on failure
            import shutil
            if version_dir.exists():
                shutil.rmtree(str(version_dir), ignore_errors=True)
            raise

    # Tag the version (outside the create lock, tag_version has its own lock)
    if tags:
        for t in tags:
            tag_version(version_hash, t, repo_path)

    return version_hash


# ---------------------------------------------------------------------------
# Version Listing & Retrieval
# ---------------------------------------------------------------------------

def list_versions(repo_path: Union[str, Path]) -> List[VersionInfo]:
    """List all versions in the repository.

    Args:
        repo_path: Path to the FLUX repository.

    Returns:
        List of VersionInfo objects with basic metadata.

    Raises:
        RepositoryNotFoundError: If repo_path is not a valid repository.
    """
    repo_path = Path(repo_path).resolve()
    _validate_repo(repo_path)

    versions_dir = repo_path / "versions"
    refs = _load_refs(repo_path)
    # Invert refs: hash -> list of tag names
    hash_to_tags: Dict[str, List[str]] = {}
    for tag_name, h in refs.items():
        hash_to_tags.setdefault(h, []).append(tag_name)

    result = []
    for version_dir in sorted(versions_dir.iterdir()):
        if not version_dir.is_dir():
            continue
        vh = version_dir.name

        # Load metadata if available
        meta_path = version_dir / "metadata.json"
        metadata = read_json(meta_path) if meta_path.exists() else {}

        # Load config if available
        config_path = version_dir / "config.json"
        config = read_json(config_path) if config_path.exists() else {}

        info = VersionInfo(
            version_hash=vh,
            raw_hash=metadata.get("raw_hash", ""),
            config_hash=metadata.get("config_hash", ""),
            config=config,
            metrics=metadata,
            tags=hash_to_tags.get(vh, []),
        )
        result.append(info)

    return result


def load_version(
    version_id: str,
    repo_path: Union[str, Path],
    data_type: str = "processed",
) -> pd.DataFrame:
    """Load a version's data as a pandas DataFrame.

    Args:
        version_id: Version hash or tag name.
        repo_path: Path to the FLUX repository.
        data_type: 'processed' (default) or 'raw'.

    Returns:
        pandas DataFrame with the requested data.

    Raises:
        VersionNotFoundError: If the version doesn't exist.
        ValueError: If data_type is invalid.
    """
    repo_path = Path(repo_path).resolve()
    _validate_repo(repo_path)

    version_hash = resolve_version(version_id, repo_path)
    version_dir = repo_path / "versions" / version_hash

    if not version_dir.exists():
        raise VersionNotFoundError(f"Version not found: {version_id}")

    if data_type == "processed":
        data_file = version_dir / "processed.csv"
    elif data_type == "raw":
        data_file = version_dir / "raw.csv"
    else:
        raise ValueError(f"Invalid data_type: '{data_type}'. Use 'processed' or 'raw'.")

    if not data_file.exists():
        raise VersionNotFoundError(f"Data file not found: {data_file}")

    return pd.read_csv(data_file, encoding="utf-8")


def get_version_info(
    version_id: str,
    repo_path: Union[str, Path],
) -> VersionInfo:
    """Get detailed information about a specific version.

    Args:
        version_id: Version hash or tag name.
        repo_path: Path to the FLUX repository.

    Returns:
        VersionInfo object with full metadata.

    Raises:
        VersionNotFoundError: If the version doesn't exist.
    """
    repo_path = Path(repo_path).resolve()
    _validate_repo(repo_path)

    version_hash = resolve_version(version_id, repo_path)
    version_dir = repo_path / "versions" / version_hash

    if not version_dir.exists():
        raise VersionNotFoundError(f"Version not found: {version_id}")

    meta_path = version_dir / "metadata.json"
    config_path = version_dir / "config.json"

    metadata = read_json(meta_path) if meta_path.exists() else {}
    config = read_json(config_path) if config_path.exists() else {}

    refs = _load_refs(repo_path)
    tags = [t for t, h in refs.items() if h == version_hash]

    return VersionInfo(
        version_hash=version_hash,
        raw_hash=metadata.get("raw_hash", ""),
        config_hash=metadata.get("config_hash", ""),
        config=config,
        metrics=metadata,
        tags=tags,
    )


def resolve_version(version_id: str, repo_path: Union[str, Path]) -> str:
    """Resolve a version ID (hash or tag) to a version hash.

    Args:
        version_id: A version hash or tag name.
        repo_path: Path to the FLUX repository.

    Returns:
        The resolved version hash.

    Raises:
        VersionNotFoundError: If the version/tag doesn't exist.
    """
    repo_path = Path(repo_path).resolve()

    # Check if it's a direct hash
    version_dir = repo_path / "versions" / version_id
    if version_dir.exists():
        return version_id

    # Check if it's a tag
    refs = _load_refs(repo_path)
    if version_id in refs:
        return refs[version_id]

    # Check for partial hash match
    versions_dir = repo_path / "versions"
    if versions_dir.exists():
        matches = [
            d.name for d in versions_dir.iterdir()
            if d.is_dir() and d.name.startswith(version_id)
        ]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise VersionNotFoundError(
                f"Ambiguous version ID '{version_id}', matches: {matches}"
            )

    raise VersionNotFoundError(f"Version or tag not found: {version_id}")


# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------

def tag_version(
    version_id: str,
    tag_name: str,
    repo_path: Union[str, Path],
) -> None:
    """Assign a tag to a version.

    Args:
        version_id: Version hash or existing tag name.
        tag_name: Human-readable tag name.
        repo_path: Path to the FLUX repository.

    Raises:
        VersionNotFoundError: If the version doesn't exist.
    """
    repo_path = Path(repo_path).resolve()
    _validate_repo(repo_path)

    version_hash = resolve_version(version_id, repo_path)
    lock_dir = repo_path / "locks"

    with FileLock(lock_dir, "refs"):
        refs = _load_refs(repo_path)
        refs[tag_name] = version_hash
        write_json(repo_path / "refs.json", refs)

    logger.info("Tagged version %s as '%s'", version_hash, tag_name)


def get_version_by_tag(tag_name: str, repo_path: Union[str, Path]) -> str:
    """Get the version hash for a tag.

    Args:
        tag_name: Tag name to look up.
        repo_path: Path to the FLUX repository.

    Returns:
        Version hash string.

    Raises:
        TagNotFoundError: If the tag doesn't exist.
    """
    repo_path = Path(repo_path).resolve()
    refs = _load_refs(repo_path)
    if tag_name not in refs:
        raise TagNotFoundError(f"Tag not found: '{tag_name}'")
    return refs[tag_name]


def list_tags(repo_path: Union[str, Path]) -> Dict[str, str]:
    """List all tags in the repository.

    Args:
        repo_path: Path to the FLUX repository.

    Returns:
        Dictionary mapping tag names to version hashes.
    """
    repo_path = Path(repo_path).resolve()
    _validate_repo(repo_path)
    return _load_refs(repo_path)


def _load_refs(repo_path: Path) -> Dict[str, str]:
    """Load refs.json from the repository.

    Args:
        repo_path: Path to the FLUX repository.

    Returns:
        Dictionary of tag -> hash mappings.
    """
    refs_path = repo_path / "refs.json"
    if refs_path.exists():
        return read_json(refs_path)
    return {}


# ---------------------------------------------------------------------------
# Export / Import
# ---------------------------------------------------------------------------

def export_version(
    version_id: str,
    output_path: Union[str, Path],
    repo_path: Union[str, Path],
) -> Path:
    """Export a version as a gzipped tarball.

    Args:
        version_id: Version hash or tag name.
        output_path: Directory to save the tarball to.
        repo_path: Path to the FLUX repository.

    Returns:
        Path to the created tarball.

    Raises:
        VersionNotFoundError: If the version doesn't exist.
    """
    repo_path = Path(repo_path).resolve()
    output_path = Path(output_path).resolve()
    _validate_repo(repo_path)

    version_hash = resolve_version(version_id, repo_path)
    version_dir = repo_path / "versions" / version_hash

    if not version_dir.exists():
        raise VersionNotFoundError(f"Version directory not found: {version_hash}")

    ensure_directory(output_path)
    tarball_path = output_path / f"{version_hash}.tar.gz"

    with tarfile.open(str(tarball_path), "w:gz") as tar:
        tar.add(str(version_dir), arcname=version_hash)

    logger.info("Exported version %s to %s", version_hash, tarball_path)
    return tarball_path


def import_version(
    tarball_path: Union[str, Path],
    repo_path: Union[str, Path],
) -> str:
    """Import a version from a gzipped tarball.

    Args:
        tarball_path: Path to the .tar.gz file.
        repo_path: Path to the FLUX repository.

    Returns:
        The imported version hash.

    Raises:
        ImportError_: If the tarball is invalid, hash mismatches, or version already exists.
    """
    repo_path = Path(repo_path).resolve()
    tarball_path = Path(tarball_path).resolve()
    _validate_repo(repo_path)

    if not tarball_path.exists():
        raise FileNotFoundError(f"Tarball not found: {tarball_path}")

    lock_dir = repo_path / "locks"

    # Extract to a temp directory first to validate
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(str(tarball_path), "r:gz") as tar:
            # Security: check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ImportError_(
                        f"Unsafe path in tarball: {member.name}"
                    )
            tar.extractall(tmpdir)

        # Find the extracted directory (should be named as the hash)
        tmpdir_path = Path(tmpdir)
        extracted_dirs = [d for d in tmpdir_path.iterdir() if d.is_dir()]

        if len(extracted_dirs) != 1:
            raise ImportError_(
                f"Expected exactly one directory in tarball, found {len(extracted_dirs)}"
            )

        version_hash = extracted_dirs[0].name

        # Verify hash format (should be hex SHA-256)
        if len(version_hash) != 64 or not all(c in "0123456789abcdef" for c in version_hash):
            raise ImportError_(
                f"Invalid version hash in tarball: {version_hash}"
            )

        # Check if already exists
        target_dir = repo_path / "versions" / version_hash
        if target_dir.exists():
            raise ImportError_(
                f"Version {version_hash} already exists in repository."
            )

        # Move to repository
        with FileLock(lock_dir, "import_version"):
            import shutil
            shutil.copytree(str(extracted_dirs[0]), str(target_dir))

    logger.info("Imported version %s from %s", version_hash, tarball_path)
    return version_hash
