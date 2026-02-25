"""Version comparison for FLUX versioning system.

Compares two dataset versions by analyzing their configs, metrics,
and processed data overlap.
"""

import difflib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from flux.core.repository import get_version_info, load_version, resolve_version, _validate_repo
from flux.models.version import ComparisonReport
from flux.utils.file_utils import read_json

logger = logging.getLogger(__name__)


def compare_versions(
    v1: str,
    v2: str,
    repo_path: Union[str, Path],
) -> str:
    """Compare two dataset versions and generate a human-readable report.

    Args:
        v1: First version hash or tag.
        v2: Second version hash or tag.
        repo_path: Path to the FLUX repository.

    Returns:
        Formatted comparison report string.
    """
    repo_path = Path(repo_path).resolve()
    _validate_repo(repo_path)

    # Resolve to hashes
    v1_hash = resolve_version(v1, repo_path)
    v2_hash = resolve_version(v2, repo_path)

    # Load metadata and configs
    v1_dir = repo_path / "versions" / v1_hash
    v2_dir = repo_path / "versions" / v2_hash

    v1_config = read_json(v1_dir / "config.json") if (v1_dir / "config.json").exists() else {}
    v2_config = read_json(v2_dir / "config.json") if (v2_dir / "config.json").exists() else {}

    v1_meta = read_json(v1_dir / "metadata.json") if (v1_dir / "metadata.json").exists() else {}
    v2_meta = read_json(v2_dir / "metadata.json") if (v2_dir / "metadata.json").exists() else {}

    # Compute config diff
    config_diff = _compute_config_diff(v1_config, v2_config)

    # Compute metrics diff
    metrics_diff = _compute_metrics_diff(v1_meta, v2_meta)

    # Compute data overlap
    data_overlap = _compute_data_overlap(v1_hash, v2_hash, repo_path)

    # Build report
    report = ComparisonReport(
        v1_hash=v1_hash,
        v2_hash=v2_hash,
        config_diff=config_diff,
        metrics_diff=metrics_diff,
        data_overlap=data_overlap,
    )

    return report.to_string()


def compare_versions_raw(
    v1: str,
    v2: str,
    repo_path: Union[str, Path],
) -> ComparisonReport:
    """Compare two versions and return the raw ComparisonReport object.

    Same as compare_versions but returns the structured object
    instead of a formatted string.
    """
    repo_path = Path(repo_path).resolve()
    _validate_repo(repo_path)

    v1_hash = resolve_version(v1, repo_path)
    v2_hash = resolve_version(v2, repo_path)

    v1_dir = repo_path / "versions" / v1_hash
    v2_dir = repo_path / "versions" / v2_hash

    v1_config = read_json(v1_dir / "config.json") if (v1_dir / "config.json").exists() else {}
    v2_config = read_json(v2_dir / "config.json") if (v2_dir / "config.json").exists() else {}

    v1_meta = read_json(v1_dir / "metadata.json") if (v1_dir / "metadata.json").exists() else {}
    v2_meta = read_json(v2_dir / "metadata.json") if (v2_dir / "metadata.json").exists() else {}

    config_diff = _compute_config_diff(v1_config, v2_config)
    metrics_diff = _compute_metrics_diff(v1_meta, v2_meta)
    data_overlap = _compute_data_overlap(v1_hash, v2_hash, repo_path)

    return ComparisonReport(
        v1_hash=v1_hash,
        v2_hash=v2_hash,
        config_diff=config_diff,
        metrics_diff=metrics_diff,
        data_overlap=data_overlap,
    )


def _compute_config_diff(config1: Dict[str, Any], config2: Dict[str, Any]) -> str:
    """Compute a textual diff between two config dictionaries.

    Args:
        config1: First config dict.
        config2: Second config dict.

    Returns:
        Unified diff string, or empty if identical.
    """
    json1 = json.dumps(config1, indent=2, sort_keys=True).splitlines(keepends=True)
    json2 = json.dumps(config2, indent=2, sort_keys=True).splitlines(keepends=True)

    diff = difflib.unified_diff(
        json1, json2,
        fromfile="v1/config.json",
        tofile="v2/config.json",
    )
    return "".join(diff).strip()


def _compute_metrics_diff(
    meta1: Dict[str, Any],
    meta2: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute differences between two metadata dictionaries.

    Args:
        meta1: First version's metadata.
        meta2: Second version's metadata.

    Returns:
        Dictionary of metric differences with absolute and percentage changes.
    """
    numeric_keys = ["num_samples", "num_unique_texts", "vocab_size", "avg_text_length"]
    diff: Dict[str, Any] = {}

    for key in numeric_keys:
        v1_val = meta1.get(key)
        v2_val = meta2.get(key)

        if v1_val is None and v2_val is None:
            continue

        entry: Dict[str, Any] = {"v1": v1_val, "v2": v2_val}

        if v1_val is not None and v2_val is not None:
            try:
                change = float(v2_val) - float(v1_val)
                entry["change"] = change
                if float(v1_val) != 0:
                    entry["pct_change"] = (change / float(v1_val)) * 100
            except (ValueError, TypeError):
                pass

        diff[key] = entry

    # Class distribution diff
    cd1 = meta1.get("class_distribution")
    cd2 = meta2.get("class_distribution")
    if cd1 is not None or cd2 is not None:
        diff["class_distribution"] = {"v1": cd1, "v2": cd2}

    return diff


def _compute_data_overlap(
    v1_hash: str,
    v2_hash: str,
    repo_path: Path,
) -> Dict[str, Any]:
    """Compute data overlap between two processed datasets.

    Args:
        v1_hash: First version hash.
        v2_hash: Second version hash.
        repo_path: Path to the FLUX repository.

    Returns:
        Dictionary with overlap statistics.
    """
    try:
        df1 = load_version(v1_hash, repo_path, data_type="processed")
        df2 = load_version(v2_hash, repo_path, data_type="processed")
    except Exception as e:
        logger.warning("Could not load data for overlap computation: %s", e)
        return {}

    if "text" not in df1.columns or "text" not in df2.columns:
        return {}

    texts1 = set(df1["text"].astype(str).tolist())
    texts2 = set(df2["text"].astype(str).tolist())

    intersection = texts1 & texts2
    union = texts1 | texts2
    only_v1 = texts1 - texts2
    only_v2 = texts2 - texts1

    jaccard = len(intersection) / len(union) if union else 1.0

    result: Dict[str, Any] = {
        "jaccard_similarity": jaccard,
        "common_rows": len(intersection),
        "only_in_v1": len(only_v1),
        "only_in_v2": len(only_v2),
    }

    # Up to 5 example rows from each side
    result["examples_only_v1"] = list(only_v1)[:5]
    result["examples_only_v2"] = list(only_v2)[:5]

    return result
