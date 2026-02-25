"""Data classes and models for FLUX versioning system."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VersionInfo:
    """Represents metadata about a dataset version.

    Attributes:
        version_hash: The SHA-256 version identifier.
        raw_hash: Hash of the raw data file.
        config_hash: Hash of the preprocessing config.
        config: The preprocessing configuration dict.
        metrics: Computed metrics dictionary.
        tags: List of tag names pointing to this version.
    """
    version_hash: str
    raw_hash: str = ""
    config_hash: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary of this version.

        Returns:
            Multi-line string summary.
        """
        lines = [
            f"Version: {self.version_hash}",
            f"  Raw Hash:    {self.raw_hash}",
            f"  Config Hash: {self.config_hash}",
        ]
        if self.tags:
            lines.append(f"  Tags:        {', '.join(self.tags)}")
        if self.metrics:
            lines.append(f"  Samples:     {self.metrics.get('num_samples', 'N/A')}")
            lines.append(f"  Unique Texts:{self.metrics.get('num_unique_texts', 'N/A')}")
            lines.append(f"  Vocab Size:  {self.metrics.get('vocab_size', 'N/A')}")
            lines.append(f"  Avg Length:  {self.metrics.get('avg_text_length', 'N/A')}")
            lines.append(f"  Created At:  {self.metrics.get('created_at', 'N/A')}")
            lines.append(f"  Created By:  {self.metrics.get('created_by', 'N/A')}")
            cd = self.metrics.get("class_distribution")
            if cd:
                lines.append(f"  Classes:     {cd}")
        return "\n".join(lines)


@dataclass
class ComparisonReport:
    """Represents a comparison between two dataset versions.

    Attributes:
        v1_hash: Hash of the first version.
        v2_hash: Hash of the second version.
        config_diff: Textual diff of configurations.
        metrics_diff: Dictionary of metric differences.
        data_overlap: Dictionary with overlap statistics.
    """
    v1_hash: str
    v2_hash: str
    config_diff: str = ""
    metrics_diff: Dict[str, Any] = field(default_factory=dict)
    data_overlap: Dict[str, Any] = field(default_factory=dict)

    def to_string(self) -> str:
        """Format comparison report as a readable string.

        Returns:
            Multi-line string report.
        """
        lines = [
            "=" * 60,
            "FLUX Version Comparison Report",
            "=" * 60,
            f"Version 1: {self.v1_hash}",
            f"Version 2: {self.v2_hash}",
            "",
            "--- Configuration Diff ---",
            self.config_diff if self.config_diff else "(identical)",
            "",
            "--- Metrics Diff ---",
        ]

        if self.metrics_diff:
            for key, diff in self.metrics_diff.items():
                lines.append(f"  {key}:")
                lines.append(f"    v1: {diff.get('v1', 'N/A')}")
                lines.append(f"    v2: {diff.get('v2', 'N/A')}")
                if "change" in diff:
                    lines.append(f"    change: {diff['change']}")
                if "pct_change" in diff:
                    lines.append(f"    pct_change: {diff['pct_change']:.2f}%")
        else:
            lines.append("  (identical)")

        lines.append("")
        lines.append("--- Data Overlap ---")

        if self.data_overlap:
            lines.append(f"  Jaccard Similarity: {self.data_overlap.get('jaccard_similarity', 'N/A'):.4f}")
            lines.append(f"  Common Rows:        {self.data_overlap.get('common_rows', 'N/A')}")
            lines.append(f"  Only in V1:         {self.data_overlap.get('only_in_v1', 'N/A')}")
            lines.append(f"  Only in V2:         {self.data_overlap.get('only_in_v2', 'N/A')}")

            examples_v1 = self.data_overlap.get("examples_only_v1", [])
            examples_v2 = self.data_overlap.get("examples_only_v2", [])

            if examples_v1:
                lines.append(f"  Example rows only in V1 (up to 5):")
                for ex in examples_v1[:5]:
                    lines.append(f"    - {ex}")
            if examples_v2:
                lines.append(f"  Example rows only in V2 (up to 5):")
                for ex in examples_v2[:5]:
                    lines.append(f"    - {ex}")
        else:
            lines.append("  (not computed)")

        lines.append("=" * 60)
        return "\n".join(lines)
