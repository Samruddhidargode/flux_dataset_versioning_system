"""Metrics computation for FLUX versioning system.

Computes summary statistics from a processed DataFrame and stores
them as structured metadata.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def compute_metrics(
    df: pd.DataFrame,
    user: Optional[str] = None,
    is_tokenized: bool = False,
) -> Dict[str, Any]:
    """Compute metrics from a processed DataFrame.

    Args:
        df: Processed DataFrame. Must contain a 'text' column.
        user: Optional user identifier (email, name, etc.).
        is_tokenized: Whether the text has been tokenized (affects how
            avg_text_length and vocab_size are computed).

    Returns:
        Dictionary of computed metrics.
    """
    metrics: Dict[str, Any] = {}

    # Number of samples
    metrics["num_samples"] = len(df)

    # Number of unique texts
    if "text" in df.columns:
        metrics["num_unique_texts"] = int(df["text"].nunique())

        # Vocabulary size and average text length
        texts = df["text"].astype(str)

        if is_tokenized:
            # Text is already tokenized (space-separated tokens)
            all_tokens = texts.str.split().explode()
            metrics["vocab_size"] = int(all_tokens.nunique())
            metrics["avg_text_length"] = float(
                texts.apply(lambda x: len(x.split())).mean()
            )
        else:
            # Approximate via simple whitespace split
            all_tokens = texts.str.split().explode()
            metrics["vocab_size"] = int(all_tokens.nunique())
            metrics["avg_text_length"] = float(texts.str.len().mean())
    else:
        metrics["num_unique_texts"] = 0
        metrics["vocab_size"] = 0
        metrics["avg_text_length"] = 0.0

    # Class distribution (if 'label' column exists)
    if "label" in df.columns:
        class_dist = df["label"].value_counts().to_dict()
        # Convert keys to strings for JSON serialization
        metrics["class_distribution"] = {str(k): int(v) for k, v in class_dist.items()}
    else:
        metrics["class_distribution"] = None

    # Timestamps and user
    metrics["created_at"] = datetime.now(timezone.utc).isoformat()
    metrics["created_by"] = user

    logger.info(
        "Metrics computed: %d samples, %d unique texts, vocab=%d, avg_len=%.2f",
        metrics["num_samples"],
        metrics["num_unique_texts"],
        metrics["vocab_size"],
        metrics["avg_text_length"],
    )

    return metrics
