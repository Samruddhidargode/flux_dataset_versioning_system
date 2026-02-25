"""Preprocessing pipeline for FLUX versioning system.

Provides built-in text preprocessing steps that are applied sequentially
to a pandas DataFrame. Steps include lowercasing, tokenization, filtering,
stopword removal, and deduplication.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from flux.exceptions import InvalidConfigError, PreprocessingError

logger = logging.getLogger(__name__)

# Built-in stopword lists (small subsets for lightweight operation)
STOPWORDS: Dict[str, Set[str]] = {
    "english": {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "need", "dare",
        "ought", "used", "it", "its", "this", "that", "these", "those",
        "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your",
        "yours", "he", "him", "his", "she", "her", "hers", "they", "them",
        "their", "theirs", "what", "which", "who", "whom", "whose",
        "not", "no", "nor", "so", "too", "very", "just", "about", "above",
        "after", "again", "all", "also", "am", "any", "because", "before",
        "below", "between", "both", "each", "few", "further", "here",
        "how", "if", "into", "more", "most", "much", "must", "now", "only",
        "other", "out", "over", "own", "same", "some", "such", "than",
        "then", "there", "through", "under", "until", "up", "when", "where",
        "while", "why", "down", "during", "off",
    },
}


def apply_pipeline(
    df: pd.DataFrame,
    pipeline: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Apply a sequence of preprocessing steps to a DataFrame.

    Args:
        df: Input DataFrame. Must contain a 'text' column.
        pipeline: List of step dictionaries, each with 'step' and optional 'params'.

    Returns:
        Preprocessed DataFrame.

    Raises:
        PreprocessingError: If a step fails.
        InvalidConfigError: If a step name is unknown or params are invalid.
    """
    if "text" not in df.columns:
        raise PreprocessingError("DataFrame must contain a 'text' column.")

    df = df.copy()

    for i, step_def in enumerate(pipeline):
        step_name = step_def.get("step")
        params = step_def.get("params", {})

        if step_name is None:
            raise InvalidConfigError(f"Step {i} is missing the 'step' field.")

        logger.info("Applying step %d: %s (params=%s)", i, step_name, params)

        try:
            if step_name == "lowercase":
                df = _step_lowercase(df, params)
            elif step_name == "tokenize":
                df = _step_tokenize(df, params)
            elif step_name == "filter_by_length":
                df = _step_filter_by_length(df, params)
            elif step_name == "remove_stopwords":
                df = _step_remove_stopwords(df, params)
            elif step_name == "deduplicate":
                df = _step_deduplicate(df, params)
            else:
                raise InvalidConfigError(f"Unknown preprocessing step: '{step_name}'")
        except (InvalidConfigError, PreprocessingError):
            raise
        except Exception as e:
            raise PreprocessingError(
                f"Step {i} ('{step_name}') failed: {e}"
            ) from e

        logger.debug("After step %d (%s): %d rows", i, step_name, len(df))

    return df.reset_index(drop=True)


def _step_lowercase(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Convert text column to lowercase.

    Args:
        df: Input DataFrame.
        params: No parameters required.

    Returns:
        DataFrame with lowercased text.
    """
    df["text"] = df["text"].astype(str).str.lower()
    return df


def _step_tokenize(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Tokenize text column.

    Args:
        df: Input DataFrame.
        params: Dictionary with optional 'method' key.
            - 'whitespace': split on whitespace (default).
            - 'regex': split on non-alphanumeric characters.

    Returns:
        DataFrame with tokenized text (space-separated tokens).
    """
    method = params.get("method", "whitespace")

    if method == "whitespace":
        df["text"] = df["text"].astype(str).apply(lambda x: " ".join(x.split()))
    elif method == "regex":
        df["text"] = df["text"].astype(str).apply(
            lambda x: " ".join(re.findall(r"[a-zA-Z0-9]+", x))
        )
    else:
        raise InvalidConfigError(f"Unknown tokenization method: '{method}'")

    return df


def _step_filter_by_length(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Filter rows by token count.

    Args:
        df: Input DataFrame.
        params: Dictionary with 'min_tokens' and/or 'max_tokens'.

    Returns:
        Filtered DataFrame.
    """
    min_tokens = params.get("min_tokens", 0)
    max_tokens = params.get("max_tokens", float("inf"))

    token_counts = df["text"].astype(str).apply(lambda x: len(x.split()))

    mask = (token_counts >= min_tokens) & (token_counts <= max_tokens)
    return df[mask]


def _step_remove_stopwords(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Remove stopwords from text.

    Args:
        df: Input DataFrame.
        params: Dictionary with 'language' (str) and/or 'custom_list' (list of str).

    Returns:
        DataFrame with stopwords removed.
    """
    stop_set: Set[str] = set()

    language = params.get("language")
    if language:
        if language not in STOPWORDS:
            raise InvalidConfigError(
                f"Unknown stopword language: '{language}'. "
                f"Available: {list(STOPWORDS.keys())}"
            )
        stop_set.update(STOPWORDS[language])

    custom_list = params.get("custom_list")
    if custom_list:
        stop_set.update(custom_list)

    if not stop_set:
        logger.warning("No stopwords specified; step has no effect.")
        return df

    def remove_stops(text: str) -> str:
        tokens = text.split()
        return " ".join(t for t in tokens if t.lower() not in stop_set)

    df["text"] = df["text"].astype(str).apply(remove_stops)
    return df


def _step_deduplicate(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Remove duplicate rows based on specified columns.

    Args:
        df: Input DataFrame.
        params: Dictionary with optional:
            - 'keep': 'first' or 'last' (default 'first').
            - 'subset': column or list of columns to consider (default ['text']).

    Returns:
        Deduplicated DataFrame.
    """
    keep = params.get("keep", "first")
    subset = params.get("subset", ["text"])

    if isinstance(subset, str):
        subset = [subset]

    # Validate columns exist
    for col in subset:
        if col not in df.columns:
            raise InvalidConfigError(
                f"Column '{col}' not found in DataFrame for deduplication. "
                f"Available columns: {list(df.columns)}"
            )

    return df.drop_duplicates(subset=subset, keep=keep)
