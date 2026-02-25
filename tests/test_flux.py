"""Unit tests for FLUX versioning system."""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure flux package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from flux.core.hasher import (
    canonicalize_config,
    compute_config_hash,
    compute_file_hash,
    compute_string_hash,
    compute_version_hash,
)
from flux.core.locker import FileLock
from flux.core.metrics import compute_metrics
from flux.core.preprocessor import apply_pipeline
from flux.core.repository import (
    create_version,
    export_version,
    get_version_by_tag,
    get_version_info,
    import_version,
    init_repo,
    list_tags,
    list_versions,
    load_version,
    resolve_version,
    tag_version,
)
from flux.core.comparator import compare_versions
from flux.exceptions import (
    InvalidConfigError,
    LockTimeoutError,
    PreprocessingError,
    RepositoryNotFoundError,
    TagNotFoundError,
    VersionNotFoundError,
)


class TestHasher(unittest.TestCase):
    """Tests for the hasher module."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_compute_file_hash(self):
        """Test SHA-256 hash computation of a file."""
        content = b"hello world\n"
        file_path = Path(self.tmpdir) / "test.txt"
        file_path.write_bytes(content)

        result = compute_file_hash(file_path)
        expected = hashlib.sha256(content).hexdigest()
        self.assertEqual(result, expected)

    def test_compute_file_hash_missing(self):
        """Test that missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            compute_file_hash(Path(self.tmpdir) / "nonexistent.txt")

    def test_canonicalize_config_deterministic(self):
        """Test that config canonicalization is deterministic regardless of key order."""
        config1 = {"b": 2, "a": 1, "c": {"z": 3, "y": 4}}
        config2 = {"a": 1, "c": {"y": 4, "z": 3}, "b": 2}
        self.assertEqual(canonicalize_config(config1), canonicalize_config(config2))

    def test_compute_config_hash(self):
        """Test config hash consistency."""
        config = {"pipeline": [{"step": "lowercase"}]}
        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA-256 hex length

    def test_compute_version_hash(self):
        """Test version hash = SHA-256(raw_hash + config_hash)."""
        raw_hash = "a" * 64
        config_hash = "b" * 64
        result = compute_version_hash(raw_hash, config_hash)
        expected = hashlib.sha256((raw_hash + config_hash).encode()).hexdigest()
        self.assertEqual(result, expected)


class TestPreprocessor(unittest.TestCase):
    """Tests for the preprocessor module."""

    def _make_df(self, texts, labels=None):
        data = {"text": texts}
        if labels:
            data["label"] = labels
        return pd.DataFrame(data)

    def test_lowercase(self):
        df = self._make_df(["Hello WORLD", "TEST"])
        result = apply_pipeline(df, [{"step": "lowercase"}])
        self.assertEqual(list(result["text"]), ["hello world", "test"])

    def test_tokenize_whitespace(self):
        df = self._make_df(["  hello   world  "])
        result = apply_pipeline(df, [{"step": "tokenize", "params": {"method": "whitespace"}}])
        self.assertEqual(result["text"].iloc[0], "hello world")

    def test_tokenize_regex(self):
        df = self._make_df(["hello, world! 123"])
        result = apply_pipeline(df, [{"step": "tokenize", "params": {"method": "regex"}}])
        self.assertEqual(result["text"].iloc[0], "hello world 123")

    def test_filter_by_length(self):
        df = self._make_df(["one", "one two three", "one two three four five six"])
        result = apply_pipeline(df, [{"step": "filter_by_length", "params": {"min_tokens": 2, "max_tokens": 4}}])
        self.assertEqual(len(result), 1)
        self.assertEqual(result["text"].iloc[0], "one two three")

    def test_remove_stopwords(self):
        df = self._make_df(["the cat is on the mat"])
        result = apply_pipeline(df, [{"step": "remove_stopwords", "params": {"language": "english"}}])
        self.assertNotIn("the", result["text"].iloc[0].split())
        self.assertIn("cat", result["text"].iloc[0].split())
        self.assertIn("mat", result["text"].iloc[0].split())

    def test_deduplicate(self):
        df = self._make_df(["hello", "world", "hello", "test"])
        result = apply_pipeline(df, [{"step": "deduplicate", "params": {"keep": "first"}}])
        self.assertEqual(len(result), 3)

    def test_pipeline_multi_step(self):
        df = self._make_df(["The CAT sat on the MAT", "The CAT sat on the MAT", "A DOG ran"])
        pipeline = [
            {"step": "lowercase"},
            {"step": "deduplicate"},
        ]
        result = apply_pipeline(df, pipeline)
        self.assertEqual(len(result), 2)
        self.assertEqual(result["text"].iloc[0], "the cat sat on the mat")

    def test_unknown_step(self):
        df = self._make_df(["test"])
        with self.assertRaises(InvalidConfigError):
            apply_pipeline(df, [{"step": "unknown_step"}])

    def test_missing_text_column(self):
        df = pd.DataFrame({"other": ["test"]})
        with self.assertRaises(PreprocessingError):
            apply_pipeline(df, [{"step": "lowercase"}])


class TestMetrics(unittest.TestCase):
    """Tests for the metrics module."""

    def test_basic_metrics(self):
        df = pd.DataFrame({"text": ["hello world", "foo bar baz", "hello world"]})
        m = compute_metrics(df)
        self.assertEqual(m["num_samples"], 3)
        self.assertEqual(m["num_unique_texts"], 2)
        self.assertIn("vocab_size", m)
        self.assertIn("avg_text_length", m)
        self.assertIn("created_at", m)
        self.assertIsNone(m["class_distribution"])

    def test_metrics_with_labels(self):
        df = pd.DataFrame({
            "text": ["a", "b", "c", "a"],
            "label": ["pos", "neg", "pos", "neg"],
        })
        m = compute_metrics(df)
        self.assertIsNotNone(m["class_distribution"])
        self.assertEqual(m["class_distribution"]["pos"], 2)
        self.assertEqual(m["class_distribution"]["neg"], 2)

    def test_metrics_tokenized(self):
        df = pd.DataFrame({"text": ["hello world foo", "bar baz"]})
        m = compute_metrics(df, is_tokenized=True)
        # avg_text_length should be avg token count
        self.assertAlmostEqual(m["avg_text_length"], 2.5)


class TestLocker(unittest.TestCase):
    """Tests for the locker module."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_lock_acquire_release(self):
        lock = FileLock(self.tmpdir, "test_lock")
        lock.acquire()
        self.assertTrue(lock._acquired)
        lock_file = Path(self.tmpdir) / "test_lock.lock"
        self.assertTrue(lock_file.exists())
        lock.release()
        self.assertFalse(lock_file.exists())

    def test_lock_context_manager(self):
        lock_file = Path(self.tmpdir) / "ctx_lock.lock"
        with FileLock(self.tmpdir, "ctx_lock") as lock:
            self.assertTrue(lock_file.exists())
        self.assertFalse(lock_file.exists())

    def test_lock_timeout(self):
        # Create a lock manually
        lock_file = Path(self.tmpdir) / "blocked.lock"
        lock_file.write_text('{"pid": 0}')

        with self.assertRaises(LockTimeoutError):
            FileLock(self.tmpdir, "blocked", timeout=0.5, retry_interval=0.1).acquire()


class TestRepository(unittest.TestCase):
    """Tests for the repository module."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.repo_path = Path(self.tmpdir) / "test_repo"
        init_repo(self.repo_path)

        # Create sample CSV
        self.csv_path = Path(self.tmpdir) / "reviews.csv"
        df = pd.DataFrame({
            "text": [
                "This movie is great!",
                "Terrible acting, bad plot.",
                "I loved the cinematography.",
                "Not worth watching.",
                "An absolute masterpiece!",
                "This movie is great!",  # duplicate
            ],
            "label": ["pos", "neg", "pos", "neg", "pos", "pos"],
        })
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_init_repo(self):
        self.assertTrue((self.repo_path / "versions").is_dir())
        self.assertTrue((self.repo_path / "locks").is_dir())
        self.assertTrue((self.repo_path / "refs.json").is_file())

    def test_init_repo_reuse(self):
        """Re-initializing should not fail."""
        result = init_repo(self.repo_path)
        self.assertEqual(result, self.repo_path.resolve())

    def test_create_version(self):
        config = {"pipeline": [{"step": "lowercase"}]}
        vh = create_version(self.csv_path, config, self.repo_path, user="test@example.com")
        self.assertEqual(len(vh), 64)

        version_dir = self.repo_path / "versions" / vh
        self.assertTrue(version_dir.exists())
        self.assertTrue((version_dir / "raw.csv").exists())
        self.assertTrue((version_dir / "config.json").exists())
        self.assertTrue((version_dir / "processed.csv").exists())
        self.assertTrue((version_dir / "metadata.json").exists())

    def test_create_version_idempotent(self):
        config = {"pipeline": [{"step": "lowercase"}]}
        vh1 = create_version(self.csv_path, config, self.repo_path)
        vh2 = create_version(self.csv_path, config, self.repo_path)
        self.assertEqual(vh1, vh2)

    def test_create_version_with_tags(self):
        config = {"pipeline": [{"step": "lowercase"}]}
        vh = create_version(self.csv_path, config, self.repo_path, tags=["main", "v1"])
        tags = list_tags(self.repo_path)
        self.assertEqual(tags["main"], vh)
        self.assertEqual(tags["v1"], vh)

    def test_list_versions(self):
        config = {"pipeline": [{"step": "lowercase"}]}
        create_version(self.csv_path, config, self.repo_path)
        versions = list_versions(self.repo_path)
        self.assertEqual(len(versions), 1)

    def test_load_version(self):
        config = {"pipeline": [{"step": "lowercase"}]}
        vh = create_version(self.csv_path, config, self.repo_path)
        df = load_version(vh, self.repo_path, data_type="processed")
        self.assertIn("text", df.columns)
        # All text should be lowercase
        for text in df["text"]:
            self.assertEqual(text, text.lower())

    def test_load_version_raw(self):
        config = {"pipeline": [{"step": "lowercase"}]}
        vh = create_version(self.csv_path, config, self.repo_path)
        df = load_version(vh, self.repo_path, data_type="raw")
        self.assertIn("text", df.columns)

    def test_load_version_by_tag(self):
        config = {"pipeline": [{"step": "lowercase"}]}
        vh = create_version(self.csv_path, config, self.repo_path, tags=["main"])
        df = load_version("main", self.repo_path)
        self.assertIn("text", df.columns)

    def test_tag_version(self):
        config = {"pipeline": [{"step": "lowercase"}]}
        vh = create_version(self.csv_path, config, self.repo_path)
        tag_version(vh, "release-1", self.repo_path)
        result = get_version_by_tag("release-1", self.repo_path)
        self.assertEqual(result, vh)

    def test_tag_overwrite(self):
        config1 = {"pipeline": [{"step": "lowercase"}]}
        config2 = {"pipeline": [{"step": "lowercase"}, {"step": "deduplicate"}]}
        vh1 = create_version(self.csv_path, config1, self.repo_path)
        vh2 = create_version(self.csv_path, config2, self.repo_path)
        tag_version(vh1, "latest", self.repo_path)
        tag_version(vh2, "latest", self.repo_path)
        result = get_version_by_tag("latest", self.repo_path)
        self.assertEqual(result, vh2)

    def test_tag_not_found(self):
        with self.assertRaises(TagNotFoundError):
            get_version_by_tag("nonexistent", self.repo_path)

    def test_version_not_found(self):
        with self.assertRaises(VersionNotFoundError):
            resolve_version("nonexistent_hash", self.repo_path)

    def test_version_info(self):
        config = {"pipeline": [{"step": "lowercase"}]}
        vh = create_version(self.csv_path, config, self.repo_path, user="tester")
        info = get_version_info(vh, self.repo_path)
        self.assertEqual(info.version_hash, vh)
        self.assertIn("num_samples", info.metrics)
        summary = info.summary()
        self.assertIn(vh, summary)

    def test_export_import(self):
        config = {"pipeline": [{"step": "lowercase"}]}
        vh = create_version(self.csv_path, config, self.repo_path)

        # Export
        export_dir = Path(self.tmpdir) / "exports"
        tarball = export_version(vh, export_dir, self.repo_path)
        self.assertTrue(tarball.exists())

        # Create a new repo and import
        repo2 = Path(self.tmpdir) / "repo2"
        init_repo(repo2)
        imported_hash = import_version(tarball, repo2)
        self.assertEqual(imported_hash, vh)

        # Verify data
        df = load_version(imported_hash, repo2)
        self.assertIn("text", df.columns)

    def test_compare_versions(self):
        config1 = {"pipeline": [{"step": "lowercase"}]}
        config2 = {"pipeline": [{"step": "lowercase"}, {"step": "deduplicate"}]}
        vh1 = create_version(self.csv_path, config1, self.repo_path)
        vh2 = create_version(self.csv_path, config2, self.repo_path)
        report = compare_versions(vh1, vh2, self.repo_path)
        self.assertIn("Comparison Report", report)
        self.assertIn("Jaccard", report)
        self.assertIn("num_samples", report)

    def test_repo_not_found(self):
        with self.assertRaises(RepositoryNotFoundError):
            list_versions(Path(self.tmpdir) / "nonexistent_repo")

    def test_full_pipeline(self):
        """End-to-end test with multi-step pipeline."""
        config = {
            "pipeline": [
                {"step": "lowercase"},
                {"step": "tokenize", "params": {"method": "regex"}},
                {"step": "remove_stopwords", "params": {"language": "english"}},
                {"step": "filter_by_length", "params": {"min_tokens": 2}},
                {"step": "deduplicate"},
            ]
        }
        vh = create_version(
            self.csv_path, config, self.repo_path,
            user="tester@test.com", tags=["full-pipeline"]
        )

        # Verify version exists
        info = get_version_info(vh, self.repo_path)
        self.assertTrue(info.metrics["num_samples"] > 0)

        # Verify can load by tag
        df = load_version("full-pipeline", self.repo_path)
        self.assertGreater(len(df), 0)


if __name__ == "__main__":
    unittest.main()
