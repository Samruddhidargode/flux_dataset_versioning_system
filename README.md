# FLUX — Dataset Versioning Made Simple

```
  ███████╗██╗     ██╗   ██╗██╗  ██╗
  ██╔════╝██║     ██║   ██║╚██╗██╔╝
  █████╗  ██║     ██║   ██║ ╚███╔╝
  ██╔══╝  ██║     ██║   ██║ ██╔██╗
  ██║     ███████╗╚██████╔╝██╔╝ ██╗
  ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝
```

**File-based Lightweight Universal Xplainable Dataset Versioning System**

---

## What is FLUX?

FLUX is a **lightweight, file-based versioning system for text datasets** — think Git, but purpose-built for NLP/ML data pipelines.

In machine learning projects, datasets evolve constantly — you clean text, remove stopwords, deduplicate entries, filter by length. But tracking *which version* of data was used for *which experiment* is a nightmare. FLUX solves this.

### The Problem

- You modify a dataset and lose the original
- You can't reproduce last week's training run
- You don't know which preprocessing was applied to which file
- Two teammates preprocess differently with no way to compare

### How FLUX Solves It

| Feature | How It Works |
|---|---|
| **Immutable Versions** | Every upload creates a snapshot that can never be changed |
| **Content-Addressed** | Version ID = SHA-256 hash of (file content + preprocessing config) |
| **Idempotent** | Upload the same file with the same config → same version, no duplicates |
| **Preprocessing Pipeline** | Built-in text preprocessing (lowercase, tokenize, stopwords, dedup, filter) |
| **Diff & Compare** | Side-by-side comparison of any two versions (config diff, metrics diff, data overlap) |
| **Tags** | Name your versions (`production`, `v1`, `experiment-3`) for easy reference |
| **Export/Import** | Share versions as `.tar.gz` tarballs across machines |
| **Zero Dependencies on Servers** | Everything is stored as files — no database, no cloud, no setup |

---

## Architecture

```
.flux/          (created by `flux init`)
├── versions/
│   └── <sha256-hash>/
│       ├── raw.csv           # Original uploaded file
│       ├── processed.csv     # After preprocessing pipeline
│       ├── config.json       # Preprocessing config used
│       └── metrics.json      # Auto-computed dataset statistics
├── refs.json                 # Tag → version hash mapping
└── locks/                    # File-based locking for concurrency
```

### Hashing Strategy

```
version_hash = SHA-256(raw_file_hash + config_hash)
```

- `raw_file_hash` = SHA-256 of the CSV file contents
- `config_hash` = SHA-256 of the canonicalized (sorted, deterministic) JSON config
- Same file + same config = **always the same version hash** (idempotent)

---

## Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/flux.git
cd flux

# Install in development mode
pip install -e .
```

**Requirements:** Python 3.10+, pandas, numpy

---

## Quick Start

### 1. Initialize a Repository

```bash
python -m flux init
```

```
  [OK] Initialized FLUX repository at: C:\Users\WCE
```

### 2. Upload a Dataset

```bash
python -m flux upload examples/reviews.csv -t "v1-raw" -u "your-name" -y
```

This uploads the CSV, applies preprocessing (interactively or via config), and creates an immutable version:

```
  [OK] Dataset loaded: examples/reviews.csv
    Rows:    20
    Columns: text, label
    Labels:  pos: 10, neg: 10

    Preview (first 3 rows):
    1. This movie is absolutely fantastic! I loved every minute of it. [pos]
    2. Terrible film. Waste of time and money. [neg]
    3. The acting was superb and the plot kept me engaged throughout. [pos]

  Creating version...

  [OK] Version created successfully!
    Hash: c7db5f01ad662fb0...
    Tag:  v1-raw

    Quick Stats:
    Samples:     20
    Unique:      20
    Vocab Size:  121
    Avg Length:  50.2
```

### 3. Upload with a Different Config

```bash
python -m flux upload examples/reviews.csv -c examples/config_full.json -t "v2-processed" -u "your-name" -y
```

The `config_full.json` applies a full pipeline: lowercase → tokenize → remove stopwords → filter by length → deduplicate. This produces a **different version** (different hash) because the preprocessing changed.

### 4. List All Versions

```bash
python -m flux list
```

```
  HASH                 SAMPLES  TAGS              CREATED
  ─────────────────────────────────────────────────────────
  c7db5f01ad66              20  v1-raw            2026-02-25T17:05:27
  ef943ab42990              19  v2-processed      2026-02-25T17:06:12

  Total: 2 version(s)
```

### 5. Show Version Details

```bash
python -m flux show v1-raw -p
```

```
  Version: c7db5f01ad662fb0d768b43d4f6dea5f...
  Tags:    v1-raw
  Raw Hash:    592d10acad71...
  Config Hash: 3f9211e7fd99...

  Metrics:
    Samples:      20
    Unique Texts: 20
    Vocab Size:   121
    Avg Length:   50.2
    Created By:   your-name
    Classes:      {'pos': 10, 'neg': 10}

  Preprocessing Pipeline:
    1. lowercase

  Data Preview (first 5 rows):
    1. this movie is absolutely fantastic! i loved every minute of it.  [pos]
    2. terrible film. waste of time and money.  [neg]
    3. the acting was superb and the plot kept me engaged throughout.  [pos]
```

### 6. Compare Two Versions

```bash
python -m flux diff v1-raw v2-processed
```

Shows:
- **Config diff** — unified diff of preprocessing pipelines
- **Metrics diff** — changes in sample count, vocab size, avg length (with % change)
- **Data overlap** — Jaccard similarity, common rows, example differences

### 7. Tag a Version

```bash
python -m flux tag v1-raw production
```

```
  [OK] Tagged v1-raw as production
```

### 8. Export & Import

```bash
# Export a version as a portable tarball
python -m flux export v1-raw ./backups

# Import on another machine
python -m flux import backups/v1-raw.tar.gz
```

### 9. Web Dashboard

```bash
python -m flux.web.app
```

Opens a browser dashboard at `http://127.0.0.1:5000` with:

- **Version Lineage Graph** — visual timeline of all versions with tags, sample counts, and pipeline info
- **Version Detail Page** — full metrics, hashes, pipeline steps, config JSON, and data preview table
- **Compare Page** — select any two versions, see color-coded config diff, metric deltas with % change, and Jaccard similarity bar
- **REST API** — `/api/versions` and `/api/compare?v1=...&v2=...` for programmatic access

```bash
# Custom port and repo path
python -m flux.web.app --port 8080 --repo /path/to/repo
```

---

## Preprocessing Pipeline

FLUX has 5 built-in preprocessing steps that can be combined in any order:

| Step | Description | Parameters |
|---|---|---|
| `lowercase` | Convert all text to lowercase | — |
| `tokenize` | Split text into tokens | `method`: `whitespace` or `regex` |
| `remove_stopwords` | Remove common English stop words | `language`: `english`, optional `custom_words` list |
| `filter_by_length` | Keep only rows within token count range | `min_tokens`, `max_tokens` |
| `deduplicate` | Remove duplicate text rows | `keep`: `first` or `last`, optional `subset` columns |

### Interactive Mode

Run `upload` without `-c` to get an interactive pipeline builder:

```bash
python -m flux upload data.csv
```

FLUX walks you through each step, asking which ones to enable and what parameters to use.

### Config File Mode

Create a JSON config:

```json
{
  "pipeline": [
    { "step": "lowercase" },
    { "step": "tokenize", "params": { "method": "regex" } },
    { "step": "remove_stopwords", "params": { "language": "english" } },
    { "step": "filter_by_length", "params": { "min_tokens": 3 } },
    { "step": "deduplicate", "params": { "keep": "first" } }
  ]
}
```

```bash
python -m flux upload data.csv -c config.json
```

---

## Command Reference

| Command | Description |
|---|---|
| `flux init [path]` | Initialize a new FLUX repository |
| `flux upload <file.csv>` | Upload dataset with interactive pipeline builder |
| `flux upload <file> -c <config.json>` | Upload with a preset config file |
| `flux list` | List all versions with tags and stats |
| `flux show <version_or_tag>` | Show version details (add `-p` for data preview) |
| `flux diff <v1> <v2>` | Compare two versions side-by-side |
| `flux tag <version> <name>` | Assign a tag name to a version |
| `flux tags` | List all tags |
| `flux export <version> <dir>` | Export version as `.tar.gz` |
| `flux import <tarball>` | Import version from `.tar.gz` |

### Flags

| Flag | Description |
|---|---|
| `-c <config.json>` | Use JSON config (skips interactive prompts) |
| `-t <tag>` | Tag this version on creation |
| `-u <user>` | Record who created this version |
| `-y` | Skip confirmation prompts |
| `-p` | Show data preview (with `show` command) |
| `-r <path>` | Use a specific repo path |
| `-v` | Verbose logging |

---

## Dataset Format

FLUX expects CSV files with at minimum a `text` column:

```csv
text,label
"This movie is fantastic!",pos
"Terrible film.",neg
```

- `text` column — **required** (the text data to version and preprocess)
- `label` column — **optional** (used for class distribution metrics)

---

## Key Design Principles

1. **Immutability** — Once created, a version can never be modified
2. **Content-Addressed Storage** — Version identity is derived from content, not timestamps
3. **Idempotency** — Same input always produces the same version (no duplicates)
4. **Explainability** — Every version stores its exact preprocessing config alongside the data
5. **Reproducibility** — Re-apply the same config to the same data → identical output
6. **Zero Infrastructure** — No database, no server — just files on disk
7. **Concurrency Safe** — File-based locking prevents race conditions

---

## Project Structure

```
flux/
├── __init__.py              # Package root, version info
├── __main__.py              # Enables `python -m flux`
├── exceptions.py            # Custom exception classes
├── core/
│   ├── hasher.py            # SHA-256 hashing (file, config, version)
│   ├── locker.py            # File-based locking with exponential backoff
│   ├── preprocessor.py      # Text preprocessing pipeline engine
│   ├── metrics.py           # Auto-computed dataset statistics
│   ├── repository.py        # Core version CRUD operations
│   └── comparator.py        # Version diff/comparison engine
├── models/
│   └── version.py           # VersionInfo & ComparisonReport dataclasses
├── cli/
│   ├── main.py              # CLI entry point with all commands
│   └── interactive.py       # Interactive prompts & pipeline builder
├── web/
│   ├── app.py               # Flask web dashboard & REST API
│   └── templates/           # HTML templates (index, version, compare)
└── utils/
    └── file_utils.py        # File I/O helpers

tests/
└── test_flux.py             # 38 unit tests (hasher, preprocessor, metrics, locker, repository)

examples/
├── reviews.csv              # Sample dataset (20 movie reviews, pos/neg)
├── config_lower.json        # Lowercase-only pipeline
├── config_lower_stop.json   # Lowercase + stopword removal
└── config_full.json         # Full pipeline (5 steps)
```

---

## Running Tests

```bash
python -m pytest tests/test_flux.py -v
```

```
38 passed ✓
```

Covers: hashing, preprocessing (all 5 steps), metrics computation, file locking, repository operations (create, idempotency, tags, export/import, compare).

---

## License

MIT

---

*Built as a prototype for explainable, reproducible dataset versioning in NLP/ML workflows.*
