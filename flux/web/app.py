"""FLUX Web Dashboard — Version lineage, comparison, and data exploration.

Run with:
    python -m flux.web.app
    # or
    python -m flux.web.app --repo /path/to/repo --port 5000
"""

import json
import os
import sys
from pathlib import Path

from flask import Flask, render_template, request, jsonify, abort

# Ensure the project root is importable
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from flux.core.repository import (
    list_versions,
    get_version_info,
    load_version,
    list_tags,
    init_repo,
)
from flux.core.comparator import compare_versions_raw
from flux.utils.file_utils import is_flux_repo

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)

# Default repo path — can be overridden via env or CLI arg
REPO_PATH = os.environ.get("FLUX_REPO", os.getcwd())


def _repo():
    """Get the current repo path."""
    return REPO_PATH


# ---- Pages ----

@app.route("/")
def index():
    """Dashboard home — version list + lineage graph."""
    if not is_flux_repo(_repo()):
        return render_template("no_repo.html", path=_repo())

    versions = list_versions(_repo())
    tags = list_tags(_repo())

    # Build timeline data for the lineage graph
    timeline = []
    for v in versions:
        created = str(v.metrics.get("created_at", ""))[:19]
        timeline.append({
            "hash": v.version_hash,
            "short_hash": v.version_hash[:12],
            "tags": v.tags,
            "samples": v.metrics.get("num_samples", 0),
            "unique": v.metrics.get("num_unique_texts", 0),
            "vocab": v.metrics.get("vocab_size", 0),
            "avg_len": round(v.metrics.get("avg_text_length", 0), 1),
            "created": created,
            "created_by": v.metrics.get("created_by", "-"),
            "pipeline_steps": len(v.config.get("pipeline", [])),
            "raw_hash": v.raw_hash[:12],
            "config_hash": v.config_hash[:12],
        })

    return render_template("index.html", versions=timeline, tags=tags)


@app.route("/version/<version_id>")
def version_detail(version_id):
    """Detailed view of a single version."""
    try:
        info = get_version_info(version_id, _repo())
    except Exception as e:
        abort(404, description=str(e))

    # Load data preview
    try:
        df = load_version(version_id, _repo(), data_type="processed")
        preview = df.head(10).to_dict(orient="records")
        columns = list(df.columns)
        total_rows = len(df)
    except Exception:
        preview = []
        columns = []
        total_rows = 0

    pipeline = info.config.get("pipeline", [])

    return render_template(
        "version.html",
        info=info,
        preview=preview,
        columns=columns,
        total_rows=total_rows,
        pipeline=pipeline,
        pipeline_json=json.dumps(info.config, indent=2),
    )


@app.route("/compare")
def compare_page():
    """Compare two versions side-by-side."""
    versions = list_versions(_repo())
    timeline = []
    for v in versions:
        timeline.append({
            "hash": v.version_hash,
            "short_hash": v.version_hash[:12],
            "tags": v.tags,
            "label": ", ".join(v.tags) if v.tags else v.version_hash[:12],
        })
    return render_template("compare.html", versions=timeline)


@app.route("/api/compare")
def api_compare():
    """API endpoint to compare two versions."""
    v1 = request.args.get("v1", "")
    v2 = request.args.get("v2", "")
    if not v1 or not v2:
        return jsonify({"error": "Both v1 and v2 are required"}), 400

    try:
        report = compare_versions_raw(v1, v2, _repo())
    except Exception as e:
        return jsonify({"error": str(e)}), 404

    result = {
        "v1_hash": report.v1_hash,
        "v2_hash": report.v2_hash,
        "config_diff": report.config_diff,
        "metrics_diff": report.metrics_diff,
        "data_overlap": {},
    }

    # Serialize data_overlap safely
    overlap = report.data_overlap
    for k, val in overlap.items():
        if isinstance(val, (int, float, str, bool)):
            result["data_overlap"][k] = val
        elif isinstance(val, list):
            result["data_overlap"][k] = [str(x) for x in val[:10]]
        else:
            result["data_overlap"][k] = str(val)

    return jsonify(result)


@app.route("/api/versions")
def api_versions():
    """API endpoint: list all versions."""
    versions = list_versions(_repo())
    data = []
    for v in versions:
        data.append({
            "hash": v.version_hash,
            "short_hash": v.version_hash[:12],
            "tags": v.tags,
            "samples": v.metrics.get("num_samples", 0),
            "created": str(v.metrics.get("created_at", ""))[:19],
        })
    return jsonify(data)


def main():
    """Entry point for the web dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="FLUX Web Dashboard")
    parser.add_argument("--repo", "-r", default=None, help="Repository path")
    parser.add_argument("--port", "-p", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    global REPO_PATH
    if args.repo:
        REPO_PATH = str(Path(args.repo).resolve())

    if not is_flux_repo(REPO_PATH):
        print(f"[!] No FLUX repo found at {REPO_PATH}")
        print(f"    Run 'python -m flux init' first, or use --repo to specify a path.")
        sys.exit(1)

    print(f"\n  FLUX Web Dashboard")
    print(f"  Repository: {REPO_PATH}")
    print(f"  Running at: http://{args.host}:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
