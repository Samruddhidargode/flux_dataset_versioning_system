"""CLI entry point for FLUX versioning system.

Provides a user-friendly command-line interface with interactive prompts.

Simple Commands:
    flux init [path]                  - Initialize a new repo
    flux upload <dataset.csv>         - Upload dataset (interactive pipeline setup)
    flux list                         - List all versions
    flux show <version_or_tag>        - Show version details
    flux diff <v1> <v2>               - Compare two versions side-by-side
    flux tag <version_or_tag> <name>  - Tag a version
    flux tags                         - List all tags
    flux export <version> <path>      - Export version as tarball
    flux import <tarball>             - Import version from tarball
    flux create <csv> -c <config>     - Advanced: create with JSON config file
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure the parent package is importable when running as a script
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from flux.cli.interactive import (
    bold,
    build_pipeline_interactive,
    dim,
    error,
    header,
    info,
    print_banner,
    print_versions_table,
    success,
    warning,
    _ask_text,
    _ask_yes_no,
)
from flux.core.comparator import compare_versions
from flux.core.repository import (
    create_version,
    export_version,
    get_version_info,
    import_version,
    init_repo,
    list_tags,
    list_versions,
    load_version,
    tag_version,
)
from flux.utils.file_utils import is_flux_repo

logger = logging.getLogger("flux")

DEFAULT_REPO_ENV = "FLUX_REPO"


def _get_repo_path(args):
    """Resolve the repository path from args, env, or current directory."""
    repo = getattr(args, "repo", None)
    if repo:
        return repo
    env_repo = os.environ.get(DEFAULT_REPO_ENV)
    if env_repo:
        return env_repo
    return os.getcwd()


def _ensure_repo(repo_path):
    """Ensure repo exists, offer to create if not."""
    if is_flux_repo(repo_path):
        return repo_path

    print()
    print("  " + warning("No FLUX repository found at:") + " " + repo_path)
    if _ask_yes_no("Initialize a new repository here?", default=True):
        result = init_repo(repo_path)
        print("  " + success("[OK]") + " Repository initialized at: " + info(str(result)))
        print()
        return str(result)
    else:
        print("  " + error("Aborted.") + " Run " + info("flux init <path>") + " to create a repo.")
        print()
        sys.exit(1)


# ---- Command Handlers ----

def cmd_init(args):
    """Handle 'flux init'."""
    repo_path = getattr(args, "repo_path", None) or os.getcwd()
    result = init_repo(repo_path)
    print()
    print("  " + success("[OK]") + " Initialized FLUX repository at: " + info(str(result)))
    print()


def cmd_upload(args):
    """Handle 'flux upload' - upload a dataset with interactive pipeline setup."""
    repo = _get_repo_path(args)
    repo = _ensure_repo(repo)

    raw_path = Path(args.dataset).resolve()
    if not raw_path.exists():
        print()
        print("  " + error("[FAIL]") + " File not found: " + str(raw_path))
        print()
        sys.exit(1)

    if not str(raw_path).lower().endswith(".csv"):
        print()
        print("  " + warning("[WARN]") + " File does not have .csv extension. Proceeding anyway...")
        print()

    # Show file info
    import pandas as pd
    try:
        df_preview = pd.read_csv(raw_path, encoding="utf-8")
    except Exception as e:
        print()
        print("  " + error("[FAIL]") + " Could not read CSV file: " + str(e))
        print()
        sys.exit(1)

    if "text" not in df_preview.columns:
        print()
        print("  " + error("[FAIL]") + " CSV must have a 'text' column.")
        print("  Found columns: " + str(list(df_preview.columns)))
        print()
        sys.exit(1)

    print()
    print("  " + success("[OK]") + " Dataset loaded: " + info(str(raw_path)))
    print("    Rows:    " + bold(str(len(df_preview))))
    print("    Columns: " + bold(", ".join(df_preview.columns)))

    has_label = "label" in df_preview.columns
    if has_label:
        class_counts = df_preview["label"].value_counts()
        classes_str = ", ".join(str(k) + ": " + str(v) for k, v in class_counts.items())
        print("    Labels:  " + bold(classes_str))

    print()
    print("    " + dim("Preview (first 3 rows):"))
    for i, (_, row) in enumerate(df_preview.head(3).iterrows(), 1):
        text_preview = str(row["text"])[:70]
        if len(str(row["text"])) > 70:
            text_preview += "..."
        label_str = " [" + str(row["label"]) + "]" if has_label else ""
        print("    " + dim(str(i) + ".") + " " + text_preview + info(label_str))
    print()

    # Use provided config or build interactively
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print()
            print("  " + error("[FAIL]") + " Config file not found: " + str(config_path))
            print()
            sys.exit(1)
        with open(config_path, "r") as f:
            config = json.load(f)
        print("  " + success("[OK]") + " Using config from: " + info(str(config_path)))
        print()
    else:
        # Interactive pipeline builder
        config = build_pipeline_interactive()

    # Confirm before proceeding
    if not getattr(args, "yes", False):
        if not _ask_yes_no("Proceed with creating this version?", default=True):
            print("  " + dim("Cancelled."))
            print()
            sys.exit(0)

    # Ask for tag
    if args.tag:
        tag = args.tag
    elif not getattr(args, "yes", False):
        tag = _ask_text("Give this version a tag name (or press Enter to skip):")
    else:
        tag = None

    # Ask for user
    if args.user:
        user_id = args.user
    elif not getattr(args, "yes", False):
        user_id = _ask_text("Your name or email (optional):")
    else:
        user_id = None

    tags = [tag] if tag else None

    # Create version
    print()
    print("  " + dim("Creating version..."))
    version_hash = create_version(
        raw_path=str(raw_path),
        config=config,
        repo_path=repo,
        user=user_id or None,
        tags=tags,
    )

    print()
    print("  " + success("[OK] Version created successfully!"))
    print("    Hash: " + info(version_hash[:16]) + "..." + dim(version_hash[16:32]))
    if tags:
        print("    Tag:  " + success(tags[0]))
    if user_id:
        print("    User: " + dim(user_id))

    # Show quick metrics
    ver_info = get_version_info(version_hash, repo)
    m = ver_info.metrics
    print()
    print("    " + bold("Quick Stats:"))
    print("    Samples:     " + str(m.get("num_samples", "?")))
    print("    Unique:      " + str(m.get("num_unique_texts", "?")))
    print("    Vocab Size:  " + str(m.get("vocab_size", "?")))
    avg_len = m.get("avg_text_length", 0)
    print("    Avg Length:  " + "{:.1f}".format(avg_len))
    print()

    # Hint for next steps
    tag_ref = tags[0] if tags else version_hash[:12]
    print("  " + dim("Next steps:"))
    print("    " + dim("*") + " View details:   " + info("flux show " + tag_ref))
    print("    " + dim("*") + " List versions:  " + info("flux list"))
    print("    " + dim("*") + " Compare:        " + info("flux diff " + tag_ref + " <other>"))
    print()


def cmd_list(args):
    """Handle 'flux list'."""
    repo = _get_repo_path(args)
    repo = _ensure_repo(repo)
    versions = list_versions(repo)
    print_versions_table(versions)


def cmd_show(args):
    """Handle 'flux show'."""
    repo = _get_repo_path(args)
    repo = _ensure_repo(repo)

    try:
        ver_info = get_version_info(args.version, repo)
    except Exception as e:
        print()
        print("  " + error("[FAIL]") + " " + str(e))
        print()
        sys.exit(1)

    print()
    vh = ver_info.version_hash
    print("  " + bold("Version:") + " " + info(vh))

    if ver_info.tags:
        print("  " + bold("Tags:") + "    " + success(", ".join(ver_info.tags)))

    print("  " + bold("Raw Hash:") + "    " + dim(ver_info.raw_hash))
    print("  " + bold("Config Hash:") + " " + dim(ver_info.config_hash))
    print()

    m = ver_info.metrics
    print("  " + bold("Metrics:"))
    print("    Samples:      " + str(m.get("num_samples", "?")))
    print("    Unique Texts: " + str(m.get("num_unique_texts", "?")))
    print("    Vocab Size:   " + str(m.get("vocab_size", "?")))
    avg_len = m.get("avg_text_length", 0)
    print("    Avg Length:   " + "{:.1f}".format(avg_len))
    created = str(m.get("created_at", "?"))[:19]
    print("    Created:      " + created)
    created_by = m.get("created_by") or "-"
    print("    Created By:   " + str(created_by))

    cd = m.get("class_distribution")
    if cd:
        print("    Classes:      " + str(cd))

    # Show pipeline
    pipeline = ver_info.config.get("pipeline", [])
    print()
    print("  " + bold("Preprocessing Pipeline:"))
    if pipeline:
        for i, step in enumerate(pipeline, 1):
            step_name = step["step"]
            params = step.get("params", {})
            params_str = "  " + dim(str(params)) if params else ""
            print("    " + success(str(i)) + ". " + step_name + params_str)
    else:
        print("    " + dim("(no preprocessing)"))

    # Preview data
    if getattr(args, "preview", False):
        try:
            df = load_version(args.version, repo, data_type="processed")
            n = min(5, len(df))
            print()
            print("  " + bold("Data Preview (first " + str(n) + " rows):"))
            for i, (_, row) in enumerate(df.head(n).iterrows(), 1):
                text = str(row["text"])[:80]
                if len(str(row["text"])) > 80:
                    text += "..."
                label_str = ""
                if "label" in df.columns:
                    label_str = "  " + info("[" + str(row["label"]) + "]")
                print("    " + dim(str(i) + ".") + " " + text + label_str)
        except Exception:
            pass

    print()


def cmd_diff(args):
    """Handle 'flux diff'."""
    repo = _get_repo_path(args)
    repo = _ensure_repo(repo)

    print()
    print("  " + dim("Comparing versions..."))
    print()
    try:
        report = compare_versions(args.v1, args.v2, repo)
    except Exception as e:
        print()
        print("  " + error("[FAIL]") + " " + str(e))
        print()
        sys.exit(1)

    print(report)
    print()


def cmd_tag(args):
    """Handle 'flux tag'."""
    repo = _get_repo_path(args)
    repo = _ensure_repo(repo)

    try:
        tag_version(args.version, args.tag_name, repo)
    except Exception as e:
        print()
        print("  " + error("[FAIL]") + " " + str(e))
        print()
        sys.exit(1)

    print()
    print("  " + success("[OK]") + " Tagged " + info(args.version) + " as " + success(args.tag_name))
    print()


def cmd_tags(args):
    """Handle 'flux tags'."""
    repo = _get_repo_path(args)
    repo = _ensure_repo(repo)
    tags = list_tags(repo)

    if not tags:
        print()
        print("  " + dim("No tags found."))
        print()
        return

    print()
    print("  " + bold("TAG") + " " * 17 + bold("VERSION HASH"))
    print("  " + "-" * 50)
    for tag_name, vh in sorted(tags.items()):
        padding = " " * max(1, 20 - len(tag_name))
        print("  " + success(tag_name) + padding + info(vh[:12]))
    print()


def cmd_export(args):
    """Handle 'flux export'."""
    repo = _get_repo_path(args)
    repo = _ensure_repo(repo)

    try:
        tarball = export_version(args.version, args.output_path, repo)
    except Exception as e:
        print()
        print("  " + error("[FAIL]") + " " + str(e))
        print()
        sys.exit(1)

    print()
    print("  " + success("[OK]") + " Exported to: " + info(str(tarball)))
    print()


def cmd_import(args):
    """Handle 'flux import'."""
    repo = _get_repo_path(args)
    repo = _ensure_repo(repo)

    try:
        vh = import_version(args.tarball_path, repo)
    except Exception as e:
        print()
        print("  " + error("[FAIL]") + " " + str(e))
        print()
        sys.exit(1)

    print()
    print("  " + success("[OK]") + " Imported version: " + info(vh))
    print()


def cmd_create(args):
    """Handle 'flux create' (advanced)."""
    repo = _get_repo_path(args)
    repo = _ensure_repo(repo)

    tags = None
    if args.tag:
        tags = [t.strip() for t in args.tag.split(",")]

    version_hash = create_version(
        raw_path=args.raw_path,
        config=args.config,
        repo_path=repo,
        user=args.user,
        tags=tags,
    )
    print()
    print("  " + success("[OK]") + " Created version: " + info(version_hash))
    if tags:
        print("  Tagged as: " + success(", ".join(tags)))
    print()


# ---- Parser ----

def build_parser():
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="flux",
        description="FLUX - Dataset Versioning Made Simple",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "\nExamples:\n"
            "  flux init                              Initialize repo in current directory\n"
            "  flux upload reviews.csv                Upload dataset (interactive setup)\n"
            "  flux upload data.csv -c config.json    Upload with preset config\n"
            "  flux list                              List all versions\n"
            "  flux show main                         Show version tagged 'main'\n"
            "  flux show main -p                      Show version with data preview\n"
            "  flux diff main dev                     Compare two versions\n"
            "  flux tag <hash> production             Tag a version\n"
            "  flux export main ./backups             Export version as .tar.gz\n"
            "  flux import backup.tar.gz              Import a version\n"
        ),
    )
    parser.add_argument(
        "--repo", "-r", default=None,
        help="Repository path (default: current directory or $FLUX_REPO)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # init
    p_init = subparsers.add_parser("init", help="Initialize a new repository")
    p_init.add_argument("repo_path", nargs="?", default=None,
                        help="Path for repo (default: current directory)")
    p_init.set_defaults(func=cmd_init)

    # upload (the main user-friendly command)
    p_upload = subparsers.add_parser(
        "upload", help="Upload a dataset and create a version",
        description="Upload a CSV dataset. Interactively choose preprocessing steps.",
    )
    p_upload.add_argument("dataset", help="Path to CSV file")
    p_upload.add_argument("--config", "-c", default=None,
                          help="Path to JSON config (skips interactive setup)")
    p_upload.add_argument("--tag", "-t", default=None, help="Tag name for this version")
    p_upload.add_argument("--user", "-u", default=None, help="Your name or email")
    p_upload.add_argument("--yes", "-y", action="store_true",
                          help="Skip confirmation prompts")
    p_upload.set_defaults(func=cmd_upload)

    # list
    p_list = subparsers.add_parser("list", help="List all versions")
    p_list.set_defaults(func=cmd_list)

    # show
    p_show = subparsers.add_parser("show", help="Show version details")
    p_show.add_argument("version", help="Version hash or tag name")
    p_show.add_argument("--preview", "-p", action="store_true",
                        help="Show data preview (first 5 rows)")
    p_show.set_defaults(func=cmd_show)

    # diff
    p_diff = subparsers.add_parser("diff", help="Compare two versions side-by-side")
    p_diff.add_argument("v1", help="First version (hash or tag)")
    p_diff.add_argument("v2", help="Second version (hash or tag)")
    p_diff.set_defaults(func=cmd_diff)

    # tag
    p_tag = subparsers.add_parser("tag", help="Tag a version with a name")
    p_tag.add_argument("version", help="Version hash or tag")
    p_tag.add_argument("tag_name", help="Tag name to assign")
    p_tag.set_defaults(func=cmd_tag)

    # tags
    p_tags = subparsers.add_parser("tags", help="List all tags")
    p_tags.set_defaults(func=cmd_tags)

    # export
    p_export = subparsers.add_parser("export", help="Export version as .tar.gz")
    p_export.add_argument("version", help="Version hash or tag")
    p_export.add_argument("output_path", help="Output directory")
    p_export.set_defaults(func=cmd_export)

    # import
    p_import = subparsers.add_parser("import", help="Import version from .tar.gz")
    p_import.add_argument("tarball_path", help="Path to tarball")
    p_import.set_defaults(func=cmd_import)

    # create (advanced/legacy)
    p_create = subparsers.add_parser("create", help="[Advanced] Create version with config file")
    p_create.add_argument("raw_path", help="Path to raw CSV")
    p_create.add_argument("--config", "-c", required=True, help="JSON config file")
    p_create.add_argument("--tag", "-t", default=None, help="Tag(s), comma-separated")
    p_create.add_argument("--user", "-u", default=None, help="User identifier")
    p_create.set_defaults(func=cmd_create)

    # compare (alias for diff)
    p_compare = subparsers.add_parser("compare", help="Compare two versions (alias for diff)")
    p_compare.add_argument("v1", help="First version")
    p_compare.add_argument("v2", help="Second version")
    p_compare.set_defaults(func=cmd_diff)

    return parser


def main():
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command is None:
        print_banner()
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print()
        print("  " + dim("Cancelled."))
        print()
        sys.exit(130)
    except Exception as e:
        logger.debug("Error details:", exc_info=True)
        print()
        print("  " + error("Error:") + " " + str(e), file=sys.stderr)
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
