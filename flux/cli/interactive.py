"""Interactive pipeline builder for FLUX.

Guides users through selecting and configuring preprocessing steps
via an interactive CLI questionnaire.
"""

from typing import Any, Dict, List, Optional


# ANSI color codes for terminal output
class Colors:
    """Terminal color codes for pretty output."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def colorize(text: str, color: str) -> str:
    """Wrap text in ANSI color codes.

    Args:
        text: Text to colorize.
        color: ANSI color code from Colors class.

    Returns:
        Colorized string.
    """
    return f"{color}{text}{Colors.RESET}"


def bold(text: str) -> str:
    return colorize(text, Colors.BOLD)


def success(text: str) -> str:
    return colorize(text, Colors.GREEN)


def warning(text: str) -> str:
    return colorize(text, Colors.YELLOW)


def error(text: str) -> str:
    return colorize(text, Colors.RED)


def info(text: str) -> str:
    return colorize(text, Colors.CYAN)


def header(text: str) -> str:
    return colorize(text, Colors.HEADER + Colors.BOLD)


def dim(text: str) -> str:
    return colorize(text, Colors.DIM)


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question.

    Args:
        prompt: Question to display.
        default: Default answer if user presses Enter.

    Returns:
        True for yes, False for no.
    """
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"  {prompt} {dim(hint)} ").strip().lower()
        if answer == "":
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print(f"  {warning('Please enter y or n.')}")


def _ask_choice(prompt: str, options: List[str], default: int = 0) -> str:
    """Ask user to pick from a list of options.

    Args:
        prompt: Question to display.
        options: List of option strings.
        default: Index of default option (0-based).

    Returns:
        Selected option string.
    """
    print(f"  {prompt}")
    for i, opt in enumerate(options):
        marker = success("→") if i == default else " "
        default_label = dim(" (default)") if i == default else ""
        print(f"    {marker} {i + 1}. {opt}{default_label}")

    while True:
        answer = input(f"  {dim('Enter choice [1-' + str(len(options)) + ']:' )} ").strip()
        if answer == "":
            return options[default]
        try:
            idx = int(answer) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print(f"  {warning('Invalid choice. Try again.')}")


def _ask_number(prompt: str, default: Optional[int] = None, min_val: int = 0) -> Optional[int]:
    """Ask user for a number.

    Args:
        prompt: Question to display.
        default: Default value if user presses Enter.
        min_val: Minimum acceptable value.

    Returns:
        The entered number, or default.
    """
    default_hint = dim(f" [{default}]") if default is not None else ""
    while True:
        answer = input(f"  {prompt}{default_hint} ").strip()
        if answer == "" and default is not None:
            return default
        if answer == "" and default is None:
            return None
        try:
            val = int(answer)
            if val >= min_val:
                return val
            print(f"  {warning(f'Must be >= {min_val}.')}")
        except ValueError:
            print(f"  {warning('Please enter a number.')}")


def _ask_text(prompt: str, default: str = "") -> str:
    """Ask user for free text input.

    Args:
        prompt: Question to display.
        default: Default value if user presses Enter.

    Returns:
        The entered text.
    """
    default_hint = dim(f" [{default}]") if default else ""
    answer = input(f"  {prompt}{default_hint} ").strip()
    return answer if answer else default


def build_pipeline_interactive() -> Dict[str, Any]:
    """Interactively build a preprocessing pipeline config.

    Guides the user through selecting and configuring each
    preprocessing step with clear prompts.

    Returns:
        A config dictionary with a 'pipeline' key.
    """
    print()
    print(header("╔══════════════════════════════════════════╗"))
    print(header("║   FLUX Preprocessing Pipeline Builder    ║"))
    print(header("╚══════════════════════════════════════════╝"))
    print()
    print(f"  {info('Choose which preprocessing steps to apply.')}")
    print(f"  {info('Steps run in order: lowercase → tokenize → stopwords → filter → dedup')}")
    print()

    pipeline: List[Dict[str, Any]] = []

    # Step 1: Lowercase
    print(f"  {bold('Step 1: Lowercase')}")
    print(f"  {dim('Converts all text to lowercase (e.g., \"Hello World\" → \"hello world\")')}")
    if _ask_yes_no("Apply lowercasing?", default=True):
        pipeline.append({"step": "lowercase"})
        print(f"  {success('✓ Lowercase enabled')}")
    else:
        print(f"  {dim('✗ Skipped')}")
    print()

    # Step 2: Tokenization
    print(f"  {bold('Step 2: Tokenization')}")
    print(f"  {dim('Splits text into clean tokens')}")
    if _ask_yes_no("Apply tokenization?", default=True):
        method = _ask_choice(
            "Tokenization method:",
            ["whitespace — split on spaces/tabs", "regex — keep only letters and numbers"],
            default=1,
        )
        method_key = "whitespace" if method.startswith("whitespace") else "regex"
        pipeline.append({"step": "tokenize", "params": {"method": method_key}})
        print(f"  {success(f'✓ Tokenization enabled ({method_key})')}")
    else:
        print(f"  {dim('✗ Skipped')}")
    print()

    # Step 3: Stopword Removal
    print(f"  {bold('Step 3: Stopword Removal')}")
    print(f"  {dim('Removes common words like \"the\", \"is\", \"and\", \"a\"')}")
    if _ask_yes_no("Remove stopwords?", default=True):
        lang = _ask_choice(
            "Stopword language:",
            ["english"],
            default=0,
        )
        params: Dict[str, Any] = {"language": lang}

        if _ask_yes_no("Add custom stopwords?", default=False):
            custom = _ask_text("Enter custom stopwords (comma-separated):")
            if custom:
                params["custom_list"] = [w.strip() for w in custom.split(",")]

        pipeline.append({"step": "remove_stopwords", "params": params})
        print(f"  {success('✓ Stopword removal enabled')}")
    else:
        print(f"  {dim('✗ Skipped')}")
    print()

    # Step 4: Filter by Length
    print(f"  {bold('Step 4: Filter by Length')}")
    print(f"  {dim('Remove texts that are too short or too long (by token count)')}")
    if _ask_yes_no("Filter by text length?", default=False):
        min_t = _ask_number("Minimum tokens per text:", default=2, min_val=0)
        max_t = _ask_number("Maximum tokens per text (leave blank for no limit):", default=None, min_val=1)
        fparams: Dict[str, Any] = {}
        if min_t is not None:
            fparams["min_tokens"] = min_t
        if max_t is not None:
            fparams["max_tokens"] = max_t
        pipeline.append({"step": "filter_by_length", "params": fparams})
        print(f"  {success('✓ Length filter enabled')}")
    else:
        print(f"  {dim('✗ Skipped')}")
    print()

    # Step 5: Deduplication
    print(f"  {bold('Step 5: Deduplication')}")
    print(f"  {dim('Remove duplicate rows based on text content')}")
    if _ask_yes_no("Remove duplicates?", default=True):
        keep = _ask_choice(
            "Which duplicate to keep?",
            ["first — keep first occurrence", "last — keep last occurrence"],
            default=0,
        )
        keep_key = "first" if keep.startswith("first") else "last"
        pipeline.append({"step": "deduplicate", "params": {"keep": keep_key}})
        print(f"  {success('✓ Deduplication enabled')}")
    else:
        print(f"  {dim('✗ Skipped')}")
    print()

    # If no steps selected, add a note
    if not pipeline:
        print(f"  {warning('No preprocessing steps selected. Raw data will be stored as-is.')}")
        print()

    # Summary
    print(header("─── Pipeline Summary ───"))
    if pipeline:
        for i, step in enumerate(pipeline, 1):
            step_name = step["step"]
            params = step.get("params", {})
            params_str = f" ({params})" if params else ""
            print(f"  {success(str(i))}. {bold(step_name)}{dim(params_str)}")
    else:
        print(f"  {dim('(no preprocessing)')}")
    print()

    return {"pipeline": pipeline}


def print_banner() -> None:
    """Print the FLUX welcome banner."""
    print()
    print(header("  ███████╗██╗     ██╗   ██╗██╗  ██╗"))
    print(header("  ██╔════╝██║     ██║   ██║╚██╗██╔╝"))
    print(header("  █████╗  ██║     ██║   ██║ ╚███╔╝ "))
    print(header("  ██╔══╝  ██║     ██║   ██║ ██╔██╗ "))
    print(header("  ██║     ███████╗╚██████╔╝██╔╝ ██╗"))
    print(header("  ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝"))
    print()
    print(f"  {dim('File-based Lightweight Universal Xplainable Dataset Versioning')}")
    print()


def print_version_card(info: Any) -> None:
    """Print a nicely formatted version info card.

    Args:
        info: VersionInfo object.
    """
    print()
    print(f"  {header('┌─────────────────────────────────────────────────────────────────────┐')}")
    print(f"  {header('│')} {bold('Version Details')}{' ' * 52}{header('│')}")
    print(f"  {header('├─────────────────────────────────────────────────────────────────────┤')}")

    vh = info.version_hash
    print(f"  {header('│')} Hash:        {info(vh[:16])}...{dim(vh[16:32])}{' ' * (37 - 3)}{header('│')}")

    if info.tags:
        tags_str = ", ".join(info.tags)
        pad = 69 - 14 - len(tags_str)
        print(f"  {header('│')} Tags:        {success(tags_str)}{' ' * max(pad, 0)}{header('│')}")

    m = info.metrics
    fields = [
        ("Samples", str(m.get("num_samples", "?"))),
        ("Unique Texts", str(m.get("num_unique_texts", "?"))),
        ("Vocab Size", str(m.get("vocab_size", "?"))),
        ("Avg Length", f"{m.get('avg_text_length', 0):.1f}"),
        ("Created", str(m.get("created_at", "?"))[:19]),
        ("Created By", str(m.get("created_by", "?"))),
    ]

    cd = m.get("class_distribution")
    if cd:
        fields.append(("Classes", str(cd)))

    for label, value in fields:
        pad = 69 - 14 - len(value)
        print(f"  {header('│')} {label + ':':<13}{value}{' ' * max(pad, 0)}{header('│')}")

    print(f"  {header('└─────────────────────────────────────────────────────────────────────┘')}")
    print()


def print_versions_table(versions: list) -> None:
    """Print a formatted table of versions.

    Args:
        versions: List of VersionInfo objects.
    """
    if not versions:
        print(f"\n  {dim('No versions found. Upload a dataset to get started:')}")
        print(f"  {info('flux upload <your_data.csv>')}\n")
        return

    print()
    print(f"  {bold('HASH'):<28} {bold('SAMPLES'):>10}  {bold('TAGS'):<20} {bold('CREATED')}")
    print(f"  {'─' * 75}")

    for v in versions:
        short_hash = v.version_hash[:12]
        samples = str(v.metrics.get("num_samples", "?"))
        tags_str = ", ".join(v.tags) if v.tags else dim("—")
        created = str(v.metrics.get("created_at", "?"))[:19]
        print(f"  {info(short_hash):<28} {samples:>10}  {success(tags_str) if v.tags else tags_str:<20} {dim(created)}")

    print(f"\n  {dim(f'Total: {len(versions)} version(s)')}\n")
