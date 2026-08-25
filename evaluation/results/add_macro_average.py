"""Add a macro-average entry to evaluation result JSON files.

By default, the script updates every ``*.json`` file in this directory. Existing
``macro_average`` entries are left unchanged, so rerunning the script is safe.
"""

import argparse
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Optional


AGGREGATE_KEYS = {"macro_average", "average_score"}


def _get_score(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        value = value.get("string_match")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return score if math.isfinite(score) else None


def calculate_macro_average(results: dict[str, Any]) -> float:
    task_scores = []
    invalid_tasks = []
    for task, result in results.items():
        if task in AGGREGATE_KEYS:
            continue
        score = _get_score(result)
        if score is None:
            invalid_tasks.append(task)
        else:
            task_scores.append(score)

    if invalid_tasks:
        names = ", ".join(invalid_tasks)
        raise ValueError(f"tasks without a numeric string_match score: {names}")
    if not task_scores:
        raise ValueError("no task scores found")
    return round(sum(task_scores) / len(task_scores), 2)


def _write_json_atomically(path: Path, results: dict[str, Any]) -> None:
    current_mode = stat.S_IMODE(path.stat().st_mode)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(results, temp_file, ensure_ascii=False, indent=2)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        temp_path.chmod(current_mode)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def add_macro_average(path: Path, *, dry_run: bool = False, overwrite: bool = False) -> str:
    with path.open(encoding="utf-8") as result_file:
        results = json.load(result_file)
    if not isinstance(results, dict):
        raise ValueError("top-level JSON value is not an object")
    if "macro_average" in results and not overwrite:
        return "already_present"

    macro_average = calculate_macro_average(results)
    if not dry_run:
        results["macro_average"] = {"string_match": macro_average}
        _write_json_atomically(path, results)
    return f"macro_average={macro_average:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing result JSON files (default: this script's directory).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing macro_average entries.")
    args = parser.parse_args()

    json_files = sorted(args.directory.glob("*.json"))
    updated = already_present = failed = 0
    for path in json_files:
        try:
            result = add_macro_average(path, dry_run=args.dry_run, overwrite=args.overwrite)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            failed += 1
            print(f"ERROR   {path.name}: {error}")
            continue

        if result == "already_present":
            already_present += 1
            print(f"SKIP    {path.name}: macro_average already present")
        else:
            updated += 1
            action = "WOULD UPDATE" if args.dry_run else "UPDATED"
            print(f"{action:<12} {path.name}: {result}")

    print(
        f"Summary: files={len(json_files)}, updated={updated}, "
        f"already_present={already_present}, failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
