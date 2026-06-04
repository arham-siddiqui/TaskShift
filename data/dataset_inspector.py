"""Inspect and validate a TaskShift dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from data.dataset_schema import CONCEPT_LABEL_KEYS, NAVIGATION_LABEL_KEYS, validate_metadata_file, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a TaskShift dataset.")
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("artifacts/prototype_dataset"),
        help="Dataset directory containing metadata.jsonl.",
    )
    args = parser.parse_args()

    inspect_dataset(args.dataset)


def inspect_dataset(dataset_dir: Path) -> None:
    metadata_path = dataset_dir / "metadata.jsonl"
    result = validate_metadata_file(metadata_path, dataset_dir)
    rows = load_jsonl(metadata_path)

    print(f"dataset: {dataset_dir}")
    print(f"rows: {result.rows_checked}")
    print(f"valid: {result.ok}")
    if result.errors:
        for error in result.errors[:20]:
            print(f"- {error}")
        return

    room_counts = Counter(row["room_type"] for row in rows)
    object_counts = Counter(item for row in rows for item in row["visible_objects"])
    action_counts = Counter(row["navigation_labels"]["best_action"] for row in rows)

    print("\nrooms")
    for room, count in room_counts.most_common():
        print(f"- {room}: {count}")

    print("\nvisible objects")
    for obj, count in object_counts.most_common():
        print(f"- {obj}: {count}")

    print("\nnavigation labels")
    for key in NAVIGATION_LABEL_KEYS:
        if key == "best_action":
            continue
        positives = sum(1 for row in rows if row["navigation_labels"][key])
        print(f"- {key}: {positives}/{len(rows)} positive")
    print(f"- best_action: {dict(action_counts)}")

    print("\nconcept labels")
    for key in CONCEPT_LABEL_KEYS:
        positives = sum(1 for row in rows if row["concept_labels"][key])
        print(f"- {key}: {positives}/{len(rows)} positive")


if __name__ == "__main__":
    main()

