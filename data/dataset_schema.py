"""Schema validation for TaskShift frame metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


PASSIVE_KEYS = ("visible_objects", "room_type")
NAVIGATION_LABEL_KEYS = (
    "door_visible",
    "path_blocked",
    "obstacle_visible",
    "best_action",
    "reachable_goal_visible",
)
CONCEPT_LABEL_KEYS = (
    "agent",
    "path",
    "obstacle",
    "landmark",
    "goal_object",
    "container",
)
BEST_ACTIONS = ("MoveAhead", "RotateLeft", "RotateRight", "Stop")


@dataclass(frozen=True)
class ValidationResult:
    """Summary of schema validation for a metadata file."""

    rows_checked: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            rows.append(row)
    return rows


def validate_metadata_file(metadata_path: Path, dataset_root: Path | None = None) -> ValidationResult:
    rows = load_jsonl(metadata_path)
    root = dataset_root or metadata_path.parent
    return validate_rows(rows, root)


def validate_rows(rows: Iterable[dict[str, Any]], dataset_root: Path) -> ValidationResult:
    errors: list[str] = []
    count = 0

    for count, row in enumerate(rows, start=1):
        prefix = f"row {count}"
        errors.extend(_validate_row(row, dataset_root, prefix))

    if count == 0:
        errors.append("metadata has no rows")

    return ValidationResult(rows_checked=count, errors=tuple(errors))


def _validate_row(row: dict[str, Any], dataset_root: Path, prefix: str) -> list[str]:
    errors: list[str] = []
    required = (
        "image_path",
        "scene",
        "agent_position",
        "agent_rotation",
        *PASSIVE_KEYS,
        "navigation_labels",
        "concept_labels",
    )

    for key in required:
        if key not in row:
            errors.append(f"{prefix}: missing {key}")

    if errors:
        return errors

    image_path = row["image_path"]
    if not isinstance(image_path, str) or not image_path:
        errors.append(f"{prefix}: image_path must be a non-empty string")
    elif not (dataset_root / image_path).exists():
        errors.append(f"{prefix}: image_path does not exist: {image_path}")

    if not isinstance(row["scene"], str) or not row["scene"]:
        errors.append(f"{prefix}: scene must be a non-empty string")

    position = row["agent_position"]
    if not _is_number_list(position, 3):
        errors.append(f"{prefix}: agent_position must be a list of 3 numbers")

    if not isinstance(row["agent_rotation"], int):
        errors.append(f"{prefix}: agent_rotation must be an integer")

    if not _is_string_list(row["visible_objects"]):
        errors.append(f"{prefix}: visible_objects must be a list of strings")

    if not isinstance(row["room_type"], str) or not row["room_type"]:
        errors.append(f"{prefix}: room_type must be a non-empty string")

    nav = row["navigation_labels"]
    if not isinstance(nav, dict):
        errors.append(f"{prefix}: navigation_labels must be an object")
    else:
        errors.extend(_validate_navigation_labels(nav, prefix))

    concepts = row["concept_labels"]
    if not isinstance(concepts, dict):
        errors.append(f"{prefix}: concept_labels must be an object")
    else:
        errors.extend(_validate_bool_map(concepts, CONCEPT_LABEL_KEYS, "concept_labels", prefix))

    return errors


def _validate_navigation_labels(labels: dict[str, Any], prefix: str) -> list[str]:
    errors = _validate_bool_map(
        labels,
        ("door_visible", "path_blocked", "obstacle_visible", "reachable_goal_visible"),
        "navigation_labels",
        prefix,
    )

    action = labels.get("best_action")
    if action not in BEST_ACTIONS:
        errors.append(f"{prefix}: navigation_labels.best_action must be one of {BEST_ACTIONS}")
    return errors


def _validate_bool_map(
    labels: dict[str, Any],
    required_keys: tuple[str, ...],
    label_group: str,
    prefix: str,
) -> list[str]:
    errors: list[str] = []
    for key in required_keys:
        if key not in labels:
            errors.append(f"{prefix}: missing {label_group}.{key}")
        elif not isinstance(labels[key], bool):
            errors.append(f"{prefix}: {label_group}.{key} must be boolean")
    return errors


def _is_number_list(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(isinstance(item, (int, float)) for item in value)
    )


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)

