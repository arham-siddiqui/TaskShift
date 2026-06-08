"""Build a TaskShift dataset from AI2-THOR scenes.

This generator preserves the same on-disk contract as the synthetic prototype:

- frames/*.png
- metadata.jsonl
- taxonomy.yaml

The labels are intentionally simple heuristics for the first real-data pass.
They give the rest of the TaskShift pipeline embodied RGB frames and simulator
metadata without changing the dataset loader, training code, probes, or plots.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from data.dataset_schema import validate_metadata_file


DEFAULT_SCENES = (
    "FloorPlan1",
    "FloorPlan2",
    "FloorPlan201",
    "FloorPlan301",
)
ROTATIONS = (0, 90, 180, 270)
HORIZONS = (-30, 0, 30)
OBSTACLE_TYPES = {
    "ArmChair",
    "Bed",
    "Chair",
    "CoffeeTable",
    "DiningTable",
    "Ottoman",
    "Safe",
    "Sofa",
    "Stool",
    "TableTopDecor",
    "TVStand",
}
LANDMARK_TYPES = {
    "Blinds",
    "Cabinet",
    "CounterTop",
    "Curtains",
    "Door",
    "Fridge",
    "Window",
}
GOAL_TYPES = {
    "Apple",
    "Bed",
    "Book",
    "Bowl",
    "Cup",
    "Fridge",
    "Laptop",
    "Microwave",
    "Mug",
    "Sink",
    "Sofa",
    "Television",
}
CONTAINER_TYPES = {
    "Box",
    "Cabinet",
    "Drawer",
    "Dresser",
    "Fridge",
    "GarbageCan",
    "Microwave",
    "Safe",
}
DOOR_TYPES = {"Door"}
ROOM_BY_SCENE_PREFIX = {
    "FloorPlan": "Kitchen",
    "FloorPlan2": "LivingRoom",
    "FloorPlan3": "Bedroom",
    "FloorPlan4": "Bathroom",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TaskShift dataset from AI2-THOR.")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path("artifacts/thor_dataset"))
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--visibility-distance", type=float, default=1.5)
    parser.add_argument("--grid-size", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_dataset(
        output_dir=args.output,
        frame_count=args.frames,
        scenes=tuple(args.scenes),
        seed=args.seed,
        width=args.width,
        height=args.height,
        visibility_distance=args.visibility_distance,
        grid_size=args.grid_size,
        overwrite=args.overwrite,
    )


def build_dataset(
    output_dir: Path,
    frame_count: int,
    scenes: tuple[str, ...] = DEFAULT_SCENES,
    seed: int = 101,
    width: int = 320,
    height: int = 240,
    visibility_distance: float = 1.5,
    grid_size: float = 0.25,
    overwrite: bool = False,
) -> None:
    if frame_count <= 0:
        raise ValueError("--frames must be greater than 0")
    if not scenes:
        raise ValueError("at least one scene is required")

    controller_cls = load_controller_class()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    frames_per_scene = distribute_frames(frame_count, len(scenes))

    controller = controller_cls(
        width=width,
        height=height,
        gridSize=grid_size,
        visibilityDistance=visibility_distance,
    )
    rows: list[dict[str, Any]] = []
    try:
        for scene, scene_frame_count in zip(scenes, frames_per_scene):
            rows.extend(
                sample_scene(
                    controller=controller,
                    scene=scene,
                    frame_count=scene_frame_count,
                    frames_dir=frames_dir,
                    start_index=len(rows),
                    rng=rng,
                )
            )
    finally:
        controller.stop()

    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    taxonomy_source = Path(__file__).with_name("concept_taxonomy.yaml")
    shutil.copyfile(taxonomy_source, output_dir / "taxonomy.yaml")

    result = validate_metadata_file(metadata_path, output_dir)
    if not result.ok:
        joined = "\n".join(result.errors[:20])
        raise RuntimeError(f"generated dataset failed validation:\n{joined}")

    print(f"wrote {result.rows_checked} THOR frames to {output_dir}")
    print(f"metadata: {metadata_path}")


def sample_scene(
    controller: Any,
    scene: str,
    frame_count: int,
    frames_dir: Path,
    start_index: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    controller.reset(scene=scene)
    reachable_event = controller.step(action="GetReachablePositions")
    reachable_positions = list(reachable_event.metadata.get("actionReturn") or [])
    if not reachable_positions:
        raise RuntimeError(f"{scene} has no reachable positions")

    rows: list[dict[str, Any]] = []
    for offset in range(frame_count):
        index = start_index + offset
        position = rng.choice(reachable_positions)
        rotation = rng.choice(ROTATIONS)
        horizon = rng.choice(HORIZONS)
        event = controller.step(
            action="TeleportFull",
            x=position["x"],
            y=position["y"],
            z=position["z"],
            rotation={"x": 0, "y": rotation, "z": 0},
            horizon=horizon,
            standing=True,
        )
        if not event.metadata.get("lastActionSuccess", False):
            continue

        visible_objects = visible_object_types(event)
        navigation_labels = derive_navigation_labels(visible_objects, event.metadata.get("objects", []))
        concept_labels = derive_concept_labels(visible_objects, navigation_labels)
        image_name = f"scene_{scene.lower()}_frame_{index:05d}.png"
        save_frame(event.frame, frames_dir / image_name)

        rows.append(
            {
                "image_path": f"frames/{image_name}",
                "scene": scene,
                "agent_position": [
                    round(float(position["x"]), 3),
                    round(float(position["y"]), 3),
                    round(float(position["z"]), 3),
                ],
                "agent_rotation": int(rotation),
                "agent_horizon": int(horizon),
                "visible_objects": visible_objects,
                "room_type": room_type_for_scene(scene),
                "navigation_labels": navigation_labels,
                "concept_labels": concept_labels,
            }
        )

    if len(rows) < frame_count:
        print(f"warning: {scene} produced {len(rows)}/{frame_count} requested frames")
    return rows


def visible_object_types(event: Any) -> list[str]:
    types = {
        str(obj.get("objectType"))
        for obj in event.metadata.get("objects", [])
        if obj.get("visible") and obj.get("objectType")
    }
    return sorted(types)


def derive_navigation_labels(
    visible_objects: list[str],
    object_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    visible_set = set(visible_objects)
    door_visible = bool(visible_set & DOOR_TYPES)
    obstacle_visible = bool(visible_set & OBSTACLE_TYPES)
    reachable_goal_visible = bool(visible_set & GOAL_TYPES)
    nearby_obstacles = [
        obj
        for obj in object_metadata
        if obj.get("visible")
        and obj.get("objectType") in OBSTACLE_TYPES
        and distance_from_agent(obj) is not None
        and distance_from_agent(obj) <= 1.25
    ]
    path_blocked = bool(nearby_obstacles)

    if path_blocked:
        best_action = "RotateLeft"
    elif reachable_goal_visible:
        best_action = "Stop"
    else:
        best_action = "MoveAhead"

    return {
        "door_visible": door_visible,
        "path_blocked": path_blocked,
        "obstacle_visible": obstacle_visible,
        "best_action": best_action,
        "reachable_goal_visible": reachable_goal_visible,
    }


def derive_concept_labels(
    visible_objects: list[str],
    navigation_labels: dict[str, Any],
) -> dict[str, bool]:
    visible_set = set(visible_objects)
    return {
        "agent": bool("Agent" in visible_set),
        "path": not navigation_labels["path_blocked"],
        "obstacle": navigation_labels["obstacle_visible"],
        "landmark": bool(visible_set & LANDMARK_TYPES),
        "goal_object": navigation_labels["reachable_goal_visible"],
        "container": bool(visible_set & CONTAINER_TYPES),
    }


def distance_from_agent(obj: dict[str, Any]) -> float | None:
    distance = obj.get("distance")
    if isinstance(distance, (int, float)):
        return float(distance)
    return None


def save_frame(frame: Any, path: Path) -> None:
    image = Image.fromarray(frame)
    image.save(path)


def room_type_for_scene(scene: str) -> str:
    digits = "".join(char for char in scene if char.isdigit())
    if not digits:
        return "Kitchen"
    number = int(digits)
    if 1 <= number <= 30:
        return "Kitchen"
    if 201 <= number <= 230:
        return "LivingRoom"
    if 301 <= number <= 330:
        return "Bedroom"
    if 401 <= number <= 430:
        return "Bathroom"
    return "Kitchen"


def distribute_frames(total: int, buckets: int) -> list[int]:
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def load_controller_class() -> Any:
    try:
        from ai2thor.controller import Controller
    except ImportError as exc:
        raise RuntimeError(
            "AI2-THOR is not installed. Install dependencies with `python3 -m pip install -r requirements.txt`."
        ) from exc
    return Controller


if __name__ == "__main__":
    main()
