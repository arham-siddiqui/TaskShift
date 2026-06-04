"""Build a small synthetic TaskShift prototype dataset.

This gives the project a concrete metadata and image contract before AI2-THOR
or ProcTHOR generation is integrated.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import shutil
import struct
import zlib

from data.dataset_schema import validate_metadata_file


ROOMS = ("Kitchen", "LivingRoom", "Bedroom", "Bathroom")
SCENES = ("FloorPlan1", "FloorPlan2", "FloorPlan201", "FloorPlan301")
OBJECTS_BY_ROOM = {
    "Kitchen": ("Door", "Chair", "Table", "Cabinet", "Sink", "Fridge", "Floor", "Wall"),
    "LivingRoom": ("Door", "Chair", "Table", "Sofa", "Cabinet", "Floor", "Wall"),
    "Bedroom": ("Door", "Chair", "Bed", "Cabinet", "Floor", "Wall"),
    "Bathroom": ("Door", "Sink", "Cabinet", "Floor", "Wall"),
}
BEST_ACTIONS = ("MoveAhead", "RotateLeft", "RotateRight", "Stop")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TaskShift prototype dataset.")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames to generate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/prototype_dataset"),
        help="Dataset output directory.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    args = parser.parse_args()

    build_dataset(args.output, args.frames, args.seed, args.overwrite)


def build_dataset(output_dir: Path, frame_count: int, seed: int, overwrite: bool = False) -> None:
    if frame_count <= 0:
        raise ValueError("--frames must be greater than 0")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for index in range(frame_count):
            room = ROOMS[index % len(ROOMS)]
            scene = SCENES[index % len(SCENES)]
            labels = _sample_labels(room, rng, index)
            image_name = f"scene_{scene.lower()}_frame_{index:05d}.png"
            image_path = frames_dir / image_name
            _write_synthetic_frame(image_path, room, labels, index)

            row = {
                "image_path": f"frames/{image_name}",
                "scene": scene,
                "agent_position": [
                    round(math.sin(index / 17) * 2.5, 3),
                    0.9,
                    round(math.cos(index / 19) * 2.5, 3),
                ],
                "agent_rotation": (index * 90) % 360,
                "visible_objects": labels["visible_objects"],
                "room_type": room,
                "navigation_labels": labels["navigation_labels"],
                "concept_labels": labels["concept_labels"],
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    taxonomy_source = Path(__file__).with_name("concept_taxonomy.yaml")
    shutil.copyfile(taxonomy_source, output_dir / "taxonomy.yaml")

    result = validate_metadata_file(metadata_path, output_dir)
    if not result.ok:
        joined = "\n".join(result.errors[:20])
        raise RuntimeError(f"generated dataset failed validation:\n{joined}")

    print(f"wrote {result.rows_checked} frames to {output_dir}")
    print(f"metadata: {metadata_path}")


def _sample_labels(room: str, rng: random.Random, index: int) -> dict[str, object]:
    objects = list(OBJECTS_BY_ROOM[room])
    rng.shuffle(objects)
    visible_objects = sorted(objects[: rng.randint(4, min(7, len(objects)))])

    door_visible = "Door" in visible_objects and rng.random() > 0.2
    obstacle_visible = any(item in visible_objects for item in ("Chair", "Table", "Sofa", "Bed"))
    path_blocked = obstacle_visible and rng.random() > 0.55
    reachable_goal_visible = any(item in visible_objects for item in ("Sink", "Fridge", "Bed", "Sofa"))

    if path_blocked:
        best_action = rng.choice(("RotateLeft", "RotateRight"))
    elif reachable_goal_visible and index % 9 == 0:
        best_action = "Stop"
    else:
        best_action = "MoveAhead"

    concept_labels = {
        "agent": False,
        "path": not path_blocked,
        "obstacle": obstacle_visible,
        "landmark": door_visible or "Cabinet" in visible_objects,
        "goal_object": reachable_goal_visible,
        "container": "Cabinet" in visible_objects or "Fridge" in visible_objects,
    }

    return {
        "visible_objects": visible_objects,
        "navigation_labels": {
            "door_visible": door_visible,
            "path_blocked": path_blocked,
            "obstacle_visible": obstacle_visible,
            "best_action": best_action,
            "reachable_goal_visible": reachable_goal_visible,
        },
        "concept_labels": concept_labels,
    }


def _write_synthetic_frame(
    path: Path,
    room: str,
    labels: dict[str, object],
    index: int,
    width: int = 160,
    height: int = 120,
) -> None:
    palette = {
        "Kitchen": (215, 229, 218),
        "LivingRoom": (221, 220, 235),
        "Bedroom": (230, 222, 210),
        "Bathroom": (214, 231, 237),
    }
    base = palette[room]
    nav = labels["navigation_labels"]
    concepts = labels["concept_labels"]
    pixels: list[tuple[int, int, int]] = []

    for y in range(height):
        for x in range(width):
            r = min(255, base[0] + y // 12)
            g = min(255, base[1] + x // 18)
            b = min(255, base[2] + (x + y + index) % 17)

            if y > height * 0.66:
                r, g, b = (174, 184, 170) if concepts["path"] else (157, 128, 118)
            if nav["door_visible"] and width * 0.68 < x < width * 0.86 and height * 0.18 < y < height * 0.75:
                r, g, b = 116, 86, 67
            if nav["obstacle_visible"] and width * 0.25 < x < width * 0.48 and height * 0.54 < y < height * 0.82:
                r, g, b = 89, 112, 145
            if concepts["goal_object"] and width * 0.08 < x < width * 0.22 and height * 0.34 < y < height * 0.56:
                r, g, b = 208, 196, 93

            pixels.append((r, g, b))

    path.write_bytes(_encode_png(width, height, pixels))


def _encode_png(width: int, height: int, pixels: list[tuple[int, int, int]]) -> bytes:
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row.extend(pixels[y * width + x])
        rows.append(bytes(row))

    raw = b"".join(rows)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw, level=6)),
            _png_chunk(b"IEND", b""),
        ]
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


if __name__ == "__main__":
    main()

