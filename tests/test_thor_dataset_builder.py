import unittest

from data.build_thor_dataset import (
    derive_concept_labels,
    derive_navigation_labels,
    distribute_frames,
    room_type_for_scene,
)


class ThorDatasetBuilderTest(unittest.TestCase):
    def test_distributes_frames_across_scenes(self) -> None:
        self.assertEqual(distribute_frames(10, 4), [3, 3, 2, 2])

    def test_room_type_for_ai2thor_scene_ranges(self) -> None:
        self.assertEqual(room_type_for_scene("FloorPlan1"), "Kitchen")
        self.assertEqual(room_type_for_scene("FloorPlan201"), "LivingRoom")
        self.assertEqual(room_type_for_scene("FloorPlan301"), "Bedroom")
        self.assertEqual(room_type_for_scene("FloorPlan401"), "Bathroom")

    def test_navigation_and_concept_heuristics(self) -> None:
        objects = ["Cabinet", "Chair", "Mug"]
        metadata = [
            {"objectType": "Chair", "visible": True, "distance": 0.9},
            {"objectType": "Mug", "visible": True, "distance": 1.1},
        ]

        navigation = derive_navigation_labels(objects, metadata)
        concepts = derive_concept_labels(objects, navigation)

        self.assertTrue(navigation["path_blocked"])
        self.assertTrue(navigation["obstacle_visible"])
        self.assertTrue(navigation["reachable_goal_visible"])
        self.assertEqual(navigation["best_action"], "RotateLeft")
        self.assertTrue(concepts["obstacle"])
        self.assertTrue(concepts["goal_object"])
        self.assertTrue(concepts["container"])
        self.assertFalse(concepts["path"])


if __name__ == "__main__":
    unittest.main()
