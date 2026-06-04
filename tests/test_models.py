import unittest

import torch

from data.taskshift_dataset import NAVIGATION_BINARY_KEYS, OBJECT_VOCAB, ROOM_VOCAB
from models.backbone import build_backbone
from models.heads import NavigationHead, PassiveHead


class ModelComponentTest(unittest.TestCase):
    def test_heads_match_expected_output_shapes(self) -> None:
        backbone = build_backbone("prototype")
        images = torch.rand(4, 3, 120, 160)
        features = backbone(images)

        passive = PassiveHead(backbone.feature_dim, len(OBJECT_VOCAB), len(ROOM_VOCAB))
        passive_outputs = passive(features)
        self.assertEqual(tuple(passive_outputs["object_logits"].shape), (4, len(OBJECT_VOCAB)))
        self.assertEqual(tuple(passive_outputs["room_logits"].shape), (4, len(ROOM_VOCAB)))

        navigation = NavigationHead(backbone.feature_dim, len(NAVIGATION_BINARY_KEYS), 4)
        navigation_outputs = navigation(features)
        self.assertEqual(
            tuple(navigation_outputs["binary_logits"].shape),
            (4, len(NAVIGATION_BINARY_KEYS)),
        )
        self.assertEqual(tuple(navigation_outputs["action_logits"].shape), (4, 4))


if __name__ == "__main__":
    unittest.main()

