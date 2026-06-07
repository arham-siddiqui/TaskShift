import unittest

import torch
from PIL import Image

from data.taskshift_dataset import NAVIGATION_BINARY_KEYS, OBJECT_VOCAB, ROOM_VOCAB
from activations.extract import resolve_backbone_name
from models.backbone import (
    BACKBONE_TRAINING_MODES,
    build_backbone,
    configure_backbone_training,
    image_transform_for_backbone,
)
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

    def test_dinov2_transform_shape_without_loading_model(self) -> None:
        transform = image_transform_for_backbone("dinov2_vits14")
        image = Image.new("RGB", (160, 120), color=(128, 128, 128))
        tensor = transform(image)

        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_legacy_prototype_checkpoint_name_resolves(self) -> None:
        checkpoint = {"backbone": {"name": "frozen_prototype_grid8"}}

        self.assertEqual(resolve_backbone_name(checkpoint), "prototype")

    def test_backbone_training_modes_require_dinov2(self) -> None:
        backbone = build_backbone("prototype")

        for mode in BACKBONE_TRAINING_MODES:
            if mode == "none":
                continue
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    configure_backbone_training(backbone, mode)


if __name__ == "__main__":
    unittest.main()
