# TaskShift
Measuring task-dependent representation shifts in vision models

## paper: https://www.biorxiv.org/content/10.64898/2026.02.19.706797v1.full

## First milestone: dataset contract

TaskShift starts by locking down the dataset schema shared by passive
recognition labels, navigation labels, and later semantic probes.

Generate a small prototype dataset:

```bash
python3 -m data.build_prototype_dataset --frames 300 --output artifacts/prototype_dataset --overwrite
```

Inspect and validate it:

```bash
python3 -m data.dataset_inspector artifacts/prototype_dataset
```

The generated prototype contains:

- `frames/`: synthetic indoor PNG frames
- `metadata.jsonl`: one metadata record per frame
- `taxonomy.yaml`: shared object, scene, navigation, affordance, and agency concepts

This synthetic dataset is a contract test for the pipeline. The next version
can replace the frame generator with AI2-THOR or ProcTHOR while preserving the
same metadata shape.

## PyTorch dataset loader

The `TaskShiftDataset` loader exposes one sample with all labels needed by the
next stages:

```python
from data.taskshift_dataset import TaskShiftDataset

dataset = TaskShiftDataset("artifacts/prototype_dataset")
sample = dataset[0]

image = sample["image"]
passive_targets = sample["passive_targets"]
navigation_targets = sample["navigation_targets"]
concept_targets = sample["concept_targets"]
```

Target groups:

- `passive_targets["objects"]`: multi-hot visible-object labels
- `passive_targets["room"]`: room class index
- `navigation_targets["binary"]`: door/path/obstacle/reachability labels
- `navigation_targets["action"]`: best-action class index
- `concept_targets`: shared semantic labels for later representation probes

Concept labels are analysis labels, not a third model-training task. Use
`dataset.active_concepts()` to skip concepts with no positive examples in the
current dataset.

Run the data tests:

```bash
python3 -m unittest tests/test_dataset_schema.py tests/test_taskshift_dataset.py
```

## Train passive and navigation heads

In this project, a **head** is the small task-specific neural network attached
to a shared visual backbone. The backbone converts an image into features; the
head converts those features into labels for one task.

For the current prototype, the backbone is a frozen PyTorch-only image
featurizer so the training pipeline can run locally without downloading DINOv2.
Later, this backbone can be replaced with DINOv2 while keeping the passive and
navigation head structure.

Train both heads:

```bash
python3 -m models.train_heads --dataset artifacts/prototype_dataset --epochs 20
```

Outputs:

- `artifacts/checkpoints/passive_head.pt`
- `artifacts/checkpoints/navigation_head.pt`

The passive head predicts visible objects and room type. The navigation head
predicts door/path/obstacle/reachability labels and best action.

Run all current tests:

```bash
python3 -m unittest tests/test_dataset_schema.py tests/test_taskshift_dataset.py tests/test_models.py
```

## Extract activations

Activation extraction runs the dataset through a trained checkpoint and saves
the internal tensors needed by probes and representation-shift metrics.

```bash
python3 -m activations.extract --dataset artifacts/prototype_dataset --checkpoint-dir artifacts/checkpoints
```

Outputs:

- `artifacts/activations/passive_activations.pt`
- `artifacts/activations/navigation_activations.pt`

Each activation artifact contains:

- `activations["backbone_features"]`: frozen visual features
- `activations["head_hidden"]`: hidden representation inside the task head
- `logits`: model outputs
- `targets`: passive, navigation, and concept labels
- `metadata`: original frame metadata
- `vocab`: column names for target tensors

Run all current tests:

```bash
python3 -m unittest tests/test_dataset_schema.py tests/test_taskshift_dataset.py tests/test_models.py tests/test_activation_extraction.py
```

## Train concept probes

Concept probes are simple linear classifiers trained after the task heads. They
ask whether a concept such as `path`, `obstacle`, or `goal_object` is linearly
decodable from a saved activation.

```bash
python3 -m analysis.linear_probes --activation-dir artifacts/activations --output-dir artifacts/probes
```

Outputs:

- `artifacts/probes/passive_concept_probes.pt`
- `artifacts/probes/navigation_concept_probes.pt`

Each probe artifact contains one weight vector per trained concept and
activation source. Concepts with too few positives, such as `agent` in the
prototype dataset, are skipped automatically.

Run all current tests:

```bash
python3 -m unittest tests/test_dataset_schema.py tests/test_taskshift_dataset.py tests/test_models.py tests/test_activation_extraction.py tests/test_linear_probes.py
```

## Compare representation shifts

Representation comparison turns probe weights and activations into the first
TaskShift metrics:

- tuning-vector correlation between passive and navigation concept probes
- concept shift magnitude for each shared concept
- linear CKA similarity between passive and navigation activations

```bash
python3 -m analysis.representation_shift --probe-dir artifacts/probes --activation-dir artifacts/activations --output-dir artifacts/shift_metrics
```

Outputs:

- `artifacts/shift_metrics/representation_shift.pt`
- `artifacts/shift_metrics/representation_shift_summary.json`

Run all current tests:

```bash
python3 -m unittest tests/test_dataset_schema.py tests/test_taskshift_dataset.py tests/test_models.py tests/test_activation_extraction.py tests/test_linear_probes.py tests/test_representation_shift.py
```
