# TaskShift
Measuring task-dependent representation shifts in vision models
paper: https://www.biorxiv.org/content/10.64898/2026.02.19.706797v1.full

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
