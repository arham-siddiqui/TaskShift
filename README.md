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

## AI2-THOR dataset builder

After the synthetic prototype pipeline is working, generate embodied RGB frames
from AI2-THOR while preserving the same TaskShift metadata contract:

```bash
python3 -m data.build_thor_dataset --frames 600 --output artifacts/thor_dataset --overwrite
```

Small smoke test:

```bash
python3 -m data.build_thor_dataset --frames 24 --scenes FloorPlan1 --output artifacts/thor_smoke_dataset --width 160 --height 120 --overwrite
python3 -m data.dataset_inspector artifacts/thor_smoke_dataset
```

The first AI2-THOR run downloads a Unity build into `~/.ai2thor`, which can take
several minutes. The generated dataset has the same files as the prototype:

- `frames/`: egocentric AI2-THOR RGB frames
- `metadata.jsonl`: scene, agent pose, visible objects, navigation labels, and concept labels
- `taxonomy.yaml`: copied TaskShift concept taxonomy

The current THOR labels combine visible-object metadata with lightweight
simulator action probes:

- passive labels come from visible AI2-THOR object types and room type
- `path_blocked` uses nearby obstacle metadata plus whether `MoveAhead`
  actually succeeds from the sampled pose
- `best_action` uses `Stop` for reachable visible goal objects, `MoveAhead`
  for clear forward motion, and otherwise chooses the better left/right turn
  by probing side views from the same position
- `reachable_goal_visible` requires a visible goal object that is interactable
  and within the visibility distance
- concept labels map visible simulator objects into `path`, `obstacle`, `landmark`, `goal_object`, and `container`

Once a THOR dataset is generated, the downstream commands are unchanged. Point
`TaskShiftDataset`, `models.train_heads`, `activations.extract`, probes, plots,
dashboards, or experiment sweeps at the THOR dataset directory.

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

Train with a real frozen DINOv2 backbone:

```bash
python3 -m models.train_heads --dataset artifacts/prototype_dataset --output-dir artifacts/checkpoints_dinov2 --backbone dinov2_vits14 --epochs 10 --batch-size 16
```

The first DINOv2 run downloads the official model through PyTorch Hub. Available
backbone names are `prototype`, `dinov2_vits14`, `dinov2_vitb14`,
`dinov2_vitl14`, and `dinov2_vitg14`.

If Python certificate setup blocks PyTorch Hub downloads, clone DINOv2 and
download the ViT-S/14 weight locally:

```bash
git clone --depth 1 https://github.com/facebookresearch/dinov2.git .external/dinov2
mkdir -p .external/dinov2/weights
curl -L https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth -o .external/dinov2/weights/dinov2_vits14_pretrain.pth
```

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

DINOv2 checkpoint extraction:

```bash
python3 -m activations.extract --dataset artifacts/prototype_dataset --checkpoint-dir artifacts/checkpoints_dinov2 --output-dir artifacts/activations_dinov2 --batch-size 16
```

Outputs:

- `artifacts/activations/passive_activations.pt`
- `artifacts/activations/navigation_activations.pt`

Each activation artifact contains:

- `activations["backbone_features"]`: frozen visual features
- `activations["head_hidden"]`: hidden representation inside the task head
- DINOv2 checkpoints also include selected transformer block activations such
  as `block_0`, `block_3`, `block_6`, `block_9`, and `block_11`
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

DINOv2 probes:

```bash
python3 -m analysis.linear_probes --activation-dir artifacts/activations_dinov2 --output-dir artifacts/probes_dinov2
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

DINOv2 representation shifts:

```bash
python3 -m analysis.representation_shift --probe-dir artifacts/probes_dinov2 --activation-dir artifacts/activations_dinov2 --output-dir artifacts/shift_metrics_dinov2
```

Outputs:

- `artifacts/shift_metrics/representation_shift.pt`
- `artifacts/shift_metrics/representation_shift_summary.json`

Run all current tests:

```bash
python3 -m unittest tests/test_dataset_schema.py tests/test_taskshift_dataset.py tests/test_models.py tests/test_activation_extraction.py tests/test_linear_probes.py tests/test_representation_shift.py
```

## Plot results

Generate static plots from the representation-shift summary:

```bash
python3 -m analysis.plots --summary artifacts/shift_metrics/representation_shift_summary.json --output-dir artifacts/plots
```

DINOv2 plots:

```bash
python3 -m analysis.plots --summary artifacts/shift_metrics_dinov2/representation_shift_summary.json --output-dir artifacts/plots_dinov2
```

Outputs:

- `artifacts/plots/backbone_features_concept_shift.png`
- `artifacts/plots/backbone_features_tuning_correlation.png`
- `artifacts/plots/head_hidden_concept_shift.png`
- `artifacts/plots/head_hidden_tuning_correlation.png`
- `artifacts/plots/cka_heatmap.png`

Run all current tests:

```bash
python3 -m unittest tests/test_dataset_schema.py tests/test_taskshift_dataset.py tests/test_models.py tests/test_activation_extraction.py tests/test_linear_probes.py tests/test_representation_shift.py tests/test_plots.py
```

## Build static dashboard

Generate a local HTML dashboard from the shift summary and plot images:

```bash
python3 -m dashboard.build_static --summary artifacts/shift_metrics/representation_shift_summary.json --plot-dir artifacts/plots --output artifacts/dashboard/index.html
```

DINOv2 dashboard:

```bash
python3 -m dashboard.build_static --summary artifacts/shift_metrics_dinov2/representation_shift_summary.json --plot-dir artifacts/plots_dinov2 --output artifacts/dashboard_dinov2/index.html
```

Open:

```text
artifacts/dashboard/index.html
```

Run all current tests:

```bash
python3 -m unittest tests/test_dataset_schema.py tests/test_taskshift_dataset.py tests/test_models.py tests/test_activation_extraction.py tests/test_linear_probes.py tests/test_representation_shift.py tests/test_plots.py tests/test_dashboard.py
```

## Run experiment sweeps

Once the single-run pipeline works, the next step is checking whether the
representation-shift results are repeatable across seeds and model conditions.
The sweep runner executes the full pipeline for each requested run:

1. generate a dataset
2. train passive and navigation heads
3. extract activations
4. train concept probes
5. compare representations
6. generate plots and a per-run dashboard
7. build a cross-run comparison dashboard

Lightweight prototype sweep:

```bash
python3 -m experiments.run_sweep --experiment prototype_seed_sweep --backbones prototype --train-backbone-modes none --seeds 17 23 31 --frames 300 --epochs 5
```

AI2-THOR frozen vs final-block DINOv2 sweep:

```bash
python3 -m experiments.run_sweep --experiment thor_dinov2_seed_sweep --dataset-kind thor --backbones dinov2_vits14 --train-backbone-modes none final_block --seeds 17 23 31 --frames 600 --epochs 3 --batch-size 8
```

DINOv2 backbone-tuning ladder:

```bash
python3 -m experiments.run_sweep --experiment dinov2_depth_sweep --backbones dinov2_vits14 --train-backbone-modes none final_block last_2_blocks last_4_blocks --seeds 17 23 31 --frames 300 --epochs 3 --batch-size 8
```

Available backbone tuning modes are `none`, `final_block`, `last_2_blocks`,
`last_4_blocks`, and `all`. The `all` mode is useful as a stress test, but it
is much slower and easier to overfit on the prototype dataset.

Outputs are grouped under:

- `artifacts/experiments/<experiment>/runs/<run_id>/`
- `artifacts/experiments/<experiment>/comparison/comparison_summary.json`
- `artifacts/experiments/<experiment>/comparison/index.html`

Each run also writes `run_manifest.json`, which records the seed, backbone,
backbone tuning mode, and artifact paths used to produce the result.

Run all current tests:

```bash
python3 -m unittest tests/test_dataset_schema.py tests/test_taskshift_dataset.py tests/test_models.py tests/test_activation_extraction.py tests/test_linear_probes.py tests/test_representation_shift.py tests/test_plots.py tests/test_dashboard.py tests/test_experiments.py
```

## Statistical validation

After a multi-seed sweep, add uncertainty estimates and a paired permutation
test to the cross-run comparison. The stats script compares matched seeds across
two conditions, bootstraps confidence intervals over seed-level differences,
and runs an exact paired sign-flip permutation test.

THOR frozen vs final-block validation:

```bash
python3 -m analysis.stats --comparison artifacts/experiments/thor_dinov2_seed_sweep/comparison/comparison_summary.json --baseline-condition dinov2_vits14:none --treatment-condition dinov2_vits14:final_block --output-dir artifacts/experiments/thor_dinov2_seed_sweep/stats
```

Outputs:

- `artifacts/experiments/<experiment>/stats/stats_summary.json`
- `artifacts/experiments/<experiment>/stats/index.html`
- `artifacts/experiments/<experiment>/stats/plots/metric_effects.png`
- `artifacts/experiments/<experiment>/stats/plots/concept_effects.png`

The reported mean difference is always `treatment - baseline`. For the THOR
example, negative CKA differences mean the final-block condition became less
similar between passive and navigation models; positive concept-shift
differences mean the final-block condition produced larger concept shifts.

Run all current tests:

```bash
python3 -m unittest tests/test_dataset_schema.py tests/test_taskshift_dataset.py tests/test_models.py tests/test_activation_extraction.py tests/test_linear_probes.py tests/test_representation_shift.py tests/test_plots.py tests/test_dashboard.py tests/test_experiments.py tests/test_stats.py
```
