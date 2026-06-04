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
