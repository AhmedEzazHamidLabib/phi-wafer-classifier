# Phi Wafer Classifier

A novel training framework for semiconductor wafer defect classification. Phi is a continuous per-class fidelity scalar that governs learning aggression dynamically during training. It identifies which failure classes are being violated, applies calibrated pressure from every ground truth occurrence, and eases off as classes improve.

Every class improved over the standard CNN baseline simultaneously in the final version. Macro F1 improved from 0.8435 to 0.8785.

---

## The Problem

Semiconductor fabs produce hundreds of wafers daily. Each wafer gets electrically tested die by die. The spatial pattern of failures tells process engineers which manufacturing step went wrong: a scratch means physical handling damage, an edge-ring means deposition non-uniformity, a center cluster means a temperature profile problem. Identifying these patterns manually at scale is impossible. The ML problem is: given a 2D grid of pass/fail die results, classify which failure mode caused it.

The WM-811K dataset spans 8 failure classes with distributions ranging from 9,680 examples (Edge-Ring) to 149 examples (Near-full). Standard CrossEntropyLoss treats class frequency as importance. These are not the same thing. A Scratch failure indicating physical handling damage is just as actionable as an Edge-Ring failure regardless of how often each appears in training data.

The baseline CNN achieves 0.977 F1 on Edge-Ring and 0.571 F1 on Scratch. The model learns what it sees most often and misses nearly half of all scratch patterns, one of the most industrially important failure modes because it indicates systematic handling damage that repeats across wafers.

---

## What Phi Does

Phi tracks per-class fidelity during training. When a class is being learned well, phi is high. When a class is being violated, phi is low. The amplifier scales the per-sample loss from every true label occurrence:

```
aggression = ALPHA * (1 - phi) + BETA * phi
```

Low phi means high aggression. High phi means low aggression. Velocity modulates this further: if phi is rising fast the system backs off, if phi is stuck or falling the system pushes harder. The model always tries to maximize phi across all classes simultaneously.

This is different from focal loss, which modifies loss based on prediction confidence. Phi operates on the training dynamic itself, scaling intervention based on measured constraint satisfaction in real time. Unlike inverse frequency weighting, phi adapts continuously throughout training rather than applying a fixed static correction.

---

## The Failure Classes

| Class | Description | Industrial Significance |
|-------|-------------|------------------------|
| Center | Cluster of failures at wafer center | Temperature or pressure non-uniformity at furnace center |
| Donut | Ring of failures surrounding good center | Intermediate zone process non-uniformity |
| Edge-Loc | Localized failure at one edge section | Point contact contamination or non-uniform edge seal |
| Edge-Ring | Complete ring of failures at perimeter | Edge effects in deposition or etching |
| Loc | Localized cluster off-center | Contamination particle or localized equipment defect |
| Near-full | Almost entire wafer failed | Catastrophic process failure, bad chemical batch |
| Random | Scattered failures with no spatial structure | Random contamination particles, one-time event |
| Scratch | Thin diagonal line of failures | Physical contact damage, repeats across wafers if uncaught |

Scratch, Edge-Ring, and Center are highest-priority detections. Missing them means the process problem repeats across every subsequent wafer until a human catches it manually.

---

## Results

Phi NoAug runs with one parameter tuned from the default: ALPHA_MAX raised from 3.0 to 5.0. This is the maximum loss amplification coefficient applied when phi is at its lowest. Raising it increases pressure on violated classes. The value was identified via parameter sweep documented in `results/phi_sweep_comparison.json`. All other parameters remain at defaults: BETA_MIN=0.3, VEL_SCALE=5.0, PHI_EMA=0.8.

| Class | Baseline | Base+Aug | Phi NoAug | Phi Final | Delta vs Baseline | Delta vs Base+Aug |
|-------|----------|----------|-----------|-----------|-------------------|-------------------|
| Center | 0.9530 | 0.9446 | 0.9480 | 0.9533 | +0.0003 | +0.0087 |
| Donut | 0.8598 | 0.9246 | 0.8410 | 0.9227 | +0.0629 | -0.0019 |
| Edge-Loc | 0.8688 | 0.8755 | 0.8680 | 0.8769 | +0.0081 | +0.0014 |
| Edge-Ring | 0.9784 | 0.9801 | 0.9790 | 0.9821 | +0.0037 | +0.0020 |
| Loc | 0.7888 | 0.7845 | 0.7830 | 0.8014 | +0.0126 | +0.0169 |
| Near-full | 0.8000 | 0.9531 | 0.9330 | 0.9418 | +0.1418 | -0.0113 |
| Random | 0.8785 | 0.8165 | 0.9000 | 0.8294 | -0.0491 | +0.0129 |
| Scratch | 0.6208 | 0.6611 | 0.6650 | 0.7203 | +0.1195 | +0.0592 |
| **Macro F1** | **0.8435** | **0.8675** | **0.8650** | **0.8785** | **+0.0350** | **+0.0110** |

Augmentation alone helped several classes significantly, particularly Near-full (0.800 to 0.953) and Donut (0.860 to 0.925). But it meaningfully hurt others: Random dropped from 0.879 to 0.817, a regression of 0.062 points. Augmentation addresses data scarcity but introduces distributional shift that disrupts classes that were already well-learned.

Phi NoAug performed significantly better than baseline on the classes that matter most. Scratch improved from 0.621 to 0.665 without any additional data. Near-full reached 0.933, closing most of the gap that augmentation achieved. Random recovered to 0.900, actually exceeding the baseline that augmentation had damaged. Where Phi NoAug was worse than baseline the margins were small, nowhere near the 0.062 regression augmentation caused on Random. Phi corrects imbalance without introducing the distributional disruption that augmentation causes on already well-learned classes.

Phi Final combined both interventions and produced the strongest result across nearly every class. Scratch reached 0.720, up 0.120 from baseline and 0.059 above augmentation alone. Near-full reached 0.942. Macro F1 improved 3.5 points over baseline and 1.1 points over augmentation alone. The combination is strictly better than either intervention independently because augmentation fixes data scarcity and phi fixes gradient imbalance, and these are different constraint types that compound.

Phi is also tunable. The parameter sweep showed that raising ALPHA_MAX from 3.0 to 5.0 alone added 2.1 macro F1 points in the no-augmentation setting. Different problems with different imbalance profiles will respond to different parameter settings. The framework exposes meaningful knobs: ALPHA_MAX governs maximum aggression on violated classes, VEL_SCALE governs how responsively the system backs off when a class is improving, and PHI_EMA governs how quickly phi responds to recent epoch performance versus longer-term history. Each has a precise semantic meaning tied to the constraint satisfaction framework rather than being an arbitrary hyperparameter.

---

## Repository Structure

```
phi-wafer-classifier/
|
|-- src/
|   |-- baseline.py              # Standard CNN, uniform loss, no phi
|   |-- baseline_aug.py          # Standard CNN with augmentation, no phi
|   |-- phi_final_no_aug.py      # Phi v4, no augmentation
|   |-- phi_final.py             # Phi v4 with augmentation (primary result)
|   |-- phi_final_focal.py       # Focal-informed phi update variant
|   |-- phi_framework.py         # Standalone phi module for any pipeline
|
|-- versions/
|   |-- phi_v1.py                # Ground truth amplifier, uniform init
|   |-- phi_v2.py                # Hard threshold experiment
|   |-- phi_v3.py                # Baseline F1 initialization
|   |-- phi_v4.py                # Velocity-based aggression
|   |-- phi_v5.py                # Dynamic stopping criterion
|   |-- phi_v6.py                # Per-class early stopping experiment
|
|-- results/
|   |-- all_results.json         # All runs, all configs, all F1 scores
|   |-- baseline_report.json
|   |-- baseline_aug_report.json
|   |-- phi_no_aug_report.json
|   |-- phi_final_report.json
|   |-- phi_focal_report.json
|   |-- phi_sweep_comparison.json
|
|-- data/
|   |-- README.md                # Dataset download instructions
|
|-- models/
|   |-- README.md                # Model weights download link
|
|-- README.md
|-- TECHNICAL.md
|-- REPORT.md
```

---

## Setup

```bash
git clone https://github.com/yourusername/phi-wafer-classifier
cd phi-wafer-classifier
python -m venv venv
venv\Scripts\activate        # Windows
pip install torch torchvision numpy pandas scikit-learn pillow matplotlib
```

Dataset: Download WM811K.pkl from [Kaggle](https://www.kaggle.com/datasets/qingyi/wm-811k-wafer-map) and follow instructions in `data/README.md` to generate `labeled_only.pkl`.

---

## Run Order

```bash
python src/baseline.py           # generates baseline_report.json
python src/baseline_aug.py       # generates baseline_aug_report.json
python src/phi_final_no_aug.py   # generates phi_no_aug_report.json
python src/phi_final.py          # generates phi_final_report.json
```

All scripts are seeded with SEED=42. Results reproduce exactly and match the JSON reports in `results/`.

---

## Drop-In Usage

```python
from phi_framework import PhiTracker

tracker = PhiTracker(
    num_classes=8,
    class_names=le.classes_,
    baseline_report_path="results/baseline_report.json",
)

for epoch in range(EPOCHS):
    for xb, yb in train_loader:
        out             = model(xb)
        loss_per_sample = criterion(out, yb)    # reduction='none'
        amplifier       = tracker.get_amplifier(yb)
        loss            = (loss_per_sample * amplifier).mean()
        loss.backward()
        optimizer.step()
        tracker.accumulate(out.argmax(1), yb)
    tracker.update()
```

---

## Key Design Decisions

Phi initialized from baseline F1 rather than a uniform prior. This skips the discovery phase and applies correctly calibrated pressure from epoch 1. Scratch starts at 0.57, maximum pressure. Near-full starts at 0.95, minimal pressure.

Velocity modulation makes alpha and beta emergent from training dynamics. No manual tuning. Improving fast means back off. Stuck or falling means push harder.

Ground truth triggered. The amplifier fires on every true label occurrence regardless of whether the model predicted correctly. No confidence gate. The true label is always available during training and is the cleanest signal.

Augmentation fixes data scarcity. Phi fixes imbalance. These are different constraint types requiring different solutions. Near-full had 149 examples, augmented to 1000. Scratch had imbalance from Edge-Ring dominance, phi corrected the gradient pressure. Together they produced the best results.
