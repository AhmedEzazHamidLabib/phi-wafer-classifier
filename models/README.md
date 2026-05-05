# Models

Trained model weights are not stored in this repository due to file size constraints.

## Available Models

| Model | Description | Macro F1 |
|-------|-------------|----------|
| baseline_cnn.pt | Standard CNN, uniform loss, no phi | 0.8435 |
| baseline_aug_cnn.pt | Standard CNN with augmentation, no phi | 0.8675 |
| phi_no_aug_cnn.pt | Phi v4, ALPHA_MAX=5.0, no augmentation | 0.8650 |
| phi_final_cnn.pt | Phi v4 with augmentation (primary result) | 0.8785 |
| phi_focal_cnn.pt | Focal-informed phi update with augmentation | 0.8760 |

## Download

Model weights are available via Google Drive: [link to be added]

## Reproducing From Scratch

All models are fully reproducible with SEED=42. Run in this order:

```bash
python src/baseline.py
python src/baseline_aug.py
python src/phi_final_no_aug.py
python src/phi_final.py
python src/phi_final_focal.py
```

Each script saves its weights to this folder automatically.
