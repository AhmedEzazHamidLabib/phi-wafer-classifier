import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from PIL import Image
import json
import random
import copy

# ── REPRODUCIBILITY ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_pickle("../data/labeled_only.pkl")

# ── 2. PREPROCESS ─────────────────────────────────────────────────────────────
TARGET_SIZE = 32

def preprocess_map(wmap):
    arr = np.array(wmap, dtype=np.float32)
    img = Image.fromarray(arr)
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)
    return np.array(img) / 2.0

df['processed'] = df['waferMap'].apply(preprocess_map)

# ── 3. ENCODE ─────────────────────────────────────────────────────────────────
le = LabelEncoder()
df['label'] = le.fit_transform(df['failureType'])
NUM_CLASSES  = len(le.classes_)

# ── 4. SPLIT ──────────────────────────────────────────────────────────────────
X = df['processed'].tolist()
y = df['label'].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# ── 5. DATASET ────────────────────────────────────────────────────────────────
class WaferDataset(Dataset):
    def __init__(self, maps, labels):
        self.maps   = torch.tensor(
            np.stack(maps), dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.maps[idx], self.labels[idx]

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    WaferDataset(X_train, y_train), batch_size=64,
    shuffle=True, generator=g)
test_loader  = DataLoader(
    WaferDataset(X_test,  y_test),  batch_size=64, shuffle=False)

# ── 6. MODEL FACTORY ──────────────────────────────────────────────────────────
def make_model():
    torch.manual_seed(SEED)
    class WaferCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return WaferCNN(NUM_CLASSES)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 7. PHI BASELINE INIT ──────────────────────────────────────────────────────
with open("../results/baseline_report.json") as f:
    baseline = json.load(f)

baseline_phi = torch.tensor([
    baseline[cls]['f1-score'] for cls in le.classes_
], dtype=torch.float32)

# ── 8. TRAIN FUNCTION ─────────────────────────────────────────────────────────
def train_phi(alpha_max, beta_min, vel_scale, phi_ema,
              vel_clamp_lo=0.5, vel_clamp_hi=2.0, epochs=20):

    model     = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    class_phi     = baseline_phi.clone()
    class_correct = torch.zeros(NUM_CLASSES)
    class_total   = torch.zeros(NUM_CLASSES)
    phi_history   = [class_phi.clone()]

    def compute_amplifier(true_labels, phi_vector, phi_vel):
        amplifier = torch.ones(len(true_labels), device=true_labels.device)
        for i in range(len(true_labels)):
            c     = true_labels[i].item()
            phi_c = phi_vector[c].item()
            vel_c = phi_vel[c].item()
            base  = alpha_max * (1 - phi_c) + beta_min * phi_c
            vf    = max(vel_clamp_lo,
                        min(vel_clamp_hi, 1.0 - vel_c * vel_scale))
            amplifier[i] = base * vf
        return amplifier

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total = 0, 0, 0

        phi_vel = (phi_history[-1] - phi_history[-2]
                   if len(phi_history) >= 2
                   else torch.zeros(NUM_CLASSES))

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out             = model(xb)
            preds           = out.argmax(1)
            loss_per_sample = nn.CrossEntropyLoss(reduction='none')(out, yb)
            amplifier       = compute_amplifier(
                yb, class_phi.clamp(0,1), phi_vel).to(device)
            loss = (loss_per_sample * amplifier).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for c in range(NUM_CLASSES):
                mask = (yb == c)
                if mask.sum() > 0:
                    class_correct[c] += (preds[mask] == yb[mask]).sum().item()
                    class_total[c]   += mask.sum().item()

            total_loss    += loss.item() * len(yb)
            total_correct += (preds == yb).sum().item()
            total         += len(yb)

        for c in range(NUM_CLASSES):
            if class_total[c] > 0:
                acc = class_correct[c] / class_total[c]
                class_phi[c] = phi_ema * class_phi[c] + (1 - phi_ema) * acc

        phi_history.append(class_phi.clone())
        class_correct.zero_()
        class_total.zero_()

        print(f"  Epoch {epoch+1:02d} | "
              f"loss {total_loss/total:.4f} | "
              f"acc {total_correct/total:.3f}")

    # evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
            all_labels.extend(yb.numpy())

    return classification_report(
        all_labels, all_preds,
        target_names=le.classes_,
        output_dict=True
    )

# ── 9. PARAMETER SWEEP ────────────────────────────────────────────────────────
configs = {
    'phi_default':     dict(alpha_max=3.0, beta_min=0.3, vel_scale=5.0, phi_ema=0.8),
    'phi_aggressive':  dict(alpha_max=5.0, beta_min=0.3, vel_scale=5.0, phi_ema=0.8),
    'phi_responsive':  dict(alpha_max=3.0, beta_min=0.3, vel_scale=3.0, phi_ema=0.7),
    'phi_combined':    dict(alpha_max=5.0, beta_min=0.3, vel_scale=3.0, phi_ema=0.7),
}

reports = {}
for name, cfg in configs.items():
    print(f"\n{'='*60}")
    print(f"Running: {name} | {cfg}")
    print('='*60)
    reports[name] = train_phi(**cfg)

# ── 10. COMPARE ───────────────────────────────────────────────────────────────
with open("../results/baseline_report.json") as f:
    baseline_r = json.load(f)

print("\n── PARAMETER SWEEP RESULTS ──")
print(f"{'Class':<12} {'Base':>7}", end="")
for name in configs:
    print(f" {name[:12]:>12}", end="")
print()
print("-" * (12 + 8 + 13 * len(configs)))

for cls in le.classes_:
    b = baseline_r[cls]['f1-score']
    print(f"{cls:<12} {b:>7.4f}", end="")
    for name in configs:
        v = reports[name][cls]['f1-score']
        d = v - b
        print(f" {'↑' if d>0 else '↓'}{abs(d):.3f}/{v:.3f}", end="")
    print()

b_macro = baseline_r['macro avg']['f1-score']
print(f"\n{'Macro':<12} {b_macro:>7.4f}", end="")
for name in configs:
    v = reports[name]['macro avg']['f1-score']
    d = v - b_macro
    print(f" {'↑' if d>0 else '↓'}{abs(d):.3f}/{v:.3f}", end="")
print()

# save best
best_name = max(reports, key=lambda n: reports[n]['macro avg']['f1-score'])
print(f"\nBest config: {best_name}")
with open("../results/phi_sweep_best_report.json", "w") as f:
    json.dump(reports[best_name], f, indent=2)