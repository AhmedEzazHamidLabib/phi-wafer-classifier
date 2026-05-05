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

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_pickle("labeled_only.pkl")
print("Loaded:", df.shape)

# ── 2. PREPROCESS ─────────────────────────────────────────────────────────────
TARGET_SIZE = 32

def preprocess_map(wmap):
    arr = np.array(wmap, dtype=np.float32)
    img = Image.fromarray(arr)
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)
    return np.array(img) / 2.0

print("Preprocessing...")
df['processed'] = df['waferMap'].apply(preprocess_map)

# ── 3. ENCODE ─────────────────────────────────────────────────────────────────
le = LabelEncoder()
df['label'] = le.fit_transform(df['failureType'])
NUM_CLASSES  = len(le.classes_)
print("Classes:", list(le.classes_))

# ── 4. SPLIT ──────────────────────────────────────────────────────────────────
X = df['processed'].tolist()
y = df['label'].tolist()

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1,
    stratify=y_trainval, random_state=42
)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

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

train_loader = DataLoader(
    WaferDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(
    WaferDataset(X_val,   y_val),   batch_size=64, shuffle=False)
test_loader  = DataLoader(
    WaferDataset(X_test,  y_test),  batch_size=64, shuffle=False)

# ── 6. MODEL ──────────────────────────────────────────────────────────────────
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

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
model     = WaferCNN(NUM_CLASSES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── 7. PHI + REGIME SETUP ─────────────────────────────────────────────────────
with open("baseline_report.json") as f:
    baseline = json.load(f)

baseline_phi = torch.tensor([
    baseline[cls]['f1-score'] for cls in le.classes_
], dtype=torch.float32)

class_phi = baseline_phi.clone()

BASELINE_CUTOFF  = 0.93
PHI_STOP         = 0.95
PHI_FREEZE_MIN   = 0.92   # phi must be above this to allow freezing
MAX_EPOCHS       = 50
PATIENCE_LIMIT   = 5

baseline_regime = [le.classes_[c] for c in range(NUM_CLASSES)
                   if baseline_phi[c].item() >= BASELINE_CUTOFF]
phi_regime      = [le.classes_[c] for c in range(NUM_CLASSES)
                   if baseline_phi[c].item() < BASELINE_CUTOFF]

print(f"\nBaseline regime: {baseline_regime}")
print(f"Phi regime:      {phi_regime}")

best_val_f1    = {c: 0.0 for c in range(NUM_CLASSES)}
patience       = {c: 0   for c in range(NUM_CLASSES)}
frozen_classes = set()

class_correct  = torch.zeros(NUM_CLASSES)
class_total    = torch.zeros(NUM_CLASSES)
phi_history    = [class_phi.clone()]

ALPHA_MAX    = 3.0
BETA_MIN     = 0.3
VEL_SCALE    = 5.0
VEL_CLAMP_LO = 0.5
VEL_CLAMP_HI = 2.0

def compute_amplifier(true_labels, phi_vector, phi_vel,
                      baseline_phi, frozen_classes):
    amplifier = torch.ones(len(true_labels), device=true_labels.device)
    for i in range(len(true_labels)):
        c = true_labels[i].item()

        # frozen — classifier output locked, conv still gets gradient
        if c in frozen_classes:
            amplifier[i] = 1.0  # keep conv healthy, classifier frozen via hook

        # baseline regime — normal CE
        elif baseline_phi[c].item() >= BASELINE_CUTOFF:
            amplifier[i] = 1.0

        # phi regime, constraint satisfied — freeze gradient
        elif phi_vector[c].item() >= PHI_STOP:
            amplifier[i] = 0.0

        # phi regime, still learning
        else:
            phi_c = phi_vector[c].item()
            vel_c = phi_vel[c].item()
            base  = ALPHA_MAX * (1 - phi_c) + BETA_MIN * phi_c
            vf    = max(VEL_CLAMP_LO,
                        min(VEL_CLAMP_HI, 1.0 - vel_c * VEL_SCALE))
            amplifier[i] = base * vf

    return amplifier

# ── 8. TRAIN ──────────────────────────────────────────────────────────────────
epoch          = 0
stopped_reason = "max epochs reached"

while epoch < MAX_EPOCHS:
    epoch += 1
    model.train()
    total_loss, total_correct, total = 0, 0, 0

    if len(phi_history) >= 2:
        phi_vel = phi_history[-1] - phi_history[-2]
    else:
        phi_vel = torch.zeros(NUM_CLASSES)

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        out   = model(xb)
        preds = out.argmax(1)

        criterion       = nn.CrossEntropyLoss(reduction='none')
        loss_per_sample = criterion(out, yb)

        phi_norm  = class_phi.clamp(0, 1)
        amplifier = compute_amplifier(
            yb, phi_norm, phi_vel, baseline_phi, frozen_classes
        ).to(device)

        loss = (loss_per_sample * amplifier).mean()

        optimizer.zero_grad()
        loss.backward()

        # freeze only classifier output weights for frozen classes
        # conv layers keep receiving gradient from all classes
        for c in frozen_classes:
            if model.classifier[-1].weight.grad is not None:
                model.classifier[-1].weight.grad[c] = 0
            if model.classifier[-1].bias.grad is not None:
                model.classifier[-1].bias.grad[c]   = 0

        optimizer.step()

        for c in range(NUM_CLASSES):
            mask = (yb == c)
            if mask.sum() > 0:
                class_correct[c] += (preds[mask] == yb[mask]).sum().item()
                class_total[c]   += mask.sum().item()

        total_loss    += loss.item() * len(yb)
        total_correct += (preds == yb).sum().item()
        total         += len(yb)

    # update phi for phi-regime classes only
    for c in range(NUM_CLASSES):
        if baseline_phi[c].item() < BASELINE_CUTOFF and class_total[c] > 0:
            acc = class_correct[c] / class_total[c]
            class_phi[c] = class_phi[c] * 0.8 + acc * 0.2

    phi_history.append(class_phi.clone())
    class_correct.zero_()
    class_total.zero_()

    # ── VALIDATION ────────────────────────────────────────────────────────────
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            val_preds.extend(
                model(xb.to(device)).argmax(1).cpu().numpy())
            val_labels.extend(yb.numpy())

    val_report = classification_report(
        val_labels, val_preds,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )
    model.train()

    newly_frozen = []
    for c in range(NUM_CLASSES):
        if c in frozen_classes:
            continue

        cls    = le.classes_[c]
        val_f1 = val_report[cls]['f1-score']
        phi_c  = class_phi[c].item()

        if val_f1 > best_val_f1[c]:
            best_val_f1[c] = val_f1
            patience[c]    = 0
        else:
            patience[c] += 1

        # BOTH conditions required to freeze
        # phi must be above PHI_FREEZE_MIN AND patience exceeded
        if patience[c] >= PATIENCE_LIMIT and phi_c >= PHI_FREEZE_MIN:
            frozen_classes.add(c)
            newly_frozen.append(cls)

    # display phi regime only
    phi_display = {
        le.classes_[c]: round(class_phi[c].item(), 3)
        for c in range(NUM_CLASSES)
        if baseline_phi[c].item() < BASELINE_CUTOFF
    }

    print(f"Epoch {epoch:02d} | "
          f"loss {total_loss/total:.4f} | "
          f"acc {total_correct/total:.3f} | "
          f"phi {phi_display}")

    if newly_frozen:
        print(f"  Frozen this epoch: {newly_frozen}")

    frozen_names = sorted([le.classes_[c] for c in frozen_classes])
    if frozen_names:
        print(f"  All frozen: {frozen_names}")

    # stopping — all classes either frozen or phi satisfied
    all_done = all(
        c in frozen_classes or
        class_phi[c].item() >= PHI_STOP
        for c in range(NUM_CLASSES)
    )

    if all_done:
        stopped_reason = f"all classes done at epoch {epoch}"
        print(f"\n  All classes satisfied or frozen — stopping.")
        break

print(f"\nStopped: {stopped_reason}")

# ── 9. EVALUATE ───────────────────────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        out = model(xb.to(device))
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(yb.numpy())

report = classification_report(
    all_labels, all_preds,
    target_names=le.classes_,
    output_dict=True
)

print("\nPhi v6 CNN per-class results:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

# ── 10. COMPARE ───────────────────────────────────────────────────────────────
with open("baseline_report.json") as f:
    baseline_r = json.load(f)
with open("phi_report.json") as f:
    phi_v1 = json.load(f)
with open("phi_v3_report.json") as f:
    phi_v3 = json.load(f)
with open("phi_v4_report.json") as f:
    phi_v4 = json.load(f)
with open("phi_v5_report.json") as f:
    phi_v5 = json.load(f)

print("\n── COMPARISON ──")
print(f"{'Class':<12} {'Base':>7} {'v1':>7} {'v3':>7} "
      f"{'v4':>7} {'v5':>7} {'v6':>7}")
print("-" * 62)

for cls in le.classes_:
    b  = baseline_r[cls]['f1-score']
    v1 = phi_v1[cls]['f1-score']
    v3 = phi_v3[cls]['f1-score']
    v4 = phi_v4[cls]['f1-score']
    v5 = phi_v5[cls]['f1-score']
    v6 = report[cls]['f1-score']
    print(f"{cls:<12} {b:>7.4f} {v1:>7.4f} {v3:>7.4f} "
          f"{v4:>7.4f} {v5:>7.4f} {v6:>7.4f}")

print(f"\n{'Macro':<12} "
      f"{baseline_r['macro avg']['f1-score']:>7.4f} "
      f"{phi_v1['macro avg']['f1-score']:>7.4f} "
      f"{phi_v3['macro avg']['f1-score']:>7.4f} "
      f"{phi_v4['macro avg']['f1-score']:>7.4f} "
      f"{phi_v5['macro avg']['f1-score']:>7.4f} "
      f"{report['macro avg']['f1-score']:>7.4f}")

with open("phi_v6_report.json", "w") as f:
    json.dump(report, f, indent=2)

torch.save(model.state_dict(), "phi_v6_cnn.pt")
print("\nSaved phi_v6_cnn.pt and phi_v6_report.json")