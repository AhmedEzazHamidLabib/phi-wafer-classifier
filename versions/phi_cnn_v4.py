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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── 5. DATASET ────────────────────────────────────────────────────────────────
class WaferDataset(Dataset):
    def __init__(self, maps, labels):
        self.maps   = torch.tensor(np.stack(maps), dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.maps[idx], self.labels[idx]

train_loader = DataLoader(WaferDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader  = DataLoader(WaferDataset(X_test,  y_test),  batch_size=64, shuffle=False)

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

# ── 7. PHI INITIALIZED FROM BASELINE F1 ──────────────────────────────────────
with open("baseline_report.json") as f:
    baseline = json.load(f)

class_phi = torch.tensor([
    baseline[cls]['f1-score'] for cls in le.classes_
], dtype=torch.float32)

print("\nPhi initialized from baseline F1:")
for c in range(NUM_CLASSES):
    print(f"  {le.classes_[c]:<12}: {class_phi[c].item():.4f}")

class_correct = torch.zeros(NUM_CLASSES)
class_total   = torch.zeros(NUM_CLASSES)

# phi history for velocity computation
phi_history   = [class_phi.clone()]

# aggression bounds — no longer fixed constants
# alpha and beta emerge from phi level and phi velocity
ALPHA_MAX     = 3.0   # maximum possible aggression
BETA_MIN      = 0.3   # minimum possible aggression
VEL_SCALE     = 5.0   # how strongly velocity modulates aggression
VEL_CLAMP_LO  = 0.5   # minimum velocity factor
VEL_CLAMP_HI  = 2.0   # maximum velocity factor

def compute_amplifier(true_labels, phi_vector, phi_vel):
    amplifier = torch.ones(len(true_labels), device=true_labels.device)
    for i in range(len(true_labels)):
        c     = true_labels[i].item()
        phi_c = phi_vector[c].item()
        vel_c = phi_vel[c].item()

        # base aggression — continuous function of phi level
        # low phi → high base, high phi → low base
        base = ALPHA_MAX * (1 - phi_c) + BETA_MIN * phi_c

        # velocity correction — continuous function of phi change rate
        # improving fast (vel > 0) → reduce aggression, mechanism working
        # stuck or falling (vel <= 0) → increase aggression, push harder
        velocity_factor = 1.0 - vel_c * VEL_SCALE
        velocity_factor = max(VEL_CLAMP_LO, min(VEL_CLAMP_HI, velocity_factor))

        amplifier[i] = base * velocity_factor

    return amplifier

# ── 8. TRAIN ──────────────────────────────────────────────────────────────────
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    total_loss, total_correct, total = 0, 0, 0

    # compute phi velocity from history
    if len(phi_history) >= 2:
        phi_vel = phi_history[-1] - phi_history[-2]
    else:
        # epoch 1 — no velocity yet, neutral
        phi_vel = torch.zeros(NUM_CLASSES)

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        out   = model(xb)
        preds = out.argmax(1)

        criterion       = nn.CrossEntropyLoss(reduction='none')
        loss_per_sample = criterion(out, yb)

        phi_norm  = class_phi.clamp(0, 1)
        amplifier = compute_amplifier(yb, phi_norm, phi_vel)
        amplifier  = amplifier.to(device)

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

    # ── UPDATE PHI ────────────────────────────────────────────────────────────
    for c in range(NUM_CLASSES):
        if class_total[c] > 0:
            acc = class_correct[c] / class_total[c]
            class_phi[c] = class_phi[c] * 0.8 + acc * 0.2

    phi_history.append(class_phi.clone())

    class_correct.zero_()
    class_total.zero_()

    # display phi and velocity per class
    print(f"Epoch {epoch+1:02d} | "
          f"loss {total_loss/total:.4f} | "
          f"acc {total_correct/total:.3f}")
    for c in range(NUM_CLASSES):
        phi_c = class_phi[c].item()
        vel_c = phi_vel[c].item()
        base  = ALPHA_MAX * (1 - phi_c) + BETA_MIN * phi_c
        vf    = max(VEL_CLAMP_LO, min(VEL_CLAMP_HI, 1.0 - vel_c * VEL_SCALE))
        eff   = base * vf
        print(f"  {le.classes_[c]:<12} phi={phi_c:.3f} "
              f"vel={vel_c:+.3f} "
              f"base={base:.3f} "
              f"vel_factor={vf:.3f} "
              f"effective_amplifier={eff:.3f}")

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

print("\nPhi v4 CNN per-class results:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

# ── 10. COMPARE ALL ───────────────────────────────────────────────────────────
with open("baseline_report.json") as f:
    baseline_r = json.load(f)
with open("phi_report.json") as f:
    phi_v1 = json.load(f)
with open("phi_v2_report.json") as f:
    phi_v2 = json.load(f)
with open("phi_v3_report.json") as f:
    phi_v3 = json.load(f)

print("\n── COMPARISON: Baseline vs v1 vs v2 vs v3 vs v4 ──")
print(f"{'Class':<12} {'Base':>7} {'v1':>7} {'v2':>7} {'v3':>7} {'v4':>7} "
      f"{'v1Δ':>7} {'v2Δ':>7} {'v3Δ':>7} {'v4Δ':>7}")
print("-" * 85)

for cls in le.classes_:
    b  = baseline_r[cls]['f1-score']
    v1 = phi_v1[cls]['f1-score']
    v2 = phi_v2[cls]['f1-score']
    v3 = phi_v3[cls]['f1-score']
    v4 = report[cls]['f1-score']
    d1 = v1 - b
    d2 = v2 - b
    d3 = v3 - b
    d4 = v4 - b
    def fmt(d): return f"{'↑' if d>0 else '↓'}{abs(d):.4f}"
    print(f"{cls:<12} {b:>7.4f} {v1:>7.4f} {v2:>7.4f} {v3:>7.4f} {v4:>7.4f} "
          f"{fmt(d1):>7} {fmt(d2):>7} {fmt(d3):>7} {fmt(d4):>7}")

with open("phi_v4_report.json", "w") as f:
    json.dump(report, f, indent=2)

torch.save(model.state_dict(), "phi_v4_cnn.pt")
print("\nSaved phi_v4_cnn.pt and phi_v4_report.json")
print("All five reports preserved.")