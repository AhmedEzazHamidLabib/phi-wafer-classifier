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
# baseline results are prior knowledge about constraint satisfaction state
# skip the discovery phase — start phi where the baseline left off
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

# aggression constants
# no threshold — continuous smooth function handles saturation naturally
# high phi → BETA fires → gentle reinforcement
# low phi → ALPHA fires → aggressive learning from ground truth
ALPHA = 3.0
BETA  = 0.3

def compute_amplifier(true_labels, phi_vector):
    amplifier = torch.ones(len(true_labels), device=true_labels.device)
    for i in range(len(true_labels)):
        c            = true_labels[i].item()
        phi_c        = phi_vector[c].item()
        amplifier[i] = ALPHA * (1 - phi_c) + BETA * phi_c
    return amplifier

# ── 8. TRAIN ──────────────────────────────────────────────────────────────────
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    total_loss, total_correct, total = 0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        out   = model(xb)
        preds = out.argmax(1)

        criterion       = nn.CrossEntropyLoss(reduction='none')
        loss_per_sample = criterion(out, yb)

        phi_norm  = class_phi.clamp(0, 1).to(device)
        amplifier = compute_amplifier(yb, phi_norm)

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

    class_correct.zero_()
    class_total.zero_()

    phi_display = {
        le.classes_[c]: round(class_phi[c].item(), 3)
        for c in range(NUM_CLASSES)
    }
    print(f"Epoch {epoch+1:02d} | "
          f"loss {total_loss/total:.4f} | "
          f"acc {total_correct/total:.3f} | "
          f"phi {phi_display}")

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

print("\nPhi v3 CNN per-class results:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

# ── 10. COMPARE ALL FOUR ──────────────────────────────────────────────────────
with open("baseline_report.json") as f:
    baseline_r = json.load(f)

with open("phi_report.json") as f:
    phi_v1 = json.load(f)

with open("phi_v2_report.json") as f:
    phi_v2 = json.load(f)

print("\n── COMPARISON: Baseline vs Phi v1 vs Phi v2 vs Phi v3 ──")
print(f"{'Class':<12} {'Base':>8} {'v1':>8} {'v2':>8} {'v3':>8} {'v1Δ':>7} {'v2Δ':>7} {'v3Δ':>7}")
print("-" * 72)

for cls in le.classes_:
    b   = baseline_r[cls]['f1-score']
    v1  = phi_v1[cls]['f1-score']
    v2  = phi_v2[cls]['f1-score']
    v3  = report[cls]['f1-score']
    d1  = v1 - b
    d2  = v2 - b
    d3  = v3 - b
    a1  = "↑" if d1 > 0 else "↓"
    a2  = "↑" if d2 > 0 else "↓"
    a3  = "↑" if d3 > 0 else "↓"
    print(f"{cls:<12} {b:>8.4f} {v1:>8.4f} {v2:>8.4f} {v3:>8.4f} "
          f"{a1}{abs(d1):.4f} {a2}{abs(d2):.4f} {a3}{abs(d3):.4f}")

with open("phi_v3_report.json", "w") as f:
    json.dump(report, f, indent=2)

torch.save(model.state_dict(), "phi_v3_cnn.pt")
print("\nSaved phi_v3_cnn.pt and phi_v3_report.json")
print("All four reports preserved.")