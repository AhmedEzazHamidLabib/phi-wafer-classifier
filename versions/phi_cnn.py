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
# clean CNN — no attention module
# phi lives entirely in the loss, not the architecture
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

# ── 7. PHI STATE ──────────────────────────────────────────────────────────────
# phi tracks per class fidelity
# starts at 0.5 — no prior, neutral
class_phi     = torch.ones(NUM_CLASSES, dtype=torch.float32) * 0.5
class_correct = torch.zeros(NUM_CLASSES)
class_total   = torch.zeros(NUM_CLASSES)

# aggression constants
# alpha: how hard we push when phi is low (constraint violated)
# beta:  how gently we reinforce when phi is high (constraint satisfied)
ALPHA = 3.0
BETA  = 0.3

def compute_amplifier(true_labels, phi_vector):
    # every true label is a learning signal
    # phi scales how hard that signal hits
    # low phi → high aggression → push hard on every ground truth occurrence
    # high phi → low aggression → gentle reinforcement
    amplifier = torch.zeros(len(true_labels), device=true_labels.device)
    for i in range(len(true_labels)):
        c         = true_labels[i].item()
        phi_c     = phi_vector[c].item()
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

        # per sample loss against ground truth
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss_per_sample = criterion(out, yb)

        # phi scales aggression directly from true label
        # no confidence gate — every ground truth occurrence counts
        phi_norm  = (class_phi / class_phi.sum() * NUM_CLASSES).clamp(0, 1)
        amplifier = compute_amplifier(yb, phi_norm)

        loss = (loss_per_sample * amplifier).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # track per class accuracy for phi update
        for c in range(NUM_CLASSES):
            mask = (yb == c)
            if mask.sum() > 0:
                class_correct[c] += (preds[mask] == yb[mask]).sum().item()
                class_total[c]   += mask.sum().item()

        total_loss    += loss.item() * len(yb)
        total_correct += (preds == yb).sum().item()
        total         += len(yb)

    # ── UPDATE PHI AFTER EACH EPOCH ───────────────────────────────────────────
    for c in range(NUM_CLASSES):
        if class_total[c] > 0:
            acc = class_correct[c] / class_total[c]
            # exponential moving average
            # phi rises when model gets it right
            # phi falls when model gets it wrong
            class_phi[c] = class_phi[c] * 0.8 + acc * 0.2

    # reset trackers
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

print("\nPhi CNN per-class results:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

# ── 10. COMPARE ───────────────────────────────────────────────────────────────
with open("baseline_report.json") as f:
    baseline = json.load(f)

print("\n── COMPARISON: Baseline vs Phi ──")
print(f"{'Class':<12} {'Baseline F1':>12} {'Phi F1':>10} {'Delta':>8}")
print("-" * 46)
for cls in le.classes_:
    b_f1  = baseline[cls]['f1-score']
    p_f1  = report[cls]['f1-score']
    delta = p_f1 - b_f1
    arrow = "↑" if delta > 0 else "↓"
    print(f"{cls:<12} {b_f1:>12.4f} {p_f1:>10.4f} {arrow} {abs(delta):.4f}")

with open("phi_report.json", "w") as f:
    json.dump(report, f, indent=2)

torch.save(model.state_dict(), "phi_cnn.pt")
print("\nSaved phi_cnn.pt and phi_report.json")