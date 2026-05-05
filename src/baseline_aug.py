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

# ── REPRODUCIBILITY ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_pickle("../data/labeled_only.pkl")
print("Loaded:", df.shape)
print(df['failureType'].value_counts())

# ── 2. AUGMENTATION ───────────────────────────────────────────────────────────
# same augmentation as phi_final — isolates augmentation contribution
# no phi — uniform CrossEntropyLoss throughout
# comparison: baseline_aug vs phi_final shows pure phi contribution on aug data

AUGMENT_TARGETS = {
    'Near-full': 1000,
    'Donut':     1000,
    'Scratch':   1500,
}

def augment_map(wmap, apply_noise=True):
    arr = np.array(wmap, dtype=np.float32)
    k   = random.choice([0, 1, 2, 3])
    arr = np.rot90(arr, k).copy()
    if random.random() > 0.5:
        arr = np.fliplr(arr).copy()
    if random.random() > 0.5:
        arr = np.flipud(arr).copy()
    if apply_noise:
        noise_mask      = np.random.random(arr.shape) < 0.02
        arr[noise_mask] = 2.0 - arr[noise_mask]
        arr             = np.clip(arr, 0, 2)
    return arr

print("\nAugmenting scarce classes...")
augmented_rows = []

for cls, target in AUGMENT_TARGETS.items():
    cls_rows    = df[df['failureType'] == cls]
    current     = len(cls_rows)
    needed      = max(0, target - current)
    apply_noise = cls != 'Scratch'
    print(f"  {cls:<12}: {current} → {current + needed} "
          f"(noise={'yes' if apply_noise else 'no'})")

    for _ in range(needed):
        row     = cls_rows.sample(1).iloc[0]
        new_map = augment_map(np.array(row['waferMap']), apply_noise)
        augmented_rows.append({
            'waferMap':       new_map,
            'failureType':    cls,
            'dieSize':        row['dieSize'],
            'lotName':        row['lotName'],
            'trainTestLabel': row['trainTestLabel'],
            'waferIndex':     row['waferIndex'],
        })

if augmented_rows:
    aug_df = pd.DataFrame(augmented_rows)
    df     = pd.concat([df, aug_df], ignore_index=True)

print(f"\nAfter augmentation:")
print(df['failureType'].value_counts())

# ── 3. PREPROCESS ─────────────────────────────────────────────────────────────
TARGET_SIZE = 32

def preprocess_map(wmap):
    arr = np.array(wmap, dtype=np.float32)
    img = Image.fromarray(arr)
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)
    return np.array(img) / 2.0

print("\nPreprocessing...")
df['processed'] = df['waferMap'].apply(preprocess_map)

# ── 4. ENCODE ─────────────────────────────────────────────────────────────────
le = LabelEncoder()
df['label'] = le.fit_transform(df['failureType'])
NUM_CLASSES  = len(le.classes_)
print("Classes:", list(le.classes_))

# ── 5. SPLIT ──────────────────────────────────────────────────────────────────
X = df['processed'].tolist()
y = df['label'].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── 6. DATASET ────────────────────────────────────────────────────────────────
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

# ── 7. MODEL ──────────────────────────────────────────────────────────────────
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
criterion = nn.CrossEntropyLoss()  # uniform — no phi
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── 8. TRAIN ──────────────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out  = model(xb)
            loss = criterion(out, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (out.argmax(1) == yb).sum().item()
            total      += len(yb)
    return total_loss / total, correct / total

EPOCHS = 20
for epoch in range(EPOCHS):
    tr_loss, tr_acc = run_epoch(
        model, train_loader, optimizer, criterion, device, train=True)
    te_loss, te_acc = run_epoch(
        model, test_loader,  optimizer, criterion, device, train=False)
    print(f"Epoch {epoch+1:02d} | "
          f"train {tr_loss:.4f} {tr_acc:.3f} | "
          f"test {te_loss:.4f} {te_acc:.3f}")

# ── 9. EVALUATE ───────────────────────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        all_preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
        all_labels.extend(yb.numpy())

report = classification_report(
    all_labels, all_preds,
    target_names=le.classes_,
    output_dict=True
)

print("\nBaseline + Augmentation per-class results:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

# ── 10. COMPARE ───────────────────────────────────────────────────────────────
with open("../results/baseline_report.json") as f:
    baseline_r = json.load(f)

print("\n── COMPARISON: Baseline vs Baseline+Aug ──")
print(f"{'Class':<12} {'Baseline':>10} {'Base+Aug':>10} {'Delta':>8}")
print("-" * 44)

for cls in le.classes_:
    b  = baseline_r[cls]['f1-score']
    a  = report[cls]['f1-score']
    d  = a - b
    print(f"{cls:<12} {b:>10.4f} {a:>10.4f} "
          f"{'↑' if d>0 else '↓'}{abs(d):.4f}")

b_macro = baseline_r['macro avg']['f1-score']
a_macro = report['macro avg']['f1-score']
d_macro = a_macro - b_macro
print(f"\n{'Macro':<12} {b_macro:>10.4f} {a_macro:>10.4f} "
      f"{'↑' if d_macro>0 else '↓'}{abs(d_macro):.4f}")

with open("../results/baseline_aug_report.json", "w") as f:
    json.dump(report, f, indent=2)

torch.save(model.state_dict(), "../models/baseline_aug_cnn.pt")
print("\nSaved baseline_aug_cnn.pt and baseline_aug_report.json")