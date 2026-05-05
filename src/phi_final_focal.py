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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── 8. PHI SETUP: FOCAL-INFORMED PHI ─────────────────────────────────────────
# phi initialized from baseline F1 as prior knowledge
# phi updated each epoch using focal-style confidence on true class
# phi measures: how confidently does the model understand this class?
# not just right/wrong — but how certain when right, how lost when wrong
with open("../results/baseline_aug_report.json") as f:
    baseline = json.load(f)

class_phi = torch.tensor([
    baseline[cls]['f1-score'] for cls in le.classes_
], dtype=torch.float32)

print("\nPhi initialized from baseline aug F1:")
for c in range(NUM_CLASSES):
    print(f"  {le.classes_[c]:<12}: {class_phi[c].item():.4f}")

phi_history    = [class_phi.clone()]

# focal hardness accumulators — reset each epoch
class_hardness = torch.zeros(NUM_CLASSES)
class_total    = torch.zeros(NUM_CLASSES)

# phi constants
ALPHA_MAX = 3.0    # maximum aggression when phi is low
BETA_MIN  = 0.3    # minimum reinforcement when phi is high
VEL_SCALE    = 5.0
VEL_CLAMP_LO = 0.5
VEL_CLAMP_HI = 2.0

# focal hardness strength
# GAMMA=0 → hardness = uniform (degenerates to accuracy-based phi)
# GAMMA=1 → linear hardness
# GAMMA=2 → standard focal loss exponent — amplifies hard examples more
GAMMA   = 2.0

# phi EMA smoothing
PHI_EMA = 0.8

def compute_amplifier(true_labels, phi_vector, phi_vel):
    """
    Compute per-sample loss amplifier from phi and velocity.
    Phi is the class-level fidelity signal.
    Amplifier scales CE loss — high amplifier on violated classes,
    low amplifier on satisfied classes.
    Roles are kept separate:
        phi    → class-level, epoch-level difficulty (long-term)
        focal  → sample-level confidence signal (used to UPDATE phi only)
    Focal signal is NOT applied directly to the amplifier — only to phi update.
    This prevents double-counting difficulty.
    """
    amplifier = torch.ones(len(true_labels), device=true_labels.device)
    for i in range(len(true_labels)):
        c     = true_labels[i].item()
        phi_c = phi_vector[c].item()
        vel_c = phi_vel[c].item()

        # base aggression — continuous function of phi level
        base = ALPHA_MAX * (1 - phi_c) + BETA_MIN * phi_c

        # velocity correction
        vf = max(VEL_CLAMP_LO,
                 min(VEL_CLAMP_HI, 1.0 - vel_c * VEL_SCALE))

        amplifier[i] = base * vf

    return amplifier

# ── 9. TRAIN ──────────────────────────────────────────────────────────────────
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    total_loss, total_correct, total = 0, 0, 0

    if len(phi_history) >= 2:
        phi_vel = phi_history[-1] - phi_history[-2]
    else:
        phi_vel = torch.zeros(NUM_CLASSES)

    # reset focal hardness accumulators each epoch
    class_hardness.zero_()
    class_total.zero_()

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        out   = model(xb)
        preds = out.argmax(1)

        # ── standard CE loss per sample ───────────────────────────────────────
        criterion       = nn.CrossEntropyLoss(reduction='none')
        loss_per_sample = criterion(out, yb)

        # ── phi class-level amplifier ─────────────────────────────────────────
        phi_norm  = class_phi.clamp(0, 1)
        amplifier = compute_amplifier(yb, phi_norm, phi_vel).to(device)

        loss = (loss_per_sample * amplifier).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── focal hardness tracking (no gradient needed) ──────────────────────
        # this runs in parallel — does NOT affect the loss
        # used only to update phi at end of epoch
        with torch.no_grad():
            probs = torch.softmax(out, dim=1)

            # probability model assigns to the TRUE class
            # low p_t = model uncertain or wrong on this sample
            p_t = probs[torch.arange(len(yb), device=device), yb]

            # focal-style hardness — penalizes low confidence more aggressively
            # (1 - p_t)^GAMMA
            # confident correct → p_t near 1 → hardness near 0
            # barely correct → p_t near 0.5 → hardness moderate
            # wrong → p_t near 0 → hardness near 1
            hardness = (1.0 - p_t) ** GAMMA

            for c in range(NUM_CLASSES):
                mask = (yb == c)
                if mask.sum() > 0:
                    class_hardness[c] += hardness[mask].sum().cpu()
                    class_total[c]    += mask.sum().cpu()

        total_loss    += loss.item() * len(yb)
        total_correct += (preds == yb).sum().item()
        total         += len(yb)

    # ── UPDATE PHI USING FOCAL-INFORMED CONFIDENCE ────────────────────────────
    # phi = EMA(class_understanding)
    # class_understanding = 1 - avg_hardness
    # high avg_hardness → model uncertain on this class → phi stays low → pressure stays on
    # low avg_hardness → model confident on this class → phi rises → pressure eases
    for c in range(NUM_CLASSES):
        if class_total[c] > 0:
            avg_hardness        = class_hardness[c] / class_total[c]
            class_understanding = 1.0 - avg_hardness.item()
            class_phi[c]        = PHI_EMA * class_phi[c] + (1 - PHI_EMA) * class_understanding

    class_phi = class_phi.clamp(0, 1)
    phi_history.append(class_phi.clone())

    phi_display = {
        le.classes_[c]: round(class_phi[c].item(), 3)
        for c in range(NUM_CLASSES)
    }
    print(f"Epoch {epoch+1:02d} | "
          f"loss {total_loss/total:.4f} | "
          f"acc {total_correct/total:.3f} | "
          f"phi {phi_display}")

# ── 10. EVALUATE ──────────────────────────────────────────────────────────────
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

print("\nPhi Final Focal per-class results:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

# ── 11. COMPARE ───────────────────────────────────────────────────────────────
with open("../results/baseline_report.json") as f:
    baseline_r = json.load(f)

with open("../results/baseline_aug_report.json") as f:
    baseline_aug_r = json.load(f)

with open("../results/phi_final_report.json") as f:
    phi_final_r = json.load(f)

print("\n── COMPARISON: Baseline vs Baseline Aug vs Phi Final vs Phi Focal ──")
print(f"{'Class':<12} {'Baseline':>10} {'Base Aug':>10} {'Phi Final':>10} {'Phi Focal':>10} "
      f"{'Δ Base':>8} {'Δ Aug':>8} {'Δ Phi':>8}")
print("-" * 88)

for cls in le.classes_:
    b  = baseline_r[cls]['f1-score']
    ba = baseline_aug_r[cls]['f1-score']
    pf = phi_final_r[cls]['f1-score']
    fc = report[cls]['f1-score']

    db  = fc - b
    dba = fc - ba
    dpf = fc - pf

    print(f"{cls:<12} {b:>10.4f} {ba:>10.4f} {pf:>10.4f} {fc:>10.4f} "
          f"{'↑' if db>0 else '↓'}{abs(db):.4f} "
          f"{'↑' if dba>0 else '↓'}{abs(dba):.4f} "
          f"{'↑' if dpf>0 else '↓'}{abs(dpf):.4f}")

b_macro  = baseline_r['macro avg']['f1-score']
ba_macro = baseline_aug_r['macro avg']['f1-score']
pf_macro = phi_final_r['macro avg']['f1-score']
fc_macro = report['macro avg']['f1-score']

db_macro  = fc_macro - b_macro
dba_macro = fc_macro - ba_macro
dpf_macro = fc_macro - pf_macro

print(f"\n{'Macro':<12} {b_macro:>10.4f} {ba_macro:>10.4f} {pf_macro:>10.4f} {fc_macro:>10.4f} "
      f"{'↑' if db_macro>0 else '↓'}{abs(db_macro):.4f} "
      f"{'↑' if dba_macro>0 else '↓'}{abs(dba_macro):.4f} "
      f"{'↑' if dpf_macro>0 else '↓'}{abs(dpf_macro):.4f}")

with open("../results/phi_focal_report.json", "w") as f:
    json.dump(report, f, indent=2)

torch.save(model.state_dict(), "../models/phi_focal_cnn.pt")
print("\nSaved phi_focal_cnn.pt and phi_focal_report.json")
