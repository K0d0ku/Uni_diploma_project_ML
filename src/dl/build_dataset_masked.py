### CURRENT V 1.4
# rebuild the data from extracted keypoints of the flawed pipeline

import numpy as np
from pathlib import Path
import random

KEYPOINTS_PATH = Path(
    r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\keypoints"
)
SAVE_PATH = Path(
    r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\dl\data"
)

SAVE_PATH.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 30
MIN_VALID_FRAMES = 10

def compute_mask(sequence):
    # frame is valid if not near-zero
    return (np.abs(sequence).sum(axis=1) > 1e-6).astype(np.float32)

def normalize_sequence(seq):
    # frame-wise centering (helps invariance)
    frame_mean = np.mean(seq, axis=1, keepdims=True)
    return seq - frame_mean

def load_sequence(file_path):
    data = np.load(file_path)

    # random crop
    if len(data) >= SEQ_LEN:
        start = random.randint(0, len(data) - SEQ_LEN)
        seq = data[start:start + SEQ_LEN]
    else:
        seq = np.zeros((SEQ_LEN, data.shape[1]), dtype=np.float32)
        seq[:len(data)] = data

    mask = compute_mask(seq)

    # FILTER BAD SEQUENCES
    if mask.sum() < MIN_VALID_FRAMES:
        return None, None

    seq = normalize_sequence(seq)

    return seq, mask

print("\nBuilding dataset (masked)...")

X, M, y_labels = [], [], []

for class_dir in sorted(KEYPOINTS_PATH.iterdir()):
    if not class_dir.is_dir():
        continue

    label = int(class_dir.name)

    for file in class_dir.glob("*.npy"):
        seq, mask = load_sequence(file)

        if seq is None:
            continue

        X.append(seq)
        M.append(mask)
        y_labels.append(label)

X = np.array(X, dtype=np.float32)
M = np.array(M, dtype=np.float32)[..., np.newaxis]
y_labels = np.array(y_labels)

print("X:", X.shape)
print("Mask:", M.shape)
print("y:", y_labels.shape)

unique_labels = sorted(np.unique(y_labels))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
inverse_map = {idx: label for label, idx in label_map.items()}

y_mapped = np.array([label_map[l] for l in y_labels])
num_classes = len(unique_labels)

print("Classes:", num_classes)

mean = np.mean(X, axis=(0, 1), keepdims=True)
std = np.std(X, axis=(0, 1), keepdims=True) + 1e-8

X = (X - mean) / std

np.save(SAVE_PATH / "X.npy", X)
np.save(SAVE_PATH / "mask.npy", M)
np.save(SAVE_PATH / "y.npy", y_mapped)

np.save(SAVE_PATH / "mean.npy", mean)
np.save(SAVE_PATH / "std.npy", std)

np.save(SAVE_PATH / "label_map.npy", label_map)
np.save(SAVE_PATH / "inverse_map.npy", inverse_map)

print("Saved to:", SAVE_PATH)