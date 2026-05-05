# IGNORE
### UNUSED

# i kept branching out in hopes of getting different results (which i eventually did) , but not with this one
import numpy as np
from pathlib import Path
import random

KEYPOINTS_PATH = Path(
    r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\keypoints"
)

SAVE_PATH = Path(
    r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\fl"
)

SAVE_PATH.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 30

def load_sequence(file_path):
    data = np.load(file_path)

    if len(data) >= SEQ_LEN:
        start = random.randint(0, len(data) - SEQ_LEN)
        return data[start:start + SEQ_LEN]
    else:
        padded = np.zeros((SEQ_LEN, data.shape[1]))
        padded[:len(data)] = data
        return padded

print("\nBuilding dataset from keypoints...")

X, y_labels = [], []

for class_dir in sorted(KEYPOINTS_PATH.iterdir()):
    if not class_dir.is_dir():
        continue

    label = int(class_dir.name)

    for file in class_dir.glob("*.npy"):
        seq = load_sequence(file)
        X.append(seq)
        y_labels.append(label)

X = np.array(X)
y_labels = np.array(y_labels)

print("Dataset built:")
print("X:", X.shape)
print("y:", y_labels.shape)

print("\nRemapping labels...")

unique_labels = sorted(np.unique(y_labels))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
inverse_map = {idx: label for label, idx in label_map.items()}

y_mapped = np.array([label_map[l] for l in y_labels])

num_classes = len(unique_labels)
print("Num classes:", num_classes)

print("\nNormalizing dataset...")

mean = np.mean(X, axis=(0, 1), keepdims=True)
std = np.std(X, axis=(0, 1), keepdims=True) + 1e-8

X = (X - mean) / std

print("\nSaving processed dataset...")

np.save(SAVE_PATH / "X.npy", X)
np.save(SAVE_PATH / "y.npy", y_mapped)

np.save(SAVE_PATH / "mean.npy", mean)
np.save(SAVE_PATH / "std.npy", std)

np.save(SAVE_PATH / "label_map.npy", label_map)
np.save(SAVE_PATH / "inverse_map.npy", inverse_map)

print("Saved to:", SAVE_PATH)
print("Done.")