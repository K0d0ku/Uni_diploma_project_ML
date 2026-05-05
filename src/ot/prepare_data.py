# IGNORE
# its just another branch of model training , tho its here i started noticing the error i made in FLAWED pipeline

import numpy as np
from pathlib import Path

from tqdm import tqdm
DATA_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\keypoints")
OUTPUT_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\ot\data")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

EXPECTED_SIZE = 1692
SEQ_LEN = 30

ZERO_THRESHOLD = 0.80   # if >80% zeros → bad frame
MIN_VALID_FRAMES = 10   # discard sequences with too few valid frames

X_data = []
y_data = []
mask_data = []

label_map = {}
label_counter = 0


def get_mask(sequence):
    # Detect bad frames based on zero ratio.
    mask = []

    for frame in sequence:
        zero_ratio = np.sum(frame == 0) / len(frame)
        if zero_ratio > ZERO_THRESHOLD:
            mask.append(0)
        else:
            mask.append(1)

    return np.array(mask, dtype=np.float32)


def normalize_sequence(sequence, mask):
    # normalize ONLY valid frames.
    valid_frames = sequence[mask == 1]

    if len(valid_frames) == 0:
        return sequence

    mean = valid_frames.mean(axis=0)
    std = valid_frames.std(axis=0) + 1e-6

    sequence = (sequence - mean) / std
    return sequence


print("Loading dataset...")

folders = sorted([f for f in DATA_PATH.iterdir() if f.is_dir()])

for folder in tqdm(folders):
    label = folder.name

    if label not in label_map:
        label_map[label] = label_counter
        label_counter += 1

    class_id = label_map[label]

    for file in folder.glob("*.npy"):
        seq = np.load(file)

        if seq.shape != (SEQ_LEN, EXPECTED_SIZE):
            continue

        mask = get_mask(seq)

        if np.sum(mask) < MIN_VALID_FRAMES:
            continue

        seq = normalize_sequence(seq, mask)

        X_data.append(seq)
        y_data.append(class_id)
        mask_data.append(mask)

X = np.array(X_data, dtype=np.float32)
y = np.array(y_data)
mask = np.array(mask_data, dtype=np.float32)

print(f"Final dataset: {X.shape}, labels: {y.shape}")

print("Computing global normalization...")

valid_values = X[mask == 1]

mean = valid_values.mean(axis=0)
std = valid_values.std(axis=0) + 1e-6

X = (X - mean) / std

np.save(OUTPUT_PATH / "X.npy", X)
np.save(OUTPUT_PATH / "y.npy", y)
np.save(OUTPUT_PATH / "mask.npy", mask)
np.save(OUTPUT_PATH / "mean.npy", mean)
np.save(OUTPUT_PATH / "std.npy", std)

print("Saved processed dataset.")