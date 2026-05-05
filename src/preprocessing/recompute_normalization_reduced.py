# IGNORE
# UNUSED AND OLD i think


import numpy as np
from pathlib import Path
DATA_PATH = Path(
    r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\fin_dataset"
)

print("Loading full dataset...")
X = np.load(DATA_PATH / "X.npy")

print("Original shape:", X.shape)  # (N, 30, 1629)

POSE_HAND_SIZE = 225

FACE_LANDMARKS = [
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324
]

face_start = POSE_HAND_SIZE

face_indices = []
for idx in FACE_LANDMARKS:
    base = face_start + idx * 3
    face_indices.extend([base, base + 1, base + 2])

face_indices = np.array(face_indices)

pose_hands = X[:, :, :POSE_HAND_SIZE]
face_selected = X[:, :, face_indices]

X_reduced = np.concatenate([pose_hands, face_selected], axis=2)

print("Reduced shape:", X_reduced.shape)  # should be (N, 30, 351) cause i tried to reduce the face

X_flat = X_reduced.reshape(-1, X_reduced.shape[-1])

mean = np.mean(X_flat, axis=0)
std = np.std(X_flat, axis=0)

std[std == 0] = 1e-6

np.save(DATA_PATH / "mean_reduced.npy", mean)
np.save(DATA_PATH / "std_reduced.npy", std)

print("Saved mean_reduced.npy and std_reduced.npy")
print("Done.")