# IGNORE
# OLD AND UNUSED
import numpy as np
from pathlib import Path
import mediapipe as mp
ORIGINAL_DATA_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed")
FIN_DATA_PATH = ORIGINAL_DATA_PATH / "fin_dataset"
FIN_DATA_PATH.mkdir(parents=True, exist_ok=True)

X_file = ORIGINAL_DATA_PATH / "X.npy"
y_file = ORIGINAL_DATA_PATH / "y.npy"

print("Loading original dataset...")
X_old = np.load(X_file)
y = np.load(y_file)
print("Old X shape:", X_old.shape)
print("y shape:", y.shape)

mp_holistic = mp.solutions.holistic

def extract_face_landmarks(holistic_results):
    if holistic_results.face_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in holistic_results.face_landmarks.landmark]).flatten()
    else:
        # if face not detected, return zeros
        return np.zeros(468 * 3)

num_samples, seq_len, old_features = X_old.shape
face_features_len = 468 * 3  # mediapipe face landmarks
new_features = old_features + face_features_len

X_new = np.zeros((num_samples, seq_len, new_features), dtype=np.float32)

with mp_holistic.Holistic(static_image_mode=True) as holistic:
    for i in range(num_samples):
        for t in range(seq_len):
            frame_data = X_old[i, t, :]
            # reconstruct minimal keypoints image if you have original video/image frames
            # for now we assume frame_data includes x,y,z positions for hands/body
            # here we only simulate face addition as zeros for placeholder
            face_landmarks = np.zeros(face_features_len)
            # concatenate old + face
            X_new[i, t, :] = np.concatenate([frame_data, face_landmarks])

np.save(FIN_DATA_PATH / "X.npy", X_new)
np.save(FIN_DATA_PATH / "y.npy", y)

print("New dataset with face landmarks saved in:", FIN_DATA_PATH)
print("X_new shape:", X_new.shape)