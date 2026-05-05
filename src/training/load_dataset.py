# MANY PURPOSES

# check the pipeline
# import cv2
# from src.utils.mediapipe_expressional import create_holistic, extract_landmarks
#
# video_path = r"F:\KSLR-FluentSigners-50\000\P0_S000_00.mp4"
#
# cap = cv2.VideoCapture(video_path)
# holistic = create_holistic()
#
# success_frames = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     landmarks = extract_landmarks(results)
#     if landmarks is not None and landmarks.sum() != 0:
#         success_frames += 1
#
# cap.release()
# print("Frames with landmarks detected:", success_frames)



### .npy shape viewer
import numpy as np
from pathlib import Path
#
# path = Path (
#     # expressional
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\kerypoints_expressional\064\P21_S064_01.npy"
#
#     #full:
#     # r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\keypoints\000\P0_S000_00.npy"
#
# )
# data = np.load(path)
# print("Shape:", data.shape)
# print(data)




# # loader
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
#
# KEYPOINTS_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\keypoints"
#     # r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\kerypoints_expressional"
# )
#
# OUTPUT_PATH = Path(
#     # r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed"
#     # r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\expressional"
# )
#
# OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
#
#
# def load_dataset():
#
#     X = []
#     y = []
#
#     class_folders = sorted([f for f in KEYPOINTS_PATH.iterdir() if f.is_dir()])
#
#     print(f"Found {len(class_folders)} classes")
#
#     for class_index, folder in enumerate(tqdm(class_folders, desc="Loading classes")):
#
#         npy_files = sorted(folder.glob("*.npy"))
#
#         for file in npy_files:
#
#             data = np.load(file)
#
#             if data.shape != (30, 1692):
#                 print(f"Skipping wrong shape: {file.name} {data.shape}")
#                 continue
#
#             X.append(data)
#             y.append(class_index)
#
#     X = np.array(X, dtype=np.float32)
#     y = np.array(y, dtype=np.int32)
#
#     print("\nDataset loaded:")
#     print("X shape:", X.shape)
#     print("y shape:", y.shape)
#
#     return X, y
#
#
# if __name__ == "__main__":
#
#     X, y = load_dataset()
#
#     print("Saving dataset...")
#
#     # np.save(OUTPUT_PATH / "X.npy", X)
#     # np.save(OUTPUT_PATH / "y.npy", y)
#
#     # print("Saved to:", OUTPUT_PATH)






# file opener:
import os

# The absolute path to your document
# FILE_PATH = r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Doc\texts\full holistics_training LIMIT.txt"
# FILE_PATH = r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Doc\texts\dataset_loading_final.txt"
# FILE_PATH = r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Doc\texts\train_unoficcial.txt"
# FILE_PATH = r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Doc\texts\dataset_loading.txt"
# FILE_PATH = r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Doc\texts\training_plateau.txt"
# FILE_PATH = r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Doc\texts\test_on_videos_sequence_30_result.txt"
FILE_PATH = r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Doc\texts\full holistics_training NO LIMIT.txt"


def display_file_contents():
    if not os.path.exists(FILE_PATH):
        print(f" Error: Could not find file at:\n{FILE_PATH}")
        return

    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as file:
            content = file.read()

            print("-" * 30)
            print(f" CONTENTS OF: {os.path.basename(FILE_PATH)}")
            print("-" * 30)
            print(content)
            print("-" * 30)

    except Exception as e:
        print(f" An error occurred: {e}")


if __name__ == "__main__":
    display_file_contents()




# DELETE ?
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
#
# KEYPOINTS_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\keypoints"
# )
#
# OUTPUT_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed"
# )
#
# OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
#
# POSE_SIZE = 33 * 4
# FACE_SIZE = 478 * 3
# HAND_SIZE = 21 * 3
#
# EXPECTED = POSE_SIZE + FACE_SIZE + HAND_SIZE + HAND_SIZE
#
#
# def transform_sequence(sequence):
#     new_seq = []
#
#     for frame in sequence:
#
#         # Split parts
#         pose_raw = frame[:POSE_SIZE]
#         face_raw = frame[POSE_SIZE:POSE_SIZE + FACE_SIZE]
#         left_raw = frame[POSE_SIZE + FACE_SIZE:
#                          POSE_SIZE + FACE_SIZE + HAND_SIZE]
#         right_raw = frame[-HAND_SIZE:]
#
#         # Reshape pose to (33,4)
#         pose = pose_raw.reshape(33, 4)[:, :3]  # drop visibility
#         left = left_raw.reshape(21, 3)
#         right = right_raw.reshape(21, 3)
#
#         # Shoulder midpoint (landmarks 11 and 12)
#         left_shoulder = pose[11]
#         right_shoulder = pose[12]
#
#         center = (left_shoulder + right_shoulder) / 2.0
#         shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
#
#         if shoulder_dist < 1e-6:
#             shoulder_dist = 1.0
#
#         # Make relative + scale normalize
#         pose = (pose - center) / shoulder_dist
#         left = (left - center) / shoulder_dist
#         right = (right - center) / shoulder_dist
#
#         combined = np.concatenate([
#             pose.flatten(),
#             left.flatten(),
#             right.flatten()
#         ])
#
#         new_seq.append(combined)
#
#     return np.array(new_seq, dtype=np.float32)
#
#
# def load_dataset():
#
#     X = []
#     y = []
#     groups = []
#
#     class_folders = sorted([f for f in KEYPOINTS_PATH.iterdir() if f.is_dir()])
#
#     print(f"Found {len(class_folders)} classes")
#
#     for class_index, folder in enumerate(tqdm(class_folders, desc="Loading classes")):
#
#         npy_files = sorted(folder.glob("*.npy"))
#
#         for file in npy_files:
#
#             data = np.load(file)
#
#             if data.shape != (30, EXPECTED):
#                 print(f"Skipping wrong shape: {file.name} {data.shape}")
#                 continue
#
#             transformed = transform_sequence(data)
#
#             signer_id = file.name.split("_")[0]
#
#             X.append(transformed)
#             y.append(class_index)
#             groups.append(signer_id)
#
#     X = np.array(X, dtype=np.float32)
#     y = np.array(y, dtype=np.int32)
#     groups = np.array(groups)
#
#     print("\nDataset loaded:")
#     print("X shape:", X.shape)  # should be (N, 30, 225)
#     print("y shape:", y.shape)
#     print("Unique signers:", len(np.unique(groups)))
#
#     return X, y, groups
#
#
# if __name__ == "__main__":
#
#     X, y, groups = load_dataset()
#
#     print("Saving dataset...")
#
#     np.save(OUTPUT_PATH / "X.npy", X)
#     np.save(OUTPUT_PATH / "y.npy", y)
#     np.save(OUTPUT_PATH / "groups.npy", groups)
#
#     print("Saved to:", OUTPUT_PATH)