# IGNORE
# # UNUSED a while after

# import numpy as np
# import keras
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
#
# X = np.load("data/X.npy")
# y = np.load("data/y.npy")
#
# num_classes = len(np.unique(y))
#
# y_cat = to_categorical(y, num_classes=num_classes)
#
# print("X:", X.shape)
# print("y:", y.shape)
# print("y_cat:", y_cat.shape)
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y_cat,
#     test_size=0.2,
#     stratify=y,
#     random_state=42
# )
#
# np.save("data/X_train.npy", X_train)
# np.save("data/X_test.npy", X_test)
# np.save("data/y_train.npy", y_train)
# np.save("data/y_test.npy", y_test)
#
# print("Train:", X_train.shape)
# print("Test:", X_test.shape)





### this one makes a split for the expressional model
# 70% train / 15% val / 15% test by signer, not by video, to avoid data leakage

# output
# Processing annotations: 100%|██████████| 80/80 [00:23<00:00,  3.35it/s]
#
# Converting to numpy arrays...
# Train: (6015, 250, 275)
# Val: (1197, 250, 275)
# Test: (1588, 250, 275)

# these pycharm ide cocmment autofills are really handy ngl
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
# import random
# import re
#
# KEYPOINTS_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\kerypoints_expressional"
# )
#
# OUTPUT_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\expressional"
# )
# OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
#
# MAX_FRAMES = 250
# FEATURE_SIZE = 275
#
# TRAIN_SPLIT = 0.70
# VAL_SPLIT = 0.15
# TEST_SPLIT = 0.15
#
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
#
#
# def pad_sequence(seq):
#     frames = seq.shape[0]
#
#     if frames > MAX_FRAMES:
#         return seq[:MAX_FRAMES]
#
#     if frames < MAX_FRAMES:
#         pad = np.zeros((MAX_FRAMES - frames, seq.shape[1]), dtype=np.float32)
#         seq = np.vstack([seq, pad])
#
#     return seq
#
# def extract_signer(filename):
#     # P12_S001_03.npy → 12
#     match = re.search(r"P(\d+)_", filename)
#     return int(match.group(1)) if match else None
#
# train_X, train_y = [], []
# val_X, val_y = [], []
# test_X, test_y = [], []
#
# class_folders = sorted([f for f in KEYPOINTS_PATH.iterdir() if f.is_dir()])
#
# print("Found classes:", len(class_folders))
#
#
# for class_index, folder in enumerate(tqdm(class_folders, desc="Processing annotations")):
#
#     files = list(folder.glob("*.npy"))
#
#     if len(files) == 0:
#         continue
#
#     signer_dict = {}
#
#     for f in files:
#         signer = extract_signer(f.name)
#         if signer is None:
#             continue
#
#         signer_dict.setdefault(signer, []).append(f)
#
#     signers = list(signer_dict.keys())
#     random.shuffle(signers)
#
#     n_signers = len(signers)
#
#     if n_signers < 3:
#         print(f"Skipping {folder.name} (too few signers: {n_signers})")
#         continue
#
#     train_end = int(TRAIN_SPLIT * n_signers)
#     val_end = train_end + int(VAL_SPLIT * n_signers)
#
#     train_signers = signers[:train_end]
#     val_signers = signers[train_end:val_end]
#     test_signers = signers[val_end:]
#
#     # safety fallback
#     if len(val_signers) == 0:
#         val_signers = train_signers[:1]
#     if len(test_signers) == 0:
#         test_signers = train_signers[:1]
#
#     def process_files(file_list, X, y):
#         for f in file_list:
#             data = np.load(f)
#
#             if data.shape[1] != FEATURE_SIZE:
#                 print(f"WRONG SHAPE: {f.name} {data.shape}")
#                 continue
#
#             data = pad_sequence(data)
#
#             X.append(data)
#             y.append(class_index)
#
#     for s in train_signers:
#         process_files(signer_dict[s], train_X, train_y)
#
#     for s in val_signers:
#         process_files(signer_dict[s], val_X, val_y)
#
#     for s in test_signers:
#         process_files(signer_dict[s], test_X, test_y)
#
# print("\nConverting to numpy arrays...")
#
# X_train = np.array(train_X, dtype=np.float32)
# y_train = np.array(train_y, dtype=np.int32)
#
# X_val = np.array(val_X, dtype=np.float32)
# y_val = np.array(val_y, dtype=np.int32)
#
# X_test = np.array(test_X, dtype=np.float32)
# y_test = np.array(test_y, dtype=np.int32)
#
# print("Train:", X_train.shape)
# print("Val:", X_val.shape)
# print("Test:", X_test.shape)
#
#
# if len(X_train) == 0:
#     raise ValueError("No training data loaded!")
#
# print("\nComputing normalization statistics...")
#
# mean = np.mean(X_train, axis=(0, 1))
# std = np.std(X_train, axis=(0, 1)) + 1e-6
#
# X_train = (X_train - mean) / std
# X_val = (X_val - mean) / std
# X_test = (X_test - mean) / std
#
#
# def shuffle_dataset(X, y):
#     idx = np.arange(len(X))
#     np.random.shuffle(idx)
#     return X[idx], y[idx]
#
# X_train, y_train = shuffle_dataset(X_train, y_train)
# X_val, y_val = shuffle_dataset(X_val, y_val)
# X_test, y_test = shuffle_dataset(X_test, y_test)
#
# print("\nSaving dataset...")
#
# unique, counts = np.unique(y_train, return_counts=True)
# print("Min samples:", counts.min())
# print("Max samples:", counts.max())
#
# print((X_train == 0).mean())
#
# np.save(OUTPUT_PATH / "X_train.npy", X_train)
# np.save(OUTPUT_PATH / "y_train.npy", y_train)
#
# np.save(OUTPUT_PATH / "X_val.npy", X_val)
# np.save(OUTPUT_PATH / "y_val.npy", y_val)
#
# np.save(OUTPUT_PATH / "X_test.npy", X_test)
# np.save(OUTPUT_PATH / "y_test.npy", y_test)
#
# np.save(OUTPUT_PATH / "mean.npy", mean)
# np.save(OUTPUT_PATH / "std.npy", std)
#
# print("Saved to:", OUTPUT_PATH)