# FULL HOLISTICS MODEL
### PREVIOUS V 1.3
# WORKIN
## this is the code of full holistics model
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATA_PATH = Path(
    r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed"
)
MODEL_PATH = Path(
    r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models"
)
MODEL_PATH.mkdir(exist_ok=True)

BEST_MODEL_FILE = MODEL_PATH / "sign_lstm_best.h5"
CONTINUED_MODEL_FILE = MODEL_PATH / "sign_lstm_supreme.keras"

print("Loading dataset...")
X = np.load(DATA_PATH / "X.npy")
y_labels = np.load(DATA_PATH / "y.npy")

mean = np.load(DATA_PATH / "mean.npy")
std = np.load(DATA_PATH / "std.npy")

X = (X - mean) / std

num_classes = len(np.unique(y_labels))
y = to_categorical(y_labels, num_classes)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_labels
)

print("X_train:", X_train.shape, "X_val:", X_val.shape)
print("Num classes:", num_classes)

print(f"Loading best model from: {BEST_MODEL_FILE}")
model = load_model(BEST_MODEL_FILE, compile=False)
model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=500, #10 min
    min_delta=0.001,  # only count improvements ≥0.1%
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=str(MODEL_PATH / "sign_lstm_supreme_best.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

print("Continuing training with smart early stopping...")
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=500, # 100 min
    batch_size=32,
    callbacks=[early_stop, checkpoint]
)

print(f"Saving final continued model to: {CONTINUED_MODEL_FILE}")
model.save(CONTINUED_MODEL_FILE)

print("Training complete. Best checkpoint and final model saved.")
























# I THINK it was for expressional one but i dont rememeber cause its been months since this
# import numpy as np
# from pathlib import Path
# import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import (
#     EarlyStopping,
#     ModelCheckpoint,
#     ReduceLROnPlateau
# )
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.model_selection import train_test_split
#
# DATA_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\fin_dataset"
# )
#
# MODEL_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models"
# )
# MODEL_PATH.mkdir(exist_ok=True)
#
# CHECKPOINT_PATH = MODEL_PATH / "checkpoints_face_reduced"
# CHECKPOINT_PATH.mkdir(exist_ok=True)
#
# FINAL_MODEL_FILE = MODEL_PATH / "sign_lstm_supreme_face_reduced_final.keras"
# BEST_MODEL_FILE = MODEL_PATH / "sign_lstm_supreme_face_reduced_best.keras"
#
# print("Loading dataset with full face landmarks...")
#
# X = np.load(DATA_PATH / "X.npy")
# y_labels = np.load(DATA_PATH / "y.npy")
# mean = np.load(DATA_PATH / "mean.npy")
# std = np.load(DATA_PATH / "std.npy")
#
# # Normalize
# X = (X - mean) / std
#
# print("Original feature shape:", X.shape)
#
# POSE_HAND_SIZE = 225
# FACE_START = POSE_HAND_SIZE
#
# FACE_LANDMARKS = [
#     70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
#     33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
#     61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
#     78, 95, 88, 178, 87, 14, 317, 402, 318, 324
# ]
#
# face_feature_indices = []
# for idx in FACE_LANDMARKS:
#     base = FACE_START + idx * 3
#     face_feature_indices.extend([base, base + 1, base + 2])
#
# face_feature_indices = np.array(face_feature_indices)
#
# pose_hands = X[:, :, :POSE_HAND_SIZE]
# face_selected = X[:, :, face_feature_indices]
#
# X = np.concatenate([pose_hands, face_selected], axis=2)
#
# print("New feature shape after reduction:", X.shape)
#
# num_classes = len(np.unique(y_labels))
# y = to_categorical(y_labels, num_classes)
#
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_labels
# )
#
# print("X_train:", X_train.shape)
# print("X_val:", X_val.shape)
# print("Num classes:", num_classes)
#
# input_dim = X.shape[2]
#
# model = Sequential([
#     LSTM(128, return_sequences=True, input_shape=(30, input_dim)),
#     Dropout(0.3),
#     LSTM(128),
#     Dropout(0.3),
#     Dense(128, activation="relu"),
#     Dropout(0.3),
#     Dense(num_classes, activation="softmax")
# ])
#
# model.compile(
#     optimizer=Adam(learning_rate=0.0005),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )
#
# model.summary()
#
#
# best_checkpoint = ModelCheckpoint(
#     str(BEST_MODEL_FILE),   # <-- FIX
#     monitor="val_accuracy",
#     save_best_only=True,
#     mode="max",
#     verbose=1
# )
#
# epoch_checkpoint = ModelCheckpoint(
#     str(CHECKPOINT_PATH / "epoch_{epoch:02d}_valacc_{val_accuracy:.4f}.keras"),  # <-- FIX
#     monitor="val_accuracy",
#     mode="max",
#     save_best_only=False,
#     verbose=0
# )
#
# early_stop = EarlyStopping(
#     monitor="val_accuracy",
#     patience=12,
#     min_delta=0.001,
#     restore_best_weights=True,
#     mode="max",
#     verbose=1
# )
#
# reduce_lr = ReduceLROnPlateau(
#     monitor="val_loss",
#     factor=0.3,
#     patience=5,
#     verbose=1,
#     min_lr=1e-6
# )
#
# callbacks = [
#     best_checkpoint,
#     epoch_checkpoint,
#     early_stop,
#     reduce_lr
# ]
#
# print("\nTraining model with reduced expressive face landmarks...\n")
#
# history = model.fit(
#     X_train,
#     y_train,
#     validation_data=(X_val, y_val),
#     epochs=80,
#     batch_size=32,
#     callbacks=callbacks
# )
#
# print("\nSaving final restored model...")
# model.save(str(FINAL_MODEL_FILE))
#
# best_val_acc = max(history.history["val_accuracy"])
# print(f"\nBest validation accuracy achieved: {best_val_acc:.4f}")
# print("Training complete.")