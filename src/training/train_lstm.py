# #  if ya using pycharm 2023 or other alike ide and see that keras lstm sequential and others are lined out
# #  red and shown as errors its not, it works even with it, i dont know whats causing it, i suspect its that lazy loading shit but god knows

# FULL HOLISTICS MODEL
### PREVIOUS V 1.3
# WORKIN
## this is the code of full holistics model
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Bidirectional
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

DATA_PATH = Path(
    r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed"
)

MODEL_PATH = Path(
    r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models"
)

MODEL_PATH.mkdir(exist_ok=True)

print("\nLoading dataset...")

X = np.load(DATA_PATH / "X.npy")
y_labels = np.load(DATA_PATH / "y.npy")

print("X shape:", X.shape)
print("y shape:", y_labels.shape)

print("\nNormalizing dataset...")

mean = np.mean(X, axis=(0, 1), keepdims=True)
std = np.std(X, axis=(0, 1), keepdims=True) + 1e-8

X = (X - mean) / std

np.save(DATA_PATH / "mean.npy", mean)
np.save(DATA_PATH / "std.npy", std)

print("Normalization saved.")

num_classes = len(np.unique(y_labels))

print("\nNumber of classes:", num_classes)

y = to_categorical(y_labels, num_classes)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y_labels
)

print("\nTrain set:", X_train.shape)
print("Validation set:", X_val.shape)

print("\nBuilding model...")

model = Sequential()
model.add(
    Bidirectional(
        LSTM(128, return_sequences=True),
        input_shape=(30, 1692)
    )
)

model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(
    Bidirectional(
        LSTM(128)
    )
)

model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.4))

model.add(Dense(num_classes, activation="softmax"))


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(

    monitor="val_accuracy",
    patience=240, # 10 min
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath=str(MODEL_PATH / "sign_lstm_best.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

print("\nTraining...")

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val, y_val),
    epochs=240, # 60 , 120
    batch_size=32,
    callbacks=[early_stop, checkpoint]
)

print("\nSaving final model...")

model.save(str(MODEL_PATH / "sign_lstm_model.h5"))
# for future
# model.save(MODEL_PATH / "sign_lstm_model.keras")

print("\nTraining complete.")
print("Best model saved as: sign_lstm_best.h5")
print("Final model saved as: sign_lstm_model.h5")
print("Normalization saved as: mean.npy, std.npy")








# v 1.4 ?? fail
# import numpy as np
# from pathlib import Path
# import re
# import shutil
#
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (
#     LSTM,
#     Dense,
#     Dropout,
#     BatchNormalization,
#     Bidirectional
# )
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import (
#     EarlyStopping,
#     ReduceLROnPlateau,
#     Callback
# )
# from sklearn.model_selection import train_test_split
#
# DATA_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\fl"
# )
#
# MODEL_BASE_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\fl"
# )
#
# CHECKPOINT_PATH = MODEL_BASE_PATH / "checkpoints"
#
# MODEL_BASE_PATH.mkdir(parents=True, exist_ok=True)
# CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
#
# class TopKCheckpoint(Callback):
#     def __init__(self, save_dir, k=2):
#         super().__init__()
#         self.save_dir = save_dir
#         self.k = k
#         self.saved = []  # (loss, filepath)
#
#     def on_epoch_end(self, epoch, logs=None):
#         val_loss = logs.get("val_loss")
#         val_acc = logs.get("val_accuracy")
#
#         if val_loss is None:
#             return
#
#         filename = f"epoch_{epoch+1:03d}_valloss_{val_loss:.4f}_valacc_{val_acc:.4f}.keras"
#         filepath = self.save_dir / filename
#
#         # Save current model
#         self.model.save(filepath)
#
#         # Track it
#         self.saved.append((val_loss, filepath))
#
#         # Sort by best (lowest loss)
#         self.saved.sort(key=lambda x: x[0])
#
#         # Keep only top K
#         if len(self.saved) > self.k:
#             worst = self.saved.pop(-1)
#             try:
#                 worst[1].unlink()
#                 print(f"Deleted worse checkpoint: {worst[1].name}")
#             except Exception as e:
#                 print(f"Warning: could not delete {worst[1]}: {e}")
#
# print("\nLoading dataset...")
#
# X = np.load(DATA_PATH / "X.npy")
# y_labels = np.load(DATA_PATH / "y.npy")
#
# label_map = np.load(DATA_PATH / "label_map.npy", allow_pickle=True).item()
# inverse_map = np.load(DATA_PATH / "inverse_map.npy", allow_pickle=True).item()
#
# mean = np.load(DATA_PATH / "mean.npy")
# std = np.load(DATA_PATH / "std.npy")
#
# print("X shape:", X.shape)
# print("y shape:", y_labels.shape)
# print("Classes:", len(label_map))
#
# assert len(X.shape) == 3
# assert X.shape[1] == 30
# assert X.shape[2] == 1692
#
# assert not np.isnan(X).any()
# assert not np.isinf(X).any()
#
# print("Sanity checks passed.")
#
# num_classes = len(label_map)
# y = to_categorical(y_labels, num_classes)
#
# X_train, X_val, y_train, y_val = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_labels
# )
#
# print("Train:", X_train.shape)
# print("Val:", X_val.shape)
#
# print("\nBuilding model...")
#
# model = Sequential()
#
# model.add(
#     Bidirectional(
#         LSTM(128, return_sequences=True),
#         input_shape=(30, 1692)
#     )
# )
#
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
#
# model.add(Bidirectional(LSTM(128)))
#
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
#
# model.add(Dense(256, activation="relu"))
# model.add(Dropout(0.4))
#
# model.add(Dense(num_classes, activation="softmax"))
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )
#
# model.summary()
#
# topk_checkpoint = TopKCheckpoint(CHECKPOINT_PATH, k=2)
#
# early_stop = EarlyStopping(
#     monitor="val_loss",
#     patience=60,
#     restore_best_weights=True
# )
#
# reduce_lr = ReduceLROnPlateau(
#     monitor="val_loss",
#     factor=0.3,
#     patience=15,
#     min_lr=1e-6,
#     verbose=1
# )
#
# print("\nTraining...")
#
# history = model.fit(
#     X_train,
#     y_train,
#     validation_data=(X_val, y_val),
#     epochs=240,
#     batch_size=32,
#     callbacks=[topk_checkpoint, early_stop, reduce_lr]
# )
#
# print("\nSaving final model...")
#
# model.save(str(MODEL_BASE_PATH / "sign_lstm_final.keras"))
#
# print("\nSelecting best model...")
#
# files = list(CHECKPOINT_PATH.glob("*.keras"))
#
# best_file = None
# best_loss = float("inf")
#
# for f in files:
#     name = f.name
#     loss = float(re.search(r"valloss_(\d+\.\d+)", name).group(1))
#
#     if loss < best_loss:
#         best_loss = loss
#         best_file = f
#
# print("Best model:", best_file)
#
# shutil.copy(
#     best_file,
#     MODEL_BASE_PATH / "sign_lstm_best.keras"
# )
#
# np.save(MODEL_BASE_PATH / "mean.npy", mean)
# np.save(MODEL_BASE_PATH / "std.npy", std)
# np.save(MODEL_BASE_PATH / "label_map.npy", label_map)
# np.save(MODEL_BASE_PATH / "inverse_map.npy", inverse_map)
#
#
# print("\nTraining complete.")
# print("Top-2 checkpoints kept in:", CHECKPOINT_PATH)
# print("Best model saved as sign_lstm_best.keras")
# print("Final model saved as sign_lstm_final.keras")






















# v 1.4 ?????? attention layer with temporal dropout
# delete the existing v 1.4 before training
# import numpy as np
# from pathlib import Path
# import random
#
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Input,
#     Dense,
#     Dropout,
#     LSTM,
#     Bidirectional,
#     BatchNormalization,
#     GaussianNoise,
#     TimeDistributed,
#     Masking,
#     Layer
# )
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.model_selection import train_test_split
#
# KEYPOINTS_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\keypoints"
# )
#
# MODEL_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models"
# )
#
# MODEL_PATH.mkdir(exist_ok=True)
#
# SEQ_LEN = 30
#
# def load_sequence(file_path):
#     data = np.load(file_path)
#
#     if len(data) >= SEQ_LEN:
#         start = random.randint(0, len(data) - SEQ_LEN)
#         return data[start:start + SEQ_LEN]
#     else:
#         padded = np.zeros((SEQ_LEN, data.shape[1]))
#         padded[:len(data)] = data
#         return padded
#
#
# print("\nBuilding dataset from keypoints...")
#
# X, y_labels = [], []
#
# for class_dir in sorted(KEYPOINTS_PATH.iterdir()):
#     if not class_dir.is_dir():
#         continue
#
#     label = int(class_dir.name)
#
#     for file in class_dir.glob("*.npy"):
#         seq = load_sequence(file)
#         X.append(seq)
#         y_labels.append(label)
#
# X = np.array(X)
# y_labels = np.array(y_labels)
#
# print("Dataset built:")
# print("X:", X.shape)
# print("y:", y_labels.shape)
#
# print("\nRemapping labels...")
#
# unique_labels = sorted(np.unique(y_labels))
# label_map = {label: idx for idx, label in enumerate(unique_labels)}
# inverse_map = {idx: label for label, idx in label_map.items()}
#
# y_mapped = np.array([label_map[l] for l in y_labels])
#
# num_classes = len(unique_labels)
# print("Num classes:", num_classes)
#
# print("\nNormalizing dataset...")
#
# mean = np.mean(X, axis=(0, 1), keepdims=True)
# std = np.std(X, axis=(0, 1), keepdims=True) + 1e-8
#
# X = (X - mean) / std
#
# np.save(MODEL_PATH / "mean.npy", mean)
# np.save(MODEL_PATH / "std.npy", std)
# np.save(MODEL_PATH / "label_map.npy", label_map)
# np.save(MODEL_PATH / "inverse_map.npy", inverse_map)
#
# y = to_categorical(y_mapped, num_classes)
#
# X_train, X_val, y_train, y_val = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_mapped
# )
#
# print("Train:", X_train.shape, "Val:", X_val.shape)
#
# def temporal_dropout(X, max_drop=10):
#     X_aug = X.copy()
#
#     for i in range(len(X_aug)):
#         drop = np.random.randint(0, max_drop)
#
#         if drop > 0:
#             X_aug[i, :drop, :] = 0
#
#     return X_aug
#
# X_train = temporal_dropout(X_train)
#
# class TemporalAttention(Layer):
#     def __init__(self):
#         super().__init__()
#
#     def build(self, input_shape):
#         self.W = self.add_weight(
#             name="att_weight",
#             shape=(input_shape[-1], 1),
#             initializer="glorot_uniform",
#             trainable=True
#         )
#         self.b = self.add_weight(
#             name="att_bias",
#             shape=(input_shape[1], 1),
#             initializer="zeros",
#             trainable=True
#         )
#
#     def call(self, x):
#         e = tf.tensordot(x, self.W, axes=1) + self.b
#         e = tf.nn.tanh(e)
#
#         alpha = tf.nn.softmax(e, axis=1)
#         context = tf.reduce_sum(x * alpha, axis=1)
#
#         return context
#
# print("\nBuilding model...")
#
# inputs = Input(shape=(SEQ_LEN, 1692))
#
# x = GaussianNoise(0.03)(inputs)
# x = Masking(mask_value=0.0)(x)
#
# x = TimeDistributed(Dense(256, activation="relu"))(x)
# x = TimeDistributed(Dropout(0.2))(x)
#
# x = Bidirectional(LSTM(128, return_sequences=True))(x)
# x = BatchNormalization()(x)
# x = Dropout(0.3)(x)
#
# x = Bidirectional(LSTM(128, return_sequences=True))(x)
# x = BatchNormalization()(x)
# x = Dropout(0.3)(x)
#
# x = TemporalAttention()(x)
#
# x = Dense(256, activation="relu")(x)
# x = Dropout(0.4)(x)
#
# outputs = Dense(num_classes, activation="softmax")(x)
#
# model = Model(inputs, outputs)
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#     loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
#     metrics=["accuracy"]
# )
#
# model.summary()
#
# early_stop = EarlyStopping(
#     monitor="val_loss",
#     patience=15,
#     restore_best_weights=True,
#     verbose=1
# )
#
# checkpoint = ModelCheckpoint(
#     filepath=str(MODEL_PATH / "sign_lstm_best_1.4.h5"),
#     monitor="val_loss",
#     save_best_only=True,
#     verbose=1
# )
#
# print("\nTraining...")
#
# model.fit(
#     X_train,
#     y_train,
#     validation_data=(X_val, y_val),
#     epochs=120,
#     batch_size=32,
#     callbacks=[early_stop, checkpoint]
# )
#
# model.save(str(MODEL_PATH / "sign_lstm_model_1.4.h5"))
#
# print("\nTraining complete.")










# I dont even know lowkey
# import numpy as np
# from pathlib import Path
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.utils import to_categorical
#
#
# DATA_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed"
# )
#
# MODEL_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models"
# )
#
# MODEL_PATH.mkdir(parents=True, exist_ok=True)
#
#
# print("\nLoading dataset...")
# X = np.load(DATA_PATH / "X.npy")
# y_labels = np.load(DATA_PATH / "y.npy")
#
# print("X shape:", X.shape)
# print("y shape:", y_labels.shape)
#
#
# from sklearn.model_selection import train_test_split
#
# X_train, X_val, y_train_labels, y_val_labels = train_test_split(
#     X,
#     y_labels,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_labels
# )
#
# print("\nTrain set:", X_train.shape)
# print("Validation set:", X_val.shape)
#
#
# print("\nNormalizing dataset...")
#
# mean = np.mean(X_train, axis=(0, 1), keepdims=True)
# std = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-8
#
# X_train = (X_train - mean) / std
# X_val = (X_val - mean) / std
#
# np.save(DATA_PATH / "mean.npy", mean)
# np.save(DATA_PATH / "std.npy", std)
#
# print("Normalization saved.")
#
#
# num_classes = len(np.unique(y_labels))
# print("\nNumber of classes:", num_classes)
#
# y_train = to_categorical(y_train_labels, num_classes)
# y_val = to_categorical(y_val_labels, num_classes)
#
#
# print("\nBuilding model...")
#
# model = Sequential()
# model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(30, X.shape[2])))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
#
# model.add(Bidirectional(LSTM(64)))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
#
# model.add(Dense(128, activation="relu"))
# model.add(Dropout(0.4))
# model.add(Dense(num_classes, activation="softmax"))
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )
#
# model.summary()
#
# checkpoint = ModelCheckpoint(
#     filepath=str(MODEL_PATH / "best_model.weights.h5"),
#     monitor="val_accuracy",
#     save_best_only=True,
#     save_weights_only=True,
#     verbose=1
# )
#
# early_stop = EarlyStopping(
#     monitor="val_accuracy",
#     patience=8,
#     restore_best_weights=True
# )
#
# print("\nTraining...")
#
# history = model.fit(
#     X_train,
#     y_train,
#     validation_data=(X_val, y_val),
#     epochs=80,
#     batch_size=32,
#     callbacks=[early_stop, checkpoint]
# )
#
#
# print("\nSaving final model...")
# model.save(str(MODEL_PATH / "final_model.keras"))
#
# print("\nTraining complete.")