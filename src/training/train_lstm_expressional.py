# IGNORE
# EXPRESSIONAL MODEL
### UNDEVELOPED OR UNFINISHED
# architectural design for the expressional model

### fail ? overfit def
# Input (250, 275)
# → Conv1D (temporal)
# → Conv1D (temporal)
# → BiLSTM
# → BiLSTM
# → Dense
# → Output
# import numpy as np
# from pathlib import Path
# import tensorflow as tf
# import os
# import re
#
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (
#     Dense,
#     Dropout,
#     BatchNormalization,
#     Bidirectional,
#     LSTM,
#     Conv1D
# )
# from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.utils import to_categorical
#
# DATA_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\expressional"
# )
#
# CHECKPOINT_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\expressional\checkpoints"
# )
#
# FINAL_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\expressional"
# )
#
# CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
# FINAL_PATH.mkdir(parents=True, exist_ok=True)
#
# print("\nLoading dataset...")
#
# X_train = np.load(DATA_PATH / "X_train.npy")
# y_train = np.load(DATA_PATH / "y_train.npy")
#
# X_val = np.load(DATA_PATH / "X_val.npy")
# y_val = np.load(DATA_PATH / "y_val.npy")
#
# X_test = np.load(DATA_PATH / "X_test.npy")
# y_test = np.load(DATA_PATH / "y_test.npy")
#
# num_classes = len(np.unique(y_train))
#
# print("Train:", X_train.shape)
# print("Val:", X_val.shape)
# print("Test:", X_test.shape)
#
# y_train = to_categorical(y_train, num_classes)
# y_val = to_categorical(y_val, num_classes)
#
# print("\nBuilding model...")
#
# model = Sequential()
#
# model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu',
#                  input_shape=(250, 275)))
# model.add(BatchNormalization())
#
# model.add(Conv1D(128, kernel_size=5, strides=2, padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
#
# model.add(Bidirectional(LSTM(128, return_sequences=True)))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
#
# model.add(Bidirectional(LSTM(128)))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
#
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.4))
#
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )
#
# model.summary()
#
# class KeepLastTwoCheckpoints(Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         filename = CHECKPOINT_PATH / f"sign_bilstm_expressional_epoch_{epoch+1}.keras"
#         self.model.save(filename)
#
#         files = sorted(CHECKPOINT_PATH.glob("sign_bilstm_expressional_epoch_*.keras"),
#                        key=os.path.getmtime)
#
#         if len(files) > 2:
#             for f in files[:-2]:
#                 os.remove(f)
#
# early_stop = EarlyStopping(
#     monitor="val_accuracy",
#     patience=25,
#     restore_best_weights=True,
#     verbose=1
# )
#
# lr_scheduler = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.3,
#     patience=10,
#     min_lr=1e-6,
#     verbose=1
# )
#
# checkpoint_cleaner = KeepLastTwoCheckpoints()
#
# print("\nTraining...")
#
# history = model.fit(
#     X_train,
#     y_train,
#     validation_data=(X_val, y_val),
#     epochs=300, # could be 500 since we got early stop
#     batch_size=32,
#     callbacks=[early_stop, lr_scheduler, checkpoint_cleaner]
# )
#
# max(history.history["val_accuracy"])
# history.history["val_loss"][-10:]
#
# print("\nSaving best and final models...")
#
# model.save(FINAL_PATH / "sign_lstm_expressional_final.keras")
#
# best_model_path = FINAL_PATH / "sign_lstm_expressional_best.keras"
# model.save(best_model_path)
#
# print("Saved:")
# print("Final:", FINAL_PATH / "sign_lstm_expressional_final.keras")
# print("Best:", best_model_path)








# aint work
# import numpy as np
# from pathlib import Path
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (
#     Dense, Dropout, BatchNormalization, Bidirectional, LSTM, Conv1D, Layer
# )
# from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import GlobalAveragePooling1D
#
# DATA_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\expressional")
# CHECKPOINT_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\expressional\checkpoints")
# FINAL_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\expressional")
# CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
# FINAL_PATH.mkdir(parents=True, exist_ok=True)
#
# y_train = np.load(DATA_PATH / "y_train.npy")
# y_val   = np.load(DATA_PATH / "y_val.npy")
# y_test  = np.load(DATA_PATH / "y_test.npy")
# num_classes = len(np.unique(y_train))
#
# y_train_cat = to_categorical(y_train, num_classes)
# y_val_cat   = to_categorical(y_val, num_classes)
#
# # --- Delta feature function ---
# # def add_delta_features(X):
# #     delta = np.diff(X, axis=1)
# #     delta = np.concatenate([np.zeros((X.shape[0], 1, X.shape[2]), dtype=np.float32), delta.astype(np.float32)], axis=1)
# #     return np.concatenate([X, delta], axis=2)
# def apply_motion_normalization(X):
#     X = X.astype(np.float32)
#     X[:, 1:] = X[:, 1:] - X[:, :-1]
#     return X
#
# def dataset_generator(X_path, y_cat, batch_size=32):
#     num_samples = y_cat.shape[0]
#     indices = np.arange(num_samples)
#     while True:
#         np.random.shuffle(indices)
#         for start in range(0, num_samples, batch_size):
#             end = min(start + batch_size, num_samples)
#             batch_idx = indices[start:end]
#
#             # X_batch = np.load(X_path)[batch_idx].astype(np.float32)  # load selected batch
#             # X_batch = add_delta_features(X_batch)                     # add delta features
#
#             # X_batch = np.load(X_path)[batch_idx].astype(np.float32)
#             X_data = np.load(X_path).astype(np.float32)
#
#             def dataset_generator(X_data, y_cat, batch_size=32):
#                 ...
#                 X_batch = X_data[batch_idx]
#
#             X_batch = apply_motion_normalization(X_batch)
#
#             y_batch = y_cat[batch_idx]
#             yield X_batch, y_batch
#
# BATCH_SIZE = 16
#
# train_dataset = tf.data.Dataset.from_generator(
#     lambda: dataset_generator(DATA_PATH / "X_train.npy", y_train_cat, batch_size=BATCH_SIZE),
#     output_signature=(
#         tf.TensorSpec(shape=(None, 250, 550), dtype=tf.float32),
#         tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
#     )
# )
#
# val_dataset = tf.data.Dataset.from_generator(
#     lambda: dataset_generator(DATA_PATH / "X_val.npy", y_val_cat, batch_size=BATCH_SIZE),
#     output_signature=(
#         tf.TensorSpec(shape=(None, 250, 550), dtype=tf.float32),
#         tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
#     )
# )
#
# class AttentionLayer(Layer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
#                                  initializer="glorot_uniform", trainable=True)
#         self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
#                                  initializer="zeros", trainable=True)
#         super().build(input_shape)
#
#     def call(self, x):
#         e = K.tanh(K.dot(x, self.W) + self.b)
#         a = K.softmax(e, axis=1)
#         output = x * a
#         return K.sum(output, axis=1)
#
# model = Sequential()
# model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu',
#                  # input_shape=(250, 550)))
#                  input_shape=(250, 275)))
# model.add(BatchNormalization())
# model.add(Conv1D(128, kernel_size=5, strides=2, padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
#
# model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
# model.add(BatchNormalization())
#
# # model.add(AttentionLayer())
# model.add(GlobalAveragePooling1D())
#
# model.add(Dropout(0.5))
#
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )
#
# model.summary()
#
# class KeepLastTwoCheckpoints(Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         filename = CHECKPOINT_PATH / f"sign_bilstm_expressional_epoch_{epoch+1}.keras"
#         self.model.save(filename)
#         files = sorted(CHECKPOINT_PATH.glob("sign_bilstm_expressional_epoch_*.keras"),
#                        key=lambda f: f.stat().st_mtime)
#         if len(files) > 2:
#             for f in files[:-2]:
#                 f.unlink()
#
# early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
# lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=10, min_lr=1e-6, verbose=1)
# checkpoint_cleaner = KeepLastTwoCheckpoints()
#
# steps_per_epoch = int(np.ceil(y_train.shape[0] / BATCH_SIZE))
# validation_steps = int(np.ceil(y_val.shape[0] / BATCH_SIZE))
#
# history = model.fit(
#     train_dataset,
#     epochs=300,
#     steps_per_epoch=steps_per_epoch,
#     validation_data=val_dataset,
#     validation_steps=validation_steps,
#     callbacks=[early_stop, lr_scheduler, checkpoint_cleaner]
# )
#
# model.save(FINAL_PATH / "sign_lstm_expressional_final.keras")
# model.save(FINAL_PATH / "sign_lstm_expressional_best.keras")
# print("Saved final and best models")






#fail
# import numpy as np
# from pathlib import Path
# import tensorflow as tf
#
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (
#     Dense, Dropout, BatchNormalization,
#     Bidirectional, LSTM, Conv1D
# )
# from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.utils import to_categorical
#
# DATA_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\expressional")
# CHECKPOINT_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\expressional\checkpoints")
# FINAL_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\expressional")
#
# CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
# FINAL_PATH.mkdir(parents=True, exist_ok=True)
#
# print("\nLoading labels...")
#
# y_train = np.load(DATA_PATH / "y_train.npy")
# y_val   = np.load(DATA_PATH / "y_val.npy")
# y_test  = np.load(DATA_PATH / "y_test.npy")
#
# num_classes = len(np.unique(y_train))
#
# y_train_cat = to_categorical(y_train, num_classes)
# y_val_cat   = to_categorical(y_val, num_classes)
#
# print("Train samples:", len(y_train))
# print("Val samples:", len(y_val))
#
# def apply_motion_normalization(X):
#     X = X.astype(np.float32)
#     X[:, 1:] = X[:, 1:] - X[:, :-1]
#     return X
#
# def dataset_generator(X_path, y_cat, batch_size=16):
#     num_samples = y_cat.shape[0]
#
#     while True:
#         indices = np.random.permutation(num_samples)
#
#         for start in range(0, num_samples, batch_size):
#             end = min(start + batch_size, num_samples)
#             batch_idx = indices[start:end]
#
#             # load ONLY needed batch
#             X_data = np.load(X_path, mmap_mode='r')
#             X_batch = X_data[batch_idx]
#
#             X_batch = apply_motion_normalization(X_batch)
#             X_batch = X_batch.astype(np.float32)
#
#             y_batch = y_cat[batch_idx]
#
#             yield X_batch, y_batch
#
# BATCH_SIZE = 16
#
# train_dataset = tf.data.Dataset.from_generator(
#     lambda: dataset_generator(DATA_PATH / "X_train.npy", y_train_cat, BATCH_SIZE),
#     output_signature=(
#         tf.TensorSpec(shape=(None, 250, 275), dtype=tf.float32),
#         tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
#     )
# )
#
# val_dataset = tf.data.Dataset.from_generator(
#     lambda: dataset_generator(DATA_PATH / "X_val.npy", y_val_cat, BATCH_SIZE),
#     output_signature=(
#         tf.TensorSpec(shape=(None, 250, 275), dtype=tf.float32),
#         tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
#     )
# )
#
# print("\nBuilding model...")
#
# model = Sequential()
#
# # conv (NO temporal downsampling)
# model.add(Conv1D(128, 5, padding='same', activation='relu',
#                  input_shape=(250, 275)))
# model.add(BatchNormalization())
#
# model.add(Conv1D(128, 5, padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
#
# model.add(Bidirectional(LSTM(128, return_sequences=True,
#                              dropout=0.3, recurrent_dropout=0.2)))
# model.add(BatchNormalization())
#
# model.add(Bidirectional(LSTM(128)))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
#
#
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.4))
#
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )
#
# model.summary()
#
# class KeepLastTwoCheckpoints(Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         filename = CHECKPOINT_PATH / f"epoch_{epoch+1}.keras"
#         self.model.save(filename)
#
#         files = sorted(CHECKPOINT_PATH.glob("epoch_*.keras"),
#                        key=lambda f: f.stat().st_mtime)
#
#         if len(files) > 2:
#             for f in files[:-2]:
#                 f.unlink()
#
# early_stop = EarlyStopping(
#     monitor="val_loss",
#     patience=12,
#     restore_best_weights=True,
#     verbose=1
# )
#
# lr_scheduler = ReduceLROnPlateau(
#     monitor="val_loss",
#     factor=0.3,
#     patience=6,
#     min_lr=1e-6,
#     verbose=1
# )
#
# print("\nTraining...")
#
# steps_per_epoch = int(np.ceil(len(y_train) / BATCH_SIZE))
# val_steps = int(np.ceil(len(y_val) / BATCH_SIZE))
#
# history = model.fit(
#     train_dataset,
#     epochs=100,
#     steps_per_epoch=steps_per_epoch,
#     validation_data=val_dataset,
#     validation_steps=val_steps,
#     callbacks=[early_stop, lr_scheduler, KeepLastTwoCheckpoints()]
# )
#
# print("\nSaving model...")
#
# model.save(FINAL_PATH / "sign_lstm_expressional_fixed.keras")
#
# print("Done.")




# hope ? ovrfit
# import numpy as np
# from pathlib import Path
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Bidirectional, LSTM
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
#
# DATA_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\expressional")
# MODEL_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\expressional")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)
#
# TARGET_FRAMES = 30
#
# def fix_sequence(seq, target_len=TARGET_FRAMES):
#     if len(seq) == target_len:
#         return seq
#     idxs = np.linspace(0, len(seq)-1, target_len).astype(int)
#     return seq[idxs]
#
# def load_and_preprocess(X_path, y_path):
#     X = np.load(X_path)
#     y = np.load(y_path)
#
#     # Resample to fixed sequence
#     X_fixed = np.array([fix_sequence(seq) for seq in X], dtype=np.float32)
#
#     # Motion normalization
#     X_fixed[:, 1:] = X_fixed[:, 1:] - X_fixed[:, :-1]
#
#     return X_fixed, y
#
# X_train, y_train = load_and_preprocess(DATA_PATH / "X_train.npy", DATA_PATH / "y_train.npy")
# X_val, y_val     = load_and_preprocess(DATA_PATH / "X_val.npy", DATA_PATH / "y_val.npy")
# X_test, y_test   = load_and_preprocess(DATA_PATH / "X_test.npy", DATA_PATH / "y_test.npy")
#
# num_classes = len(np.unique(y_train))
# y_train_cat = to_categorical(y_train, num_classes)
# y_val_cat   = to_categorical(y_val, num_classes)
# y_test_cat  = to_categorical(y_test, num_classes)
#
# mean = np.mean(X_train, axis=(0,1), keepdims=True)
# std = np.std(X_train, axis=(0,1), keepdims=True) + 1e-6
#
# X_train = (X_train - mean) / std
# X_val   = (X_val - mean) / std
# X_test  = (X_test - mean) / std
#
# model = Sequential([
#     Bidirectional(LSTM(128, return_sequences=True), input_shape=(TARGET_FRAMES, X_train.shape[2])),
#     BatchNormalization(),
#     Dropout(0.3),
#
#     Bidirectional(LSTM(128)),
#     BatchNormalization(),
#     Dropout(0.4),
#
#     Dense(256, activation='relu'),
#     Dropout(0.4),
#
#     Dense(num_classes, activation='softmax')
# ])
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#     loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),  # reduces overconfidence
#     metrics=['accuracy']
# )
#
# model.summary()
#
# checkpoint = ModelCheckpoint(
#     filepath=str(MODEL_PATH / "sign_lstm_expressional_best.keras"),  # <-- convert to string
#     monitor="val_accuracy",
#     save_best_only=True,
#     verbose=1
# )
#
# early_stop = EarlyStopping(
#     monitor="val_accuracy",
#     patience=10,
#     restore_best_weights=True,
#     verbose=1
# )
#
# history = model.fit(
#     X_train, y_train_cat,
#     validation_data=(X_val, y_val_cat),
#     epochs=200,
#     batch_size=32,
#     callbacks=[checkpoint, early_stop]
# )
#
# model.save(MODEL_PATH / "sign_lstm_expressional_final.keras")
# print("Training complete. Best and final models saved.")






# overfit
# import numpy as np
# from pathlib import Path
# import tensorflow as tf
#
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Bidirectional, LSTM
# from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.regularizers import l2
#
# from sklearn.utils.class_weight import compute_class_weight
#
# DATA_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\expressional")
# MODEL_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\expressional")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)
#
# TARGET_FRAMES = 30
#
# def fix_sequence(seq, target_len=TARGET_FRAMES):
#     if len(seq) == target_len:
#         return seq
#     idxs = np.linspace(0, len(seq)-1, target_len).astype(int)
#     return seq[idxs]
#
# def load_and_preprocess(X_path, y_path):
#     X = np.load(X_path)
#     y = np.load(y_path)
#
#     # Fix sequence length
#     X_fixed = np.array([fix_sequence(seq) for seq in X], dtype=np.float32)
#
#     # Motion normalization
#     X_fixed[:, 1:] = X_fixed[:, 1:] - X_fixed[:, :-1]
#
#     return X_fixed, y
#
# X_train, y_train = load_and_preprocess(DATA_PATH / "X_train.npy", DATA_PATH / "y_train.npy")
# X_val, y_val     = load_and_preprocess(DATA_PATH / "X_val.npy", DATA_PATH / "y_val.npy")
# X_test, y_test   = load_and_preprocess(DATA_PATH / "X_test.npy", DATA_PATH / "y_test.npy")
#
# num_classes = len(np.unique(y_train))
#
# y_train_cat = to_categorical(y_train, num_classes)
# y_val_cat   = to_categorical(y_val, num_classes)
# y_test_cat  = to_categorical(y_test, num_classes)
#
# mean = np.mean(X_train, axis=(0,1), keepdims=True)
# std  = np.std(X_train, axis=(0,1), keepdims=True) + 1e-6
#
# X_train = (X_train - mean) / std
# X_val   = (X_val - mean) / std
# X_test  = (X_test - mean) / std
#
# class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(y_train),
#     y=y_train
# )
# class_weights = dict(enumerate(class_weights))
#
# print("Class weights:", class_weights)
#
# class GeneralizationEarlyStopping(Callback):
#     def __init__(self, patience=12, min_delta=0.001):
#         super().__init__()
#         self.patience = patience
#         self.min_delta = min_delta
#         self.best_score = None
#         self.wait = 0
#
#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#
#         train_acc = logs.get('accuracy', 0)
#         val_acc   = logs.get('val_accuracy', 0)
#
#         gap = abs(train_acc - val_acc)
#
#         score = val_acc - (0.5 * gap)
#
#         print(f"Epoch {epoch+1} → val_acc: {val_acc:.4f}, gap: {gap:.4f}, score: {score:.4f}")
#
#         if self.best_score is None or score > self.best_score + self.min_delta:
#             self.best_score = score
#             self.wait = 0
#         else:
#             self.wait += 1
#             if self.wait >= self.patience:
#                 print(f"Stopping early due to overfitting.")
#                 self.model.stop_training = True
#
# model = Sequential([
#     Bidirectional(LSTM(128, return_sequences=True), input_shape=(TARGET_FRAMES, X_train.shape[2])),
#     BatchNormalization(),
#     Dropout(0.4),
#
#     Bidirectional(LSTM(128)),
#     BatchNormalization(),
#     Dropout(0.5),
#
#     Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
#     Dropout(0.5),
#
#     Dense(num_classes, activation='softmax')
# ])
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#     loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
#     metrics=['accuracy']
# )
#
# model.summary()
#
# checkpoint = ModelCheckpoint(
#     filepath=str(MODEL_PATH / "sign_lstm_expressional_best.keras"),
#     monitor="val_accuracy",
#     save_best_only=True,
#     verbose=1
# )
#
# gen_early_stop = GeneralizationEarlyStopping(patience=12)
#
# reduce_lr = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,
#     patience=5,
#     min_lr=1e-6,
#     verbose=1
# )
#
# history = model.fit(
#     X_train, y_train_cat,
#     validation_data=(X_val, y_val_cat),
#     epochs=200,
#     batch_size=32,
#     class_weight=class_weights,
#     callbacks=[checkpoint, gen_early_stop, reduce_lr]
# )
#
# model.save(MODEL_PATH / "sign_lstm_expressional_final.keras")
#
# print("Training complete. Best and final models saved.")





# import numpy as np
# from pathlib import Path
# import tensorflow as tf
#
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.utils import to_categorical
#
# from sklearn.utils.class_weight import compute_class_weight
#
# DATA_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\expressional")
# MODEL_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\expressional")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)
#
# TARGET_FRAMES = 30
# NOISE_LEVEL = 0.01
# FRAME_DROP_PROB = 0.1
#
#
# def fix_sequence(seq, target_len=TARGET_FRAMES):
#     if len(seq) == target_len:
#         return seq
#     idxs = np.linspace(0, len(seq)-1, target_len).astype(int)
#     return seq[idxs]
#
# def add_noise(X, noise_level=NOISE_LEVEL):
#     noise = np.random.normal(0, noise_level, X.shape)
#     return X + noise
#
# def random_frame_mask(X, drop_prob=FRAME_DROP_PROB):
#     mask = np.random.rand(*X.shape[:2]) > drop_prob
#     mask = mask[..., None]
#     return X * mask
#
# def load_and_preprocess(X_path, y_path, augment=False):
#     X = np.load(X_path)
#     y = np.load(y_path)
#
#     # Fix sequence length
#     X_fixed = np.array([fix_sequence(seq) for seq in X], dtype=np.float32)
#
#     # Motion normalization
#     X_fixed[:, 1:] = X_fixed[:, 1:] - X_fixed[:, :-1]
#
#     return X_fixed, y
#
# X_train, y_train = load_and_preprocess(DATA_PATH / "X_train.npy", DATA_PATH / "y_train.npy")
# X_val, y_val     = load_and_preprocess(DATA_PATH / "X_val.npy", DATA_PATH / "y_val.npy")
# X_test, y_test   = load_and_preprocess(DATA_PATH / "X_test.npy", DATA_PATH / "y_test.npy")
#
# num_classes = len(np.unique(y_train))
#
# y_train_cat = to_categorical(y_train, num_classes)
# y_val_cat   = to_categorical(y_val, num_classes)
# y_test_cat  = to_categorical(y_test, num_classes)
#
# mean = np.mean(X_train, axis=(0,1), keepdims=True)
# std  = np.std(X_train, axis=(0,1), keepdims=True) + 1e-6
#
# X_train = (X_train - mean) / std
# X_val   = (X_val - mean) / std
# X_test  = (X_test - mean) / std
#
# X_train = add_noise(X_train)
# X_train = random_frame_mask(X_train)
#
# class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(y_train),
#     y=y_train
# )
# class_weights = dict(enumerate(class_weights))
#
# print("Class weights:", class_weights)
#
# model = Sequential([
#     LSTM(64, return_sequences=False, input_shape=(TARGET_FRAMES, X_train.shape[2])),
#     Dropout(0.3),
#
#     Dense(64, activation='relu'),
#     Dropout(0.3),
#
#     Dense(num_classes, activation='softmax')
# ])
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#     loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
#     metrics=['accuracy']
# )
#
# model.summary()
#
# checkpoint = ModelCheckpoint(
#     filepath=str(MODEL_PATH / "sign_lstm_expressional_best.keras"),
#     monitor="val_accuracy",
#     save_best_only=True,
#     verbose=1
# )
#
# early_stop = EarlyStopping(
#     monitor='val_loss',
#     patience=15,
#     restore_best_weights=True,
#     verbose=1
# )
#
# reduce_lr = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,
#     patience=5,
#     min_lr=1e-6,
#     verbose=1
# )
#
# history = model.fit(
#     X_train, y_train_cat,
#     validation_data=(X_val, y_val_cat),
#     epochs=200,
#     batch_size=32,
#     class_weight=class_weights,
#     callbacks=[checkpoint, early_stop, reduce_lr]
# )
#
# print(X_train.shape)
# print(np.mean(X_train), np.std(X_train))
# print(np.min(X_train), np.max(X_train))
#
# model.save(MODEL_PATH / "sign_lstm_expressional_final.keras")
#
# print("Training complete.")




# import numpy as np
# from pathlib import Path
# import tensorflow as tf
#
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.utils import to_categorical
#
# from sklearn.utils.class_weight import compute_class_weight
#
# DATA_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\expressional")
# MODEL_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\expressional")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)
#
# TARGET_FRAMES = 30
# NOISE_LEVEL = 0.01
# FRAME_DROP_PROB = 0.1
# CLIP_MIN, CLIP_MAX = -5.0, 5.0
# APPLY_MOTION_DIFF = False  # toggle motion difference
#
# POSE_LANDMARKS = list(range(33))   # example, update to your actual indices
# FACE_LANDMARKS = list(range(468))  # example
# HAND_LANDMARKS = list(range(21))   # per hand
#
# def fix_sequence(seq, target_len=TARGET_FRAMES):
#     if len(seq) == target_len:
#         return seq
#     idxs = np.linspace(0, len(seq)-1, target_len).astype(int)
#     return seq[idxs]
#
# def add_noise(X, noise_level=NOISE_LEVEL):
#     noise = np.random.normal(0, noise_level, X.shape)
#     return X + noise
#
# def random_frame_mask(X, drop_prob=FRAME_DROP_PROB):
#     mask = np.random.rand(*X.shape[:2]) > drop_prob
#     mask = mask[..., None]
#     return X * mask
#
# pose_size = len(POSE_LANDMARKS) * 4  # x,y,z,visibility
# face_size = len(FACE_LANDMARKS) * 3
# hand_size = len(HAND_LANDMARKS) * 3
#
# def split_features(X):
#     pose = X[..., :pose_size]
#     face = X[..., pose_size:pose_size+face_size]
#     lh   = X[..., pose_size+face_size:pose_size+face_size+hand_size]
#     rh   = X[..., -hand_size:]
#     return pose, face, lh, rh
#
# def stats(name, x):
#     print(f"{name}: min={x.min():.2f}, max={x.max():.2f}, mean={x.mean():.6f}, std={x.std():.6f}")
#
# def normalize_group(x):
#     mean = np.mean(x, axis=(0,1), keepdims=True)
#     std  = np.std(x, axis=(0,1), keepdims=True) + 1e-6
#     return (x - mean) / std
#
# def preprocess(X):
#     pose, face, lh, rh = split_features(X)
#
#     # optional motion difference
#     if APPLY_MOTION_DIFF:
#         pose[:, 1:] = pose[:, 1:] - pose[:, :-1]
#         face[:, 1:] = face[:, 1:] - face[:, :-1]
#         lh[:, 1:]   = lh[:, 1:] - lh[:, :-1]
#         rh[:, 1:]   = rh[:, 1:] - rh[:, :-1]
#
#     # normalize per group
#     pose = normalize_group(pose)
#     face = normalize_group(face)
#     lh   = normalize_group(lh)
#     rh   = normalize_group(rh)
#
#     # concatenate back
#     X_proc = np.concatenate([pose, face, lh, rh], axis=-1)
#
#     # clip extremes
#     X_proc = np.clip(X_proc, CLIP_MIN, CLIP_MAX)
#
#     return X_proc
#
# def load_and_preprocess(X_path, y_path, augment=False):
#     X = np.load(X_path)
#     y = np.load(y_path)
#
#     # Fix sequence length
#     X_fixed = np.array([fix_sequence(seq) for seq in X], dtype=np.float32)
#
#     # Preprocessing
#     X_proc = preprocess(X_fixed)
#
#     # Apply augmentation only for training
#     if augment:
#         X_proc = add_noise(X_proc)
#         X_proc = random_frame_mask(X_proc)
#
#     return X_proc, y
#
# X_train, y_train = load_and_preprocess(DATA_PATH / "X_train.npy", DATA_PATH / "y_train.npy", augment=True)
# X_val, y_val     = load_and_preprocess(DATA_PATH / "X_val.npy", DATA_PATH / "y_val.npy")
# X_test, y_test   = load_and_preprocess(DATA_PATH / "X_test.npy", DATA_PATH / "y_test.npy")
#
# num_classes = len(np.unique(y_train))
#
# y_train_cat = to_categorical(y_train, num_classes)
# y_val_cat   = to_categorical(y_val, num_classes)
# y_test_cat  = to_categorical(y_test, num_classes)
#
# class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(y_train),
#     y=y_train
# )
# class_weights = dict(enumerate(class_weights))
# print("Class weights:", class_weights)
#
# model = Sequential([
#     LSTM(64, return_sequences=False, input_shape=(TARGET_FRAMES, X_train.shape[2])),
#     Dropout(0.3),
#
#     Dense(64, activation='relu'),
#     Dropout(0.3),
#
#     Dense(num_classes, activation='softmax')
# ])
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#     loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
#     metrics=['accuracy']
# )
#
# model.summary()
#
# checkpoint = ModelCheckpoint(
#     filepath=str(MODEL_PATH / "sign_lstm_expressional_best.keras"),
#     monitor="val_accuracy",
#     save_best_only=True,
#     verbose=1
# )
#
# early_stop = EarlyStopping(
#     monitor='val_loss',
#     patience=15,
#     restore_best_weights=True,
#     verbose=1
# )
#
# reduce_lr = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,
#     patience=5,
#     min_lr=1e-6,
#     verbose=1
# )
#
# history = model.fit(
#     X_train, y_train_cat,
#     validation_data=(X_val, y_val_cat),
#     epochs=200,
#     batch_size=32,
#     class_weight=class_weights,
#     callbacks=[checkpoint, early_stop, reduce_lr]
# )
#
# print(X_train.shape)
# print(np.mean(X_train), np.std(X_train))
# print(np.min(X_train), np.max(X_train))
#
# model.save(MODEL_PATH / "sign_lstm_expressional_final.keras")
# print(" Training complete.")








# dont touch for now
# import numpy as np
# from pathlib import Path
# import tensorflow as tf
# import os
#
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (
#     Dense, Dropout, BatchNormalization, Bidirectional, LSTM, Conv1D, Layer
# )
# from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras import backend as K
#
# DATA_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\expressional")
# CHECKPOINT_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\expressional\checkpoints")
# FINAL_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\expressional")
#
# CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
# FINAL_PATH.mkdir(parents=True, exist_ok=True)
#
# print("\nLoading dataset...")
#
# X_train = np.load(DATA_PATH / "X_train.npy").astype(np.float32)
# y_train = np.load(DATA_PATH / "y_train.npy")
# X_val   = np.load(DATA_PATH / "X_val.npy").astype(np.float32)
# y_val   = np.load(DATA_PATH / "y_val.npy")
# X_test  = np.load(DATA_PATH / "X_test.npy").astype(np.float32)
# y_test  = np.load(DATA_PATH / "y_test.npy")
#
# num_classes = len(np.unique(y_train))
#
# def add_delta_features(X):
#     delta = np.diff(X, axis=1)
#     delta = np.concatenate([np.zeros((X.shape[0], 1, X.shape[2]), dtype=np.float32), delta.astype(np.float32)], axis=1)
#     X_aug = np.concatenate([X, delta], axis=2)
#     return X_aug
#
# X_train = add_delta_features(X_train)
# X_val   = add_delta_features(X_val)
# X_test  = add_delta_features(X_test)
#
# print("Train shape after delta:", X_train.shape)  # should be (6015, 250, 1100)
# print("Val shape after delta:", X_val.shape)
# print("Test shape after delta:", X_test.shape)
#
# y_train = to_categorical(y_train, num_classes)
# y_val   = to_categorical(y_val, num_classes)
#
# class AttentionLayer(Layer):
#     def __init__(self, **kwargs):
#         super(AttentionLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.W = self.add_weight(name="att_weight",
#                                  shape=(input_shape[-1], 1),
#                                  initializer="glorot_uniform",
#                                  trainable=True)
#         self.b = self.add_weight(name="att_bias",
#                                  shape=(input_shape[1], 1),
#                                  initializer="zeros",
#                                  trainable=True)
#         super(AttentionLayer, self).build(input_shape)
#
#     def call(self, x):
#         e = K.tanh(K.dot(x, self.W) + self.b)
#         a = K.softmax(e, axis=1)
#         output = x * a
#         return K.sum(output, axis=1)
#
# print("\nBuilding model...")
#
# model = Sequential()
# model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(BatchNormalization())
# model.add(Conv1D(128, kernel_size=5, strides=2, padding='same', activation='relu'))  # downsample time dimension
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
#
# model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.4, recurrent_dropout=0.3)))
# model.add(BatchNormalization())
#
# model.add(AttentionLayer())
# model.add(Dropout(0.5))
#
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )
#
# model.summary()
#
# class KeepLastTwoCheckpoints(Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         filename = CHECKPOINT_PATH / f"sign_bilstm_expressional_epoch_{epoch+1}.keras"
#         self.model.save(filename)
#         files = sorted(CHECKPOINT_PATH.glob("sign_bilstm_expressional_epoch_*.keras"),
#                        key=os.path.getmtime)
#         if len(files) > 2:
#             for f in files[:-2]:
#                 os.remove(f)
#
# early_stop = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=1)
# lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, min_lr=1e-6, verbose=1)
# checkpoint_cleaner = KeepLastTwoCheckpoints()
#
# print("\nTraining...")
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=300,
#     batch_size=16,
#     callbacks=[early_stop, lr_scheduler, checkpoint_cleaner]
# )
#
# print("\nSaving best and final models...")
# model.save(FINAL_PATH / "sign_lstm_expressional_final.keras")
# model.save(FINAL_PATH / "sign_lstm_expressional_best.keras")
# print("Saved:")
# print("Final:", FINAL_PATH / "sign_lstm_expressional_final.keras")
# print("Best:", FINAL_PATH / "sign_lstm_expressional_best.keras")