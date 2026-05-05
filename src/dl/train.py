### CURRENT V 1.4
# model with what i think with fixed workaround on flawed dataset, using mask to zero out padded frames

import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Bidirectional,
    Input,
    Multiply
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

DATA_PATH = Path(
    r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\dl\data"
)

MODEL_PATH = Path(
    r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\dl\model"
)

MODEL_PATH.mkdir(exist_ok=True)

print("\nLoading dataset...")

X = np.load(DATA_PATH / "X.npy")
M = np.load(DATA_PATH / "mask.npy")
y_labels = np.load(DATA_PATH / "y.npy")

print("X:", X.shape)
print("Mask:", M.shape)
print("y:", y_labels.shape)

num_classes = len(np.unique(y_labels))
y = to_categorical(y_labels, num_classes)

X_train, X_val, M_train, M_val, y_train, y_val = train_test_split(
    X, M, y,
    test_size=0.2,
    random_state=42,
    stratify=y_labels
)

print("\nTrain:", X_train.shape)
print("Val:", X_val.shape)

input_seq = Input(shape=(30, 1692))
input_mask = Input(shape=(30, 1))

masked_seq = Multiply()([input_seq, input_mask])

x = Bidirectional(LSTM(512, return_sequences=True))(masked_seq) # 15.6 mil parameters
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Bidirectional(LSTM(512))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)

output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=[input_seq, input_mask], outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=240,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath=str(MODEL_PATH / "sign_lstm_masked_best.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

print("\nTraining...")

history = model.fit(
    [X_train, M_train],
    y_train,
    validation_data=([X_val, M_val], y_val),
    epochs=240,
    batch_size=32,
    callbacks=[early_stop, checkpoint]
)

model.save(str(MODEL_PATH / "sign_lstm_masked_final.keras"))
print("\nDone.")