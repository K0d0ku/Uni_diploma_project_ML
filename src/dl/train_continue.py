# CURRENT V 1.4
# long story short i wanted to try "Model Grokking" but due to the limited computational hardware capability
# i was limited by time and could train only for 15.6 Mil parameters, and 250 + 500 epochs, and even that took nearly
# 40 real life hours tho considering it trained on "flawed" data it still made great improvements

import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATA_PATH = Path(
    r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\dl\data"
)

MODEL_PATH = Path(
    r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\dl\model"
)

BEST_MODEL_FILE = MODEL_PATH / "sign_lstm_masked_best.keras"
FINAL_MODEL_FILE = MODEL_PATH / "sign_lstm_masked_supreme.keras"

print("Loading dataset...")

X = np.load(DATA_PATH / "X.npy")
M = np.load(DATA_PATH / "mask.npy")
y_labels = np.load(DATA_PATH / "y.npy")

num_classes = len(np.unique(y_labels))
y = to_categorical(y_labels, num_classes)

X_train, X_val, M_train, M_val, y_train, y_val = train_test_split(
    X, M, y,
    test_size=0.2,
    random_state=42,
    stratify=y_labels
)

print("Loading model...")

model = load_model(BEST_MODEL_FILE)

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=500,
    min_delta=0.001,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath=str(MODEL_PATH / "sign_lstm_masked_supreme_best.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

print("Continuing training...")

history = model.fit(
    [X_train, M_train],
    y_train,
    validation_data=([X_val, M_val], y_val),
    epochs=500,
    batch_size=32,
    callbacks=[early_stop, checkpoint]
)

model.save(FINAL_MODEL_FILE)

print("Done.")