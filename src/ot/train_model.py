# IGNORE
# its just another unsuccessfull branch of model training

import numpy as np
from pathlib import Path
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, Callback
)
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
DATA_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\ot\data")
MODEL_PATH = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\ot\models")

MODEL_PATH.mkdir(parents=True, exist_ok=True)


X = np.load(DATA_PATH / "X.npy")
y = np.load(DATA_PATH / "y.npy")
mask = np.load(DATA_PATH / "mask.npy")

num_classes = len(np.unique(y))
y_cat = to_categorical(y, num_classes)


X_train, X_val, y_train, y_val, m_train, m_val = train_test_split(
    X, y_cat, mask,
    test_size=0.2,
    random_state=42,
    stratify=y
)


inputs = Input(shape=(30, 1692))

x = Bidirectional(LSTM(96, return_sequences=True))(inputs)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Bidirectional(LSTM(64))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

class TopK(Callback):
    def __init__(self, path, k=2):
        super().__init__()
        self.path = path
        self.k = k
        self.saved = []

    def on_epoch_end(self, epoch, logs=None):
        loss = logs['val_loss']
        fname = self.path / f"ep{epoch:03d}_loss{loss:.4f}.keras"
        self.model.save(fname)

        self.saved.append((loss, fname))
        self.saved.sort(key=lambda x: x[0])

        if len(self.saved) > self.k:
            worst = self.saved.pop(-1)
            try:
                worst[1].unlink()
            except:
                pass

callbacks = [
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=12, min_lr=1e-6),
    TopK(MODEL_PATH)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

model.save(MODEL_PATH / "final_model_v2.keras")