# OLD AND UNUSED

# import numpy as np
# from tensorflow.keras.models import load_model
# from pathlib import Path
# import json
#
# MODEL_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\sign_lstm_supreme.keras")
# ANNOTATIONS_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\annotations")
#
#
# def load_annotations(language="kazakh"):
#     file_map = {
#         "kazakh": "kazakh.json",
#         "russian": "russian.json",
#         "gloss": "gloss.json"
#     }
#     path = ANNOTATIONS_PATH / file_map.get(language, "kazakh.json")
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)
#
#
# def predict_class(model, X_sample, language="kazakh"):
#     pred_idx = int(np.argmax(model.predict(np.expand_dims(X_sample, axis=0))))
#     annotations = load_annotations(language)
#     return pred_idx, annotations.get(str(pred_idx), "UNKNOWN")
#
#
# if __name__ == "__main__":
#     model = load_model(MODEL_PATH)
#
#     # Load a sample from your dataset (replace with actual sample)
#     X_sample = np.load(r"c:\path\to\X_sample.npy")
#
#     pred_id, pred_text = predict_class(model, X_sample, language="kazakh")
#     print(f"Predicted ID: {pred_id}, Annotation: {pred_text}")










# import numpy as np
# import tensorflow as tf
# from pathlib import Path
# import json
#
# MODEL_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\sign_lstm_supreme_face_reduced_best.keras")
# ANNOTATIONS_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\annotations")
#
# FACE_LANDMARKS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373,
#                   380, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
# FACE_INDICES = np.concatenate([np.arange(225 + i * 3, 225 + i * 3 + 3) for i in FACE_LANDMARKS])
#
#
# def load_annotations(language="kazakh"):
#     path = ANNOTATIONS_PATH / f"{language}.json"
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)
#
#
# def predict_class(model, X_sample, language="kazakh"):
#     # X_sample expected to be (30, 351) already normalized
#     # or (1, 30, 351)
#     if len(X_sample.shape) == 2:
#         X_sample = np.expand_dims(X_sample, axis=0)
#
#     prediction = model.predict(X_sample, verbose=0)
#     pred_idx = int(np.argmax(prediction))
#     conf = np.max(prediction)
#
#     annotations = load_annotations(language)
#     return pred_idx, annotations.get(str(pred_idx), "UNKNOWN"), conf
#
#
# if __name__ == "__main__":
#     model = tf.keras.models.load_model(MODEL_PATH, compile=False)
#     # Testing with dummy data of correct shape (1, 30, 351)
#     X_test = np.random.rand(1, 30, 351)
#     pid, text, c = predict_class(model, X_test)
#     print(f"Predicted: {text} ({c:.2%})")

    ### FIXME


# import numpy as np
# import tensorflow as tf
# from pathlib import Path
# import json
#
# MODEL_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\sign_lstm_supreme_face_reduced_final.keras")
# ANNOTATIONS_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\annotations")
#
# POSE_HAND_SIZE = 225
# FACE_START = 225
# FACE_LANDMARKS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373,
#                   380, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
# face_indices = []
# for idx in FACE_LANDMARKS:
#     base = FACE_START + idx * 3
#     face_indices.extend([base, base + 1, base + 2])
# face_indices = np.array(face_indices)
#
#
# def load_annotations(language="kazakh"):
#     path = ANNOTATIONS_PATH / f"{language}.json"
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)
#
#
# def predict_class(model, X_sample, language="kazakh"):
#     """
#     X_sample: numpy array of shape (30, 1692)
#     """
#     # 1. Reduce features if input is the full 1692
#     if X_sample.shape[-1] == 1692:
#         pose_hands = X_sample[:, :POSE_HAND_SIZE]
#         face_selected = X_sample[:, face_indices]
#         X_input = np.concatenate([pose_hands, face_selected], axis=1)
#     else:
#         X_input = X_sample
#
#     # 2. Add batch dimension (1, 30, 351)
#     X_input = np.expand_dims(X_input, axis=0)
#
#     # 3. Predict
#     prediction = model.predict(X_input, verbose=0)
#     pred_idx = int(np.argmax(prediction))
#
#     annotations = load_annotations(language)
#     return pred_idx, annotations.get(str(pred_idx), "UNKNOWN"), np.max(prediction)
#
#
# if __name__ == "__main__":
#     # Load model
#     model = tf.keras.models.load_model(MODEL_PATH)
#
#     # Load a sample (make sure this path exists)
#     sample_path = Path(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\fin_dataset\X.npy")
#     if sample_path.exists():
#         X_test = np.load(sample_path)
#         pred_id, pred_text, conf = predict_class(model, X_test)
#         print(f"Result: {pred_text} (ID: {pred_id}) with {conf:.2%} confidence")
#     else:
#         print("Sample file not found. Please update sample_path.")
















### fixme
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path
import json

MODEL_PATH = Path(
    r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\sign_lstm_supreme_face_reduced_best.keras"
)

DATA_PATH = Path(
    r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\fin_dataset"
)

ANNOTATIONS_PATH = Path(
    r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\annotations"
)

SEQUENCE_LENGTH = 30
POSE_HAND_SIZE = 225
FULL_FEATURE_SIZE = 1692
REDUCED_FEATURE_SIZE = 351

FACE_LANDMARKS = [
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324
]

mean = np.load(DATA_PATH / "mean.npy").reshape(-1)
std = np.load(DATA_PATH / "std.npy").reshape(-1)
std[std == 0] = 1e-6


def load_annotations(language="kazakh"):
    file_map = {
        "kazakh": "kazakh.json",
        "russian": "russian.json",
        "gloss": "gloss.json"
    }

    path = ANNOTATIONS_PATH / file_map.get(language, "kazakh.json")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def reduce_features(X_full):

    if len(X_full.shape) == 3:
        if X_full.shape[0] > 1:
            print("⚠️ Dataset detected. Using first sample only.")
        X_full = X_full[0]

    if X_full.shape[0] != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected {SEQUENCE_LENGTH} frames, got {X_full.shape[0]}"
        )

    full_feature_size = X_full.shape[1]
    reduced_sequence = []
    for frame in X_full:

        pose_hands = frame[:POSE_HAND_SIZE]
        face_start = POSE_HAND_SIZE
        face_indices = []
        for idx in FACE_LANDMARKS:
            base = face_start + idx * 3
            if base + 2 < full_feature_size:
                face_indices.extend([base, base + 1, base + 2])

        face_selected = frame[face_indices]
        reduced = np.concatenate([pose_hands, face_selected])
        reduced_sequence.append(reduced)

    return np.array(reduced_sequence)

def predict_class(model, X_full_sample, language="kazakh"):

    # 1. Reduce
    X_reduced = reduce_features(X_full_sample)

    # 2. Normalize
    X_reduced = (X_reduced - mean) / std

    # 3. Add batch dimension
    X_reduced = np.expand_dims(X_reduced, axis=0)

    prediction = model.predict(X_reduced, verbose=0)[0]
    pred_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    annotations = load_annotations(language)

    return pred_idx, annotations.get(str(pred_idx), "UNKNOWN"), confidence

if __name__ == "__main__":

    model = load_model(str(MODEL_PATH), compile=False)

    X_dataset = np.load(r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\fin_dataset\X.npy")
    X_full_sample = X_dataset[0]

    pred_id, pred_text, confidence = predict_class(
        model,
        X_full_sample,
        language="kazakh"
    )

    print(f"Predicted ID: {pred_id}")
    print(f"Prediction: {pred_text}")
    print(f"Confidence: {confidence:.2%}")