## PREVIOUS V 1.3 Video tests
# video test for FULL HOLISTICS MODEL

# WORKIN 1 no holistics and not visible text
# import random
# import cv2
# import numpy as np
# from pathlib import Path
# import tensorflow as tf
# import mediapipe as mp
# import json
#
# VIDEO_DATASET = Path(r"F:\KSLR-FluentSigners-50")
#
# DATA_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed"
# )
#
# MODEL_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models"
# )
#
# ANNOTATIONS_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\annotations\kazakh.json"
# )
#
# SEQUENCE_LENGTH = 30
# FEATURE_SIZE = 1692
# NUM_TESTS = 10
#
# mean = np.load(DATA_PATH / "mean.npy").reshape(-1)
# std = np.load(DATA_PATH / "std.npy").reshape(-1)
#
# std[std == 0] = 1e-6
#
# with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
#     annotations = json.load(f)
#
# annotations = {int(k): v for k, v in annotations.items()}
#
# mp_holistic = mp.solutions.holistic
#
#
# def extract_keypoints(results):
#
#     pose = (
#         np.array([[res.x, res.y, res.z, res.visibility]
#                   for res in results.pose_landmarks.landmark]).flatten()
#         if results.pose_landmarks else np.zeros(33 * 4)
#     )
#
#     face = (
#         np.array([[res.x, res.y, res.z]
#                   for res in results.face_landmarks.landmark]).flatten()
#         if results.face_landmarks else np.zeros(468 * 3)
#     )
#
#     lh = (
#         np.array([[res.x, res.y, res.z]
#                   for res in results.left_hand_landmarks.landmark]).flatten()
#         if results.left_hand_landmarks else np.zeros(21 * 3)
#     )
#
#     rh = (
#         np.array([[res.x, res.y, res.z]
#                   for res in results.right_hand_landmarks.landmark]).flatten()
#         if results.right_hand_landmarks else np.zeros(21 * 3)
#     )
#
#     keypoints = np.concatenate([pose, face, lh, rh])
#
#     if keypoints.shape[0] != FEATURE_SIZE:
#         return np.zeros(FEATURE_SIZE)
#
#     return keypoints
#
# def video_to_sequence(video_path):
#
#     cap = cv2.VideoCapture(str(video_path))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     if total_frames <= 0:
#         cap.release()
#         return None
#
#     frame_indices = np.linspace(
#         0,
#         total_frames - 1,
#         SEQUENCE_LENGTH,
#         dtype=int
#     )
#
#     sequence = []
#
#     with mp_holistic.Holistic(
#             static_image_mode=False,
#             model_complexity=1,
#             smooth_landmarks=True,
#             enable_segmentation=False,
#             refine_face_landmarks=True
#     ) as holistic:
#
#         for idx in frame_indices:
#
#             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#             ret, frame = cap.read()
#
#             if not ret:
#                 sequence.append(np.zeros(FEATURE_SIZE))
#                 continue
#
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = holistic.process(image)
#
#             keypoints = extract_keypoints(results)
#             sequence.append(keypoints)
#
#     cap.release()
#
#     if len(sequence) != SEQUENCE_LENGTH:
#         return None
#
#     cleaned_sequence = []
#     for frame in sequence:
#         if frame.shape[0] != FEATURE_SIZE:
#             cleaned_sequence.append(np.zeros(FEATURE_SIZE))
#         else:
#             cleaned_sequence.append(frame)
#
#     sequence = np.array(cleaned_sequence)  # (30, 1692)
#
#     sequence = (sequence - mean) / std
#
#     # Add batch dimension
#     sequence = np.expand_dims(sequence, axis=0)  # (1, 30, 1692)
#
#     return sequence
#
# def test_model(model_file):
#
#     print("\nTesting:", model_file.name)
#
#     model = tf.keras.models.load_model(model_file, compile=False)
#
#     folders = [f for f in VIDEO_DATASET.iterdir() if f.is_dir()]
#     chosen_folders = random.sample(folders, NUM_TESTS)
#
#     correct = 0
#
#     for i, folder in enumerate(chosen_folders):
#
#         videos = list(folder.glob("*.mp4"))
#         if not videos:
#             continue
#
#         video = random.choice(videos)
#         sequence = video_to_sequence(video)
#
#         if sequence is None:
#             print("Skipped corrupted video:", video.name)
#             continue
#
#         print("Final sequence shape:", sequence.shape)
#
#         prediction = model.predict(sequence, verbose=0)[0]
#         pred_id = int(np.argmax(prediction))
#         confidence = float(np.max(prediction))
#
#         true_id = int(folder.name)
#         true_text = annotations.get(true_id, "UNKNOWN")
#         pred_text = annotations.get(pred_id, "UNKNOWN")
#
#         result = "PASS" if pred_id == true_id else "FAIL"
#         if result == "PASS":
#             correct += 1
#
#         print(f"""
# VIDEO {i+1}/{NUM_TESTS}
#
# Opened: {video.name}
# True ID: {true_id}
# True: {true_text}
# Pred ID: {pred_id}
# Pred: {pred_text}
# Confidence: {confidence:.2%}
# RESULT: {result}
# """)
#
#         # Show preview
#         cap = cv2.VideoCapture(str(video))
#         ret, frame = cap.read()
#
#         if ret:
#             overlay = frame.copy()
#             cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
#             frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
#
#             cv2.putText(frame, f"True: {true_text}", (10, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
#             cv2.putText(frame, f"Pred: {pred_text}", (10, 80),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#
#             cv2.putText(frame, result, (10, 110),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                         (0, 255, 0) if result == "PASS" else (0, 0, 255), 2)
#
#             cv2.imshow("Prediction", frame)
#             cv2.waitKey(1500)
#
#         cap.release()
#
#     cv2.destroyAllWindows()
#     return correct
#
#
#
# best_score = test_model(MODEL_PATH / "sign_lstm_best.h5") # previous gen model at peak performance before overfit
# # supreme_score = test_model(MODEL_PATH / "sign_lstm_supreme.keras") # final model tent to overfit
# supreme_score = test_model(MODEL_PATH / "sign_lstm_supreme_best.keras") # same model before overfit at its peak
#
# print("\nFINAL COMPARISON")
#
# print(f"""
# Model                  Score
# sign_lstm_best        {best_score}/{NUM_TESTS}
# sign_lstm_supreme     {supreme_score}/{NUM_TESTS}
# """)


# WORKIN 2 with holistics and full on text
import random
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
import mediapipe as mp
import json
from PIL import Image, ImageDraw, ImageFont

VIDEO_DATASET = Path(r"F:\KSLR-FluentSigners-50")

DATA_PATH = Path(
    # v 1.3
    r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed"

    # V 1.4
    # r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models"
)

MODEL_PATH = Path(
    r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models"
)

ANNOTATIONS_PATH = Path(
    r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\annotations\kazakh.json"
)

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 1692
NUM_TESTS = 10

FONT_PATH = "C:/Windows/Fonts/arial.ttf"
font = ImageFont.truetype(FONT_PATH, 28)

mean = np.load(DATA_PATH / "mean.npy").reshape(-1)
std = np.load(DATA_PATH / "std.npy").reshape(-1)
std[std == 0] = 1e-6

with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
    annotations = json.load(f)

annotations = {int(k): v for k, v in annotations.items()}

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):

    pose = (
        np.array([[res.x, res.y, res.z, res.visibility]
                  for res in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 4)
    )

    face = (
        np.array([[res.x, res.y, res.z]
                  for res in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks else np.zeros(468 * 3)
    )

    lh = (
        np.array([[res.x, res.y, res.z]
                  for res in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )

    rh = (
        np.array([[res.x, res.y, res.z]
                  for res in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )

    keypoints = np.concatenate([pose, face, lh, rh])

    if keypoints.shape[0] != FEATURE_SIZE:
        return np.zeros(FEATURE_SIZE)

    return keypoints

def video_to_sequence(video_path):

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH,
        dtype=int
    )

    sequence = []

    with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True
    ) as holistic:

        for idx in frame_indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                sequence.append(np.zeros(FEATURE_SIZE))
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints = extract_keypoints(results)

            sequence.append(keypoints)

    cap.release()

    if len(sequence) != SEQUENCE_LENGTH:
        return None

    sequence = np.array(sequence)
    sequence = (sequence - mean) / std
    sequence = np.expand_dims(sequence, axis=0)

    return sequence

def preview_video(video_path, true_text, pred_text, confidence, result):

    cap = cv2.VideoCapture(str(video_path))

    with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
            )

            mp_drawing.draw_landmarks(
                frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION
            )

            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )

            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)

            draw.rectangle([(0, 0), (frame.shape[1], 110)], fill=(0, 0, 0))

            draw.text((10, 10), f"True: {true_text}", font=font, fill=(0, 255, 0))
            draw.text((10, 45), f"Pred: {pred_text}", font=font, fill=(255, 255, 0))

            draw.text(
                (10, 80),
                f"{result} ({confidence:.2%})",
                font=font,
                fill=(0, 255, 0) if result == "PASS" else (255, 0, 0)
            )

            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            cv2.imshow("Prediction", frame)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def test_model(model_file):

    print("\nTesting:", model_file.name)

    model = tf.keras.models.load_model(model_file, compile=False)

    folders = [f for f in VIDEO_DATASET.iterdir() if f.is_dir()]
    chosen_folders = random.sample(folders, NUM_TESTS)

    correct = 0

    for i, folder in enumerate(chosen_folders):

        videos = list(folder.glob("*.mp4"))
        if not videos:
            continue

        video = random.choice(videos)
        sequence = video_to_sequence(video)

        if sequence is None:
            print("Skipped corrupted video:", video.name)
            continue

        prediction = model.predict(sequence, verbose=0)[0]
        pred_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        true_id = int(folder.name)
        true_text = annotations.get(true_id, "UNKNOWN")
        pred_text = annotations.get(pred_id, "UNKNOWN")

        result = "PASS" if pred_id == true_id else "FAIL"
        if result == "PASS":
            correct += 1

        print(f"""
VIDEO {i+1}/{NUM_TESTS}

Opened: {video.name}
True ID: {true_id}
True: {true_text}
Pred ID: {pred_id}
Pred: {pred_text}
Confidence: {confidence:.2%}
RESULT: {result}
""")

        preview_video(video, true_text, pred_text, confidence, result)

    return correct

#  V 1.3
best_score = test_model(MODEL_PATH / "sign_lstm_best.h5")
supreme_score = test_model(MODEL_PATH / "sign_lstm_supreme_best.keras")

#  V 1.4
# best_score = test_model(MODEL_PATH / "sign_lstm_best_1.4.h5")
# supreme_score = test_model(MODEL_PATH / "sign_lstm_model_1.4.h5")

print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

print(f"""
Model                  Score

sign_lstm_best        {best_score}/{NUM_TESTS}
sign_lstm_model     {supreme_score}/{NUM_TESTS}
""")




#### V 1.4 ?
# this is a video test for another one of unsuccessful branches of model training
# import random
# import cv2
# import numpy as np
# from pathlib import Path
# import tensorflow as tf
# import mediapipe as mp
# import json
# from PIL import Image, ImageDraw, ImageFont
#
# VIDEO_DATASET = Path(r"F:\KSLR-FluentSigners-50")
#
# DATA_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\fl"
# )
#
# MODEL_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\fl"
# )
#
# ANNOTATIONS_PATH = Path(
#     r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\annotations\kazakh.json"
# )
#
# SEQUENCE_LENGTH = 30
# FEATURE_SIZE = 1692
# NUM_TESTS = 10
#
# FONT_PATH = "C:/Windows/Fonts/arial.ttf"
# font = ImageFont.truetype(FONT_PATH, 28)
#
#
# mean = np.load(DATA_PATH / "mean.npy")
# std = np.load(DATA_PATH / "std.npy")
# std[std == 0] = 1e-6
#
# inverse_map = np.load(DATA_PATH / "inverse_map.npy", allow_pickle=True).item()
#
# with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
#     annotations = json.load(f)
#
# annotations = {int(k): v for k, v in annotations.items()}
#
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils
#
# def extract_keypoints(results):
#
#     pose = (
#         np.array([[res.x, res.y, res.z, res.visibility]
#                   for res in results.pose_landmarks.landmark]).flatten()
#         if results.pose_landmarks else np.zeros(33 * 4)
#     )
#
#     face = (
#         np.array([[res.x, res.y, res.z]
#                   for res in results.face_landmarks.landmark]).flatten()
#         if results.face_landmarks else np.zeros(468 * 3)
#     )
#
#     lh = (
#         np.array([[res.x, res.y, res.z]
#                   for res in results.left_hand_landmarks.landmark]).flatten()
#         if results.left_hand_landmarks else np.zeros(21 * 3)
#     )
#
#     rh = (
#         np.array([[res.x, res.y, res.z]
#                   for res in results.right_hand_landmarks.landmark]).flatten()
#         if results.right_hand_landmarks else np.zeros(21 * 3)
#     )
#
#     keypoints = np.concatenate([pose, face, lh, rh])
#
#     if keypoints.shape[0] != FEATURE_SIZE:
#         return np.zeros(FEATURE_SIZE)
#
#     return keypoints
#
# def video_to_sequence(video_path):
#
#     cap = cv2.VideoCapture(str(video_path))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     if total_frames <= 0:
#         cap.release()
#         return None
#
#     frame_indices = np.linspace(
#         0,
#         total_frames - 1,
#         SEQUENCE_LENGTH,
#         dtype=int
#     )
#
#     sequence = []
#
#     with mp_holistic.Holistic(
#             static_image_mode=False,
#             model_complexity=1,
#             smooth_landmarks=True,
#             enable_segmentation=False,
#             refine_face_landmarks=True
#     ) as holistic:
#
#         for idx in frame_indices:
#
#             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#             ret, frame = cap.read()
#
#             if not ret:
#                 sequence.append(np.zeros(FEATURE_SIZE))
#                 continue
#
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = holistic.process(image)
#             keypoints = extract_keypoints(results)
#
#             sequence.append(keypoints)
#
#     cap.release()
#
#     if len(sequence) != SEQUENCE_LENGTH:
#         return None
#
#     sequence = np.array(sequence)
#
#     # IMPORTANT: same normalization as training
#     sequence = (sequence - mean) / std
#     # sequence = np.expand_dims(sequence, axis=0)
#     # ensure shape is exactly (1, 30, 1692)
#     sequence = sequence.reshape(1, SEQUENCE_LENGTH, FEATURE_SIZE)
#
#     print("Sequence shape:", sequence.shape)
#
#     return sequence
#
# def preview_video(video_path, true_text, pred_text, confidence, result):
#
#     cap = cv2.VideoCapture(str(video_path))
#
#     with mp_holistic.Holistic(
#             static_image_mode=False,
#             model_complexity=1,
#             smooth_landmarks=True,
#             enable_segmentation=False,
#             refine_face_landmarks=True
#     ) as holistic:
#
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = holistic.process(image)
#
#             mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#             mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
#             mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#             mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#
#             frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             draw = ImageDraw.Draw(frame_pil)
#
#             draw.rectangle([(0, 0), (frame.shape[1], 110)], fill=(0, 0, 0))
#
#             draw.text((10, 10), f"True: {true_text}", font=font, fill=(0, 255, 0))
#             draw.text((10, 45), f"Pred: {pred_text}", font=font, fill=(255, 255, 0))
#
#             draw.text(
#                 (10, 80),
#                 f"{result} ({confidence:.2%})",
#                 font=font,
#                 fill=(0, 255, 0) if result == "PASS" else (255, 0, 0)
#             )
#
#             frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
#             cv2.imshow("Prediction", frame)
#
#             if cv2.waitKey(20) & 0xFF == ord('q'):
#                 break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# def test_model(model_file):
#
#     print("\nTesting:", model_file.name)
#
#     model = tf.keras.models.load_model(model_file, compile=False)
#
#     folders = [f for f in VIDEO_DATASET.iterdir() if f.is_dir()]
#     chosen_folders = random.sample(folders, NUM_TESTS)
#
#     correct = 0
#
#     for i, folder in enumerate(chosen_folders):
#
#         videos = list(folder.glob("*.mp4"))
#         if not videos:
#             continue
#
#         video = random.choice(videos)
#         sequence = video_to_sequence(video)
#
#         if sequence is None:
#             print("Skipped corrupted video:", video.name)
#             continue
#
#         prediction = model.predict(sequence, verbose=0)[0]
#
#         pred_index = int(np.argmax(prediction))
#         pred_id = inverse_map[pred_index]
#
#         confidence = float(np.max(prediction))
#
#         true_id = int(folder.name)
#
#         true_text = annotations.get(true_id, "UNKNOWN")
#         pred_text = annotations.get(pred_id, "UNKNOWN")
#
#         result = "PASS" if pred_id == true_id else "FAIL"
#         if result == "PASS":
#             correct += 1
#
#         print(f"""
# VIDEO {i+1}/{NUM_TESTS}
#
# Opened: {video.name}
# True ID: {true_id}
# True: {true_text}
# Pred ID: {pred_id}
# Pred: {pred_text}
# Confidence: {confidence:.2%}
# RESULT: {result}
# """)
#
#         preview_video(video, true_text, pred_text, confidence, result)
#
#     return correct
#
# score = test_model(MODEL_PATH / "sign_lstm_best.keras")
#
# print("\nFINAL RESULT")
# print(f"Score: {score}/{NUM_TESTS}")








######## WORKIN ? prob not i fkin forgot
### test on videos for the expressional model
# import random
# import cv2
# import numpy as np
# from pathlib import Path
# import tensorflow as tf
# import json
# from PIL import Image, ImageDraw, ImageFont
#
# from src.utils.mediapipe_expressional import (
#     create_holistic,
#     extract_landmarks,
#     EXPECTED_SIZE
# )
#
# VIDEO_DATASET = Path(r"F:\KSLR-FluentSigners-50")
#
# DATA_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\expressional"
# )
#
# MODEL_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\expressional"
# )
#
# ANNOTATIONS_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\annotations\kazakh.json"
# )
#
# SEQUENCE_LENGTH = 250
# FEATURE_SIZE = EXPECTED_SIZE
# NUM_TESTS = 10
#
# FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
#
# mean = np.load(DATA_PATH / "mean.npy").reshape(-1)
# std = np.load(DATA_PATH / "std.npy").reshape(-1)
# std[std == 0] = 1e-6
#
# with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
#     annotations = {int(k): v for k, v in json.load(f).items()}
#
# def video_to_sequence(video_path):
#
#     cap = cv2.VideoCapture(str(video_path))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     if total_frames <= 0:
#         cap.release()
#         return None
#
#     indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
#
#     sequence = []
#     prev_landmarks = None
#
#     holistic = create_holistic()
#
#     for idx in indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ret, frame = cap.read()
#
#         if not ret:
#             sequence.append(np.zeros(FEATURE_SIZE))
#             continue
#
#         try:
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = holistic.process(image)
#
#             landmarks = extract_landmarks(results)
#
#             if landmarks.shape[0] != FEATURE_SIZE:
#                 sequence.append(np.zeros(FEATURE_SIZE))
#                 continue
#
#             # temporal recovery (same as training)
#             if prev_landmarks is not None:
#                 mask = landmarks == 0
#                 landmarks[mask] = prev_landmarks[mask]
#
#             prev_landmarks = landmarks
#             sequence.append(landmarks)
#
#         except Exception:
#             sequence.append(np.zeros(FEATURE_SIZE))
#
#     cap.release()
#
#     if len(sequence) != SEQUENCE_LENGTH:
#         return None
#
#     sequence = np.array(sequence, dtype=np.float32)
#     sequence = (sequence - mean) / std
#
#     return np.expand_dims(sequence, axis=0)
#
# def preview(video_path, true_text, pred_text, confidence, result):
#
#     cap = cv2.VideoCapture(str(video_path))
#     holistic = create_holistic()
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = holistic.process(image)
#
#         frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         draw = ImageDraw.Draw(frame_pil)
#
#         draw.rectangle([(0, 0), (frame.shape[1], 110)], fill=(0, 0, 0))
#
#         draw.text((10, 10), f"True: {true_text}", font=FONT, fill=(0, 255, 0))
#         draw.text((10, 45), f"Pred: {pred_text}", font=FONT, fill=(255, 255, 0))
#         draw.text(
#             (10, 80),
#             f"{result} ({confidence:.2%})",
#             font=FONT,
#             fill=(0, 255, 0) if result == "PASS" else (255, 0, 0)
#         )
#
#         frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
#         cv2.imshow("Prediction", frame)
#
#         if cv2.waitKey(20) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# def test_model(model_file):
#
#     print(f"\n=== Testing: {model_file.name} ===")
#
#     model = tf.keras.models.load_model(model_file, compile=False)
#
#     folders = [f for f in VIDEO_DATASET.iterdir() if f.is_dir()]
#     selected = random.sample(folders, NUM_TESTS)
#
#     correct = 0
#
#     for i, folder in enumerate(selected, 1):
#
#         videos = list(folder.glob("*.mp4"))
#         if not videos:
#             continue
#
#         video = random.choice(videos)
#         sequence = video_to_sequence(video)
#
#         if sequence is None:
#             print(f"[{i}] Skipped: {video.name}")
#             continue
#
#         prediction = model.predict(sequence, verbose=0)[0]
#
#         pred_id = int(np.argmax(prediction))
#         confidence = float(np.max(prediction))
#
#         true_id = int(folder.name)
#
#         true_text = annotations.get(true_id, "UNKNOWN")
#         pred_text = annotations.get(pred_id, "UNKNOWN")
#
#         is_correct = pred_id == true_id
#         correct += int(is_correct)
#
#         result = "PASS" if is_correct else "FAIL"
#
#         print(f"[{i}/{NUM_TESTS}] {video.name}")
#         print(f"  True: {true_text}")
#         print(f"  Pred: {pred_text} ({confidence:.2%}) → {result}\n")
#
#         preview(video, true_text, pred_text, confidence, result)
#
#     return correct
#
# best_score = test_model(MODEL_PATH / "sign_lstm_expressional_best.keras")
# final_score = test_model(MODEL_PATH / "sign_lstm_expressional_final.keras")
#
# print("\n FINAL COMPARISON")
# print(f"""
# Best Model     : {best_score}/{NUM_TESTS}
# Final Model    : {final_score}/{NUM_TESTS}
# """)





# import random
# import cv2
# import numpy as np
# from pathlib import Path
# import tensorflow as tf
# import json
#
# from src.utils.mediapipe_expressional import (
#     create_holistic,
#     extract_landmarks,
#     EXPECTED_SIZE
# )
#
# VIDEO_DATASET = Path(r"F:\KSLR-FluentSigners-50")
#
# MODEL_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\expressional\sign_lstm_expressional_fixed.keras"
# )
#
# ANNOTATIONS_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\annotations\kazakh.json"
# )
#
# SEQUENCE_LENGTH = 250
# FEATURE_SIZE = EXPECTED_SIZE
# NUM_TESTS = 10
#
# with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
#     annotations = {int(k): v for k, v in json.load(f).items()}
#
# def apply_motion_normalization(X):
#     X = X.astype(np.float32)
#     X[1:] = X[1:] - X[:-1]
#     return X
#
# def video_to_sequence(video_path):
#
#     cap = cv2.VideoCapture(str(video_path))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     if total_frames <= 0:
#         return None
#
#     indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
#
#     sequence = []
#     prev_landmarks = None
#
#     holistic = create_holistic()
#
#     for idx in indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ret, frame = cap.read()
#
#         if not ret:
#             sequence.append(np.zeros(FEATURE_SIZE))
#             continue
#
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = holistic.process(image)
#
#         landmarks = extract_landmarks(results)
#
#         if landmarks.shape[0] != FEATURE_SIZE:
#             sequence.append(np.zeros(FEATURE_SIZE))
#             continue
#
#         # temporal recovery
#         if prev_landmarks is not None:
#             mask = landmarks == 0
#             landmarks[mask] = prev_landmarks[mask]
#
#         prev_landmarks = landmarks
#         sequence.append(landmarks)
#
#     cap.release()
#
#     if len(sequence) != SEQUENCE_LENGTH:
#         return None
#
#     sequence = np.array(sequence, dtype=np.float32)
#
#     sequence = apply_motion_normalization(sequence)
#
#     return np.expand_dims(sequence, axis=0)
#
# def test_model(model_path):
#
#     print(f"\n=== Testing {model_path.name} ===")
#
#     model = tf.keras.models.load_model(model_path, compile=False)
#
#     folders = [f for f in VIDEO_DATASET.iterdir() if f.is_dir()]
#     selected = random.sample(folders, NUM_TESTS)
#
#     correct = 0
#
#     for i, folder in enumerate(selected, 1):
#
#         videos = list(folder.glob("*.mp4"))
#         if not videos:
#             continue
#
#         video = random.choice(videos)
#         sequence = video_to_sequence(video)
#
#         if sequence is None:
#             print(f"[{i}] Skipped")
#             continue
#
#         prediction = model.predict(sequence, verbose=0)[0]
#
#         pred_id = int(np.argmax(prediction))
#         confidence = float(np.max(prediction))
#
#         true_id = int(folder.name)
#
#         is_correct = pred_id == true_id
#         correct += int(is_correct)
#
#         print(f"[{i}/{NUM_TESTS}] {video.name}")
#         print(f"  True: {annotations.get(true_id)}")
#         print(f"  Pred: {annotations.get(pred_id)} ({confidence:.2%})")
#         print(f"  {'PASS' if is_correct else 'FAIL'}\n")
#
#     return correct
#
# score = test_model(MODEL_PATH)
# print(f"\nAccuracy: {score}/{NUM_TESTS}")