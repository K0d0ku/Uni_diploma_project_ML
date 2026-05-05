### CURRENT V 1.4
# video testing , opens a random dataset video then predicts, then compares it by id's that it gets from the file
# and compares with prediction, for video ones to display the text it just gets the id and text from file and kazakh.json annotation
## 3 modes of testing
# 1 mode simple no video just open file give output
# 2 mode video and output but holistics dont render
# 3 mode video and output and holistics also render

# 1 mode
# TEST NO VIDEO PREVIEW NO LANDMARK RENDER
# usually used only for me to see the results fast
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
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\dl\data"
# )
#
# MODEL_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\dl\model"
# )
#
# ANNOTATIONS_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\annotations\kazakh.json"
# )
#
# SEQUENCE_LENGTH = 30
# FEATURE_SIZE = 1692
# NUM_TESTS = 10
#
# mean = np.load(DATA_PATH / "mean.npy")
# std = np.load(DATA_PATH / "std.npy")
#
# label_map = np.load(DATA_PATH / "label_map.npy", allow_pickle=True).item()
# inverse_map = np.load(DATA_PATH / "inverse_map.npy", allow_pickle=True).item()
#
# with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
#     annotations = json.load(f)
#
# annotations = {int(k): v for k, v in annotations.items()}
#
# mp_holistic = mp.solutions.holistic
#
# def normalize_sequence(seq):
#     # SAME as training
#     frame_mean = np.mean(seq, axis=1, keepdims=True)
#     return seq - frame_mean
#
# def compute_mask(sequence):
#     return (np.abs(sequence).sum(axis=1) > 1e-6).astype(np.float32)
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
#         if results.face_landmarks else np.zeros(478 * 3)  # FIXED
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
#     return keypoints.astype(np.float32)
#
# def video_to_sequence(video_path):
#
#     cap = cv2.VideoCapture(str(video_path))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     if total_frames <= 0:
#         cap.release()
#         return None, None
#
#     indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
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
#         for idx in indices:
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
#     sequence = np.array(sequence, dtype=np.float32)
#
#     # 1. Frame normalization (CRITICAL)
#     sequence = normalize_sequence(sequence)
#
#     # 2. Mask
#     mask = compute_mask(sequence)
#     mask = mask[..., np.newaxis]
#
#     # 3. Global normalization
#     sequence = (sequence - mean) / std
#
#     # 4. Shape fix
#     sequence = sequence.reshape(1, 30, 1692)
#     mask = mask.reshape(1, 30, 1)
#
#     return sequence, mask
#
# def test_model(model_file):
#
#     print("Testing:", model_file.name)
#
#     model = tf.keras.models.load_model(model_file)
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
#
#         sequence, mask = video_to_sequence(video)
#
#         if sequence is None:
#             print("Skipped:", video.name)
#             continue
#
#         prediction = model.predict([sequence, mask], verbose=0)[0]
#
#         pred_mapped = int(np.argmax(prediction))
#         confidence = float(np.max(prediction))
#
#         # map back to original annotation ID
#         pred_id = inverse_map[pred_mapped]
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
# File: {video.name}
# True ID: {true_id}
# True: {true_text}
# Pred ID: {pred_id}
# Pred: {pred_text}
# Confidence: {confidence:.2%}
# Result: {result}
# """)
#
#     return correct
#
# best_score = test_model(MODEL_PATH / "sign_lstm_masked_best.keras")
# supreme_score = test_model(MODEL_PATH / "sign_lstm_masked_supreme_best.keras")
#
# print("FINAL")
# print(f"""
# Masked Best:     {best_score}/{NUM_TESTS}
# Masked Supreme:  {supreme_score}/{NUM_TESTS}
# """)






# 2 mode
# TEST VIDEO PREVIEW NO LANDMARK
# import random
# import cv2
# import numpy as np
# from pathlib import Path
# import tensorflow as tf
# import mediapipe as mp
# import json
#
# VIDEO_DATASET = Path(r"F:\KSLR-FluentSigners-50")
# DATA_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\dl\data"
# )
# MODEL_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\dl\model"
# )
# ANNOTATIONS_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\annotations\kazakh.json"
# )
#
# SEQUENCE_LENGTH = 30
# FEATURE_SIZE = 1692
# NUM_TESTS = 10
#
# mean = np.load(DATA_PATH / "mean.npy")
# std = np.load(DATA_PATH / "std.npy")
#
# inverse_map = np.load(DATA_PATH / "inverse_map.npy", allow_pickle=True).item()
#
# with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
#     annotations = json.load(f)
#
# annotations = {int(k): v for k, v in annotations.items()}
# mp_holistic = mp.solutions.holistic
#
# def normalize_sequence(seq):
#     frame_mean = np.mean(seq, axis=1, keepdims=True)
#     return seq - frame_mean
#
# def compute_mask(sequence):
#     return (np.abs(sequence).sum(axis=1) > 1e-6).astype(np.float32)
#
# def extract_keypoints(results):
#     pose = (
#         np.array([[res.x, res.y, res.z, res.visibility]
#                   for res in results.pose_landmarks.landmark]).flatten()
#         if results.pose_landmarks else np.zeros(33 * 4)
#     )
#     face = (
#         np.array([[res.x, res.y, res.z]
#                   for res in results.face_landmarks.landmark]).flatten()
#         if results.face_landmarks else np.zeros(478 * 3)
#     )
#     lh = (
#         np.array([[res.x, res.y, res.z]
#                   for res in results.left_hand_landmarks.landmark]).flatten()
#         if results.left_hand_landmarks else np.zeros(21 * 3)
#     )
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
#     return keypoints.astype(np.float32)
#
# def video_to_sequence(video_path):
#     cap = cv2.VideoCapture(str(video_path))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     if total_frames <= 0:
#         cap.release()
#         return None, None
#
#     indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
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
#         for idx in indices:
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
#     sequence = np.array(sequence, dtype=np.float32)
#
#     sequence = normalize_sequence(sequence)
#     mask = compute_mask(sequence)[..., np.newaxis]
#     sequence = (sequence - mean) / std
#
#     sequence = sequence.reshape(1, 30, 1692)
#     mask = mask.reshape(1, 30, 1)
#
#     return sequence, mask
#
# from PIL import Image, ImageDraw, ImageFont
#
# FONT_PATH = "C:/Windows/Fonts/times.ttf"
# font = ImageFont.truetype(FONT_PATH, 18)
#
# def preview_video(video_path, true_id, pred_id, true_text, pred_text, confidence, result):
#     cap = cv2.VideoCapture(str(video_path))
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # convert to PIL
#         frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         draw = ImageDraw.Draw(frame_pil)
#
#         # background bar
#         draw.rectangle([(0, 0), (frame.shape[1], 110)], fill=(0, 0, 0))
#
#         color = (0, 255, 0) if result == "PASS" else (255, 0, 0)
#
#         draw.text((10, 10), f"True [{true_id}]: {true_text}", font=font, fill=(0, 255, 0))
#         draw.text((10, 45), f"Pred [{pred_id}]: {pred_text}", font=font, fill=(255, 255, 0))
#         draw.text((10, 80), f"{result} ({confidence:.2%})", font=font, fill=color)
#
#         frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
#
#         cv2.imshow("Prediction", frame)
#
#         if cv2.waitKey(20) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# def test_model(model_file):
#     print("Testing:", model_file.name)
#
#     model = tf.keras.models.load_model(model_file)
#
#     folders = [f for f in VIDEO_DATASET.iterdir() if f.is_dir()]
#     chosen_folders = random.sample(folders, NUM_TESTS)
#
#     correct = 0
#
#     for i, folder in enumerate(chosen_folders):
#         videos = list(folder.glob("*.mp4"))
#         if not videos:
#             continue
#
#         video = random.choice(videos)
#
#         sequence, mask = video_to_sequence(video)
#
#         if sequence is None:
#             print("Skipped:", video.name)
#             continue
#
#         prediction = model.predict([sequence, mask], verbose=0)[0]
#
#         pred_mapped = int(np.argmax(prediction))
#         confidence = float(np.max(prediction))
#
#         pred_id = inverse_map[pred_mapped]
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
# File: {video.name}
# True ID: {true_id}
# True: {true_text}
# Pred ID: {pred_id}
# Pred: {pred_text}
# Confidence: {confidence:.2%}
# Result: {result}
# """)
#
#         preview_video(video, true_id, pred_id, true_text, pred_text, confidence, result)
#
#     return correct
#
# best_score = test_model(MODEL_PATH / "sign_lstm_masked_best.keras")
# supreme_score = test_model(MODEL_PATH / "sign_lstm_masked_supreme_best.keras")
#
# print("FINAL")
#
# print(f"""
# Masked Best:     {best_score}/{NUM_TESTS}
# Masked Supreme:  {supreme_score}/{NUM_TESTS}
# """)




# 3 mode
# TEST VIDEO PREVIEW AND LANDMARK RENDER
import random
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
import mediapipe as mp
import json

VIDEO_DATASET = Path(r"F:\KSLR-FluentSigners-50")
DATA_PATH = Path(
    r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\dl\data"
)
MODEL_PATH = Path(
    r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\dl\model"
)
ANNOTATIONS_PATH = Path(
    r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\annotations\kazakh.json"
)

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 1692
NUM_TESTS = 10

mean = np.load(DATA_PATH / "mean.npy")
std = np.load(DATA_PATH / "std.npy")

inverse_map = np.load(DATA_PATH / "inverse_map.npy", allow_pickle=True).item()

with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
    annotations = json.load(f)

annotations = {int(k): v for k, v in annotations.items()}

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def normalize_sequence(seq):
    frame_mean = np.mean(seq, axis=1, keepdims=True)
    return seq - frame_mean

def compute_mask(sequence):
    return (np.abs(sequence).sum(axis=1) > 1e-6).astype(np.float32)


def extract_keypoints(results):

    pose = (
        np.array([[res.x, res.y, res.z, res.visibility]
                  for res in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 4)
    )

    face = (
        np.array([[res.x, res.y, res.z]
                  for res in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks else np.zeros(478 * 3)
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

    return keypoints.astype(np.float32)


def video_to_sequence(video_path):

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None, None

    indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)

    sequence = []

    with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True
    ) as holistic:

        for idx in indices:

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

    sequence = np.array(sequence, dtype=np.float32)

    # match training
    sequence = normalize_sequence(sequence)
    mask = compute_mask(sequence)[..., np.newaxis]
    sequence = (sequence - mean) / std

    sequence = sequence.reshape(1, 30, 1692)
    mask = mask.reshape(1, 30, 1)

    return sequence, mask

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "C:/Windows/Fonts/times.ttf"
font = ImageFont.truetype(FONT_PATH, 18)


def preview_video(video_path, true_id, pred_id, true_text, pred_text, confidence, result):

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

            # draw landmarks (OpenCV still fine)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # convert to PIL
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)

            # background bar
            draw.rectangle([(0, 0), (frame.shape[1], 110)], fill=(0, 0, 0))

            color = (0, 255, 0) if result == "PASS" else (255, 0, 0)

            # Unicode-safe text
            draw.text((10, 10), f"True [{true_id}]: {true_text}", font=font, fill=(0, 255, 0))
            draw.text((10, 45), f"Pred [{pred_id}]: {pred_text}", font=font, fill=(255, 255, 0))
            draw.text((10, 80), f"{result} ({confidence:.2%})", font=font, fill=color)

            # back to OpenCV
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            cv2.imshow("Prediction", frame)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def test_model(model_file):

    print("Testing:", model_file.name)

    model = tf.keras.models.load_model(model_file)

    folders = [f for f in VIDEO_DATASET.iterdir() if f.is_dir()]
    chosen_folders = random.sample(folders, NUM_TESTS)

    correct = 0

    for i, folder in enumerate(chosen_folders):

        videos = list(folder.glob("*.mp4"))
        if not videos:
            continue

        video = random.choice(videos)

        sequence, mask = video_to_sequence(video)

        if sequence is None:
            print("Skipped:", video.name)
            continue

        prediction = model.predict([sequence, mask], verbose=0)[0]

        pred_mapped = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        pred_id = inverse_map[pred_mapped]
        true_id = int(folder.name)

        true_text = annotations.get(true_id, "UNKNOWN")
        pred_text = annotations.get(pred_id, "UNKNOWN")

        result = "PASS" if pred_id == true_id else "FAIL"
        if result == "PASS":
            correct += 1

        print(f"""
VIDEO {i+1}/{NUM_TESTS}
File: {video.name}
True ID: {true_id}
True: {true_text}
Pred ID: {pred_id}
Pred: {pred_text}
Confidence: {confidence:.2%}
Result: {result}
""")

        preview_video(video, true_id, pred_id, true_text, pred_text, confidence, result)

    return correct

best_score = test_model(MODEL_PATH / "sign_lstm_masked_best.keras")
supreme_score = test_model(MODEL_PATH / "sign_lstm_masked_supreme_best.keras")

print("FINAL")

print(f"""
Masked Best:     {best_score}/{NUM_TESTS}
Masked Supreme:  {supreme_score}/{NUM_TESTS}
""")