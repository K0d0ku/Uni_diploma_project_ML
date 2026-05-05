### CURRENT V 1.4

# MODE 1 simple live prediction
# # TEST CAMERA SIMPLE NO ID ENTERING
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
from pathlib import Path
import json

from PIL import Image, ImageDraw, ImageFont

SHOW_LANDMARKS = True

SEQ_LEN = 30
FEATURE_SIZE = 1692
PRED_INTERVAL = 5  # predict every N frames (stability control)

DATA_PATH = Path(
    r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\dl\data"
)
MODEL_PATH = Path(
    r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\dl\model\sign_lstm_masked_supreme_best.keras"
)
ANNOTATIONS_PATH = Path(
    r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\annotations\kazakh.json"
)

mean = np.load(DATA_PATH / "mean.npy")
std = np.load(DATA_PATH / "std.npy")

inverse_map = np.load(DATA_PATH / "inverse_map.npy", allow_pickle=True).item()

with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
    annotations = json.load(f)
annotations = {int(k): v for k, v in annotations.items()}

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded.")

FONT_PATH = r"C:\Windows\Fonts\arial.ttf"
font = ImageFont.truetype(FONT_PATH, 32)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(results):

    pose = (
        np.array([[r.x, r.y, r.z, r.visibility]
                  for r in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 4)
    )

    face = (
        np.array([[r.x, r.y, r.z]
                  for r in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks else np.zeros(478 * 3)
    )

    lh = (
        np.array([[r.x, r.y, r.z]
                  for r in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )

    rh = (
        np.array([[r.x, r.y, r.z]
                  for r in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )

    return np.concatenate([pose, face, lh, rh]).astype(np.float32)


def normalize_sequence(seq):
    frame_mean = np.mean(seq, axis=1, keepdims=True)
    return seq - frame_mean


def compute_mask(seq):
    return (np.abs(seq).sum(axis=1) > 1e-6).astype(np.float32)


cap = cv2.VideoCapture(0)

buffer = deque(maxlen=SEQ_LEN)
frame_count = 0

last_prediction = "..."
last_conf = 0.0

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

        keypoints = extract_keypoints(results)
        buffer.append(keypoints)

        if SHOW_LANDMARKS:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        frame_count += 1

        if len(buffer) == SEQ_LEN and frame_count % PRED_INTERVAL == 0:

            seq = np.array(buffer, dtype=np.float32)

            seq = normalize_sequence(seq)
            mask = compute_mask(seq)[..., np.newaxis]
            seq = (seq - mean) / std

            seq_input = seq.reshape(1, SEQ_LEN, FEATURE_SIZE)
            mask_input = mask.reshape(1, SEQ_LEN, 1)

            pred = model.predict([seq_input, mask_input], verbose=0)[0]

            pred_idx = int(np.argmax(pred))
            conf = float(np.max(pred))

            pred_id = inverse_map[pred_idx]
            last_prediction = annotations.get(pred_id, "UNKNOWN")
            last_conf = conf

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)

        # Convert OpenCV → PIL
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        text = f"{last_prediction} ({last_conf:.2%})"
        draw.text((10, 30), text, font=font, fill=(0, 255, 0))
        # Convert back → OpenCV
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow("Live Sign Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()








## MODE 2 ON DEVELOPMENT
# TODO TEST ON CAM COMPLEX ID ENTERING, there are timing problems
# import cv2
# import numpy as np
# import tensorflow as tf
# import mediapipe as mp
# from collections import deque
# from pathlib import Path
# import json
#
# from PIL import Image, ImageDraw, ImageFont
#
# SHOW_LANDMARKS = True
#
# SEQ_LEN = 30
# FEATURE_SIZE = 1692
# PRED_INTERVAL = 5
#
# DATA_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\dl\data"
# )
# MODEL_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\dl\model\sign_lstm_masked_supreme_best.keras"
# )
# ANNOTATIONS_PATH = Path(
#     r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\annotations\kazakh.json"
# )
#
# mean = np.load(DATA_PATH / "mean.npy")
# std = np.load(DATA_PATH / "std.npy")
#
# inverse_map = np.load(DATA_PATH / "inverse_map.npy", allow_pickle=True).item()
#
# with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
#     annotations = json.load(f)
# annotations = {int(k): v for k, v in annotations.items()}
#
# model = tf.keras.models.load_model(MODEL_PATH, compile=False)
# print("Model loaded.")
#
# FONT_PATH = r"C:\Windows\Fonts\arial.ttf"
# font = ImageFont.truetype(FONT_PATH, 28)
#
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils
#
# def extract_keypoints(results):
#
#     pose = (
#         np.array([[r.x, r.y, r.z, r.visibility]
#                   for r in results.pose_landmarks.landmark]).flatten()
#         if results.pose_landmarks else np.zeros(33 * 4)
#     )
#
#     face = (
#         np.array([[r.x, r.y, r.z]
#                   for r in results.face_landmarks.landmark]).flatten()
#         if results.face_landmarks else np.zeros(478 * 3)
#     )
#
#     lh = (
#         np.array([[r.x, r.y, r.z]
#                   for r in results.left_hand_landmarks.landmark]).flatten()
#         if results.left_hand_landmarks else np.zeros(21 * 3)
#     )
#
#     rh = (
#         np.array([[r.x, r.y, r.z]
#                   for r in results.right_hand_landmarks.landmark]).flatten()
#         if results.right_hand_landmarks else np.zeros(21 * 3)
#     )
#
#     return np.concatenate([pose, face, lh, rh]).astype(np.float32)
#
#
# def normalize_sequence(seq):
#     frame_mean = np.mean(seq, axis=1, keepdims=True)
#     return seq - frame_mean
#
#
# def compute_mask(seq):
#     return (np.abs(seq).sum(axis=1) > 1e-6).astype(np.float32)
#
# cap = cv2.VideoCapture(0)
#
# buffer = deque(maxlen=SEQ_LEN)
# frame_count = 0
#
# last_prediction = "..."
# last_conf = 0.0
#
# expected_id = None
# eval_result = ""
# eval_color = (255, 255, 255)
#
# print("\nMANUAL EVALUATION MODE")
# print("Enter expected class ID in terminal.")
# print("Press SPACE to evaluate.")
# print("Press Q to quit.\n")
#
# # initial input
# while expected_id is None:
#     try:
#         user_input = input("Enter expected class ID: ").strip()
#         if user_input.lower() == 'q':
#             exit()
#         expected_id = int(user_input)
#         if expected_id not in annotations:
#             print("Invalid ID. Try again.")
#             expected_id = None
#     except:
#         continue
#
# with mp_holistic.Holistic(
#     static_image_mode=False,
#     model_complexity=1,
#     smooth_landmarks=True,
#     enable_segmentation=False,
#     refine_face_landmarks=True
# ) as holistic:
#
#     while cap.isOpened():
#
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = holistic.process(image)
#
#         keypoints = extract_keypoints(results)
#         buffer.append(keypoints)
#
#         if SHOW_LANDMARKS:
#             mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#             mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
#             mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#             mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#
#         frame_count += 1
#
#         if len(buffer) == SEQ_LEN and frame_count % PRED_INTERVAL == 0:
#
#             seq = np.array(buffer, dtype=np.float32)
#             seq = normalize_sequence(seq)
#             mask = compute_mask(seq)[..., np.newaxis]
#             seq = (seq - mean) / std
#
#             seq_input = seq.reshape(1, SEQ_LEN, FEATURE_SIZE)
#             mask_input = mask.reshape(1, SEQ_LEN, 1)
#
#             pred = model.predict([seq_input, mask_input], verbose=0)[0]
#
#             pred_idx = int(np.argmax(pred))
#             conf = float(np.max(pred))
#
#             pred_id = inverse_map[pred_idx]
#             last_prediction = annotations.get(pred_id, "UNKNOWN")
#             last_conf = conf
#
#
#         key = cv2.waitKey(1) & 0xFF
#
#         # ---- SPACE = evaluate ----
#         if key == 32 and len(buffer) == SEQ_LEN:
#
#             seq = np.array(buffer, dtype=np.float32)
#             seq = normalize_sequence(seq)
#             mask = compute_mask(seq)[..., np.newaxis]
#             seq = (seq - mean) / std
#
#             seq_input = seq.reshape(1, SEQ_LEN, FEATURE_SIZE)
#             mask_input = mask.reshape(1, SEQ_LEN, 1)
#
#             pred = model.predict([seq_input, mask_input], verbose=0)[0]
#
#             pred_idx = int(np.argmax(pred))
#             conf = float(np.max(pred))
#             pred_id = inverse_map[pred_idx]
#
#             true_text = annotations.get(expected_id, "UNKNOWN")
#             pred_text = annotations.get(pred_id, "UNKNOWN")
#
#             if pred_id == expected_id:
#                 eval_result = f"PASS ({conf:.2%})"
#                 eval_color = (0, 255, 0)
#             else:
#                 eval_result = f"FAIL ({conf:.2%})"
#                 eval_color = (255, 0, 0)
#
#             print("\nRESULT")
#             print(f"TRUE: {expected_id} - {true_text}")
#             print(f"PRED: {pred_id} - {pred_text}")
#             print(f"CONF: {conf:.2%}")
#             print(eval_result)
#             print("================\n")
#
#             # ask for next test
#             expected_id = None
#             while expected_id is None:
#                 try:
#                     user_input = input("Enter next expected class ID (or q to quit): ").strip()
#                     if user_input.lower() == 'q':
#                         cap.release()
#                         cv2.destroyAllWindows()
#                         exit()
#                     expected_id = int(user_input)
#                     if expected_id not in annotations:
#                         print("Invalid ID. Try again.")
#                         expected_id = None
#                 except:
#                     continue
#
#         if key == ord('q'):
#             break
#
#         img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         draw = ImageDraw.Draw(img_pil)
#
#         draw.rectangle([(0, 0), (frame.shape[1], 120)], fill=(0, 0, 0))
#
#         draw.text((10, 10), f"Live: {last_prediction} ({last_conf:.2%})", font=font, fill=(255, 255, 0))
#         draw.text((10, 50), f"Expected ID: {expected_id}", font=font, fill=(0, 255, 255))
#         draw.text((10, 85), f"Result: {eval_result}", font=font, fill=eval_color)
#
#         frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
#
#         cv2.imshow("Camera Evaluation", frame)
#
# cap.release()
# cv2.destroyAllWindows()