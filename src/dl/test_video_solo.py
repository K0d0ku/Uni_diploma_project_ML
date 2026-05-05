## CURRENT V 1.4
# needs to be updated a bit, the video preview is too fast

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
    r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\dl\data"
)
MODEL_PATH = Path(
    r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\src\dl\model\sign_lstm_masked_supreme_best.keras"
)
ANNOTATIONS_PATH = Path(
    r"C:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\annotations\kazakh.json"
)

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 1692
NUM_TESTS = 10

TARGET_WIDTH = 960
TARGET_HEIGHT = 720

FONT_PATH = "C:/Windows/Fonts/times.ttf"

mode = int(input("Mode 1 Output"
                 "\nMode 2 Video preview"
                 "\nMode 3 Video preview + Holistics"
                 "\nSelect mode (1-3): "))

SHOW_VIDEO = mode in [2, 3]
SHOW_LANDMARKS = mode == 3

mean = np.load(DATA_PATH / "mean.npy")
std = np.load(DATA_PATH / "std.npy")

inverse_map = np.load(DATA_PATH / "inverse_map.npy", allow_pickle=True).item()

with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
    annotations = json.load(f)

annotations = {int(k): v for k, v in annotations.items()}

font = ImageFont.truetype(FONT_PATH, 18)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def resize_with_padding(frame, target_w, target_h):
    h, w = frame.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas

def normalize_sequence(seq):
    frame_mean = np.mean(seq, axis=1, keepdims=True)
    return seq - frame_mean

def compute_mask(sequence):
    return (np.abs(sequence).sum(axis=1) > 1e-6).astype(np.float32)

def extract_keypoints(results):
    pose = (
        np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 4)
    )
    face = (
        np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks else np.zeros(478 * 3)
    )
    lh = (
        np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )
    rh = (
        np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )

    keypoints = np.concatenate([pose, face, lh, rh])

    if keypoints.shape[0] != FEATURE_SIZE:
        return np.zeros(FEATURE_SIZE)

    return keypoints.astype(np.float32)


def process_video(video_path):

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None, None, None

    indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)

    sequence = []
    preview_frames = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=True
    ) as holistic:

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                sequence.append(np.zeros(FEATURE_SIZE))
                preview_frames.append(None)
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            if SHOW_VIDEO:
                frame_copy = frame.copy()

                if SHOW_LANDMARKS:
                    mp_drawing.draw_landmarks(frame_copy, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame_copy, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
                    mp_drawing.draw_landmarks(frame_copy, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame_copy, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                preview_frames.append(frame_copy)

    cap.release()

    sequence = np.array(sequence, dtype=np.float32)
    sequence = normalize_sequence(sequence)

    mask = compute_mask(sequence)[..., np.newaxis]
    sequence = (sequence - mean) / std

    sequence = sequence.reshape(1, 30, 1692)
    mask = mask.reshape(1, 30, 1)

    return sequence, mask, preview_frames


def show_preview(frames, true_id, pred_id, true_text, pred_text, confidence, result):

    cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Prediction", TARGET_WIDTH, TARGET_HEIGHT)

    for frame in frames:
        if frame is None:
            continue

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        draw.rectangle([(0, 0), (frame.shape[1], 110)], fill=(0, 0, 0))

        color = (0, 255, 0) if result == "PASS" else (255, 0, 0)

        draw.text((10, 10), f"True [{true_id}]: {true_text}", font=font, fill=(0, 255, 0))
        draw.text((10, 45), f"Pred [{pred_id}]: {pred_text}", font=font, fill=(255, 255, 0))
        draw.text((10, 80), f"{result} ({confidence:.2%})", font=font, fill=color)

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        frame = resize_with_padding(frame, TARGET_WIDTH, TARGET_HEIGHT)

        cv2.imshow("Prediction", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def test_model():

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    folders = [f for f in VIDEO_DATASET.iterdir() if f.is_dir()]
    chosen_folders = random.sample(folders, NUM_TESTS)

    correct = 0

    for i, folder in enumerate(chosen_folders):

        videos = list(folder.glob("*.mp4"))
        if not videos:
            continue

        video = random.choice(videos)

        sequence, mask, frames = process_video(video)

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

        if SHOW_VIDEO:
            show_preview(frames, true_id, pred_id, true_text, pred_text, confidence, result)

    print("\nFINAL")
    print(f"Score: {correct}/{NUM_TESTS}")

if __name__ == "__main__":
    test_model()