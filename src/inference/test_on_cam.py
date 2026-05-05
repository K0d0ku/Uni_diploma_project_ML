## Previous V 1.3
## WAS ON DEVELOPMENT


import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
import mediapipe as mp
import json
from collections import deque

SHOW_LANDMARKS = True          # Toggle holistic drawing (workin ? prob not)
SEQUENCE_LENGTH = 30
MOTION_THRESHOLD = 0.01        # detect movement
GESTURE_STILL_FRAMES = 5       # frames of minimal motion to consider gesture complete
FEATURE_SIZE = 1692

DATA_PATH = Path(r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed")
MODEL_PATH = Path(r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\sign_lstm_supreme_best.keras")
ANNOTATIONS_PATH = Path(r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\annotations\kazakh.json")

mean = np.load(DATA_PATH / "mean.npy").reshape(-1)
std = np.load(DATA_PATH / "std.npy").reshape(-1)
std[std == 0] = 1e-6

with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
    annotations = json.load(f)
annotations = {int(k): v for k, v in annotations.items()}

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded.")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z]
                     for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(478*3)
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

results_log = []

cap = cv2.VideoCapture(0)
sequence = deque(maxlen=SEQUENCE_LENGTH)
prev_keypoints = None
still_counter = 0
expected_id = None

print("Type expected class ID in console before performing gesture. Press Q to quit.")

with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True, # False True
        enable_segmentation=False,
        refine_face_landmarks=True # False True
) as holistic:

    while cap.isOpened():
        # get expected id if none
        if expected_id is None:
            try:
                expected_id = input("Enter expected class ID: ").strip()
                if expected_id.lower() == 'q':
                    break
                expected_id = int(expected_id)
                if expected_id not in annotations:
                    print("Invalid ID, try again.")
                    expected_id = None
                    continue
                print("Perform gesture now...")
            except:
                continue

        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        keypoints = extract_keypoints(results)

        # detect motion
        motion_detected = False
        if prev_keypoints is not None:
            delta = np.mean(np.abs(keypoints - prev_keypoints))
            if delta > MOTION_THRESHOLD:
                motion_detected = True

        prev_keypoints = keypoints

        # append keypoints if moving
        if motion_detected:
            sequence.append(keypoints)
            still_counter = 0
        else:
            if sequence:
                still_counter += 1

        # if enough frames collected and gesture is still
        if len(sequence) == SEQUENCE_LENGTH and still_counter >= GESTURE_STILL_FRAMES:
            seq_array = np.array(sequence)
            seq_array = (seq_array - mean)/std
            seq_array = np.expand_dims(seq_array, axis=0)
            prediction = model.predict(seq_array, verbose=0)[0]
            pred_id = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            true_text = annotations.get(expected_id, "UNKNOWN")
            pred_text = annotations.get(pred_id, "UNKNOWN")
            result = "PASS" if pred_id == expected_id else "FAIL"

            results_log.append({
                "true_id": expected_id,
                "true_text": true_text,
                "pred_id": pred_id,
                "pred_text": pred_text,
                "confidence": confidence,
                "result": result
            })

            print("\nResult:")
            print("True:", expected_id, "-", true_text)
            print("Pred:", pred_id, "-", pred_text)
            print("Confidence:", f"{confidence:.2%}")
            print("RESULT:", result, "\n")

            sequence.clear()
            expected_id = None
            still_counter = 0

        # draw landmarks
        if SHOW_LANDMARKS:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow("Webcam Evaluation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Webcam Evaluation", cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()

print("\n==============================\nFINAL RESULTS\n==============================\n")
correct = 0
for i, entry in enumerate(results_log, 1):
    print(f"Attempt {i}:")
    print(" True:", entry["true_id"], "-", entry["true_text"])
    print(" Pred:", entry["pred_id"], "-", entry["pred_text"])
    print(" Confidence:", f"{entry['confidence']:.2%}")
    print(" RESULT:", entry["result"], "\n")
    if entry["result"] == "PASS":
        correct += 1
print(f"FINAL SCORE: {correct}/{len(results_log)}")