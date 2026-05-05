### CURRENT PIPELINE (FLAWED)
## this is the code of full holistics model

### BAD DATA TRAINED AS GOOD
# its a mistake from my side zeroing was big mistake and with tight deadline and
# limited computational power (core i5 6300U no gpu, win 10 , 8gb ram, intel hd 520 128mb laptop) i could not afford to
# rebuild the pipeline, too bad i realized it too late in near end
import numpy as np
import mediapipe as mp
mp_holistic = mp.solutions.holistic
def create_holistic():
    return mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True
    )

POSE = 33 * 4       # 132  (x, y, z, visibility)
FACE = 478 * 3       # 1434 (x, y, z) — 478 cuz refine_face_landmarks=True adds 10 iris landmarks for fucks sake
HAND = 21 * 3        # 63   (x, y, z)

EXPECTED_SIZE = POSE + FACE + HAND + HAND  # 132 + 1434 + 63 + 63 = 1692

def extract_landmarks(results):

    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility]
             for lm in results.pose_landmarks.landmark]
        ).flatten()
    else:
        pose = np.zeros(POSE)

    if results.face_landmarks:
        face = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.face_landmarks.landmark]
        ).flatten()
    else:
        face = np.zeros(FACE)

    if results.left_hand_landmarks:
        left = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.left_hand_landmarks.landmark]
        ).flatten()
    else:
        left = np.zeros(HAND)

    if results.right_hand_landmarks:
        right = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.right_hand_landmarks.landmark]
        ).flatten()
    else:
        right = np.zeros(HAND)

    landmarks = np.concatenate([pose, face, left, right])

    return landmarks.astype(np.float32)
