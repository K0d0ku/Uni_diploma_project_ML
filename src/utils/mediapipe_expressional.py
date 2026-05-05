# IGNORE
## UNFINISHED EXPRESSIONAL MODEL PIPELINE

# workin version
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

# upper body pose landmarks only
POSE_LANDMARKS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]

# facial expression subset
FACE_LANDMARKS = [
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,     # eyebrows
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, # eyes
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324 # mouth
]

# hands full
HAND_LANDMARKS = list(range(21))

POSE_SIZE = len(POSE_LANDMARKS) * 4  # x,y,z,visibility
FACE_SIZE = len(FACE_LANDMARKS) * 3  # x,y,z
HAND_SIZE = len(HAND_LANDMARKS) * 3  # x,y,z

EXPECTED_SIZE = POSE_SIZE + FACE_SIZE + HAND_SIZE + HAND_SIZE

print("USING EXPRESSIONAL PIPELINE:", EXPECTED_SIZE)


def normalize_landmarks(landmarks):
    """
    normalize coordinates relative to shoulder center and scale by shoulder distance.
    Handles mixed dimensions safely (pose=4D, face/hands=3D).
    """
    if landmarks.size != EXPECTED_SIZE:
        return landmarks

    # split sections
    pose_end = POSE_SIZE
    face_end = pose_end + FACE_SIZE
    left_end = face_end + HAND_SIZE

    pose = landmarks[:pose_end].reshape(-1, 4)   # (N, 4)
    face = landmarks[pose_end:face_end].reshape(-1, 3)
    left = landmarks[face_end:left_end].reshape(-1, 3)
    right = landmarks[left_end:].reshape(-1, 3)

    try:
        # get shoulder indices INSIDE reduced pose set
        LEFT_SHOULDER_INDEX = POSE_LANDMARKS.index(11)
        RIGHT_SHOULDER_INDEX = POSE_LANDMARKS.index(12)

        left_shoulder = pose[LEFT_SHOULDER_INDEX][:3]
        right_shoulder = pose[RIGHT_SHOULDER_INDEX][:3]

        # compute normalization params
        center = (left_shoulder + right_shoulder) / 2.0
        scale = np.linalg.norm(left_shoulder - right_shoulder)

        if scale < 1e-6:
            scale = 1.0

        # normalize pose (only xyz, keep visibility as is), face, and hands
        pose[:, :3] = (pose[:, :3] - center) / scale
        face = (face - center) / scale
        left = (left - center) / scale
        right = (right - center) / scale

    except ValueError:
        # shoulders missing → skip normalization safely
        pass

    # recombine
    return np.concatenate([
        pose.flatten(),
        face.flatten(),
        left.flatten(),
        right.flatten()
    ]).astype(np.float32)


def extract_landmarks(results):
    # Extract reduced expressional landmarks, fill missing with zeros.
    #  ↑ shame my dumbass zeroed here too without knowing the consequences

    # pose
    if results.pose_landmarks:
        pose = np.array([
            [
                results.pose_landmarks.landmark[i].x,
                results.pose_landmarks.landmark[i].y,
                results.pose_landmarks.landmark[i].z,
                results.pose_landmarks.landmark[i].visibility
            ]
            for i in POSE_LANDMARKS
        ], dtype=np.float32).flatten()
    else:
        pose = np.zeros(POSE_SIZE, dtype=np.float32)

    # face
    if results.face_landmarks:
        face = np.array([
            [
                results.face_landmarks.landmark[i].x,
                results.face_landmarks.landmark[i].y,
                results.face_landmarks.landmark[i].z
            ]
            for i in FACE_LANDMARKS
        ], dtype=np.float32).flatten()
    else:
        face = np.zeros(FACE_SIZE, dtype=np.float32)

    # hand L
    if results.left_hand_landmarks:
        left = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.left_hand_landmarks.landmark
        ], dtype=np.float32).flatten()
    else:
        left = np.zeros(HAND_SIZE, dtype=np.float32)

    # hand R
    if results.right_hand_landmarks:
        right = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.right_hand_landmarks.landmark
        ], dtype=np.float32).flatten()
    else:
        right = np.zeros(HAND_SIZE, dtype=np.float32)

    # combine
    landmarks = np.concatenate([pose, face, left, right])

    # check size before normalization to avoid silent errors
    if landmarks.size != EXPECTED_SIZE:
        raise ValueError(f"Landmark size mismatch: got {landmarks.size}, expected {EXPECTED_SIZE}")

    # normalization
    landmarks = normalize_landmarks(landmarks)
    return landmarks