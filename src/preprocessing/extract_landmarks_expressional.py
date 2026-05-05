# IGNORE
# EXPRESSIONAL MODEL
# UNDEVELOPED OR UNFINISHED


# THIS PIPELINE WAS FOR the expressional model that didnt get to be fully developed, the pipeline
# featured lightweight reduced facial landmarks but thrown off development
# and for the time being i tho i was solving the errors of FULL HOLISTICS MODEL PIPELINE tho now when im near finishing i dont think so
import cv2
import numpy as np
from tqdm import tqdm
from configs.settings import DATASET_PATH, OUTPUT_KEYPOINTS_PATH
# from src.utils.mediapipe_utils import create_holistic, extract_landmarks, EXPECTED_SIZE # this line costed me 2 days, shouldve checked it
from src.utils.mediapipe_expressional import create_holistic, extract_landmarks, EXPECTED_SIZE  # use the expressional one for this

MAX_FRAMES = 250
MIN_GOOD_FRAMES = 3
TARGET_PER_CLASS = 110 #could be 80 for extraction, and 1 for testing
FALLBACK_ALL = True


def extract_video(video_path, holistic):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "cant open"

    sequence = []
    prev_landmarks = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if len(sequence) >= MAX_FRAMES:
            break

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            landmarks = extract_landmarks(results)

            if landmarks.shape[0] != EXPECTED_SIZE:
                continue

            # recover missing landmarks from previous frame
            if prev_landmarks is not None:
                mask = landmarks == 0
                landmarks[mask] = prev_landmarks[mask]

            prev_landmarks = landmarks
            sequence.append(landmarks)

        except Exception:
            continue

    cap.release()

    if len(sequence) < MIN_GOOD_FRAMES:
        return None, f"too_few_frames({len(sequence)})"

    return np.array(sequence, dtype=np.float32), "ok"


def _process_videos(videos, output_folder, holistic, existing, ok_count, target):
    skipped = 0
    reasons = []

    needed = max(0, target - ok_count)
    pbar = tqdm(total=needed, desc=f"  {output_folder.name}", leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} extracted [{elapsed}<{remaining}]")

    for video in videos:
        if ok_count >= target:
            break

        if video.stem in existing:
            continue

        sequence, status = extract_video(video, holistic)

        if sequence is None:
            skipped += 1
            reasons.append(f"{video.name}: {status}")
            continue

        output_file = output_folder / (video.stem + ".npy")
        np.save(output_file, sequence)
        ok_count += 1
        existing.add(video.stem)
        pbar.update(1)

    pbar.close()
    return ok_count, skipped, reasons


def extract_dataset():
    """Two-pass extraction for each annotation class."""
    OUTPUT_KEYPOINTS_PATH.mkdir(parents=True, exist_ok=True)
    holistic = create_holistic()
    folders = sorted([f for f in DATASET_PATH.iterdir() if f.is_dir()])
    summary = {}

    for folder in tqdm(folders, desc="Annotations"):
        output_folder = OUTPUT_KEYPOINTS_PATH / folder.name
        output_folder.mkdir(exist_ok=True)
        existing = set(f.stem for f in output_folder.glob("*.npy"))
        ok_count = len(existing)

        all_videos = sorted(folder.glob("*.mp4"))
        total_available = len(all_videos)

        # First pass: target per class
        first_batch = all_videos[:TARGET_PER_CLASS]
        ok_count, skipped, reasons = _process_videos(
            first_batch, output_folder, holistic, existing, ok_count, target=TARGET_PER_CLASS
        )

        # Second pass: fallback to all remaining videos if needed
        used_fallback = False
        if FALLBACK_ALL and ok_count < TARGET_PER_CLASS and len(all_videos) > TARGET_PER_CLASS:
            remaining = all_videos[TARGET_PER_CLASS:]
            used_fallback = True
            tqdm.write(f"  {folder.name}: only {ok_count}/{TARGET_PER_CLASS} after first pass, "
                       f"trying {len(remaining)} remaining videos...")
            ok_count, extra_skipped, extra_reasons = _process_videos(
                remaining, output_folder, holistic, existing, ok_count, target=TARGET_PER_CLASS
            )
            skipped += extra_skipped
            reasons.extend(extra_reasons)

        summary[folder.name] = {
            "extracted": ok_count,
            "available": total_available,
            "skipped": skipped,
            "used_fallback": used_fallback,
            "reasons": reasons[:5],
        }

    # summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)

    low_classes = []
    for cls, info in sorted(summary.items()):
        tag = "✓" if info["extracted"] >= TARGET_PER_CLASS else "✗"
        fb = " (fallback)" if info["used_fallback"] else ""
        line = (f"  {tag} {cls}: {info['extracted']}/{info['available']} "
                f"extracted, {info['skipped']} skipped{fb}")

        if info["extracted"] < TARGET_PER_CLASS:
            low_classes.append(cls)
            print(line)
            for r in info["reasons"]:
                print(f"       └─ {r}")
        else:
            print(line)

    print(f"\nTotal classes: {len(summary)}")
    print(f"Classes with >= {TARGET_PER_CLASS} extractions: "
          f"{len(summary) - len(low_classes)}/{len(summary)}")
    if low_classes:
        print(f"Classes below target: {low_classes}")


if __name__ == "__main__":
    extract_dataset()