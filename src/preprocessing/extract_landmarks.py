## this is the code of full holistics model

### CURRENT PIPELINE

### FLAWED
# zero mapping was a mistake that costed me some time and effort
import cv2
import numpy as np
from tqdm import tqdm
from configs.settings import DATASET_PATH, OUTPUT_KEYPOINTS_PATH
from src.utils.mediapipe_utils import create_holistic, extract_landmarks, EXPECTED_SIZE

FRAME_SKIP = 2            # process every N'th frame (1 = every frame)
TARGET_LENGTH = 30         # fixed output sequence length (frames)
MIN_GOOD_FRAMES = 3

TARGET_PER_CLASS = 80
FALLBACK_ALL = True


def extract_video(video_path, holistic):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "cannot_open"

    sequence = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if FRAME_SKIP > 1 and frame_idx % FRAME_SKIP != 0:
            continue

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            landmarks = extract_landmarks(results)

            if landmarks is not None and landmarks.shape[0] == EXPECTED_SIZE:
                sequence.append(landmarks)
        except Exception:
            continue

    cap.release()

    if len(sequence) < MIN_GOOD_FRAMES:
        return None, f"too_few_frames({len(sequence)})"

    sequence = np.array(sequence, dtype=np.float32)

    if len(sequence) < TARGET_LENGTH:
        padding = np.zeros(
            (TARGET_LENGTH - len(sequence), EXPECTED_SIZE), dtype=np.float32
        )
        sequence = np.vstack([sequence, padding])
    elif len(sequence) > TARGET_LENGTH:
        indices = np.linspace(0, len(sequence) - 1, TARGET_LENGTH, dtype=int)
        sequence = sequence[indices]

    return sequence, "ok"


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
    ### IMPORTANT, any fucker tryna replicate this should read
    """Two-pass extraction:

    Pass 1: Try the first TARGET_PER_CLASS videos per annotation.
            Most classes should reach 40-60+ extractions from 80 videos.
    Pass 2: If a class is still short, try ALL remaining videos as fallback.
    """
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

        # TARGET_PER_CLASS videos
        first_batch = all_videos[:TARGET_PER_CLASS]
        ok_count, skipped, reasons = _process_videos(
            first_batch, output_folder, holistic, existing, ok_count,
            target=TARGET_PER_CLASS
        )

        # if we haven't extracted enough, try the rest
        used_fallback = False
        if FALLBACK_ALL and ok_count < TARGET_PER_CLASS and len(all_videos) > TARGET_PER_CLASS:
            remaining = all_videos[TARGET_PER_CLASS:]
            used_fallback = True
            tqdm.write(f"  {folder.name}: only {ok_count}/{TARGET_PER_CLASS} after first pass, "
                       f"trying {len(remaining)} remaining videos...")
            ok_count, extra_skipped, extra_reasons = _process_videos(
                remaining, output_folder, holistic, existing, ok_count,
                target=TARGET_PER_CLASS
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