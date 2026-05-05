### OLD BUT REAL USEFUL

import json
import cv2
from pathlib import Path
from statistics import mean
from tqdm import tqdm

from configs.settings import DATASET_PATH, OUTPUT_STATS_PATH, SAMPLE_VIDEOS_PER_CLASS

def analyze_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "frames": frame_count,
        "duration": duration,
        "resolution": [width, height]
    }


def analyze_dataset():

    OUTPUT_STATS_PATH.mkdir(parents=True, exist_ok=True)

    stats = {}

    folders = sorted([f for f in DATASET_PATH.iterdir() if f.is_dir()])

    for folder in tqdm(folders, desc="Analyzing annotations"):

        videos = sorted(folder.glob("*.mp4"))

        if not videos:
            continue

        videos = videos[:SAMPLE_VIDEOS_PER_CLASS]

        fps_list = []
        frame_list = []
        duration_list = []
        resolution_list = []

        for video in videos:

            info = analyze_video(video)

            if info is None:
                continue

            fps_list.append(info["fps"])
            frame_list.append(info["frames"])
            duration_list.append(info["duration"])
            resolution_list.append(tuple(info["resolution"]))

        if fps_list:
            stats[folder.name] = {
                "avg_fps": mean(fps_list),
                "avg_frames": mean(frame_list),
                "avg_duration": mean(duration_list),
                "resolution": resolution_list[0]
            }

    output_file = OUTPUT_STATS_PATH / "annotation_stats.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print(f"\nSaved stats to {output_file}")


if __name__ == "__main__":
    analyze_dataset()
