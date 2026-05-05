### OLD

import os
import cv2
import numpy as np
from pathlib import Path

DATASET_PATH = Path(r"F:\KSLR-FluentSigners-50")
KEYPOINTS_PATH = Path(r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\keypoints")

print("=" * 60)
print("1. CHECKING DATASET PATH")
print("=" * 60)

if not DATASET_PATH.exists():
    print(f"Dataset path does NOT exist: {DATASET_PATH}")
else:
    print(f" Dataset path exists: {DATASET_PATH}")
    folders = sorted([f for f in DATASET_PATH.iterdir() if f.is_dir()])
    print(f"  Total annotation folders: {len(folders)}")
    
    print(f"\n  First 5 folders (video counts):")
    for folder in folders[:5]:
        mp4s = list(folder.glob("*.mp4"))
        all_files = list(folder.iterdir())
        print(f"    {folder.name}: {len(mp4s)} .mp4 files, {len(all_files)} total files")
        if mp4s:

            v = str(mp4s[0])
            cap = cv2.VideoCapture(v)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ret, frame = cap.read()
                print(f"      First video: {mp4s[0].name}, FPS={fps}, Frames={frames}, Read OK={ret}")
                if ret:
                    print(f"      Frame shape: {frame.shape}")
                cap.release()
            else:
                print(f"Cannot open first video: {mp4s[0].name}")

print()
print("=" * 60)
print("2. CHECKING EXISTING .NPY FILES")
print("=" * 60)

for folder in sorted(KEYPOINTS_PATH.iterdir()):
    if not folder.is_dir():
        continue
    npy_files = sorted(folder.glob("*.npy"))
    if npy_files:
        for npy in npy_files[:2]:
            data = np.load(npy)
            zero_frames = np.sum(np.all(data == 0, axis=1))
            print(f"  {folder.name}/{npy.name}: shape={data.shape}, zero_frames={zero_frames}/{data.shape[0]}")
        if len(npy_files) > 2:
            print(f"  ... ({len(npy_files)} total)")
        break

print()
print("=" * 60)
print("3. TESTING VIDEO EXTRACTION (first 5 videos from class 000)")
print("=" * 60)

if DATASET_PATH.exists():
    test_folder = sorted([f for f in DATASET_PATH.iterdir() if f.is_dir()])[0]
    videos = sorted(test_folder.glob("*.mp4"))[:5]
    
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f" Cannot open: {video.name}")
            continue
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        readable_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            readable_frames += 1
        
        cap.release()
        print(f"  {video.name}: reported_frames={total_frames}, actually_readable={readable_frames}, fps={fps}")
