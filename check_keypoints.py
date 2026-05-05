### OLD

from pathlib import Path

keypoints_dir = Path(r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\keypoints")

folders = sorted(keypoints_dir.iterdir())
total_classes = 0
counts = []

for folder in folders:
    if folder.is_dir():
        npy_count = len(list(folder.glob("*.npy")))
        counts.append((folder.name, npy_count))
        total_classes += 1

print(f"Total annotation classes: {total_classes}\n")

from collections import Counter
count_vals = [c for _, c in counts]
dist = Counter(count_vals)
print("=== Distribution of .npy counts ===")
for k in sorted(dist.keys()):
    print(f"  {k:>3} files: {dist[k]} classes")

print(f"\n  Min: {min(count_vals)}, Max: {max(count_vals)}, Avg: {sum(count_vals)/len(count_vals):.1f}")

low = [(n, c) for n, c in counts if c < 10]
print(f"\n=== Classes with < 10 extracted files ({len(low)}) ===")
for name, cnt in low:
    print(f"  {name}: {cnt}")

empty = [(n, c) for n, c in counts if c == 0]
if empty:
    print(f"\n=== Empty classes ({len(empty)}) ===")
    for name, cnt in empty:
        print(f"  {name}: {cnt}")
