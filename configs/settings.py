from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = Path(r"F:\KSLR-FluentSigners-50")

DATA_PATH = PROJECT_ROOT / "data"
OUTPUT_STATS_PATH = DATA_PATH / "stats"
OUTPUT_KEYPOINTS_PATH = DATA_PATH / "keypoints" # 1st one full holistics
# OUTPUT_KEYPOINTS_PATH = DATA_PATH / "kerypoints_expressional" # 2nd one expressional holistics

SAMPLE_VIDEOS_PER_CLASS = 10
EXTRACT_VIDEOS_PER_CLASS = 100



# pip freeze > requirements.txt