### OLD
# used to get the class names / id's (in the dataset the folder structure is the annotation id)
import json
from pathlib import Path

KEYPOINTS_PATH = Path(
    r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\keypoints"
)

OUTPUT = Path(
    r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\models\labels.json"
)

folders = sorted([f.name for f in KEYPOINTS_PATH.iterdir() if f.is_dir()])

with open(OUTPUT, "w") as f:
    json.dump(folders, f, indent=4)

print("Saved labels:", len(folders))