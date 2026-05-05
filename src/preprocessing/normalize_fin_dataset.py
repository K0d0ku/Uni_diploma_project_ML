# IGNORE
# UNUSED AND OLD

import numpy as np
from pathlib import Path
DATA_PATH = Path(
    r"c:\Users\Bekal\Documents\my_own_documents\Uni_Diploma\Project\Model_training\KrSL_FluenSigners-50_Kuro\data\processed\fin_dataset"
)

X = np.load(DATA_PATH / "X.npy")

# Shape of X: (num_samples, timesteps, features)
mean = X.mean(axis=(0, 1))  # avg over samples and  timesteps
std = X.std(axis=(0, 1))    # std over samples and timesteps

std[std == 0] = 1.0

np.save(DATA_PATH / "mean.npy", mean)
np.save(DATA_PATH / "std.npy", std)

print("Normalization stats computed and saved:")
print("Mean shape:", mean.shape)
print("Std shape:", std.shape)