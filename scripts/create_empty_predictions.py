import h5py
import numpy as np
from pathlib import Path
from beneuro_pose_estimation.config import _load_config

config = _load_config()

animal = "M078"
session = "M078_2025_08_08_10_30"
missing_camera = "Camera_Back_Right"
template_camera = "Camera_Front_Right"

pred_dir = config.predictions2D/ animal / session / f"{session}_pose_estimation"

existing_file = Path(pred_dir) / template_camera / f"{session}_{template_camera}.analysis.h5"
new_file = Path(pred_dir) / missing_camera / f"{session}_{missing_camera}.analysis.h5"
new_file.parent.mkdir(parents=True, exist_ok=True)


# Load shape info from existing camera
with h5py.File(existing_file, "r") as f:
    tracks = f["tracks"]
    n_nodes = tracks.shape[2]
    n_frames = tracks.shape[3]
    dtype = tracks.dtype

# Create empty arrays
empty_tracks = np.full((1, 2, n_nodes, n_frames), np.nan, dtype=dtype)
empty_scores = np.zeros((1, n_nodes, n_frames), dtype=dtype)  

# Save to new file
with h5py.File(new_file, "w") as f:
    f.create_dataset("tracks", data=empty_tracks, compression="gzip")
    f.create_dataset("point_scores", data=empty_scores, compression="gzip")

print(f"Empty camera file created at {new_file}")