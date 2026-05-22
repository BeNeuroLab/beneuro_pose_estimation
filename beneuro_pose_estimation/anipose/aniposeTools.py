"""
Module to carry out Anipose operations
TBD:
- needs checking after the last changes
-------------------------------
conda activate bnp
bnp init    # to create the .env file
bnp pose session_name(s)


"""
from aniposelib.boards import CharucoBoard
import logging
import cv2

import matplotlib as plt

import beneuro_pose_estimation.sleap.sleapTools as sleapTools
import beneuro_pose_estimation.tools as tools
from beneuro_pose_estimation import params

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
import os
import subprocess
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sleap_anipose as slap
import toml
from anipose.compute_angles import compute_angles
from aniposelib.cameras import CameraGroup
from typing import Optional
from beneuro_pose_estimation import params, set_logging
from beneuro_pose_estimation.config import _load_config
from beneuro_pose_estimation.tools import (
    get_calib_for_session,
    _ensure_calib_file_local,
)

config = _load_config()

logger = set_logging(__name__)

def get_calib_file(calib_videos_dir: Path, calib_save_path: Optional[Path] = None) -> Path:
    """
    Gets or generates calibration file using ChArUco board videos with local/remote sync.
    
    Process:
    1. Derives expected calibration file name
    2. Checks if calibration exists on remote - if yes, raises error
    3. Checks local cache - if exists, returns it
    4. If not found anywhere, generates new calibration file from videos
    5. Saves locally and syncs to remote

    Parameters
    ----------
    calib_videos_dir : Path
        Directory path containing ChArUco videos for calibration
    calib_save_path : Path, optional
        Explicit path to save the calibration file. If None, derives from folder name.

    Returns
    -------
    Path
        Path to the calibration file (local copy)
    
    Raises
    ------
    FileExistsError
        If calibration file already exists on remote
    """
    # Determine calibration file name
    if calib_save_path is None:
        # Extract date from folder name (format: camera_calibration_YYYY_MM_DD_HH_MM)
        try:
            date_part = "_".join(calib_videos_dir.name.split("_")[2:])
            calib_file_name = f"calibration_{date_part}.toml"
        except Exception:
            calib_file_name = "calibration.toml"
    else:
        calib_file_name = calib_save_path.name if isinstance(calib_save_path, Path) else Path(calib_save_path).name
    
  
    # 2. Check if already in local cache
    if calib_save_path.exists():
        logger.info(f"Calibration already exists: {calib_save_path}")
        # Try to copy to remote if not already there
        
        return logger.info
    
    # 3. Generate new calibration file
    logger.info(f"Generating new calibration file from videos in {calib_videos_dir}...")
    board = params.board

    video_files = list(calib_videos_dir.iterdir())
    cam_names, vidnames = [], []
    reversed_mapping = {v: k for k, v in params.camera_name_mapping.items()}
    for video_file in video_files:
        if video_file.suffix in [".avi", ".mp4"]:  # Check file extension
            camera = video_file.stem
            if camera == "Camera_3":
                continue
            cam_name = reversed_mapping.get(camera, camera)
            vidnames.append([str(video_file)])  # Convert to str if required by downstream methods
            cam_names.append(cam_name)

    # Save to local repo first
    calib_save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize and configure CharucoBoard and CameraGroup
    cgroup = CameraGroup.from_names(cam_names, fisheye=params.fisheye)
    cgroup.calibrate_videos(vidnames, board)
    cgroup.dump(calib_save_path)
    logger.info(f"Calibration file generated and saved locally: {calib_save_path}")

    
    
    return calib_save_path

def new_calib(calib_folder_name: str = None, start_date: str = None) -> str:
    """
    Creates a new calibration registration.
    
    Process:
    1. Resolves calibration folder (uses most recent if not specified)
    2. Extracts/validates start date
    3. Gets or generates calibration file (checks remote first, regenerates if needed)
    4. Registers in calibration registry with remote sync
    
    Parameters
    ----------
    calib_folder_name : str, optional
        Name of calibration video folder. If None, uses most recent.
    start_date : str, optional
        Date when calibration becomes valid (YYYY-MM-DD). If None, extracts from folder name.
    
    Returns
    -------
    str
        The calibration ID (e.g., "calibration_2026_05_21")
    
    Raises
    ------
    Exception
        If folder not found, date cannot be extracted, or other errors occur
    """
    from beneuro_pose_estimation.tools import (
        _register_calibration,_copy_file_to_remote
    )
    
    # Resolve calibration folder
    if calib_folder_name is None:
        # Find most recent folder in config.calibration_videos
        try:
            calib_videos_dir = config.calibration_videos
            all_folders = [f for f in calib_videos_dir.iterdir() if f.is_dir()]
            if not all_folders:
                raise FileNotFoundError(f"No calibration video folders found in {calib_videos_dir}")
            # Sort by name (which includes timestamp) in descending order
            calib_folder = max(all_folders, key=lambda x: x.name)
            calib_folder_name = calib_folder.name
            logger.info(f"Using most recent calibration folder: {calib_folder_name}")
        except Exception as e:
            logger.error(f"Error finding calibration folders: {e}")
            raise
    else:
        calib_folder = config.calibration_videos / calib_folder_name
    
    # Validate folder exists
    if not calib_folder.exists():
        raise FileNotFoundError(f"Calibration folder not found: {calib_folder}")
    
    # Resolve start date
    if start_date is None:
        # Extract date from folder name (format: camera_calibration_YYYY_MM_DD_HH_MM)
        try:
            date_part = "_".join(calib_folder_name.split("_")[2:])
            parsed_datetime = datetime.strptime(date_part, "%Y_%m_%d_%H_%M")
            start_date = parsed_datetime.strftime("%Y-%m-%d")
            logger.info(f"Extracted start date from folder: {start_date}")
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Could not extract date from folder name '{calib_folder_name}': {e}. "
                f"Please provide start date with --start-date flag (YYYY-MM-DD format)"
            )
    else:
        # Validate date format
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            logger.info(f"Using provided start date: {start_date}")
        except ValueError:
            raise ValueError(f"Invalid date format: {start_date}. Use YYYY-MM-DD format.")
    
    # Check if calibration file already exists on remote (before attempting to create)
    try:
        date_part = "_".join(calib_folder_name.split("_")[2:])
        calib_file_name = f"calibration_{date_part}.toml"
    except Exception:
        calib_file_name = "calibration.toml"
    
    remote_calib_path = config.calibration / calib_file_name
    if remote_calib_path.exists():
        logger.info(
            f"Calibration file already exists on remote: {remote_calib_path}. "
           
        )
    else:
        calib_path = get_calib_file(calib_folder,calib_save_path=config.calibration_local / calib_file_name)

        if not _copy_file_to_remote(calib_path, remote_calib_path):
            logger.warning(
                f"[bold yellow]Could not copy calibration to remote ({remote_calib_path}). "
                f"Local copy available at {calib_path}[/bold yellow]"
            )

    
    # # Get or generate calibration file (checks local first, generates if needed)
    # logger.info(f"Getting calibration file from videos in {calib_folder}...") 

    logger.info(f"Calibration file ready: {calib_path.name}")
    
    # if not calib_path.exists():
    #     raise FileNotFoundError(f"Calibration file not found: {local_calib_path}")
    
    # Register in calibration registry (with remote sync)
    logger.info(f"Registering calibration with start_date={start_date}...")
    calib_id = _register_calibration(calib_folder_name, start_date, calib_path)
    logger.info(f"Calibration registered successfully: {calib_id}")
    
    return calib_id




def evaluate_reprojection(reprojection_path, predictions_2D_dir, histogram_path=None):
    """
    Plots histogram of reprojection error.
    Not tested
    """
    # Generate histogram using data from the combined file
    detection_data = load_predictions_2D(predictions_2D_dir)
    reprojection_data = np.load(reprojection_path)
    slap.make_histogram(detection_data, reprojection_data, save_path=histogram_path)
    logging.info(f"Reprojection histogram saved at {histogram_path}")

    return


def load_predictions_2D(predictions_2D_dir):
    """
    TBD - need to check that the order of cameras is the same as in reprojections
    """
    predictions_list = []
    cameras = params.default_cameras
    session = os.path.basename(predictions_2D_dir)
    for camera in cameras:
        h5_file = f"{predictions_2D_dir}/{session}_{camera}.analysis.h5"
        with h5py.File(h5_file, "r") as f:
            # Shape of 'tracks': (1, 2, n_nodes, n_frames)
            tracks = f["tracks"][:]  # Read the data into memory
            tracks = np.squeeze(tracks)  # Remove the leading dimension if it is 1
            # Rearrange the dimensions to (n_frames, n_nodes, 2)
            tracks = np.moveaxis(tracks, 1, -1)
            predictions_list.append(tracks)

    # Combine predictions from all cameras along a new first dimension (n_cams)
    concatenated_predictions = np.stack(predictions_list, axis=0)

    return concatenated_predictions


def get_frame_count(h5_analysis_file):
    with h5py.File(h5_analysis_file, "r") as f:
        return f["tracks"].shape[-1]


def get_most_recent_calib(session):
    # Parse session date and time
    try:
        session_datetime = datetime.strptime(
            "_".join(session.split("_")[1:]), "%Y_%m_%d_%H_%M"
        )
    except ValueError:
        logging.error(f"Invalid session format: {session}")
        return None

    # Iterate over all calibration folders and extract timestamps
    calib_folders = []
    calib_vid_dir = config.calibration_videos
    for folder in calib_vid_dir.iterdir():
        if folder.is_dir():
            try:
                # Extract the datetime from the "Recording_<datetime>" format
                calib_datetime = datetime.strptime(
                    "_".join(folder.name.split("_")[2:]), "%Y_%m_%d_%H_%M"
                )
                calib_folders.append((calib_datetime, folder))
            except ValueError:
                logging.warning(
                    f"Invalid calibration folder format: {folder.name}. Skipping."
                )
                continue

    # Sort calibration folders by datetime in descending order
    calib_folders = sorted(calib_folders, key=lambda x: x[0], reverse=True)

    # Find the most recent calibration folder before the session
    recent_calib_folder = None
    for calib_datetime, folder in calib_folders:
        if calib_datetime < session_datetime:
            recent_calib_folder = folder
            break

    if recent_calib_folder is None:
        logging.warning(f"No valid calibration folders found before session {session}.")
        return None

    logging.info(
        f"Using calibration folder: {recent_calib_folder} for session {session}"
    )
    

    # Generate calibration file path
    calib_file_name = Path(
        f"calibration_{calib_datetime.strftime('%Y_%m_%d_%H_%M')}.toml"
    )
    calib_file_path = config.calibration / calib_file_name
    return recent_calib_folder, calib_file_path


def convert_2Dpred_to_h5(
    sessions,
    cameras=params.default_cameras,
    input_dir=None,
    output_dir=None,
):
    """
    Converts .slp.predictions.slp files to .h5 analysis files for each session and camera.
    """
    if isinstance(sessions, str):
        sessions = [sessions]
    if isinstance(cameras, str):
        cameras = [cameras]
    if input_dir is None:
        input_dir = config.predictions2D
    if output_dir is None:
        output_dir = config.predictions3D

    for session in sessions:
    
        for camera in cameras:
            input_file = input_dir/camera /f"{session}_{camera}.slp.predictions.slp"
            output_file = output_dir/camera / f"{session}_{camera}.analysis.h5"
            if output_file.exists():
                logging.info(f"Output file {output_file} already exists. Skipping...")
            else:
                try:
                    subprocess.run(
                        [
                            "sleap-convert",
                            "--format",
                            "analysis",
                            "-o",
                            output_file,
                            input_file,
                        ],
                        check=True,
                    )
                    logging.info(f"Converted {input_file} to {output_file}")
                except subprocess.CalledProcessError as e:
                    logging.error(f"Error during conversion for {input_file}: {e}")
           


def compute_3Dpredictions(
    session, pred_dir, output_dir, calib_file_path, frame_window=params.frame_window, eval=False
):
    """
    Triangulates 3D predictions from 2D predictions for each session in windows and then combines them.
    """
    # cgroup = CameraGroup.load(calib_file_path)
    # n_frames = get_frame_count(
    #     f"{project_dir}/{params.default_cameras[0]}/{session}_{params.default_cameras[0]}.analysis.h5"
    # )
    frame_counts = []
    for cam in params.default_cameras:
        file_path = Path(f"{pred_dir}/{cam}/{session}_{cam}.analysis.h5")
        if file_path.exists():
            count = get_frame_count(file_path)
            frame_counts.append(count)
        
    # Check that all frame counts are consistent
    if len(set(frame_counts)) != 1:
        logging.warning(f"Frame counts across cameras are not consistent: {frame_counts}. Using the minimum value.")

    # Use the consistent frame count
    n_frames = np.min(frame_counts)
    windows = np.arange(0, n_frames, frame_window)
    windows = np.append(windows, n_frames)
    reprojections_list = []  # List to store reprojections from each window
    # breakpoint()
    for start, end in zip(windows[:-1], windows[1:]):
        logging.info(f"Processing frames {start} to {end}")
        output_file = Path(f"{output_dir}/{session}_triangulation_{start}_{end}.h5")
        if output_file.exists():
            logging.info(f"Output file {output_file} already exists. Skipping...")
            continue
        # breakpoint()
        slap.triangulate(
            p2d=str(pred_dir),
            calib=str(calib_file_path),
            fname=str(output_file),
            disp_progress=True,
            frames=(start, end),
            ransac = params.triangulation_params["ransac"],
            constraints=params.constraints,
            scale_smooth=params.triangulation_params["scale_smooth"],
            scale_length=params.triangulation_params["scale_length"],
            scale_length_weak=params.triangulation_params["scale_length_weak"],
            reproj_error_threshold=params.triangulation_params[
                "reproj_error_threshold"
            ],
            # excluded_views=["Camera_6"],
            reproj_loss=params.triangulation_params["reproj_loss"],
            n_deriv_smooth=params.triangulation_params["n_deriv_smooth"],
        )
       
        logging.info(f"3D prediction file created: {output_file}")

        if eval:
            reproj_output = slap.reproject(
                p3d=output_file, calib=calib_file_path, frames=(start, end)
            )
            logging.info(f"Reprojection created for frames {start} to {end}")
            reprojections_list.append(reproj_output)

    if reprojections_list:
        reprojections_array = np.concatenate(
            reprojections_list, axis=1
        )  # Concatenate along the frames axis
        logging.info(
            f"Reprojections concatenated with shape: {reprojections_array.shape}"
        )

        save_path = f"{output_dir}/{session}_reprojections.npy"
        np.save(save_path, reprojections_array)
        logging.info(f"Reprojections saved to {save_path}")
        # histogram_path = f"{project_dir}/{session}_reprojection_histogram.pdf"
        # evaluate_reprojection(
        #     reprojection_path=save_path,
        #     predictions_2D_dir=project_dir,
        #     histogram_path=histogram_path,
        # )

    combine_h5_files(session, windows, output_dir, eval)


def combine_h5_files(session, windows, project_dir, eval=False):
    """
    Combines multiple .h5 files into one.
    """
    combined_file = f"{project_dir}/{session}_pose_estimation_combined.h5"
    with h5py.File(combined_file, "w") as combined_h5:
        for start, end in zip(windows[:-1], windows[1:]):
            fname = f"{project_dir}/{session}_triangulation_{start}_{end}.h5"
            if os.path.exists(fname):
                with h5py.File(fname, "r") as f:
                    # points3d_data = f['points3d'][:]
                    points3d_data = f["tracks"]
                    # logging.info(f"points3d_data shape: {points3d_data.shape}")
                    if "points3d" not in combined_h5:
                        combined_dataset = combined_h5.create_dataset(
                            "points3d",
                            data=points3d_data,
                            maxshape=(None,) + points3d_data.shape[1:],
                        )
                    else:
                        combined_dataset.resize(
                            (combined_dataset.shape[0] + points3d_data.shape[0]), axis=0
                        )
                        combined_dataset[-points3d_data.shape[0] :] = points3d_data

    logging.info(f"Combined 3D predictions saved at {combined_file}")


def save_to_csv(session, h5_file_path, csv_path):
    """
    Saves 3D prediction data to CSV format.
    add _error, _score, _ncams,add fnum
    """
    points3d = h5py.File(h5_file_path, "r")["points3d"][:]
    fnum = np.arange(points3d.shape[0])  # Generate frame numbers
    points3d_flat = points3d.reshape((points3d.shape[0], -1))

    # Prepare column names for the 3D points
    columns = [
        f"{part}_{axis}" for part in params.body_parts for axis in ("x", "y", "z")
    ]

    # Create the base DataFrame with points3d and frame numbers
    df = pd.DataFrame(points3d_flat, columns=columns)
    df.insert(0, "fnum", fnum)  # Add 'fnum' column as the first column

    # Prepare '_error' and '_score' columns for all body parts in bulk
    error_columns = {
        f"{part}_error": 0 for part in params.body_parts
    }  # All '_error' set to 0
    score_columns = {
        f"{part}_score": 1 for part in params.body_parts
    }  # All '_score' set to 1

    # Create DataFrames for error and score columns
    error_df = pd.DataFrame(error_columns, index=df.index)
    score_df = pd.DataFrame(score_columns, index=df.index)

    # Concatenate all DataFrames: df (points3d + fnum), error_df, and score_df
    df = pd.concat([df, error_df, score_df], axis=1)

    # Save the DataFrame to CSV
    df.to_csv(csv_path, index=False)
    logging.info(f"3D predictions saved to CSV at {csv_path}")


def get_body_part_connections(constraints, keypoints_dict):
    return [[keypoints_dict[start], keypoints_dict[end]] for start, end in constraints]


# def create_config_file(project_dir, config_file_name="config.toml", body_parts = params.body_parts, constraints = params.constraints, triangulation_params = params.triangulation_params,angles = params.angles, frame_window = params.frame_window):
#     """
#     Creates a configuration file in the project directory if it doesn't already exist.
#     Uses parameters defined in params.py for setup.
#     """
#     config_path = f"{project_dir}/{config_file_name}"

#     # Check if configuration file already exists
#     if os.path.isfile(config_path):
#         logging.info(f"Configuration file already exists at {config_path}.")
#         return config_path
#     body_part_connections =  get_body_part_connections(constraints,params.keypoints_dict)
#     # Create configuration dictionary
#     config_dict = {
#         "body_parts": body_parts,
#         "body_part_connections": body_part_connections,
#         "constraints": constraints,
#         "triangulation_params": triangulation_params,
#         "angles": angles,
#         "frame_window": frame_window,
#     }

#     # Save configuration to a JSON file
#     with open(config_path, "w") as config_file:
#         json.dump(config_dict, config_file, indent=4)
#         logging.info(f"Configuration file created at {config_path}.")

#     return config_path


def create_config_file(config_path, angles=params.angles):
    """
    Creates a configuration file in the project directory if it doesn't already exist.
    Uses parameters defined in params.py for setup.
    """

    if not os.path.isfile(config_path):
        config = {"angles": angles}
        with open(config_path, "w") as f:
            toml.dump(config, f)

        logging.info(f"Configuration file created at {config_path}")
    else:
        logging.info(f"Configuration file already exists at {config_path}")

    return config_path


def plot_behaviour(
    data_path,
    columns_to_plot,
    windwow=params.frame_window,
    frame_start=None,
    frame_end=None,
):
    """
    Plots selected body points or angles over time from a CSV file.
    """
    data = pd.read_csv(data_path)

    plt.figure(figsize=(10, 6))
    for column in columns_to_plot:
        if column in data.columns:
            plt.plot(data.index, data[column], label=column)
        else:
            logging.warning(f"Column {column} not found in data.")

    plt.xlabel("Frame Index")
    plt.ylabel("Value")
    plt.title("Behaviour over time")
    plt.legend()
    plt.grid(True)
    plt.show()


def extract_date(session_name):
    return "".join(session_name.split("_")[1:4])

def pad_all_analysis_h5(
    csv_path: str,
    in_root: Path,
    out_root: Path,
    n_slots: int = 414_000,
    bin_size: int = 10_000_000,
):
    """
    For each camera under `in_root`, reads its <camera>.analysis.h5,
    pads both 'tracks' (shape 1×2×n_nodes×n_raw) and 'scores'
    (shape 1×n_nodes×n_raw) out to length n_slots,
    filling missing frames with zeros (i.e. no‐detection, zero confidence),
    and writes to parallel structure under `out_root`.
    """
    # Load timestamp CSV once
    df = pd.read_csv(csv_path, sep=";")
    half = bin_size // 2

    inv_map = {v: k for k, v in params.camera_name_mapping.items()}
    # Determine camera names from CSV
    cameras = df["frame_camera_name"].unique()

    for cam in cameras:
        sub = (
            df[df["frame_camera_name"] == cam]
            .loc[:, ["frame_timestamp", "frame_id"]]
            .sort_values("frame_timestamp")
            .reset_index(drop=True)
        )
        # sub["sub_idx"] = sub.index  # 0 … n_raw‑1
        sub["sub_idx"] = np.arange(len(sub))
        if sub.shape[0] == 0:
            print(f"{inv_map[cam]} no rows → skipping")
            continue

        # Build the 10ms grid of frame_ids
        start, end = sub["frame_timestamp"].min(), sub["frame_timestamp"].max()
        grid_ticks = np.round(np.linspace(start, end, n_slots)).astype(np.int64)
        merged = pd.merge_asof(
            pd.DataFrame({"bin_ts": grid_ticks}),
            sub.rename(columns={"frame_timestamp": "actual_ts"}),
            left_on="bin_ts",
            right_on="actual_ts",
            direction="nearest",
            tolerance=half,
        )
        aligned_idxs = merged["sub_idx"].to_numpy()# floats with NaN where missing
        valid_mask  = ~np.isnan(aligned_idxs )
        valid_idxs  = aligned_idxs [valid_mask].astype(int)

        # Locate the single .analysis.h5 in this camera's folder
        cam_in_dir = in_root / inv_map[cam]
        h5_files = list(cam_in_dir.glob("*.analysis.h5"))
        if not h5_files:
            print(f"[{cam}] no .analysis.h5 → skipping")
            continue
        in_h5 = h5_files[0]

        # Read raw datasets
        with h5py.File(in_h5, "r") as src:
            raw_tracks = src["tracks"][:]  # (1,2,n_nodes,n_raw)
            raw_scores = src["point_scores"][:]  # (1,n_nodes,n_raw), if present

        # Prepare new zero‐filled arrays
        _, dim, n_nodes, _ = raw_tracks.shape
        new_tracks = np.zeros((1, dim, n_nodes, n_slots), dtype=raw_tracks.dtype)
        new_scores = np.zeros((1,     n_nodes, n_slots), dtype=raw_scores.dtype)

        # Copy valid frames into their slots
        new_tracks[..., valid_mask] = raw_tracks[..., valid_idxs]
        new_scores[..., valid_mask] = raw_scores[..., valid_idxs]

        # Write out to out_h5_root/<cam>/<same filename>
        cam_out_dir = out_root / inv_map[cam]
        cam_out_dir.mkdir(parents=True, exist_ok=True)
        out_h5 = cam_out_dir / in_h5.name
        with h5py.File(out_h5, "w") as dst:
            dst.create_dataset("tracks", data=new_tracks, compression="gzip")
            dst.create_dataset("point_scores", data=new_scores, compression="gzip")
def pad_and_interp_analysis_h5_dynamic(
    csv_path: str,
    in_root: Path,
    out_root: Path,
    bin_size: int = 10_000_000,      # 10 ms ticks
):
    """
    Pads & interpolates every camera's .analysis.h5 so each has exactly `n_slots` bins,
    where n_slots = min over cameras of ((end_ts - start_ts)//bin_size + 1).
    Both tracks and point_scores are linearly interpolated, and
    an npy of interpolated slot-indices is saved per camera.
    """
    df      = pd.read_csv(csv_path, sep=";")
    half    = bin_size // 2


    # invert mapping "friendly"->"Camera_N"
    inv_map = {v: k for k, v in params.camera_name_mapping.items()}

    # 1) compute expected slot count per camera
    cam_slots = {}
    for cam_code in df["frame_camera_name"].unique():
        sub = df[df["frame_camera_name"] == cam_code]
        if sub.empty:
            continue
        start, end = sub["frame_timestamp"].min(), sub["frame_timestamp"].max()
        cam_slots[cam_code] = int((end - start) // bin_size) + 1

    if not cam_slots:
        raise RuntimeError("No cameras found in CSV!")
    n_slots = min(cam_slots.values())
    print(f"Using n_slots={n_slots} (min across cameras)")

    # 2) process each camera
    for cam_code, slots in cam_slots.items():
        folder = inv_map.get(cam_code)
        if folder is None:
            print(f"[WARN] no folder mapping for '{cam_code}', skip")
            continue

        cam_out = out_root / folder
        interp_npy = cam_out / f"{cam_code}_interp_idxs.npy"

        if interp_npy.exists():
            print(f"[{cam_code}] found {interp_npy.name} → skipping (already interpolated)")
            continue

        # isolate & sort
        sub = (
            df[df["frame_camera_name"] == cam_code]
            .loc[:, ["frame_timestamp","frame_id"]]
            .sort_values("frame_timestamp")
            .reset_index(drop=True)
        )
        sub["sub_idx"] = np.arange(len(sub))  # 0..n_raw-1

        # build exact-10ms grid of length n_slots
        start_ts = sub["frame_timestamp"].iloc[0]
        grid_ticks = start_ts + np.arange(n_slots) * bin_size
        grid = pd.DataFrame({"bin_ts": grid_ticks})

        # merge_asof to find nearest detection sub_idx
        merged = pd.merge_asof(
            grid,
            sub.rename(columns={"frame_timestamp":"actual_ts"}),
            left_on="bin_ts", right_on="actual_ts",
            direction="nearest",
            tolerance=half
        )
        aligned = merged["sub_idx"].to_numpy()    # floats, NaN for missing
        valid   = ~np.isnan(aligned)
        pos     = aligned[valid].astype(int)
        interp_idxs = np.where(~valid)[0]         # slots we will interpolate

        # load raw .analysis.h5
        cam_in = next((in_root/folder).glob("*.analysis.h5"), None)
        if cam_in is None:
            print(f"[{folder}] no .analysis.h5 → skipping")
            continue
        with h5py.File(cam_in, "r") as src:
            raw_tracks      = src["tracks"][:]         # (1,2,n_nodes,n_raw)
            raw_point_scores= src["point_scores"][:]   # (1,n_nodes,n_raw)

        # allocate full arrays
        _, dim, n_nodes, _ = raw_tracks.shape
        T = n_slots
        tracks = np.zeros((1,dim,n_nodes,T),      dtype=raw_tracks.dtype)
        scores = np.zeros((1,n_nodes,   T),      dtype=raw_point_scores.dtype)

        # copy real detections
        tracks[..., valid] = raw_tracks[...,      pos]
        scores[..., valid] = raw_point_scores[...,pos]

        # interpolate over time
        x_valid = np.where(valid)[0]
        x_all   = np.arange(T)
        for node in range(n_nodes):
            # confidence
            y = scores[0,node,x_valid]
            scores[0,node] = np.interp(x_all, x_valid, y)
            # x,y coords
            for d in range(dim):
                y = tracks[0,d,node,x_valid]
                tracks[0,d,node] = np.interp(x_all, x_valid, y)

        # write padded+interpolated H5
        cam_out.mkdir(parents=True, exist_ok=True)
        out_h5 = cam_out/cam_in.name
        with h5py.File(out_h5,"w") as dst:
            dst.create_dataset("tracks",       data=tracks, compression="gzip")
            dst.create_dataset("point_scores", data=scores, compression="gzip")

        # save interpolated slot indices
        np.save(interp_npy, interp_idxs)

        print(f"[{cam_code}] → {out_h5}  (interpolated {len(interp_idxs)} slots)")
def session_datetime(session: str) -> datetime:
    """
    Expected format: M0XX_year_month_day_hour_minute
    Example: M063_2025_03_21_14_00
    """
    parts = session.split("_")
    if len(parts) != 6:
        raise ValueError(f"Unexpected session format: {session!r}")

    _, y, m, d, hh, mm = parts
    return datetime(int(y), int(m), int(d), int(hh), int(mm))

def run_pose_estimation(
    sessions,
    custom_model_name = None,
    eval=False
):
    """
    Main routing from videos to 3D keypoints and angles.
    """

    if isinstance(sessions, str):
        sessions = [sessions]
    

    for session in sessions:
        logging.info(f"Running pose estimation on {session}")
        animal = session.split("_")[0]
        session_dir = config.predictions3D / animal / session ############
        predictions_dir = session_dir / f"{session}_pose_estimation" # TODO change to pose_estimation
        predictions_dir.mkdir(parents=True, exist_ok=True)
        

        # get 2D predictions - slp files saved in predictions2D/animal/session/predictions
        sleapTools.get_2Dpredictions(session, custom_model_name = custom_model_name)

        # convert 2D predictions to h5 files - h5 files saved in predictions2D/animal/session/predictions
        convert_2Dpred_to_h5(session,input_dir=predictions_dir, output_dir=predictions_dir) 
        metadata_path = config.recordings/animal/session/ f"{session}_cameras"/ "metadata.csv"

        try: 
            pad_and_interp_analysis_h5_dynamic(
            csv_path    = metadata_path,
            in_root  = predictions_dir,
            out_root = predictions_dir,
            )
        except Exception as e:
            logger.error(f"Skipping frame alignment, {e}")

        ###############################################

        # Get calibration file for this session using the registry
        calib_file_path = get_calib_for_session(session)
        
        if calib_file_path is None:
            logging.error(
                f"Could not find calibration file for session {session}. "
                f"Please register a calibration using: bnp new-calib"
            )
            raise ValueError(f"No calibration found for session {session}")
        
        logging.info(f"Found calibration file: {calib_file_path}")
        
        # Ensure calibration file is available locally (download if needed)
        local_calib_path = _ensure_calib_file_local(calib_file_path)
        logging.info(f"Using local calibration copy: {local_calib_path} for triangulation")
            
        compute_3Dpredictions(
            session, calib_file_path=local_calib_path, pred_dir=predictions_dir, output_dir=predictions_dir, eval=eval
        )
        labels_fname = predictions_dir/f"{session}_3dpts.csv"
        save_to_csv(
            session,
            predictions_dir/f"{session}_pose_estimation_combined.h5",
            labels_fname,
        )
        config_path = config.angles_config/"config.toml"
        if not config_path.exists():
            config_path = create_config_file(config_path)
        config_angles = toml.load(config_path)
        angles_csv = predictions_dir/f"{session}_angles.csv"
        # labels_data = pd.read_csv(labels_xfname)
        # logging.debug(labels_data.columns)
        compute_angles(config_angles, labels_fname, angles_csv)
        
        pose_data = pd.read_csv(labels_fname)
        angles_data = pd.read_csv(angles_csv)

        # Combine pose data and angles data
        combined_data = pd.concat([pose_data, angles_data], axis=1)
        combined_csv = predictions_dir/f"{session}_3dpts_angles.csv"
        # Save the updated CSV
        combined_data.to_csv(combined_csv, index=False)
        logging.info(f"Angles computed and combined CSV saved at {combined_csv}.")
        try:
            if labels_fname.exists():
                labels_fname.unlink()
                logger.info(f"Deleted intermediate CSV: {labels_fname.name}")
            if angles_csv.exists():
                angles_csv.unlink()
                logger.info(f"Deleted intermediate CSV: {angles_csv.name}")
        except Exception as e:
            logger.error(f"Error deleting intermediate CSVs: {e}")
        

        triangulation_files = list(predictions_dir.glob("**/*triangulation*.h5"))
        if triangulation_files:
            for tri_file in triangulation_files:
                try:
                    tri_file.unlink()
                    logger.info(f"Deleted: {tri_file}")
                except Exception as e:
                    logger.error(f"Error deleting {tri_file}: {e}")
        logging.info(f"Pose estimation completed for {session}.")



def run_pose_test(session, test_name = None, cameras=params.default_cameras, force_new_videos=False, 
                 start_frame=None, duration_seconds=10):
    """
    Runs a test of the pose estimation pipeline on short videos.
    """
    try:
        # 1. Create test videos
        logger.info("Creating test videos...")
        tests_dir = tools.create_test_videos(session, cameras, duration_seconds, 
                                    force_new=force_new_videos, start_frame=start_frame)
        
        if test_name is None:
            test_name = session + "_test"
        test_dir = tests_dir / test_name
        # 2. Run 2D predictions on test videos
        logger.info("Running 2D predictions...")
        sleapTools.get_2Dpredictions(session, cameras, test_name = test_name)
        
        # 3. Convert predictions to h5 format
        logger.info("Converting predictions to h5 format...")
        convert_2Dpred_to_h5(session, cameras, input_dir=test_dir, output_dir=test_dir)
        
        # Get calibration file for this session using the registry
        calib_file_path = get_calib_for_session(session)
        
        if calib_file_path is None:
            logging.error(
                f"Could not find calibration file for session {session}. "
                f"Please register a calibration using: bnp new-calib"
            )
            raise ValueError(f"No calibration found for session {session}")
        
        logging.info(f"Using calibration file: {calib_file_path} for session {session}")
        
        compute_3Dpredictions(
            session, calib_file_path=calib_file_path, pred_dir = test_dir, output_dir=test_dir, eval=False
        )
        labels_fname = test_dir/f"{session}_3dpts.csv"
        save_to_csv(
            session,
            test_dir/f"{session}_pose_estimation_combined.h5",
            labels_fname,
        )
        config_path = config.angles_config/"config.toml"
        if not config_path.exists():
            config_path = create_config_file(config_path)
        config_angles = toml.load(config_path)
        angles_csv = test_dir/f"{session}_angles.csv"
      

        compute_angles(config_angles, labels_fname, angles_csv)
        
        pose_data = pd.read_csv(labels_fname)
        angles_data = pd.read_csv(angles_csv)

        # Combine pose data and angles data
        combined_data = pd.concat([pose_data, angles_data], axis=1)
        combined_csv = test_dir/f"{session}_3dpts_angles.csv"
        # Save the updated CSV
        combined_data.to_csv(combined_csv, index=False)
        logging.info(f"Angles computed and combined CSV saved at {combined_csv}.")
        try:
            if labels_fname.exists():
                labels_fname.unlink()
                logger.info(f"Deleted intermediate CSV: {labels_fname.name}")
            if angles_csv.exists():
                angles_csv.unlink()
                logger.info(f"Deleted intermediate CSV: {angles_csv.name}")
        except Exception as e:
            logger.error(f"Error deleting intermediate CSVs: {e}")
        
        triangulation_files = list(tests_dir.glob("**/*triangulation*.h5"))
        if triangulation_files:
            for tri_file in triangulation_files:
                try:
                    tri_file.unlink()
                    logger.info(f"Deleted: {tri_file}")
                except Exception as e:
                    logger.error(f"Error deleting {tri_file}: {e}")
        logging.info(f"Pose estimation completed for {session}.")
        return test_dir

        
        

    except Exception as e:
        logger.error(f"Error in pose test for {session}: {e}")
        raise
