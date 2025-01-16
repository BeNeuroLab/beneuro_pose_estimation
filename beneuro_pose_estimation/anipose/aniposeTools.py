"""
Module to carry out Anipose operations 
TBD:
- test evaluate_reprojection
- check logging
-------------------------------
conda activate bnp
-> in dev:
python -m beneuro_pose_estimation.cli pose-estimation --sessions session_name(s) 
-> after package installation:
pose pose-estimation --sessions session_name(s) 

"""
import json
import matplotlib as plt
import logging
from beneuro_pose_estimation import params, set_logging
import beneuro_pose_estimation.sleap.sleapTools as sleapTools
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from pathlib import Path
import sleap
import numpy as np
import os
import h5py
from aniposelib.boards import CharucoBoard
from aniposelib.cameras import CameraGroup
import sleap_anipose as slap
import argparse
import pandas as pd
import subprocess
from datetime import datetime
from anipose.compute_angles import compute_angles
import toml



def evaluate_reprojection(reprojection_path, predictions_2D_dir,histogram_path = None):
    """
    Plots histogram of reprojection error.
    Not tested
    """
    # Generate histogram using data from the combined file
    detection_data = load_2D_predictions(predictions_2D_dir)
    reprojection_data = np.load(reprojection_path)
    slap.make_histogram(detection_data, reprojection_data, save_path=histogram_path)
    logging.info(f"Reprojection histogram saved at {histogram_path}")
        
    return 

def load_2D_predictions(predictions_2D_dir):
    '''
    TBD - need to check that the order of cameras is the same as in reprojections
    '''
    predictions_list = []
    cameras = params.default_cameras
    session = os.path.basename(predictions_2D_dir)
    for camera in cameras:
        h5_file = f"{predictions_2D_dir}/{session}_{camera}.analysis.h5"
        with h5py.File(h5_file, 'r') as f:
            # Shape of 'tracks': (1, 2, n_nodes, n_frames)
            tracks = f['tracks'][:]  # Read the data into memory
            tracks = np.squeeze(tracks)  # Remove the leading dimension if it is 1
            # Rearrange the dimensions to (n_frames, n_nodes, 2)
            tracks = np.moveaxis(tracks, 1, -1)
            predictions_list.append(tracks)

    # Combine predictions from all cameras along a new first dimension (n_cams)
    concatenated_predictions = np.stack(predictions_list, axis=0)

    return concatenated_predictions


def get_frame_count(h5_analysis_file):
    with h5py.File(h5_analysis_file, 'r') as f:
        return f['tracks'].shape[-1]

def get_most_recent_calib(session):
    # Parse session date and time
    try:
        session_datetime = datetime.strptime("_".join(session.split('_')[1:]), "%Y_%m_%d_%H_%M")
    except ValueError:
        logging.error(f"Invalid session format: {session}")
        return None

    # Iterate over all calibration folders and extract timestamps
    calib_folders = []
    calib_vid_dir = Path(params.calib_vid_dir)
    for folder in calib_vid_dir.iterdir():
        if folder.is_dir():
            try:
                # Extract the datetime from the "Recording_<datetime>" format
                calib_datetime = datetime.strptime("_".join(folder.name.split('_')[2:]), "%Y_%m_%d_%H_%M")
                calib_folders.append((calib_datetime, folder))
            except ValueError:
                logging.warning(f"Invalid calibration folder format: {folder.name}. Skipping.")
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

    logging.info(f"Using calibration folder: {recent_calib_folder} for session {session}")

    # Generate calibration file path
    calib_file_name = Path(f"calibration_{calib_datetime.strftime('%Y_%m_%d_%H_%M')}.toml")
    calib_file_path = Path(params.calibration_dir) / calib_file_name
    print(calib_file_path)
    # Create calibration configuration if it doesn't exist
    if not calib_file_path.exists():
        get_calib_file(recent_calib_folder, calib_file_path)
        logging.info(f"Created new calibration file: {calib_file_path}")
    else:
        logging.info(f"Calibration file already exists: {calib_file_path}")

    return calib_file_path

def get_calib_file(calib_videos_dir, calib_save_path, board=params.board):
    """
    Generates calibration file using ChArUco board videos. - get most recent calibration 

    Parameters
    ----------

    calib_videos_dir : str
        Directory path containing ChArUco videos for calibration 
    calib_save_path : str
        Path to save the calibration file
    board : CharucoBoard
        Configuration of the ChArUco board.

    -------

    """
     
    calib_videos_dir = next(calib_videos_dir.iterdir(), None) # might want to change this
    video_files = os.listdir(calib_videos_dir) 
    cam_names, vidnames = [], []
    reversed_mapping = {v: k for k, v in params.camera_name_mapping.items()}
    for video_file in video_files:
        if video_file.endswith('.avi') or video_file.endswith('.mp4'):
            # cam_name = "_".join(video_file.split('_')[:2])
            camera = Path(video_file).stem
            if camera =="Camera_3":
                continue
            cam_name = reversed_mapping.get(camera, camera)
            vidnames.append([f"{calib_videos_dir}/{video_file}"])
            cam_names.append(cam_name)
    
    # Initialize and configure CharucoBoard and CameraGroup
    cgroup = CameraGroup.from_names(cam_names, fisheye=params.fisheye)
    cgroup.calibrate_videos(vidnames, board)
    cgroup.dump(calib_save_path)
    logging.info(f"Calibration file saved at {calib_save_path}")
    return 


def convert_2Dpred_to_h5(sessions, cameras=params.default_cameras, input_dir=params.predictions_dir, output_dir=params.complete_projects_dir):
    """
    Converts .slp.predictions.slp files to .h5 analysis files for each session and camera.
    """
    if isinstance(sessions, str):
        sessions = [sessions]
    if isinstance(cameras, str):
        cameras = [cameras]
    for session in sessions:
        session_dir = f"{output_dir}/{session}"
        os.makedirs(session_dir, exist_ok=True)
        for camera in cameras:
            input_file = f"{input_dir}/{session}_{camera}.slp.predictions.slp"
            os.makedirs(f"{session_dir}/{camera}", exist_ok=True)
            output_file = f"{session_dir}/{camera}/{session}_{camera}.analysis.h5"
            if os.path.exists(output_file):
                logging.info(f"Output file {output_file} already exists. Skipping...")
            else:
                try:
                    subprocess.run(
                        ["sleap-convert", "--format", "analysis", "-o", output_file, input_file],
                        check=True
                    )
                    logging.info(f"Converted {input_file} to {output_file}")
                except subprocess.CalledProcessError as e:
                    logging.error(f"Error during conversion for {input_file}: {e}")


def compute_3Dpredictions(session, project_dir, calib_file_path, frame_window=params.frame_window,eval = False):
    """
    Triangulates 3D predictions from 2D predictions for each session in windows and then combines them.
    """
    cgroup = CameraGroup.load(calib_file_path)
    n_frames = get_frame_count(f"{project_dir}/{params.default_cameras[0]}/{session}_{params.default_cameras[0]}.analysis.h5")
    windows = np.arange(0, n_frames, frame_window)
    windows = np.append(windows, n_frames)
    reprojections_list = []  # List to store reprojections from each window

    for start, end in zip(windows[:-1], windows[1:]):
        print(f"Processing frames {start} to {end}")
        output_file = f"{project_dir}/{session}_triangulation_{start}_{end}.h5"
        
        slap.triangulate(
            p2d=project_dir,
            calib=calib_file_path,
            fname=output_file,
            disp_progress=True,
            frames=(start, end),
            constraints=params.constraints,
            scale_smooth=params.triangulation_params["scale_smooth"],
            scale_length=params.triangulation_params["scale_length"],
            scale_length_weak=params.triangulation_params["scale_length_weak"],
            reproj_error_threshold=params.triangulation_params["reproj_error_threshold"],
            reproj_loss=params.triangulation_params["reproj_loss"],
            n_deriv_smooth=params.triangulation_params["n_deriv_smooth"]
        )
        
        logging.info(f"3D prediction file created: {output_file}")

        if eval:
            reproj_output = slap.reproject(
                p3d=output_file,
                calib=calib_file_path,
                frames=(start, end)
            )
            logging.info(f"Reprojection created for frames {start} to {end}")
            reprojections_list.append(reproj_output)

    if reprojections_list:
        reprojections_array = np.concatenate(reprojections_list, axis=1)  # Concatenate along the frames axis
        logging.info(f"Reprojections concatenated with shape: {reprojections_array.shape}")

        save_path = f"{project_dir}/{session}_reprojections.npy"
        np.save(save_path, reprojections_array)
        logging.info(f"Reprojections saved to {save_path}")
        histogram_path = f"{project_dir}/{session}_reprojection_histogram.pdf"
        evaluate_reprojection(reprojection_path = save_path,predictions_2D_dir = project_dir, histogram_path = histogram_path)


    combine_h5_files(session, windows, project_dir,eval)
    
    
    


def combine_h5_files(session, windows, project_dir,eval = False):
    """
    Combines multiple .h5 files into one.
    """
    combined_file = f"{project_dir}/{session}_pose_estimation_combined.h5"
    with h5py.File(combined_file, 'w') as combined_h5:
        for start, end in zip(windows[:-1], windows[1:]):
            fname = f"{project_dir}/{session}_triangulation_{start}_{end}.h5"
            if os.path.exists(fname):
                with h5py.File(fname, 'r') as f:
                    # points3d_data = f['points3d'][:]
                    points3d_data = f['tracks']
                    if 'points3d' not in combined_h5:
                        combined_dataset = combined_h5.create_dataset('points3d', data=points3d_data, maxshape=(None,) + points3d_data.shape[1:])
                    else:
                        combined_dataset.resize((combined_dataset.shape[0] + points3d_data.shape[0]), axis=0)
                        combined_dataset[-points3d_data.shape[0]:] = points3d_data
            
    logging.info(f"Combined 3D predictions saved at {combined_file}")


def save_to_csv(session, h5_file_path, csv_path):
    """
    Saves 3D prediction data to CSV format.
    add _error, _score, _ncams,add fnum
    """
    points3d = h5py.File(h5_file_path, 'r')['points3d'][:]
    fnum = np.arange(points3d.shape[0])  # Generate frame numbers
    points3d_flat = points3d.reshape((points3d.shape[0], -1))

    # Prepare column names for the 3D points
    columns = [f"{part}_{axis}" for part in params.body_parts for axis in ("x", "y", "z")]

    # Create the base DataFrame with points3d and frame numbers
    df = pd.DataFrame(points3d_flat, columns=columns)
    df.insert(0, 'fnum', fnum)  # Add 'fnum' column as the first column

    # Prepare '_error' and '_score' columns for all body parts in bulk
    error_columns = {f"{part}_error": 0 for part in params.body_parts}  # All '_error' set to 0
    score_columns = {f"{part}_score": 1 for part in params.body_parts}  # All '_score' set to 1

    # Create DataFrames for error and score columns
    error_df = pd.DataFrame(error_columns, index=df.index)
    score_df = pd.DataFrame(score_columns, index=df.index)

    # Concatenate all DataFrames: df (points3d + fnum), error_df, and score_df
    df = pd.concat([df, error_df, score_df], axis=1)

    # Save the DataFrame to CSV
    df.to_csv(csv_path, index=False)
    logging.info(f"3D predictions saved to CSV at {csv_path}")


def get_body_part_connections(constraints, keypoints_dict):
    return [
        [keypoints_dict[start], keypoints_dict[end]] for start, end in constraints
    ]

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

def create_config_file(config_path, angles = params.angles):
    """
    Creates a configuration file in the project directory if it doesn't already exist.
    Uses parameters defined in params.py for setup.
    """

    if not os.path.isfile(config_path):
        config = {
            "angles": angles
        }
        with open(config_path, "w") as f:
            toml.dump(config, f)

        logging.info(f"Configuration file created at {config_path}")
    else:
        logging.info(f"Configuration file already exists at {config_path}")

    return config_path


def plot_behaviour(data_path, columns_to_plot, windwow = params.frame_window, frame_start = None, frame_end = None):
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


def run_pose_estimation(sessions, log_file = None, projects_dir=params.complete_projects_dir,videos_folder = None, eval = False):
    """
    Main routing from videos to 3D keypoints and angles.
    """
    # set_logging(log_file)
    if isinstance(sessions, str):
        sessions = [sessions]
    for session in sessions:
        logging.info(f"Running pose estimation on {session}")
        project_dir = f"{projects_dir}/{session}"
        os.makedirs(project_dir, exist_ok=True)
        sleapTools.get_2Dpredictions(session,input_file = videos_folder)
        convert_2Dpred_to_h5(session)
        ###############################################
        # calib_file_path = get_most_recent_calib("M045_2024_11_20_11_35")
        calib_file_path = get_most_recent_calib(session)
        compute_3Dpredictions(session, calib_file_path=calib_file_path, project_dir = project_dir,eval = eval)
        labels_fname = f"{project_dir}/{session}_3d_predictions.csv"
        save_to_csv(session,f"{project_dir}/{session}_pose_estimation_combined.h5", labels_fname)
        config_path = f"{project_dir}/config.toml"
        if not os.path.exists(config_path):
            config_path = create_config_file(config_path)
        config = toml.load(config_path)
        angles_csv = f"{project_dir}/{session}_angles.csv"
        labels_data = pd.read_csv(labels_fname)
        print(labels_data.columns)
        compute_angles(config,labels_fname, angles_csv  )
        logging.info(f"Pose estimation completed for {session}")
        pose_data = pd.read_csv(labels_fname)
        angles_data = pd.read_csv(angles_csv)

        # Combine pose data and angles data
        combined_data = pd.concat([pose_data, angles_data], axis=1)
        combined_csv = f"{project_dir}/{session}_pose_and_angles.csv"
        # Save the updated CSV
        combined_data.to_csv(combined_csv, index=False)
        logging.info(f"Angles computed and combined CSV saved at {combined_csv}")


