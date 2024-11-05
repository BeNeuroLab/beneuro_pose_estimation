"""
Module to carry out Anipose operations 
TBD
- not tested yet
- add angle calculation 
"""
import logging
from beneuro_pose_estimation import set_logging
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
import params
import sleapTools
import subprocess



def get_frame_count(h5_analysis_file):
    with h5py.File(h5_analysis_file, 'r') as f:
        # Access the 'track_occupancy' dataset to get frame count
        frame_count = f['track_occupancy'].shape[1]  # frames dimension

    return frame_count

def get_calib_file(calibration_videos_dir, calib_save_path, board=params.board):
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
    video_files = os.listdir(calib_videos_dir)
    cam_names, vidnames = [], []

    for video_file in video_files:
        if video_file.endswith('.avi') or video_file.endswith('.mp4'):
            cam_name = "_".join(video_file.split('_')[:3])
            vidnames.append([f"{calib_videos_dir}/{video_file}"])
            cam_names.append(cam_name)

    # Initialize and configure CharucoBoard and CameraGroup
    cgroup = CameraGroup.from_names(cam_names, fisheye=params.fisheye)
    cgroup.calibrate_videos(vidnames, board)
    cgroup.dump(calib_save_path)
    logging.info(f"Calibration file saved at {calib_save_path}")
    return 


def convert_2Dpred_to_h5(sessions, cameras=params.default_cameras, input_dir=params.predictions_path, output_dir=params.complete_projects_dir):
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
            output_file = f"{session_dir}/{session}_{camera}.analysis.h5"
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


def compute_3Dpredictions(session, project_dir, calib_file_path, body_parts=params.body_parts, constraints=params.constraints, frame_window=1000):
    """
    Triangulates 3D predictions from 2D predictions for each session in windows and then combines them.
    """
    cgroup = CameraGroup.load(calib_file_path)
    n_frames = get_frame_count(f"{project_dir}/{session}_{params.default_cameras[0]}.analysis.h5")
    windows = np.arange(0, n_frames, frame_window)
    windows = np.append(windows, n_frames)

    for start, end in zip(windows[:-1], windows[1:]):
        output_file = f"{project_dir}/{session}_triangulation_{start}_{end}.h5"
        slap.triangulate(
            p2d=project_dir, calib=calibration_file, fname=output_file,
            frames=(start, end), constraints=constraints, disp_progress=True
        )
        logging.info(f"3D prediction file created: {output_file}")

    combine_h5_files(session, windows, project_dir)


def combine_h5_files(session, windows, project_dir):
    """
    Combines multiple .h5 files into one.
    """
    combined_file = f"{project_dir}/{session}_pose_estimation_combined.h5"
    with h5py.File(combined_file, 'w') as combined_h5:
        for start, end in zip(windows[:-1], windows[1:]):
            fname = f"{project_dir}/{session}_triangulation_{start}_{end}.h5"
            if os.path.exists(fname):
                with h5py.File(fname, 'r') as f:
                    points3d_data = f['points3d'][:]
                    if 'points3d' not in combined_h5:
                        combined_dataset = combined_h5.create_dataset('points3d', data=points3d_data, maxshape=(None,) + points3d_data.shape[1:])
                    else:
                        combined_dataset.resize((combined_dataset.shape[0] + points3d_data.shape[0]), axis=0)
                        combined_dataset[-points3d_data.shape[0]:] = points3d_data
    logging.info(f"Combined 3D predictions saved at {combined_file}")


def save_to_csv(session, h5_file_path, csv_path):
    """
    Saves 3D prediction data to CSV format.
    """
    points3d = h5py.File(h5_file_path, 'r')['points3d'][:]
    points3d_flat = points3d.reshape((points3d.shape[0], -1))
    columns = [f"{part}_{axis}" for part in params.body_parts for axis in ("x", "y", "z")]
    pd.DataFrame(points3d_flat, columns=columns).to_csv(csv_path, index=False)
    logging.info(f"3D predictions saved to CSV at {csv_path}")


def extract_date(session_name):
    return "".join(session_name.split("_")[1:4])


def run_pose_estimation(sessions, log_file = None), projects_dir=params.complete_projects_dir):
    """
    Main routing from videos to 3D keypoints and angles.
    """
    set_logging(log_file)
    for session in sessions:
        logging.info(f"Running pose estimation on {session}")
        project_dir = f"{projects_dir}/{session}"
        os.makedirs(session_dir, exist_ok=True)
        sleapTools.get_2Dpredictions(session)
        convert_predictions_to_h5(session)

        session_date = extract_date(session)
        calib_file_path = f"{params.calibration_dir}/calibration_{session_date}.toml"
        calib_videos_dir = f"{params.calibration_vid_dir}_{session_date}"
        if not os.path.exists(calib_file_path):
            get_calib_file(calib_videos_dir,calib_file_path)
        
        compute_3Dpredictions(session, calib_file_path=calib_file_path, project_dir = project_dir)
        save_to_csv(f"{project_dir}/{session}_pose_estimation_combined.h5", f"{project_dir}/{session}_3d_predictions.csv")
        logging.info(f"Pose estimation completed for {session}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Run pose estimation for specified sessions.")
#     parser.add_argument('sessions', nargs='+', help="List of session names to process")
#     args = parser.parse_args()
#     run_pose_estimation(sessions=args.sessions)
