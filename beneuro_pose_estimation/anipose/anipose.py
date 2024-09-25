"""
Module to carry out anipose operations
"""
import logging
from pathlib import Path
from beneuro_pose_estimation import set_logging


def calibrate_cameras(path_to_charuco_video: Path):
    return


def compute_3d_predictions():
    """
    Function to run 3d predictions from 2d Sleap models

    Returns
    -------

    """
    return


def save_to_csv() -> None:
    return


def run_pose_estimation(session_name: str, log_file: str | None = None) -> None:
    """
    Main conversion routing from videos to 2d keypoints and angles

    Steps:

    1. Access videos from session
    2. Load sleap models (previously trained) and compute 2d prediction on the new sessions
    3. Use 2d predictions to compute 3D predictions
    3. Save to a csv.


    Parameters
    ----------
    session_name
    log_file


    Returns
    -------

    """

    set_logging(log_file)
    logging.info(f'Running pose estimation on {session_name}')

    return


if __name__ == '__main__':
    run_pose_estimation('M00_2024_01_01_10_00')