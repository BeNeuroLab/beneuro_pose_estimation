import sys
import os

# Add the parent directory of 'scripts' to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
import beneuro_pose_estimation.anipose.aniposeTools as aniposeTools
from pathlib import Path
recent_calib_folder = Path("/mnt/rds/raw/pose-estimation/calibration-videos/camera_calibration_2026_02_03_10_00/")
calib_file_path = Path("/home/il620/repos/beneuro_pose_estimation/beneuro_pose_estimation/calibration_2026_02_03_10_00.toml")
aniposeTools.get_calib_file(recent_calib_folder, calib_file_path)


recent_calib_folder = Path("/mnt/rds/raw/pose-estimation/calibration-videos/camera_calibration_2026_01_21_15_00/")
calib_file_path = Path("/home/il620/repos/beneuro_pose_estimation/beneuro_pose_estimation/calibration_2026_01_21_15_00.toml")
aniposeTools.get_calib_file(recent_calib_folder, calib_file_path)