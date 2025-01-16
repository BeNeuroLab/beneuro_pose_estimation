import os
import sys

# Add the parent directory of 'scripts' to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
import beneuro_pose_estimation.anipose.aniposeTools as aniposeTools

recent_calib_folder = "/mnt/rds/bb2020/projects/beneuro/live/raw/pose-estimation/calibration-videos/camera_calibration_2024_11_20_11_25/Recording_2024-11-20T113135"
calib_file_path = "/home/il620/beneuro_pose_estimation/projects/calibrations/calibration_2024_11_20_11_25.toml"
aniposeTools.get_calib_file(recent_calib_folder, calib_file_path)
