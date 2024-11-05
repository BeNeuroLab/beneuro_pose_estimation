from sleap.info.feature_suggestions import (
    FeatureSuggestionPipeline,
    ParallelFeaturePipeline,
)
from aniposelib.boards import CharucoBoard

####### CONFIGURATIONS

#### PATHS
repo_dir = "/home/il620/beneuro_pose_estimation"
recordings_dir = "/mnt/rds/bb2020/projects/beneuro/live/raw" 
# file format: "M043/M043_2024_10_23_11_15/M043_2024_10_23_11_15_cameras/M043_2024_10_23_11_15_camera_1.avi"
projects_dir = "/home/il620/beneuro_pose_estimation/beneuro_pose_estimation/projects/" #?

## SLEAP paths
slp_annotations_dir = "/home/il620/beneuro_pose_estimation/beneuro_pose_estimation/projects/annotations"
predictions_dir = "/home/il620/beneuro_pose_estimation/beneuro_pose_estimation/projects/predictions" #2D
slp_models_dir = "/mnt/rds/bb2020/projects/beneuro/live/raw/pose-estimation/models/h1_new_setup"
skeleton_path = f"{repo_path}/beneuro_pose_estimation/sleap/skeleton.json"
predicition_eval_dir = "/home/il620/beneuro_pose_estimation/beneuro_pose_estimation/predictions/evaluation"

input_2Dpred = slp_annotations_dir # can be recordings_dir or projects_dir or slp_annotations_dir

## Anipose paths
# path to 3D pose estimation directory
complete_projects_dir = "/home/il620/beneuro_pose_estimation/beneuro_pose_estimation/projects/complete_projects"
# path to calibration videos directory
calibration_vid_dir = "/home/il620/beneuro_pose_estimation/calibration-videos/ChAruCo_W5_H4" #?
# path to the calibration output file directory
calibration_dir = complete_projects_dir 




#### CAMERAS
default_cameras = [
                    "Camera_Top_Left",
                    "Camera_Side_Left",
                    "Camera_Front_Right",
                    "Camera_Front_Left",
                    "Camera_Side_Right",
                    "Camera_Back_Right"
                    ]
        

camera_name_mapping = {
    "Camera_Top_Left": "camera_0",
    "Camera_Side_Left": "camera_1",
    "Camera_Front_Right": "camera_2",
    "Camera_Face": "camera_3",
    "Camera_Front_Left": "camera_4",
    "Camera_Side_Right": "camera_5",
    "Camera_Back_Right": "camera_6"
}

#### SLEAP config

## SLEAP annotation
sessions_to_annotate = default_sessions
frame_selection_pipeline = FeatureSuggestionPipeline(
    per_video=50,
    scale=0.25,
    sample_method="stride",
    feature_type="hog",
    brisk_threshold=10,
    n_components=10,
    n_clusters=10,
    per_cluster=5,
    )
## SLEAP training
training_sessions = default_sessions
## SLEAP 2D predictions
sessions_to_predict = default_sessions

# SLEAP tracking 
frames_to_predict = None
tracking_options = None

#### ANIPOSE config

body_parts = [
                    "shoulder_center",
                    "left_shoulder",
                    "left_paw",
                    "right_shoulder",
                    "right_elbow",
                    "right_paw",
                    "hip_center",
                    "left_knee",
                    "left_ankle",
                    "left_foot",
                    "right_knee",
                    "right_ankle",
                    "right_foot",
                    "tail_base",
                    "tail_middle",
                    "tail_tip",
                    "left_elbow",
                    "left_wrist",
                    "right_wrist"
                ]

keypoints_dict = {
    0: "shoulder_center",
    1: "left_shoulder",
    2: "left_paw",
    3: "right_shoulder",
    4: "right_elbow",
    5: "right_paw",
    6: "hip_center",
    7: "left_knee",
    8: "left_ankle",
    9: "left_foot",
    10: "right_knee",
    11: "right_ankle",
    12: "right_foot",
    13: "tail_base",
    14: "tail_middle",
    15: "tail_tip",
    16: "left_elbow",
    17: "left_wrist",
    18: "right_wrist"
}


constraints = [[0,1],[0,3],[2,17],[16,17],[1,16],[5,18],[4,18],[6,7],[6,10],[7,8],[8,9],[10,11],[11,12],[6,13],[13,14],[14,15]]

board = CharucoBoard(5, 4, square_length=10, marker_length=6, marker_bits=4, dict_size=250)
fisheye = False


