from aniposelib.boards import CharucoBoard
from sleap.info.feature_suggestions import (
    FeatureSuggestionPipeline,
)
import cv2


############### CONFIGURATIONS

#### CAMERAS
default_cameras = [
    "Camera_Top_Left",
    "Camera_Side_Left",
    "Camera_Front_Right",
    "Camera_Front_Left",
    "Camera_Side_Right",
    "Camera_Back_Right",
]


camera_name_mapping = {
    "Camera_Top_Left": "camera_0",
    "Camera_Side_Left": "camera_1",
    "Camera_Front_Right": "camera_2",
    "Camera_Face": "camera_3",
    "Camera_Front_Left": "camera_4",
    "Camera_Side_Right": "camera_5",
    "Camera_Back_Right": "camera_6",
}

#### SLEAP config

## SLEAP annotation parameters
frame_selection_pipeline = FeatureSuggestionPipeline(
    per_video=100,
    scale=0.25,
    sample_method="stride",
    feature_type="brisk", # hog
    brisk_threshold=10,
    n_components=10,
    n_clusters=10,
    per_cluster=5,
)


## SLEAP tracking
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
    "right_wrist",
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
    18: "right_wrist",
}


constraints = [
    [0, 1],
    [0, 3],
    [2, 17],
    [16, 17],
    [1, 16],
    [5, 18],
    [4, 18],
    [6, 7],
    [6, 10],
    [7, 8],
    [8, 9],
    [10, 11],
    [11, 12],
    [6, 13],
    [13, 14],
    [14, 15],
]

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

board = cv2.aruco.CharucoBoard((5, 4), 10, 6, aruco_dict)
# board = CharucoBoard(
#     5, 4, square_length=10, marker_length=6, marker_bits=4, dict_size=250
# )
fisheye = False


# Triangulation parameters
triangulation_params = {
    "scale_smooth": 5,
    "scale_length": 4,
    "scale_length_weak": 1,
    "reproj_error_threshold": 5,
    "reproj_loss": "l2",
    "n_deriv_smooth": 2,
}
frame_window = 1000


# Angle calculation config
angles = {
    "right_knee": ["hip_center", "right_knee", "right_ankle"],
    "left_knee": ["hip_center", "left_knee", "left_ankle"],
    "right_ankle": ["right_knee", "right_ankle", "right_foot"],
    "left_ankle": ["left_knee", "left_ankle", "left_foot"],
    "right_wrist": ["right_elbow", "right_wrist", "right_paw"],
    "left_wrist": ["left_elbow", "left_wrist", "left_paw"],
    "right_elbow": ["right_shoulder", "right_elbow", "right_wrist"],
    "left_elbow": ["left_shoulder", "left_elbow", "left_wrist"],
}
